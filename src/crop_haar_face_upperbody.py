"""
approach1_face_plus_upperbody_ULTRA_DETAILED.py

Approach 1 (Classical CV):
1) Detect faces using Haar cascade.
2) Detect upper bodies using Haar cascade.
3) For each face, find the upper-body detection that overlaps the face the most.
4) Crop using union(face, upperbody) + padding.
5) If no upperbody is matched, fallback to expanding face bbox using heuristics.

Outputs:
  OUTPUT_DIR/
    debug/overlay.jpg
    crops/person_XX_crop.jpg
    report.json
    report.csv

This version contains very explicit comments INSIDE every function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------
# We use .env (via src/config.py) to store configuration.
# This keeps "hardcoded constants" out of the script while preserving readability.
try:
    from .config import load_config
    from .utils import ensure_dir, write_csv, write_json
except ImportError:  # Allows running directly: python src/crop_haar_face_upperbody.py
    from config import load_config
    from utils import ensure_dir, write_csv, write_json


cfg = load_config()


# =============================================================================
# CONFIG
# =============================================================================
INPUT_IMAGE = cfg.input_image
OUTPUT_DIR = cfg.output_dir_a1

MIN_FACE_SIZE = cfg.min_face_size
MIN_UPPER_SIZE = cfg.min_upper_size

CROP_WIDTH = cfg.crop_width
CROP_HEIGHT = cfg.crop_height

# Used only if upper-body detection is unavailable
FACE_X_MARGIN_RATIO = cfg.face_x_margin_ratio
FACE_TOP_MARGIN_RATIO = cfg.face_top_margin_ratio
FACE_BOTTOM_MARGIN_RATIO = cfg.face_bottom_margin_ratio

# Padding used when upperbody is available
UPPERBODY_PAD_RATIO = cfg.upperbody_pad_ratio  # 8%

# Used to remove duplicate detections
NMS_IOU_THRESH = cfg.nms_iou_thresh


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("A1_face_upperbody_ultra")


# =============================================================================
# DATA STRUCTURE
# =============================================================================
@dataclass(frozen=True)
class Det:
    """
    Represents a detection box.

    box: (x, y, w, h)  => top-left corner (x,y) + width w + height h
    score: a confidence score. Haar cascade doesn't return a confidence score,
           so we just keep 1.0 and use it for stable sorting in NMS.
    """
    box: Tuple[int, int, int, int]
    score: float = 1.0


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================
def to_xyxy(box_xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Convert a box from (x, y, w, h) format to (x1, y1, x2, y2) format.

    Why?
      Many overlap computations are easier in (x1,y1,x2,y2) format.

    Example:
      (x=10, y=20, w=30, h=40)
      => (x1=10, y1=20, x2=40, y2=60)
    """
    x, y, w, h = box_xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute IoU (Intersection over Union) between two boxes a and b.

    Inputs:
      a, b are arrays [x1, y1, x2, y2]

    IoU = area(intersection(a,b)) / area(union(a,b))

    IoU is between 0 and 1:
      0  => no overlap
      1  => perfect overlap (identical boxes)

    This is used in NMS to remove near-duplicate detections.
    """

    # 1) Compute coordinates of the intersection rectangle:
    # Intersection's top-left corner is max of top-left coords
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))

    # Intersection's bottom-right corner is min of bottom-right coords
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    # 2) Compute width and height of intersection rectangle
    # If boxes do not overlap, x2 < x1 or y2 < y1.
    # We clamp to 0 to avoid negative sizes.
    iw = max(0.0, x2 - x1 + 1.0)
    ih = max(0.0, y2 - y1 + 1.0)

    # Intersection area
    inter = iw * ih

    # 3) Compute area of each box separately
    # Area = width * height
    area_a = max(0.0, float(a[2] - a[0] + 1.0)) * max(0.0, float(a[3] - a[1] + 1.0))
    area_b = max(0.0, float(b[2] - b[0] + 1.0)) * max(0.0, float(b[3] - b[1] + 1.0))

    # 4) Union area = area_a + area_b - intersection_area
    # Add tiny epsilon to avoid divide-by-zero.
    union = area_a + area_b - inter + 1e-9

    # 5) IoU = inter / union
    return inter / union


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """
    Non-Max Suppression (NMS).

    Problem:
      Haar detector can return multiple overlapping boxes for the same face/upperbody.

    Solution (NMS):
      - Sort detections by score descending
      - Keep the highest-score box
      - Remove all other boxes that overlap too much (IoU > threshold)
      - Repeat until no boxes left

    Returns:
      A list of indices of boxes to keep.
    """
    if boxes_xyxy.size == 0:
        return []

    # Sort indices by score in descending order
    # scores.argsort() gives ascending indices, so [::-1] reverses
    order = scores.argsort()[::-1]

    keep: List[int] = []

    while order.size > 0:
        # Take the first index in 'order' => highest score among remaining
        i = int(order[0])
        keep.append(i)

        # If this was the last box, we are done
        if order.size == 1:
            break

        # Remaining indices after removing the chosen one
        rest = order[1:]

        # Compute IoU of chosen box i with every other remaining box
        ious = np.array([iou_xyxy(boxes_xyxy[i], boxes_xyxy[j]) for j in rest], dtype=np.float32)

        # Keep only those boxes whose IoU <= threshold
        # i.e., remove boxes that overlap too much with the best one
        rest = rest[ious <= iou_thresh]

        # Continue with only the remaining boxes
        order = rest

    return keep


def intersection_area_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    """
    Compute intersection area (not IoU) between two boxes in XYXY format.

    We use intersection area (instead of IoU) to match face to upperbody:
      "Choose the upperbody box that covers the face the most."

    Returns:
      intersection area in pixels (integer).
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Intersection rectangle coords:
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    # Width/height clamped to 0 if no overlap
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    return int(iw * ih)


# =============================================================================
# CASCADE LOADING
# =============================================================================
def load_cascade(filename: str) -> cv2.CascadeClassifier:
    """
    Load a Haar cascade classifier from OpenCV built-in path.

    OpenCV stores these XML files in:
      cv2.data.haarcascades

    If classifier fails to load, clf.empty() is True.
    """
    cascade_path = str(Path(cv2.data.haarcascades) / filename)

    clf = cv2.CascadeClassifier(cascade_path)

    # If empty => file missing or OpenCV install broken
    if clf.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    return clf


# =============================================================================
# HAAR DETECTION
# =============================================================================
def detect_haar(img_bgr: np.ndarray, cascade: cv2.CascadeClassifier, min_size: int) -> List[Det]:
    """
    Run Haar cascade detection and return cleaned detections.

    Steps:
    (A) Convert BGR image to grayscale.
        Haar features (Haar-like rectangles) operate on intensity patterns.
    (B) Equalize histogram:
        - improves contrast
        - can help detection in uneven lighting
    (C) detectMultiScale():
        - scans the image at multiple scales
        - returns a list of bounding boxes
    (D) Apply NMS to remove duplicates
    (E) Sort left-to-right so person IDs remain stable.
    """

    # (A) Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # (B) Contrast normalization
    gray = cv2.equalizeHist(gray)

    # (C) Run detector
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,             # how much we shrink image at each pyramid step
        minNeighbors=5,              # higher => stricter detection, fewer false positives
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Convert raw boxes to Det objects
    dets = [Det((int(x), int(y), int(w), int(h))) for (x, y, w, h) in boxes]

    # (D) Deduplicate with NMS
    if len(dets) > 1:
        # Convert to XYXY arrays for IoU computations
        xyxy_list = []
        score_list = []
        for d in dets:
            x, y, w, h = d.box
            xyxy_list.append([x, y, x + w, y + h])
            score_list.append(d.score)

        xyxy_arr = np.array(xyxy_list, dtype=np.float32)
        score_arr = np.array(score_list, dtype=np.float32)

        keep_idx = nms(xyxy_arr, score_arr, iou_thresh=NMS_IOU_THRESH)
        dets = [dets[i] for i in keep_idx]

    # (E) Stable ordering: sort by x coordinate (left to right)
    dets.sort(key=lambda d: d.box[0])

    return dets


def match_upperbody(face_box: Tuple[int, int, int, int], upper_boxes: List[Det]) -> Optional[Tuple[int, int, int, int]]:
    """
    Match a face box to the best upperbody box.

    Logic:
      For each upperbody detection:
        - compute overlap area with the face region
      choose the upperbody with the largest overlap area.

    If no overlap area is > 0, return None.
    """

    # Convert face box to XYXY for overlap calculation
    face_xyxy = to_xyxy(face_box)

    best_upper: Optional[Tuple[int, int, int, int]] = None
    best_overlap = 0

    for ub in upper_boxes:
        # Convert upperbody to XYXY format
        ub_xyxy = to_xyxy(ub.box)

        # Compute overlap pixels between face and this upperbody
        overlap = intersection_area_xyxy(face_xyxy, ub_xyxy)

        # Keep the upperbody with maximum overlap
        if overlap > best_overlap:
            best_overlap = overlap
            best_upper = ub.box

    # If overlap is zero => no meaningful match
    if best_overlap <= 0:
        return None

    return best_upper


# =============================================================================
# CROPPING
# =============================================================================
def crop_from_face_fallback(img_bgr: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Fallback method when upperbody is missing/unmatched.

    Idea:
      Expand the face bounding box to include:
        - headroom above
        - shoulders + torso below
        - shoulders on left/right

    This uses ratios relative to face size.
    """

    img_h, img_w = img_bgr.shape[:2]
    x, y, w, h = face_box

    # Compute expansion margins from face dimensions
    x_margin = int(FACE_X_MARGIN_RATIO * w)
    top_margin = int(FACE_TOP_MARGIN_RATIO * h)
    bottom_margin = int(FACE_BOTTOM_MARGIN_RATIO * h)

    # Expand coordinates, then clamp to image boundaries
    x1 = max(0, x - x_margin)
    y1 = max(0, y - top_margin)
    x2 = min(img_w, x + w + x_margin)
    y2 = min(img_h, y + h + bottom_margin)

    # Defensive check: if bbox became invalid, use original face bbox
    if x2 <= x1 or y2 <= y1:
        x1, y1 = x, y
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

    # Crop region from image
    crop = img_bgr[y1:y2, x1:x2]

    # If crop is empty (rare), return blank image to prevent crash
    if crop.size == 0:
        return np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)

    # Resize to standard output size
    return cv2.resize(crop, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)


def crop_smart(
    img_bgr: np.ndarray,
    face_box: Tuple[int, int, int, int],
    upper_box: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Preferred method: use upperbody box if available.

    Steps if upper_box exists:
      1) Compute union rectangle that contains BOTH face and upperbody boxes.
      2) Add a small padding around union rectangle.
      3) Clamp to image bounds.
      4) Crop and resize.

    If upper_box is None:
      Use face fallback method.
    """

    # If no upperbody -> fallback to face expansion
    if upper_box is None:
        return crop_from_face_fallback(img_bgr, face_box)

    img_h, img_w = img_bgr.shape[:2]

    # Unpack face + upperbody boxes
    fx, fy, fw, fh = face_box
    ux, uy, uw, uh = upper_box

    # Union rectangle coordinates
    # top-left: min of both top-lefts
    x1 = min(fx, ux)
    y1 = min(fy, uy)

    # bottom-right: max of both bottom-rights
    x2 = max(fx + fw, ux + uw)
    y2 = max(fy + fh, uy + uh)

    # Padding derived from upperbody size (more stable than padding from face size)
    pad = int(UPPERBODY_PAD_RATIO * max(uw, uh))

    # Expand union with padding and clamp to image boundaries
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)

    return cv2.resize(crop, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)


# =============================================================================
# DEBUG OVERLAY
# =============================================================================
def draw_overlay(img_bgr: np.ndarray, faces: List[Det], uppers: List[Det]) -> np.ndarray:
    """
    Create a debug image:
      - draw upperbody boxes in BLUE
      - draw face boxes in GREEN and label person_XX

    This helps visually verify:
      - Are faces being detected correctly?
      - Are upper bodies detected?
      - Are there false positives?
    """
    out = img_bgr.copy()

    # Draw upperbodies
    for ub in uppers:
        x, y, w, h = ub.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw faces + IDs
    for i, f in enumerate(faces, start=1):
        x, y, w, h = f.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"person_{i:02d}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return out


# =============================================================================
# MAIN FOR CALLING IN SCRIPTS AND CLI
# =============================================================================
def main() -> None:
    """
    End-to-end pipeline:
    1) Validate input path
    2) Create output directories
    3) Read image
    4) Load cascades
    5) Detect faces + upper bodies
    6) Save debug overlay
    7) For each face:
         - match upperbody
         - crop smart
         - save crop
         - log metadata
    8) Save report.json and report.csv
    """

    in_path = Path(INPUT_IMAGE)
    out_dir = Path(OUTPUT_DIR)

    # Validate input exists
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Create output folders
    ensure_dir(out_dir)
    ensure_dir(out_dir / "debug")
    ensure_dir(out_dir / "crops")

    # Read image using OpenCV
    img = cv2.imread(str(in_path))
    if img is None:
        raise ValueError(f"OpenCV failed to read: {in_path}")

    # Load Haar cascades
    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    upper_cascade = load_cascade("haarcascade_upperbody.xml")

    # Detect faces and upper bodies
    faces = detect_haar(img, face_cascade, MIN_FACE_SIZE)
    uppers = detect_haar(img, upper_cascade, MIN_UPPER_SIZE)

    logger.info("Faces=%d | Upperbodies=%d", len(faces), len(uppers))

    # Save overlay debug image
    overlay = draw_overlay(img, faces, uppers)
    cv2.imwrite(str(out_dir / "debug" / "overlay.jpg"), overlay)

    rows: List[Dict] = []

    # For each detected face (person), produce a crop
    for i, face_det in enumerate(faces, start=1):
        pid = f"person_{i:02d}"

        # Match this face to an upperbody detection (if any overlaps)
        matched_upper = match_upperbody(face_det.box, uppers)

        # Crop contact photo using the best available method
        crop = crop_smart(img, face_det.box, matched_upper)

        # Save crop
        crop_path = out_dir / "crops" / f"{pid}_crop.jpg"
        cv2.imwrite(str(crop_path), crop)

        # Record metadata
        fx, fy, fw, fh = face_det.box

        rec: Dict = {
            "person_id": pid,
            "face_bbox_x": int(fx),
            "face_bbox_y": int(fy),
            "face_bbox_w": int(fw),
            "face_bbox_h": int(fh),
            "upperbody_found": bool(matched_upper is not None),
            "upper_bbox_x": None,
            "upper_bbox_y": None,
            "upper_bbox_w": None,
            "upper_bbox_h": None,
            "crop_path": str(crop_path.relative_to(out_dir)),
        }

        # If upperbody matched, store its bbox too
        if matched_upper is not None:
            ux, uy, uw, uh = matched_upper
            rec["upper_bbox_x"] = int(ux)
            rec["upper_bbox_y"] = int(uy)
            rec["upper_bbox_w"] = int(uw)
            rec["upper_bbox_h"] = int(uh)

        rows.append(rec)

    # Save reports
    write_json(out_dir / "report.json", rows)
    write_csv(out_dir / "report.csv", rows)

    logger.info("Saved %d crops -> %s", len(rows), str(out_dir / "crops"))
    logger.info("Done -> %s", str(out_dir))

# =============================================================================
# API-FRIENDLY WRAPPER
# =============================================================================

def run_crop_face_upperbody(input_image_path: str, output_dir: str) -> List[Dict]:
    """
    API-friendly wrapper around the pipeline.

    IMPORTANT:
    - We do NOT remove the original main() or its comments.
    - This function simply runs the same logic with caller-provided paths.
    - All detection/cropping behavior remains identical (uses module constants).

    Returns:
      rows: list of dict records (same as report.json content)
    """
    in_path = Path(input_image_path)
    out_dir = Path(output_dir)

    # Validate input exists
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Create output folders
    ensure_dir(out_dir)
    ensure_dir(out_dir / "debug")
    ensure_dir(out_dir / "crops")

    # Read image using OpenCV
    img = cv2.imread(str(in_path))
    if img is None:
        raise ValueError(f"OpenCV failed to read: {in_path}")

    # Load Haar cascades
    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    upper_cascade = load_cascade("haarcascade_upperbody.xml")

    # Detect faces and upper bodies
    faces = detect_haar(img, face_cascade, MIN_FACE_SIZE)
    uppers = detect_haar(img, upper_cascade, MIN_UPPER_SIZE)

    logger.info("Faces=%d | Upperbodies=%d", len(faces), len(uppers))

    # Save overlay debug image
    overlay = draw_overlay(img, faces, uppers)
    cv2.imwrite(str(out_dir / "debug" / "overlay.jpg"), overlay)

    rows: List[Dict] = []

    # For each detected face (person), produce a crop
    for i, face_det in enumerate(faces, start=1):
        pid = f"person_{i:02d}"

        matched_upper = match_upperbody(face_det.box, uppers)
        crop = crop_smart(img, face_det.box, matched_upper)

        crop_path = out_dir / "crops" / f"{pid}_crop.jpg"
        cv2.imwrite(str(crop_path), crop)

        fx, fy, fw, fh = face_det.box

        rec: Dict = {
            "person_id": pid,
            "face_bbox_x": int(fx),
            "face_bbox_y": int(fy),
            "face_bbox_w": int(fw),
            "face_bbox_h": int(fh),
            "upperbody_found": bool(matched_upper is not None),
            "upper_bbox_x": None,
            "upper_bbox_y": None,
            "upper_bbox_w": None,
            "upper_bbox_h": None,
            "crop_path": str(crop_path.relative_to(out_dir)),
        }

        if matched_upper is not None:
            ux, uy, uw, uh = matched_upper
            rec["upper_bbox_x"] = int(ux)
            rec["upper_bbox_y"] = int(uy)
            rec["upper_bbox_w"] = int(uw)
            rec["upper_bbox_h"] = int(uh)

        rows.append(rec)

    write_json(out_dir / "report.json", rows)
    write_csv(out_dir / "report.csv", rows)

    logger.info("Saved %d crops -> %s", len(rows), str(out_dir / "crops"))
    logger.info("Done -> %s", str(out_dir))

    return rows


if __name__ == "__main__":
    main()
