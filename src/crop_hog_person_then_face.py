"""
approach2_hog_person_then_face_ULTRA_DETAILED.py

================================================================================
WHAT THIS SCRIPT DOES (END-TO-END)
================================================================================
Goal:
  From a single "group photo", create one "contact-style" portrait crop per person:
    - face + some upper-body (head + torso region)
    - consistent size (e.g., 600x800) for downstream enhancement and UI display

Constraint (2022/23 realistic):
  No modern foundation vision models (no YOLOv8 / RetinaFace / etc.)
  We use classical / traditional OpenCV techniques:
    1) HOG + Linear SVM (OpenCV default people detector)
    2) Haar cascade face detector
    3) Simple geometric assignment + cropping heuristics

High-level logic:
  1) Detect person boxes using HOG+SVM (rectangles that roughly cover a human body).
  2) Detect face boxes using Haar (rectangles over faces).
  3) For each detected face, find the *most plausible* person box:
       - choose the smallest person box that contains the face center
       - "smallest" avoids assigning a face to a huge box that covers multiple people
  4) Crop:
       - If person box found: crop top part of person box => natural "contact photo"
       - Else: fallback crop by expanding face box (common production trick)
  5) Save:
       - crops/person_XX_crop.jpg
       - debug/overlay.jpg (draw person + face boxes)
       - report.json + report.csv (metadata for reproducibility)

Outputs:
  OUTPUT_DIR/
    debug/overlay.jpg
    crops/person_01_crop.jpg, person_02_crop.jpg, ...
    report.json
    report.csv

================================================================================
IMPORTANT REAL-WORLD NOTE ABOUT HOG PERSON DETECTOR
================================================================================
The HOG person detector works best when the full/upper body is visible and
people are not extremely close together.

In tightly cropped group photos:
  - Faces may be detected reliably (Haar), but
  - Person boxes may be missed (HOG)

Thats why this pipeline ALWAYS includes a fallback crop from face bbox.

================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Config + utils
# -----------------------------------------------------------------------------
try:
    from .config import load_config
    from .utils import clamp, ensure_dir, write_csv, write_json
except ImportError:  # Allows running directly: python src/crop_hog_person_then_face.py
    from config import load_config
    from utils import clamp, ensure_dir, write_csv, write_json


cfg = load_config()


# =============================================================================
# CONFIG SECTION (VERY IMPORTANT)
# =============================================================================
# These parameters control how detection + cropping behave.
# If results are not good, you tune THESE first.

# -----------------------------------------------------------------------------
# Input / Output paths
# -----------------------------------------------------------------------------
INPUT_IMAGE = cfg.input_image
OUTPUT_DIR = cfg.output_dir_a2


# -----------------------------------------------------------------------------
# Face detection config (Haar)
# -----------------------------------------------------------------------------
MIN_FACE_SIZE = cfg.min_face_size


# -----------------------------------------------------------------------------
# Output crop resolution
# -----------------------------------------------------------------------------
CROP_WIDTH = cfg.crop_width
CROP_HEIGHT = cfg.crop_height


# -----------------------------------------------------------------------------
# Person-driven crop settings (when a person box is found)
# -----------------------------------------------------------------------------
PERSON_CROP_TOP_RATIO = cfg.person_crop_top_ratio
PERSON_CROP_BOTTOM_RATIO = cfg.person_crop_bottom_ratio
PERSON_PAD_RATIO = cfg.person_pad_ratio


# -----------------------------------------------------------------------------
# Face-based fallback crop (when no person box is found)
# -----------------------------------------------------------------------------
FACE_X_MARGIN_RATIO = cfg.face_x_margin_ratio
FACE_TOP_MARGIN_RATIO = cfg.face_top_margin_ratio
FACE_BOTTOM_MARGIN_RATIO = cfg.face_bottom_margin_ratio


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hog_person_then_face_ultra")


# =============================================================================
# FACE DETECTION (HAAR CASCADE)
# =============================================================================
def load_face_cascade() -> cv2.CascadeClassifier:
    """
    Load the Haar face detector from OpenCV's built-in haarcascades directory.

    If clf.empty() is True => cascade file not found or OpenCV install issue.
    """
    path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"Failed to load face cascade: {path}")
    return clf


def detect_faces(
    img_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    min_face: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image.

    Input:
      img_bgr      : color image in BGR format (OpenCV default)
      face_cascade : Haar cascade object
      min_face     : minimum face size in pixels

    Output:
      List of face bboxes in (x, y, w, h), sorted left-to-right.
      Left-to-right sorting ensures stable person_01, person_02 naming.
    """

    # Haar detectors operate on grayscale intensity patterns, not color.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Histogram equalization improves contrast, helps in shadows/uneven lighting.
    gray = cv2.equalizeHist(gray)

    # detectMultiScale:
    # - scans image at multiple scales
    # - returns possible face rectangles
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,               # pyramid scaling step
        minNeighbors=5,                # false positive control
        minSize=(min_face, min_face),  # ignore smaller boxes
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Convert to Python ints for safety and consistency.
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    # Sort by x coordinate => stable ordering by left-to-right in image.
    boxes.sort(key=lambda b: b[0])
    return boxes


# =============================================================================
# PERSON DETECTION (HOG + SVM)
# =============================================================================
def detect_people_hog(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect people using OpenCV's built-in HOG + SVM person detector.

    What is HOG?
      Histogram of Oriented Gradients:
        - captures edge/gradient structure typical of human silhouette.
      A linear SVM classifier is trained to classify "person" vs "not person".

    Pros:
      - classical and widely used in 2010s/early 2020s
      - no GPU required

    Cons:
      - misses people if bodies are small, partially visible, or crowded
      - can produce false positives in clutter

    Output:
      List of person bboxes (x, y, w, h), sorted left-to-right.
    """

    # Create HOG descriptor object
    hog = cv2.HOGDescriptor()

    # Attach pretrained people detector weights that OpenCV ships with
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detectMultiScale:
    # - similar concept: scan at multiple scales
    # winStride controls scanning step size
    # padding adds margin around window to help detection
    # scale controls pyramid scale step
    rects, weights = hog.detectMultiScale(
        img_bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    # rects: array of (x, y, w, h)
    people = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

    # Sort for stable debug rendering
    people.sort(key=lambda b: b[0])
    return people


# =============================================================================
# FACE -> PERSON ASSIGNMENT
# =============================================================================
def face_center(face_xywh: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Compute center point of face box.

    If face bbox is (x,y,w,h):
      center_x = x + w/2
      center_y = y + h/2

    Why use face center?
      It's a robust reference point that should lie inside the person's bbox.
    """
    x, y, w, h = face_xywh
    return (x + w // 2, y + h // 2)


def point_inside_box(pt: Tuple[int, int], box_xywh: Tuple[int, int, int, int]) -> bool:
    """
    Check if a point lies inside a rectangle box.

    pt = (px, py)
    box = (x, y, w, h)

    returns True if:
      x <= px <= x+w  AND  y <= py <= y+h
    """
    px, py = pt
    x, y, w, h = box_xywh
    return (x <= px <= x + w) and (y <= py <= y + h)


def assign_face_to_person(
    face_xywh: Tuple[int, int, int, int],
    people_xywh: List[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Assign each face to the most plausible person bbox.

    Step-by-step:
      1) Compute face center (cx, cy).
      2) Find all person boxes that contain this center.
         (If none contain it => return None)
      3) Choose the smallest area person box among candidates.
         Why smallest?
           If there is a giant person box covering multiple people,
           smallest box is more likely to correspond to the actual person.

    Returns:
      person bbox (x,y,w,h) or None.
    """
    c = face_center(face_xywh)

    # Filter only those person boxes where face center lies inside
    candidates = [p for p in people_xywh if point_inside_box(c, p)]

    # If no candidates => HOG failed / person bbox not covering face
    if not candidates:
        return None

    # Choose smallest bbox area => best specific match
    candidates.sort(key=lambda b: b[2] * b[3])
    return candidates[0]


# =============================================================================
# CROPPING
# =============================================================================
def crop_from_face_fallback(img_bgr: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Fallback cropping method when no person box is available.

    It expands face bbox into a contact-photo crop.

    Steps:
      1) Compute expansion margins relative to face size.
      2) Expand bbox.
      3) Clamp to image boundaries.
      4) Crop and resize.

    Output:
      Standardized crop of shape (CROP_HEIGHT, CROP_WIDTH, 3).
    """
    img_h, img_w = img_bgr.shape[:2]
    x, y, w, h = face_box

    # Calculate margins
    x_margin = int(FACE_X_MARGIN_RATIO * w)
    top_margin = int(FACE_TOP_MARGIN_RATIO * h)
    bottom_margin = int(FACE_BOTTOM_MARGIN_RATIO * h)

    # Expand crop coordinates
    x1 = max(0, x - x_margin)
    y1 = max(0, y - top_margin)
    x2 = min(img_w, x + w + x_margin)
    y2 = min(img_h, y + h + bottom_margin)

    # Defensive: if invalid bbox, use original face bbox
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = x, y, min(img_w, x + w), min(img_h, y + h)

    crop = img_bgr[y1:y2, x1:x2]

    # Defensive: return blank if crop failed (prevents crashing)
    if crop.size == 0:
        return np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)

    return cv2.resize(crop, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)


def crop_from_person(img_bgr: np.ndarray, person_box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Person-driven cropping when we have a person bbox.

    Why crop only upper portion?
      Person bbox may include legs/background.
      For a contact photo, we want:
        head + torso, not full body.

    Steps:
      1) Compute y1,y2 inside the person bbox using ratios.
      2) Use full width of person bbox.
      3) Add padding.
      4) Clamp to image boundaries.
      5) Crop and resize.
    """
    img_h, img_w = img_bgr.shape[:2]
    x, y, w, h = person_box

    # Vertical crop boundaries inside person bbox
    # Example: 0.00 -> top, 0.70 -> 70% height
    y1 = y + int(PERSON_CROP_TOP_RATIO * h)
    y2 = y + int(PERSON_CROP_BOTTOM_RATIO * h)

    # Horizontal crop uses whole person bbox width
    x1 = x
    x2 = x + w

    # Add padding around crop region
    pad = int(PERSON_PAD_RATIO * max(w, h))
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    # Clamp crop coordinates to stay within image
    x1 = clamp(x1, 0, img_w)
    x2 = clamp(x2, 0, img_w)
    y1 = clamp(y1, 0, img_h)
    y2 = clamp(y2, 0, img_h)

    # If bbox became invalid, return blank to avoid crashing
    if x2 <= x1 or y2 <= y1:
        return np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)

    return cv2.resize(crop, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)


# =============================================================================
# DEBUG OVERLAY IMAGE
# =============================================================================
def draw_overlay(
    img_bgr: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    people: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Create a debug image where we visualize detections.

    - People boxes: ORANGE
    - Face boxes: GREEN + label face_XX

    Why helpful?
      If results are wrong, debug overlay tells you:
        - HOG missing people?
        - Haar faces correct?
        - false positives?
    """
    out = img_bgr.copy()

    # Draw people bboxes
    for (x, y, w, h) in people:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)

    # Draw face bboxes
    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"face_{i:02d}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return out


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main() -> None:
    """
    End-to-end steps:
      1) Validate input image exists
      2) Create output folders
      3) Read image (cv2.imread)
      4) Load face cascade
      5) Detect people (HOG) + faces (Haar)
      6) Save debug overlay
      7) For each face:
           - assign to person bbox
           - crop using person (or fallback)
           - save crop
           - store metadata
      8) Save report.json and report.csv
    """

    # Convert string paths to Path objects for safe path operations
    in_path = Path(INPUT_IMAGE)
    out_dir = Path(OUTPUT_DIR)

    # (1) Validate input
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # (2) Create output directories
    ensure_dir(out_dir)
    ensure_dir(out_dir / "debug")
    ensure_dir(out_dir / "crops")

    # (3) Read image
    img = cv2.imread(str(in_path))
    if img is None:
        raise ValueError(f"OpenCV failed to read: {in_path}")

    # (4) Load Haar face cascade
    face_cascade = load_face_cascade()

    # (5) Detect people and faces
    # People boxes come from HOG+SVM
    people = detect_people_hog(img)

    # Face boxes come from Haar
    faces = detect_faces(img, face_cascade, MIN_FACE_SIZE)

    logger.info("People(HOG)=%d | Faces(Haar)=%d", len(people), len(faces))

    # (6) Save debug overlay image
    overlay = draw_overlay(img, faces, people)
    cv2.imwrite(str(out_dir / "debug" / "overlay.jpg"), overlay)

    # (7) Generate a crop for each detected face
    rows: List[Dict] = []

    for i, face in enumerate(faces, start=1):
        pid = f"person_{i:02d}"

        # Try to assign face to a person bbox
        person_match = assign_face_to_person(face, people)

        # Report dict: keep stable schema with fixed keys
        rec: Dict = {
            "person_id": pid,

            # Face bbox (always exists for each row because we loop over faces)
            "face_bbox_x": int(face[0]),
            "face_bbox_y": int(face[1]),
            "face_bbox_w": int(face[2]),
            "face_bbox_h": int(face[3]),

            # Person bbox info (may be None)
            "person_found": bool(person_match is not None),
            "used_person_crop": bool(person_match is not None),
            "person_bbox_x": None,
            "person_bbox_y": None,
            "person_bbox_w": None,
            "person_bbox_h": None,

            # Crop path (filled after saving)
            "crop_path": None,
        }

        # Choose cropping method
        if person_match is not None:
            # Use person-driven crop (upper part of person bbox)
            crop = crop_from_person(img, person_match)

            # Store person bbox in report
            px, py, pw, ph = person_match
            rec["person_bbox_x"] = int(px)
            rec["person_bbox_y"] = int(py)
            rec["person_bbox_w"] = int(pw)
            rec["person_bbox_h"] = int(ph)
        else:
            # Fallback: expand face bbox to contact crop
            crop = crop_from_face_fallback(img, face)
            rec["used_person_crop"] = False

        # Save crop
        crop_path = out_dir / "crops" / f"{pid}_crop.jpg"
        cv2.imwrite(str(crop_path), crop)

        # Store relative path (so report is portable if you move the folder)
        rec["crop_path"] = str(crop_path.relative_to(out_dir))

        rows.append(rec)

    # (8) Save reports
    write_json(out_dir / "report.json", rows)
    write_csv(out_dir / "report.csv", rows)

    logger.info("Saved %d crop(s) -> %s", len(rows), str(out_dir / "crops"))
    logger.info("Done -> %s", str(out_dir))


def run_crop_hog_person_then_face(input_image_path: str, output_dir: str) -> List[Dict]:
    """
    API-friendly wrapper around the pipeline.

    IMPORTANT:
    - We keep original main() untouched.
    - This function runs identical pipeline steps but uses caller-provided paths.

    Returns:
      rows: list of dict records (same as report.json content)
    """
    in_path = Path(input_image_path)
    out_dir = Path(output_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    ensure_dir(out_dir)
    ensure_dir(out_dir / "debug")
    ensure_dir(out_dir / "crops")

    img = cv2.imread(str(in_path))
    if img is None:
        raise ValueError(f"OpenCV failed to read: {in_path}")

    face_cascade = load_face_cascade()

    people = detect_people_hog(img)
    faces = detect_faces(img, face_cascade, MIN_FACE_SIZE)

    logger.info("People(HOG)=%d | Faces(Haar)=%d", len(people), len(faces))

    overlay = draw_overlay(img, faces, people)
    cv2.imwrite(str(out_dir / "debug" / "overlay.jpg"), overlay)

    rows: List[Dict] = []

    for i, face in enumerate(faces, start=1):
        pid = f"person_{i:02d}"

        person_match = assign_face_to_person(face, people)

        rec: Dict = {
            "person_id": pid,
            "face_bbox_x": int(face[0]),
            "face_bbox_y": int(face[1]),
            "face_bbox_w": int(face[2]),
            "face_bbox_h": int(face[3]),
            "person_found": bool(person_match is not None),
            "used_person_crop": bool(person_match is not None),
            "person_bbox_x": None,
            "person_bbox_y": None,
            "person_bbox_w": None,
            "person_bbox_h": None,
            "crop_path": None,
        }

        if person_match is not None:
            crop = crop_from_person(img, person_match)
            px, py, pw, ph = person_match
            rec["person_bbox_x"] = int(px)
            rec["person_bbox_y"] = int(py)
            rec["person_bbox_w"] = int(pw)
            rec["person_bbox_h"] = int(ph)
        else:
            crop = crop_from_face_fallback(img, face)
            rec["used_person_crop"] = False

        crop_path = out_dir / "crops" / f"{pid}_crop.jpg"
        cv2.imwrite(str(crop_path), crop)
        rec["crop_path"] = str(crop_path.relative_to(out_dir))

        rows.append(rec)

    write_json(out_dir / "report.json", rows)
    write_csv(out_dir / "report.csv", rows)

    logger.info("Saved %d crop(s) -> %s", len(rows), str(out_dir / "crops"))
    logger.info("Done -> %s", str(out_dir))

    return rows


if __name__ == "__main__":
    main()
