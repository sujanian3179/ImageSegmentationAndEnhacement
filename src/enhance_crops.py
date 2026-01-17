"""
02_enhance_crops_BETTER_ULTRA_DETAILED.py  (NO CLI args; hardcoded paths)

================================================================================
WHAT THIS SCRIPT DOES
================================================================================
This script enhances ("improves quality of") cropped portrait images produced by
your Step-1 cropping pipeline (Haar-only / Haar+Upperbody / HOG+Face).

It expects:
  OUTPUT_DIR/crops/*.jpg

It produces:
  OUTPUT_DIR/enhanced/*.jpg               # enhanced versions of each crop
  OUTPUT_DIR/enhance_report.json          # per-image enhancement metadata
  OUTPUT_DIR/enhance_report.csv           # same metadata in tabular form

================================================================================
WHY THIS IS 2022/23-REALISTIC
================================================================================
No modern foundation vision models are used.

Everything is done using classical image processing techniques:
  1) Exposure correction via Adaptive Gamma (LUT-based)
  2) Contrast correction via CLAHE (only on luminance channel LAB->L)
  3) White balance via Gray-World assumption
  4) Noise reduction via edge-preserving smoothing
  5) Sharpening via edge-only unsharp mask (avoid sharpening skin noise)
  6) Optional upscaling via bicubic interpolation (not SR, still classical)

================================================================================
IMPORTANT REALITY CHECK (INTERVIEWER-FRIENDLY)
================================================================================
Enhancement cannot magically "restore" missing details.
If the crop is very blurry or low-resolution, classical methods can:
  - improve perceived clarity
  - improve exposure
  - reduce noise
  - make edges look sharper
But it cannot reconstruct true lost texture like modern SR models.

This is why we:
  - sharpen adaptively using blur score
  - avoid oversharpening skin
  - keep parameters conservative (portfolio-friendly, not artifact-heavy)
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Config + utils
# -----------------------------------------------------------------------------
try:
    from .config import load_config
    from .utils import ensure_dir, write_csv, write_json
except ImportError:  # Allows running directly: python src/enhance_crops.py
    from config import load_config
    from utils import ensure_dir, write_csv, write_json


cfg = load_config()


# =============================================================================
# CONFIG SECTION (THIS IS WHERE YOU TUNE RESULTS)
# =============================================================================

# -----------------------------------------------------------------------------
# OUTPUT_DIR: where your crops exist
# -----------------------------------------------------------------------------
OUTPUT_DIR = cfg.output_dir_enhance


# -----------------------------------------------------------------------------
# UPSCALE_2X: optional resolution increase
# -----------------------------------------------------------------------------
UPSCALE_2X = cfg.upscale_2x


# -----------------------------------------------------------------------------
# GAMMA_TARGET_MEAN: adaptive exposure correction
# -----------------------------------------------------------------------------
GAMMA_TARGET_MEAN = cfg.gamma_target_mean


# -----------------------------------------------------------------------------
# CLAHE settings: contrast on luminance only
# -----------------------------------------------------------------------------
CLAHE_CLIP = cfg.clahe_clip
CLAHE_GRID: Tuple[int, int] = cfg.clahe_grid


# -----------------------------------------------------------------------------
# Edge-preserving smoothing: noise reduction without killing edges
# -----------------------------------------------------------------------------
EDGE_PRESERVE_SIGMA_S = cfg.edge_preserve_sigma_s
EDGE_PRESERVE_SIGMA_R = cfg.edge_preserve_sigma_r


# -----------------------------------------------------------------------------
# Edge sharpening settings
# -----------------------------------------------------------------------------
CANNY_T1 = cfg.canny_t1
CANNY_T2 = cfg.canny_t2
SHARPEN_SIGMA = cfg.sharpen_sigma


# -----------------------------------------------------------------------------
# Blur thresholds for adaptive sharpening
# -----------------------------------------------------------------------------
BLUR_THRESH_BLURRY = cfg.blur_thresh_blurry
BLUR_THRESH_OK = cfg.blur_thresh_ok


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("enhance_crops_better_ultra")


# =============================================================================
# METRICS (MEASUREMENTS)
# =============================================================================
def blur_score_variance_of_laplacian(img_bgr: np.ndarray) -> float:
    """
    Compute a classic blur metric: Variance of Laplacian.

    How it works:
      - Laplacian operator detects edges (second derivative)
      - If image is sharp: edges have strong responses => high variance
      - If image is blurry: edges are weak => low variance

    Output:
      A single float.
      Higher => sharper, lower => blurrier.

    Why used here?
      We use it to decide how much sharpening to apply.
    """
    # Convert to grayscale (edge operators work on intensity)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian; CV_64F avoids overflow and keeps precision
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    # Variance is used as "sharpness score"
    return float(lap.var())


# =============================================================================
# ENHANCEMENT BUILDING BLOCKS
# =============================================================================
def gamma_correct(img_bgr: np.ndarray, target_mean: float) -> np.ndarray:
    """
    Adaptive Gamma Correction.

    Purpose:
      Fix exposure more naturally than simply increasing brightness.

    Steps:
      1) Compute mean brightness of grayscale image.
      2) Compute ratio = mean / target_mean
         - If mean < target => ratio < 1 => image is darker than desired.
      3) gamma = clamp(ratio, 0.60..1.60)
         - this avoids extreme over/under correction.
      4) Use LUT (lookup table) to apply gamma efficiently:
         new_pixel = (pixel/255)^(1/gamma) * 255

    Key intuition:
      - gamma < 1 brightens shadows more gently
      - gamma > 1 darkens highlights slightly
    """

    # 1) Compute mean brightness from grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())

    # 2) ratio to decide direction of correction
    ratio = max(1e-6, mean_val / float(target_mean))

    # 3) Bound gamma to avoid artifacts
    gamma = float(np.clip(ratio, 0.60, 1.60))

    # 4) Create LUT for fast mapping of [0..255]
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)

    # Apply LUT to each channel identically
    return cv2.LUT(img_bgr, lut)


def clahe_on_luminance(img_bgr: np.ndarray, clip_limit: float, grid: Tuple[int, int]) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance.

    Why only luminance?
      If we apply contrast enhancement directly on RGB/BGR channels,
      colors can shift unnaturally. LAB separates brightness (L) from color (A,B).

    Steps:
      1) Convert BGR -> LAB
      2) Split channels: L, A, B
      3) Apply CLAHE to L only
      4) Merge back and convert LAB -> BGR

    clip_limit:
      Controls how aggressive local contrast is.
      Too high => halos/noise amplification.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-World White Balance.

    Assumption:
      In a well-balanced image, average color should be neutral gray.
      So we scale channels so that mean(B) ~= mean(G) ~= mean(R).

    Steps:
      1) Convert to float to avoid overflow
      2) Compute mean of each channel
      3) Compute target mean = average of the three means
      4) Scale each channel to match the target
      5) Clip and convert back to uint8

    Pros:
      Simple, fast, classical.
    Cons:
      If the image naturally has strong dominant color, may overcorrect slightly.
    """
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)

    b_mean, g_mean, r_mean = float(b.mean()), float(g.mean()), float(r.mean())
    target = (b_mean + g_mean + r_mean) / 3.0

    b = b * (target / (b_mean + 1e-6))
    g = g * (target / (g_mean + 1e-6))
    r = r * (target / (r_mean + 1e-6))

    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def edge_preserving_smooth(img_bgr: np.ndarray, sigma_s: int, sigma_r: float) -> np.ndarray:
    """
    Edge-preserving smoothing (OpenCV's edgePreservingFilter).

    Why?
      We want to reduce noise but preserve facial edges (eyes, lips, hairline).
      Normal blur removes noise but also blurs edges (bad for portraits).

    Parameters:
      sigma_s: spatial smoothing (size of neighborhood)
      sigma_r: range smoothing (how much intensities can differ before smoothing stops)

    Output:
      A denoised image with edges preserved.
    """
    return cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)


def edge_only_sharpen(
    img_bgr: np.ndarray,
    canny_t1: int,
    canny_t2: int,
    sharpen_sigma: float,
    sharpen_amount: float,
) -> np.ndarray:
    """
    Edge-only sharpening using unsharp mask + edge mask.

    Problem:
      If we sharpen everywhere, we also sharpen noise on skin => ugly texture.
    Solution:
      Sharpen only edges (eyes/hair/lips) using an edge mask.

    Steps:
      1) Convert to grayscale
      2) Use Canny to detect edges => binary edge map
      3) Dilate edge map slightly (thicken edges so sharpening affects nearby pixels)
      4) Create "sharp" image using unsharp mask:
           sharp = img*(1+amount) + blurred*(-amount)
      5) Replace pixels in original image only where edge map is True

    Output:
      Sharper edges, less noise amplification in smooth skin areas.
    """
    # 1) grayscale for edge detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) Canny edges
    edges = cv2.Canny(gray, canny_t1, canny_t2)

    # 3) Dilate edges to cover a small neighborhood (more natural sharpening)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 4) Unsharp mask computation
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sharpen_sigma, sigmaY=sharpen_sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)

    # 5) Apply sharpened pixels only on edge areas
    mask = edges > 0
    out = img_bgr.copy()
    out[mask] = sharp[mask]
    return out


def upscale_2x(img_bgr: np.ndarray) -> np.ndarray:
    """
    2x Upscaling using bicubic interpolation.

    Why it helps portfolio:
      Upscaled images look "higher quality" and are easier to view.
    But:
      It does NOT invent true details like SR models.
      It mainly smooths and makes pixels bigger.
    """
    return cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


# =============================================================================
# FULL PIPELINE (ENHANCE ONE IMAGE)
# =============================================================================
def enhance_portrait(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Enhance a single portrait crop.

    Returns:
      enhanced_img: final enhanced image
      debug_info : dict containing useful metrics & decisions for reporting

    Pipeline order (why this order):
      1) Gamma correction: fix exposure first (so later steps work on better lighting)
      2) CLAHE on luminance: improve local contrast (safer after gamma)
      3) White balance: correct color cast (better after contrast)
      4) Edge-preserving smooth: reduce noise after contrast/wb (otherwise noise gets boosted)
      5) Edge-only sharpen: restore crispness on eyes/hair (after denoise)
      6) Optional upscale: done at end so it doesn't interfere with edge detection
    """
    debug: Dict = {}

    # Measure blur before enhancement (for decision-making and reporting)
    b0 = blur_score_variance_of_laplacian(img_bgr)
    debug["blur_before"] = float(b0)

    # -------------------------------------------------------------------------
    # 1) Exposure correction (Adaptive Gamma)
    # -------------------------------------------------------------------------
    out = gamma_correct(img_bgr, target_mean=GAMMA_TARGET_MEAN)

    # -------------------------------------------------------------------------
    # 2) Contrast enhancement (CLAHE on L)
    # -------------------------------------------------------------------------
    out = clahe_on_luminance(out, clip_limit=CLAHE_CLIP, grid=CLAHE_GRID)

    # -------------------------------------------------------------------------
    # 3) White balance (gray world)
    # -------------------------------------------------------------------------
    out = gray_world_white_balance(out)

    # -------------------------------------------------------------------------
    # 4) Noise reduction (edge-preserving)
    # -------------------------------------------------------------------------
    out = edge_preserving_smooth(out, sigma_s=EDGE_PRESERVE_SIGMA_S, sigma_r=EDGE_PRESERVE_SIGMA_R)

    # -------------------------------------------------------------------------
    # 5) Adaptive sharpening:
    #    Use blur score to choose sharpen strength.
    # -------------------------------------------------------------------------
    if b0 < BLUR_THRESH_BLURRY:
        # very blurry => stronger sharpening (still edge-only)
        sharpen_amount = 1.35
        debug["sharpen_level"] = "high"
    elif b0 < BLUR_THRESH_OK:
        # moderately sharp => medium sharpening
        sharpen_amount = 1.05
        debug["sharpen_level"] = "medium"
    else:
        # already sharp => low sharpening to avoid halos
        sharpen_amount = 0.75
        debug["sharpen_level"] = "low"

    out = edge_only_sharpen(
        out,
        canny_t1=CANNY_T1,
        canny_t2=CANNY_T2,
        sharpen_sigma=SHARPEN_SIGMA,
        sharpen_amount=sharpen_amount,
    )

    # -------------------------------------------------------------------------
    # 6) Optional upscale (portfolio-friendly)
    # -------------------------------------------------------------------------
    if UPSCALE_2X:
        out = upscale_2x(out)
        debug["upscale_2x"] = True
    else:
        debug["upscale_2x"] = False

    # Measure blur after processing (not perfect metric, but useful)
    b1 = blur_score_variance_of_laplacian(out)
    debug["blur_after"] = float(b1)

    return out, debug


# =============================================================================
# MAIN (BATCH PROCESS CROPS)
# =============================================================================
def run() -> None:
    """
    Batch enhancement pipeline:
      1) Verify crops folder exists
      2) Create enhanced folder
      3) Read each crop
      4) Enhance it
      5) Save enhanced image
      6) Write per-image metadata to JSON + CSV
    """
    out_dir = Path(OUTPUT_DIR)
    crops_dir = out_dir / "crops"

    # (1) Validate that crops folder exists
    # This folder should be created by step-1 crop script.
    if not crops_dir.exists():
        raise ValueError(
            f"Missing crops folder: {crops_dir}. "
            f"Run step-1 first so OUTPUT_DIR/crops exists."
        )

    # (2) Create output folder for enhanced results
    enhanced_dir = out_dir / "enhanced"
    ensure_dir(enhanced_dir)

    # (3) Gather crop files
    crop_files = sorted(crops_dir.glob("*.jpg"))

    logger.info("Enhancing crops from: %s", str(crops_dir))
    logger.info("Found %d crop(s).", len(crop_files))

    rows: List[Dict] = []

    for f in crop_files:
        # Read crop image
        img = cv2.imread(str(f))
        if img is None:
            # If unreadable, skip to prevent crash
            logger.warning("Skipping unreadable crop: %s", f.name)
            continue

        # Enhance one image + get debug info
        enhanced, dbg = enhance_portrait(img)

        # Output naming convention:
        #   person_01_crop.jpg -> person_01_enhanced.jpg
        out_name = f.name.replace("_crop", "_enhanced")
        out_path = enhanced_dir / out_name

        # Save enhanced image
        cv2.imwrite(str(out_path), enhanced)

        # Save metadata row for report
        rows.append(
            {
                "crop_file": f.name,
                "enhanced_file": out_name,
                "crop_path": str(f.relative_to(out_dir)),
                "enhanced_path": str(out_path.relative_to(out_dir)),
                "blur_before": dbg.get("blur_before"),
                "blur_after": dbg.get("blur_after"),
                "sharpen_level": dbg.get("sharpen_level"),
                "upscale_2x": dbg.get("upscale_2x"),
            }
        )

    # Write reports
    write_json(out_dir / "enhance_report.json", rows)
    write_csv(out_dir / "enhance_report.csv", rows)

    logger.info("Saved enhanced images to: %s", str(enhanced_dir))
    logger.info("Reports: %s , %s", str(out_dir / "enhance_report.json"), str(out_dir / "enhance_report.csv"))
    logger.info("Done.")

def run_enhance_crops(output_dir: str) -> List[Dict]:
    """
    API-friendly wrapper around the enhancer.

    IMPORTANT:
    - Keeps the original run() method unchanged.
    - This function runs the same batch enhancement but uses caller-provided output_dir.

    Returns:
      rows: list of dict records (same as enhance_report.json content)
    """
    out_dir = Path(output_dir)
    crops_dir = out_dir / "crops"

    # (1) Validate that crops folder exists
    if not crops_dir.exists():
        raise ValueError(
            f"Missing crops folder: {crops_dir}. "
            f"Run crop step first so OUTPUT_DIR/crops exists."
        )

    # (2) Create output folder for enhanced results
    enhanced_dir = out_dir / "enhanced"
    ensure_dir(enhanced_dir)

    # (3) Gather crop files
    crop_files = sorted(crops_dir.glob("*.jpg"))

    logger.info("Enhancing crops from: %s", str(crops_dir))
    logger.info("Found %d crop(s).", len(crop_files))

    rows: List[Dict] = []

    for f in crop_files:
        img = cv2.imread(str(f))
        if img is None:
            logger.warning("Skipping unreadable crop: %s", f.name)
            continue

        enhanced, dbg = enhance_portrait(img)

        out_name = f.name.replace("_crop", "_enhanced")
        out_path = enhanced_dir / out_name

        cv2.imwrite(str(out_path), enhanced)

        rows.append(
            {
                "crop_file": f.name,
                "enhanced_file": out_name,
                "crop_path": str(f.relative_to(out_dir)),
                "enhanced_path": str(out_path.relative_to(out_dir)),
                "blur_before": dbg.get("blur_before"),
                "blur_after": dbg.get("blur_after"),
                "sharpen_level": dbg.get("sharpen_level"),
                "upscale_2x": dbg.get("upscale_2x"),
            }
        )

    write_json(out_dir / "enhance_report.json", rows)
    write_csv(out_dir / "enhance_report.csv", rows)

    logger.info("Saved enhanced images to: %s", str(enhanced_dir))
    logger.info("Reports: %s , %s", str(out_dir / "enhance_report.json"), str(out_dir / "enhance_report.csv"))
    logger.info("Done.")

    return rows


if __name__ == "__main__":
    run()
