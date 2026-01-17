"""
config.py

Loads configuration from .env and environment variables.

IMPORTANT:
- This file is intentionally explicit and verbose.
- Defaults are provided so scripts still run even without .env.
- Uses separate output dirs for each pipeline:
    OUTPUT_DIR_A1, OUTPUT_DIR_A2, OUTPUT_DIR_ENHANCE
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv


def _get_bool(name: str, default: bool) -> bool:
    """
    Parse boolean environment variables.

    Accepts common values:
      true/false, 1/0, yes/no (case-insensitive)
    """
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "t")


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def _get_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v


# Load .env once at import time (simple and common for scripts).
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    # Paths
    input_image: str
    output_dir_a1: str
    output_dir_a2: str
    output_dir_enhance: str

    # Crop size
    crop_width: int
    crop_height: int

    # Shared fallback crop
    face_x_margin_ratio: float
    face_top_margin_ratio: float
    face_bottom_margin_ratio: float

    # Approach 1
    min_face_size: int
    min_upper_size: int
    upperbody_pad_ratio: float
    nms_iou_thresh: float

    # Approach 2
    person_crop_top_ratio: float
    person_crop_bottom_ratio: float
    person_pad_ratio: float

    # Enhancement
    upscale_2x: bool
    gamma_target_mean: float
    clahe_clip: float
    clahe_grid: Tuple[int, int]
    edge_preserve_sigma_s: int
    edge_preserve_sigma_r: float
    canny_t1: int
    canny_t2: int
    sharpen_sigma: float
    blur_thresh_blurry: float
    blur_thresh_ok: float


def load_config() -> AppConfig:
    """
    Create a config object from environment variables.

    This keeps config centralized and prevents “config drift”
    between scripts.

    Backward-compat note:
      If OUTPUT_DIR is set (single output path), scripts can still work
      even if OUTPUT_DIR_A1/A2/ENHANCE are not set.
    """
    output_dir_common = _get_str("OUTPUT_DIR", "outputs/photo")

    return AppConfig(
        input_image=_get_str("INPUT_IMAGE", "/content/group_photo.jpg"),

        output_dir_a1=_get_str("OUTPUT_DIR_A1", output_dir_common),
        output_dir_a2=_get_str("OUTPUT_DIR_A2", output_dir_common),
        output_dir_enhance=_get_str("OUTPUT_DIR_ENHANCE", output_dir_common),

        crop_width=_get_int("CROP_WIDTH", 600),
        crop_height=_get_int("CROP_HEIGHT", 800),

        face_x_margin_ratio=_get_float("FACE_X_MARGIN_RATIO", 0.60),
        face_top_margin_ratio=_get_float("FACE_TOP_MARGIN_RATIO", 0.50),
        face_bottom_margin_ratio=_get_float("FACE_BOTTOM_MARGIN_RATIO", 1.60),

        min_face_size=_get_int("MIN_FACE_SIZE", 40),
        min_upper_size=_get_int("MIN_UPPER_SIZE", 80),
        upperbody_pad_ratio=_get_float("UPPERBODY_PAD_RATIO", 0.08),
        nms_iou_thresh=_get_float("NMS_IOU_THRESH", 0.35),

        person_crop_top_ratio=_get_float("PERSON_CROP_TOP_RATIO", 0.00),
        person_crop_bottom_ratio=_get_float("PERSON_CROP_BOTTOM_RATIO", 0.70),
        person_pad_ratio=_get_float("PERSON_PAD_RATIO", 0.05),

        upscale_2x=_get_bool("UPSCALE_2X", True),
        gamma_target_mean=_get_float("GAMMA_TARGET_MEAN", 135.0),
        clahe_clip=_get_float("CLAHE_CLIP", 2.0),
        clahe_grid=(
            _get_int("CLAHE_GRID_W", 8),
            _get_int("CLAHE_GRID_H", 8),
        ),
        edge_preserve_sigma_s=_get_int("EDGE_PRESERVE_SIGMA_S", 60),
        edge_preserve_sigma_r=_get_float("EDGE_PRESERVE_SIGMA_R", 0.40),
        canny_t1=_get_int("CANNY_T1", 60),
        canny_t2=_get_int("CANNY_T2", 120),
        sharpen_sigma=_get_float("SHARPEN_SIGMA", 1.0),
        blur_thresh_blurry=_get_float("BLUR_THRESH_BLURRY", 80.0),
        blur_thresh_ok=_get_float("BLUR_THRESH_OK", 150.0),
    )
