"""
api.py

FastAPI server exposing:
1) /crop    -> run cropping pipeline (Approach 1 or Approach 2), store outputs
2) /enhance -> run enhancement pipeline on stored crops, store outputs

Input style:
- user sends filesystem paths (already stored on server machine)
- server reads/writes on disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import pipeline wrapper functions (must exist in these modules)
try:
    from .crop_haar_face_upperbody import run_crop_face_upperbody
    from .crop_hog_person_then_face import run_crop_hog_person_then_face
    from .enhance_crops import run_enhance_crops
except ImportError:
    from crop_haar_face_upperbody import run_crop_face_upperbody
    from crop_hog_person_then_face import run_crop_hog_person_then_face
    from enhance_crops import run_enhance_crops


# IMPORTANT: Uvicorn expects this variable name (unless you specify a different one).
app = FastAPI(title="Group Photo Crop + Enhance API", version="1.0.0")


class CropRequest(BaseModel):
    image_path: str = Field(..., description="Path to stored group image on server disk.")
    output_dir: str = Field(..., description="Output folder to store crop results.")
    approach: Literal["a1", "a2"] = Field("a1", description="Cropping approach selector.")


class EnhanceRequest(BaseModel):
    output_dir: str = Field(..., description="Folder containing crops/ to enhance.")


def _validate_file_exists(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {p}")
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {p}")
    return p


def _validate_or_create_dir(path_str: str) -> Path:
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {p}")
    return p


@app.post("/crop")
def crop(req: CropRequest):
    img_path = _validate_file_exists(req.image_path)
    out_dir = _validate_or_create_dir(req.output_dir)

    try:
        if req.approach == "a1":
            rows = run_crop_face_upperbody(str(img_path), str(out_dir))
        else:
            rows = run_crop_hog_person_then_face(str(img_path), str(out_dir))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cropping failed: {type(e).__name__}: {e}")

    return {
        "status": "ok",
        "approach": req.approach,
        "input_image": str(img_path),
        "output_dir": str(out_dir),
        "debug_overlay": str(out_dir / "debug" / "overlay.jpg"),
        "crops_dir": str(out_dir / "crops"),
        "report_json": str(out_dir / "report.json"),
        "report_csv": str(out_dir / "report.csv"),
        "num_crops": len(rows),
    }


@app.post("/enhance")
def enhance(req: EnhanceRequest):
    out_dir = _validate_or_create_dir(req.output_dir)
    crops_dir = out_dir / "crops"

    if not crops_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Missing crops folder: {crops_dir}. Run /crop first or create crops/ manually.",
        )

    try:
        rows = run_enhance_crops(str(out_dir))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {type(e).__name__}: {e}")

    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "crops_dir": str(out_dir / "crops"),
        "enhanced_dir": str(out_dir / "enhanced"),
        "report_json": str(out_dir / "enhance_report.json"),
        "report_csv": str(out_dir / "enhance_report.csv"),
        "num_enhanced": len(rows),
    }
