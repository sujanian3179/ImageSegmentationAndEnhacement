"""
utils.py

Shared helper utilities reused across scripts.

IMPORTANT:
- These functions are moved from the original scripts.
- Their docstrings and explanatory comments are intentionally preserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def ensure_dir(p: Path) -> None:
    """
    Ensures that directory 'p' exists.

    - parents=True  => create parent directories if needed
    - exist_ok=True => do not throw error if it already exists
    """
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    """
    Save any Python object (dict/list) into JSON file.

    indent=2 makes it human readable.
    ensure_ascii=False keeps unicode characters intact.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict]) -> None:
    """
    Write list of dicts to CSV.

    Why we do "union of keys":
      Some rows may contain optional keys (like upper_bbox_*),
      and csv.DictWriter will crash if later dict has new keys.

    So:
      fieldnames = union of all keys across all rows
    """
    if not rows:
        return

    # union of all keys across rows => stable schema
    fieldnames = sorted({k for r in rows for k in r.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def clamp(v: int, lo: int, hi: int) -> int:
    """
    Clamp integer v to [lo, hi].

    Used to ensure crop coordinates never go out-of-bounds.
    """
    return max(lo, min(hi, v))
