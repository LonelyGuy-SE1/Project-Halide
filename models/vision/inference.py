"""Vision inference pipeline. Takes a film scan and returns defect JSON.

The model emits bboxes in the [0, 999] integer grid (aligned with the VLM's
pre-training). This module converts them to normalized [0.0, 1.0] floats for
the downstream pipeline. If the model ever emits float 0-1 bboxes (e.g.,
before fine-tuning), we still accept them and pass them through unchanged.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from data.schemas import ALLOWED_LABELS
from models.vision.minicpm_wrapper import get_detector

logger = logging.getLogger(__name__)


def _normalize_bbox(bbox: list) -> list[float] | None:
    """Normalize a bbox to float [0, 1].

    Accepts either int 0-999 grid or float 0-1. Returns None if invalid.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x_min, y_min, x_max, y_max = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    if x_max <= x_min or y_max <= y_min:
        return None

    all_int = all(isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                  for v in bbox)
    max_val = max(x_min, y_min, x_max, y_max)

    if all_int and max_val > 1.5:
        scale = 999.0
    else:
        scale = 1.0

    if scale == 999.0:
        x_min /= 999.0
        y_min /= 999.0
        x_max /= 999.0
        y_max /= 999.0

    if not (0.0 <= x_min <= 1.0 and 0.0 <= y_min <= 1.0):
        return None
    if not (0.0 <= x_max <= 1.0 and 0.0 <= y_max <= 1.0):
        return None

    return [round(x_min, 6), round(y_min, 6), round(x_max, 6), round(y_max, 6)]


def extract_defects(image: Any) -> dict:
    """Run defect extraction on a PIL image. Returns defect dict + metadata."""
    started = time.perf_counter()
    detector = get_detector()
    raw = detector.detect(image)
    elapsed = time.perf_counter() - started

    defects = raw.get("defects", [])
    if not isinstance(defects, list):
        logger.warning("Model output 'defects' is not a list: %r", type(defects))
        defects = []

    cleaned: list[dict] = []
    dropped = 0
    for d in defects:
        if not isinstance(d, dict):
            dropped += 1
            continue
        label = d.get("label")
        bbox = d.get("bbox")
        if label not in ALLOWED_LABELS:
            dropped += 1
            continue
        norm = _normalize_bbox(bbox)
        if norm is None:
            dropped += 1
            continue
        cleaned.append({"label": label, "bbox": norm})

    label_counts: dict[str, int] = {}
    for d in cleaned:
        label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1

    return {
        "defects": cleaned,
        "defect_count": len(cleaned),
        "label_counts": label_counts,
        "dropped_count": dropped,
        "inference_seconds": round(elapsed, 3),
        "model_path": detector.model_path,
    }


def extract_defects_from_path(image_path: str | Path) -> dict:
    """Convenience: open image from path and run extraction."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    return extract_defects(img)
