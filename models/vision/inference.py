"""Vision inference pipeline. Takes a film scan and returns defect JSON."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from models.vision.minicpm_wrapper import get_detector

logger = logging.getLogger(__name__)

ALLOWED_LABELS = {"dust", "dirt", "scratch", "long_hair", "short_hair"}


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
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            dropped += 1
            continue
        try:
            x_min, y_min, x_max, y_max = (float(v) for v in bbox)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (0.0 <= x_min <= 1.0 and 0.0 <= y_min <= 1.0):
            dropped += 1
            continue
        if not (0.0 <= x_max <= 1.0 and 0.0 <= y_max <= 1.0):
            dropped += 1
            continue
        if x_max <= x_min or y_max <= y_min:
            dropped += 1
            continue
        cleaned.append({"label": label, "bbox": [x_min, y_min, x_max, y_max]})

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
