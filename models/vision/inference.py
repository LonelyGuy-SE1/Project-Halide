"""Vision inference pipeline. Takes a film scan and returns defect JSON."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from config import get_vision_config
from data.schemas import clean_defects, dedupe_defects, label_counts
from data.preprocessing import load_image
from models.vision.minicpm_wrapper import get_detector

logger = logging.getLogger(__name__)


def extract_defects(image: Any) -> dict:
    """Run defect extraction on a PIL image. Returns defect dict + metadata."""
    started = time.perf_counter()
    detector = get_detector()
    input_image = load_image(image)
    model_image, resized_for_model = _resize_for_model(input_image)
    raw = detector.detect(model_image)
    elapsed = time.perf_counter() - started

    if not isinstance(raw, dict):
        logger.warning("Model output is not a dict: %r", type(raw))
        raw = {"defects": [], "_parse_error": "non_dict_output"}

    cleaned, dropped = clean_defects(raw.get("defects", []))
    cleaned, duplicate_count = dedupe_defects(cleaned)
    counts = label_counts(cleaned)

    return {
        "defects": cleaned,
        "defect_count": len(cleaned),
        "label_counts": counts,
        "dropped_count": dropped,
        "duplicate_count": duplicate_count,
        "inference_seconds": round(elapsed, 3),
        "model_path": detector.model_path,
        "parse_error": raw.get("_parse_error"),
        "resized_for_model": resized_for_model,
    }


def extract_defects_from_path(image_path: str | Path) -> dict:
    """Convenience: open image from path and run extraction."""
    img = load_image(image_path)
    return extract_defects(img)


def _resize_for_model(image: Any) -> tuple[Any, bool]:
    cfg = get_vision_config()
    max_pixels = max(1, int(cfg.max_input_pixels or 0))
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image, False

    scale = (max_pixels / float(pixels)) ** 0.5
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size), True
