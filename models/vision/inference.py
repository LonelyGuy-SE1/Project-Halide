"""Vision inference pipeline. Takes a film scan and returns defect JSON."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from config import get_vision_config
from data.schemas import (
    BBox,
    clean_defects,
    dedupe_defects,
    filter_edge_artifacts,
    label_counts,
    normalize_bbox,
)
from data.preprocessing import load_image
from models.vision.classical_assist import detect_classical_defects
from models.vision.minicpm_wrapper import get_detector

logger = logging.getLogger(__name__)


def extract_defects(image: Any) -> dict:
    """Run defect extraction on a PIL image. Returns defect dict + metadata."""
    started = time.perf_counter()
    detector = get_detector()
    input_image = load_image(image)
    model_image, resized_for_model = _resize_for_model(input_image)
    raw = detector.detect(model_image)

    if not isinstance(raw, dict):
        logger.warning("Model output is not a dict: %r", type(raw))
        raw = {"defects": [], "_parse_error": "non_dict_output"}

    cleaned, dropped = clean_defects(raw.get("defects", []))
    full_frame_count = len(cleaned)
    tile_fallback_used = False
    tile_count = 0
    tile_parse_errors: list[str] = []
    classical_assist_count = 0
    classical_assist_used = False
    edge_artifact_count = 0

    cfg = get_vision_config()
    if _should_run_tile_fallback(input_image, cleaned):
        tile_fallback_used = True
        tile_defects: list[dict[str, Any]] = list(cleaned)
        for tile_index, (tile_image, tile_box) in enumerate(_iter_tiles(input_image), start=1):
            tile_count = tile_index
            tile_model_image, tile_resized = _resize_for_model(tile_image)
            resized_for_model = resized_for_model or tile_resized
            tile_raw = detector.detect(tile_model_image)
            if not isinstance(tile_raw, dict):
                tile_parse_errors.append("non_dict_output")
                dropped += 1
                continue
            if tile_raw.get("_parse_error"):
                tile_parse_errors.append(str(tile_raw.get("_parse_error")))
            tile_cleaned, tile_dropped = clean_defects(tile_raw.get("defects", []))
            dropped += tile_dropped
            tile_defects.extend(
                _remap_tile_defects(
                    tile_cleaned,
                    tile_box=tile_box,
                    image_size=input_image.size,
                )
            )
            if tile_count >= max(1, int(cfg.tile_max_tiles)):
                break
        cleaned = tile_defects

    if cfg.classical_assist_enabled:
        classical_raw = detect_classical_defects(
            input_image,
            max_defects=cfg.classical_assist_max_defects,
        )
        classical_cleaned, classical_dropped = clean_defects(classical_raw)
        dropped += classical_dropped
        classical_cleaned = [
            defect for defect in classical_cleaned if defect.get("label") == "scratch"
        ]
        classical_assist_count = len(classical_cleaned)
        classical_assist_used = bool(classical_cleaned)
        cleaned.extend(classical_cleaned)

    cleaned, edge_artifact_count = filter_edge_artifacts(cleaned)
    cleaned, duplicate_count = dedupe_defects(cleaned)
    counts = label_counts(cleaned)
    elapsed = time.perf_counter() - started

    return {
        "defects": cleaned,
        "defect_count": len(cleaned),
        "label_counts": counts,
        "dropped_count": dropped,
        "duplicate_count": duplicate_count,
        "edge_artifact_count": edge_artifact_count,
        "inference_seconds": round(elapsed, 3),
        "model_path": detector.model_path,
        "parse_error": raw.get("_parse_error"),
        "resized_for_model": resized_for_model,
        "tile_fallback_used": tile_fallback_used,
        "tile_count": tile_count,
        "full_frame_defect_count": full_frame_count,
        "tile_parse_errors": tile_parse_errors,
        "classical_assist_used": classical_assist_used,
        "classical_assist_count": classical_assist_count,
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


def _should_run_tile_fallback(image: Any, defects: list[dict[str, Any]]) -> bool:
    cfg = get_vision_config()
    if not cfg.tile_fallback_enabled:
        return False
    if len(defects) >= max(0, int(cfg.tile_fallback_min_defects)):
        return False
    width, height = image.size
    if max(width, height) < max(1, int(cfg.tile_min_side)):
        return False
    return True


def _iter_tiles(image: Any) -> list[tuple[Any, tuple[int, int, int, int]]]:
    cfg = get_vision_config()
    width, height = image.size
    tile_side = min(max(1, int(cfg.tile_max_side)), max(width, height))
    tile_width = min(width, tile_side)
    tile_height = min(height, tile_side)
    overlap = max(0.0, min(0.85, float(cfg.tile_overlap)))
    xs = _axis_positions(width, tile_width, overlap)
    ys = _axis_positions(height, tile_height, overlap)
    tiles: list[tuple[Any, tuple[int, int, int, int]]] = []
    center = ((width - tile_width) // 2, (height - tile_height) // 2)
    ordered_positions = [(x, y) for y in ys for x in xs]
    ordered_positions.insert(0, center)
    seen: set[tuple[int, int]] = set()
    for x, y in ordered_positions:
        x = max(0, min(width - tile_width, x))
        y = max(0, min(height - tile_height, y))
        if (x, y) in seen:
            continue
        seen.add((x, y))
        box = (x, y, x + tile_width, y + tile_height)
        tiles.append((image.crop(box), box))
        if len(tiles) >= max(1, int(cfg.tile_max_tiles)):
            break
    return tiles


def _axis_positions(length: int, tile_length: int, overlap: float) -> list[int]:
    if length <= tile_length:
        return [0]
    stride = max(1, int(round(tile_length * (1.0 - overlap))))
    limit = length - tile_length
    positions = list(range(0, limit + 1, stride))
    positions.extend([limit, limit // 2])
    return sorted(set(max(0, min(limit, pos)) for pos in positions))


def _remap_tile_defects(
    defects: list[dict[str, Any]],
    *,
    tile_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    x0, y0, x1, y1 = tile_box
    tile_width = max(1, x1 - x0)
    tile_height = max(1, y1 - y0)
    remapped: list[dict[str, Any]] = []
    for defect in defects:
        bbox = normalize_bbox(defect.get("bbox"))
        if bbox is None:
            continue
        gx0, gy0, gx1, gy1 = _remap_bbox(
            bbox,
            x0=x0,
            y0=y0,
            tile_width=tile_width,
            tile_height=tile_height,
            image_width=image_width,
            image_height=image_height,
        )
        out = {
            "label": defect.get("label"),
            "bbox": [gx0, gy0, gx1, gy1],
        }
        if defect.get("confidence") is not None:
            out["confidence"] = defect.get("confidence")
        remapped.append(out)
    return remapped


def _remap_bbox(
    bbox: BBox,
    *,
    x0: int,
    y0: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
) -> BBox:
    bx0, by0, bx1, by1 = bbox
    return (
        round((x0 + bx0 * tile_width) / image_width, 6),
        round((y0 + by0 * tile_height) / image_height, 6),
        round((x0 + bx1 * tile_width) / image_width, 6),
        round((y0 + by1 * tile_height) / image_height, 6),
    )
