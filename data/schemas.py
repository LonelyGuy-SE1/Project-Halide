"""Defect schema and geometry helpers for Project Halide."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

ALLOWED_LABELS = frozenset(
    {
        "dust",
        "dirt",
        "scratch",
        "long_hair",
        "short_hair",
        "emulsion_damage",
        "chemical_stain",
        "light_leak",
    }
)

DEDUP_IOU_THRESHOLD = 0.72

LABEL_DISPLAY_NAMES = {
    "dust": "Dust",
    "dirt": "Dirt",
    "scratch": "Scratch",
    "long_hair": "Long hair",
    "short_hair": "Short hair",
    "emulsion_damage": "Emulsion damage",
    "chemical_stain": "Chemical stain",
    "light_leak": "Light leak",
}

DEFECT_CLASSES_KNOWN = {
    "dust": 0,
    "dirt": 1,
    "scratch": 2,
    "long_hair": 3,
    "short_hair": 4,
    "light_leak": 5,
    "chemical_stain": 6,
    "emulsion_damage": 7,
}

BBox = tuple[float, float, float, float]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


SUBJECT_HAIR_CONFIDENCE_MAX = 0.5
MIN_DEFECT_CONFIDENCE = _env_float("HALIDE_MIN_DEFECT_CONFIDENCE", 0.45)


@dataclass(frozen=True)
class Defect:
    label: str
    bbox: BBox
    confidence: float | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": self.label,
            "bbox": [round(v, 6) for v in self.bbox],
        }
        if self.confidence is not None:
            out["confidence"] = round(float(self.confidence), 4)
        return out


def _unwrap_bbox(bbox: Any) -> Any:
    """Accept a single nested bbox from imperfect model JSON."""
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 1
        and isinstance(bbox[0], (list, tuple))
    ):
        return bbox[0]
    return bbox


def normalize_bbox(bbox: Any) -> BBox | None:
    """Normalize a bbox to float [0, 1].

    Accepts either [0, 999] integer grid values or normalized [0, 1] floats.
    Returns None for malformed, reversed, or out-of-range boxes.
    """
    bbox = _unwrap_bbox(bbox)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x_min, y_min, x_max, y_max = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    if x_max <= x_min or y_max <= y_min:
        return None

    max_val = max(x_min, y_min, x_max, y_max)
    all_whole = all(
        isinstance(v, int) or (isinstance(v, float) and v.is_integer())
        for v in bbox
    )
    scale = 999.0 if all_whole and max_val > 1.5 else 1.0
    if scale == 999.0:
        x_min /= scale
        y_min /= scale
        x_max /= scale
        y_max /= scale
        if not all(-0.001 <= v <= 1.002 for v in (x_min, y_min, x_max, y_max)):
            return None
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

    if not all(0.0 <= v <= 1.0 for v in (x_min, y_min, x_max, y_max)):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None

    return (
        round(x_min, 6),
        round(y_min, 6),
        round(x_max, 6),
        round(y_max, 6),
    )


def validate_defect(raw: Any, min_confidence: float = MIN_DEFECT_CONFIDENCE) -> Defect | None:
    if not isinstance(raw, dict):
        return None
    label = raw.get("label")
    if label not in ALLOWED_LABELS:
        return None
    bbox = normalize_bbox(raw.get("bbox"))
    if bbox is None:
        return None
    confidence = raw.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
    if confidence is not None and confidence < min_confidence:
        return None
    if is_likely_subject_hair(label, bbox, confidence):
        return None
    return Defect(label=label, bbox=bbox, confidence=confidence)


def is_likely_subject_hair(
    label: str,
    bbox: BBox,
    confidence: float | None,
) -> bool:
    """Drop central hair-like subject detail before it reaches diagnosis."""
    if label not in {"long_hair", "short_hair"}:
        return False
    if confidence is not None and confidence >= SUBJECT_HAIR_CONFIDENCE_MAX:
        return False

    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return False

    aspect_ratio = max(width / height, height / width)
    fully_inside_subject_zone = (
        x_min > 0.16
        and x_max < 0.84
        and y_min > 0.10
        and y_max < 0.90
    )
    return fully_inside_subject_zone and aspect_ratio >= 7.5


def clean_defects(
    raw_defects: Any,
    min_confidence: float = MIN_DEFECT_CONFIDENCE,
) -> tuple[list[dict[str, Any]], int]:
    """Return valid defect dicts and number of dropped records."""
    if not isinstance(raw_defects, list):
        return [], 1 if raw_defects else 0

    cleaned: list[dict[str, Any]] = []
    dropped = 0
    for raw in raw_defects:
        defect = validate_defect(raw, min_confidence=min_confidence)
        if defect is None:
            dropped += 1
        else:
            cleaned.append(defect.to_json())
    return cleaned, dropped


def label_counts(defects: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for defect in defects:
        label = defect.get("label")
        if label in ALLOWED_LABELS:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _defect_confidence(defect: dict[str, Any]) -> float:
    value = defect.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.5


def _serialize_defect(label: str, bbox: BBox, source: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "bbox": [round(v, 6) for v in bbox]}
    if source.get("confidence") is not None:
        out["confidence"] = round(_defect_confidence(source), 4)
    return out


def dedupe_defects(
    defects: Iterable[dict[str, Any]],
    iou_threshold: float = DEDUP_IOU_THRESHOLD,
) -> tuple[list[dict[str, Any]], int]:
    """Drop exact and heavily overlapping duplicates from already-clean defects."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    duplicate_count = 0

    for defect in defects:
        label = str(defect.get("label", ""))
        bbox = normalize_bbox(defect.get("bbox"))
        if label not in ALLOWED_LABELS or bbox is None:
            continue
        key = (label, bbox)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(_serialize_defect(label, bbox, defect))

    merged: list[dict[str, Any]] = []
    for defect in unique:
        label = str(defect.get("label", ""))
        bbox = normalize_bbox(defect.get("bbox"))
        if bbox is None:
            continue
        replaced = False
        for index, existing in enumerate(merged):
            if existing.get("label") != label:
                continue
            if bbox_iou(existing.get("bbox"), bbox) < iou_threshold:
                continue
            duplicate_count += 1
            if _defect_confidence(defect) > _defect_confidence(existing):
                merged[index] = defect
            replaced = True
            break
        if not replaced:
            merged.append(defect)

    return merged, duplicate_count


def bbox_area(bbox: Any) -> float:
    norm = normalize_bbox(bbox)
    if norm is None:
        return 0.0
    x_min, y_min, x_max, y_max = norm
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def bbox_iou(a: Any, b: Any) -> float:
    box_a = normalize_bbox(a)
    box_b = normalize_bbox(b)
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    if union <= 0:
        return 0.0
    return round(inter / union, 6)


def bbox_to_pixels(bbox: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    norm = normalize_bbox(bbox)
    if norm is None:
        return None
    x_min, y_min, x_max, y_max = norm
    return (
        int(round(x_min * width)),
        int(round(y_min * height)),
        int(round(x_max * width)),
        int(round(y_max * height)),
    )


def spatial_summary(defects: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute compact spatial cues for the reasoning model."""
    defects_list = list(defects)
    if not defects_list:
        return {
            "edge_defects": 0,
            "center_defects": 0,
            "largest_labels": [],
        }

    edge_count = 0
    center_count = 0
    largest: list[tuple[float, str]] = []
    for defect in defects_list:
        bbox = normalize_bbox(defect.get("bbox"))
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        if x_min < 0.08 or y_min < 0.08 or x_max > 0.92 or y_max > 0.92:
            edge_count += 1
        if 0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65:
            center_count += 1
        largest.append((bbox_area(bbox), str(defect.get("label", "unknown"))))

    largest_labels = [
        label for _, label in sorted(largest, reverse=True)[:5]
    ]
    return {
        "edge_defects": edge_count,
        "center_defects": center_count,
        "largest_labels": largest_labels,
    }


__all__ = [
    "ALLOWED_LABELS",
    "BBox",
    "DEFECT_CLASSES_KNOWN",
    "DEDUP_IOU_THRESHOLD",
    "Defect",
    "LABEL_DISPLAY_NAMES",
    "MIN_DEFECT_CONFIDENCE",
    "bbox_area",
    "bbox_iou",
    "bbox_to_pixels",
    "clean_defects",
    "dedupe_defects",
    "label_counts",
    "normalize_bbox",
    "spatial_summary",
    "validate_defect",
]
