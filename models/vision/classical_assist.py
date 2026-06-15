"""Conservative image-analysis assist for obvious film defects.

This module does not run a model. It uses local contrast only to catch clear
linear scratches and compact bright debris that MiniCPM can miss on real scans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from data.schemas import bbox_area, bbox_iou, normalize_bbox


@dataclass(frozen=True)
class Candidate:
    label: str
    bbox: tuple[float, float, float, float]
    score: float

    def to_json(self) -> dict[str, Any]:
        confidence = min(0.61, max(0.48, 0.46 + self.score * 0.48))
        return {
            "label": self.label,
            "bbox": [round(v, 6) for v in self.bbox],
            "confidence": round(confidence, 4),
        }


def detect_classical_defects(
    image: Any,
    *,
    max_defects: int = 32,
    include_compact_debris: bool = False,
) -> list[dict[str, Any]]:
    """Return high-precision defect candidates from image structure only."""
    pil_image = ImageOps.exif_transpose(image.convert("RGB"))
    width, height = pil_image.size
    if width < 64 or height < 64:
        return []

    work = _resize_work_image(pil_image)
    gray = _gray_array(work)
    blur = _blur_array(gray, work.size)
    residual = gray - blur
    mask = _bright_residual_mask(gray, residual)

    candidates: list[Candidate] = []
    candidates.extend(_linear_candidates(mask, residual))
    if include_compact_debris:
        candidates.extend(_compact_debris_candidates(mask, residual))

    candidates = _dedupe_candidates(candidates, max_defects=max(1, int(max_defects)))
    return [candidate.to_json() for candidate in candidates]


def _resize_work_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_side = 1100
    scale = min(1.0, max_side / float(max(width, height)))
    if scale >= 1.0:
        return image.copy()
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _gray_array(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image).astype("float32") / 255.0
    return (
        0.2126 * arr[:, :, 0]
        + 0.7152 * arr[:, :, 1]
        + 0.0722 * arr[:, :, 2]
    )


def _blur_array(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    radius = max(2, int(round(max(width, height) * 0.006)))
    source = Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255)))
    blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred).astype("float32") / 255.0


def _bright_residual_mask(gray: np.ndarray, residual: np.ndarray) -> np.ndarray:
    threshold = max(0.045, float(np.percentile(residual, 99.2)))
    mask = (residual >= threshold) & (gray > float(np.percentile(gray, 42)))
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    return mask


def _linear_candidates(mask: np.ndarray, residual: np.ndarray) -> list[Candidate]:
    height, width = mask.shape
    min_horizontal = max(22, int(round(width * 0.035)))
    min_vertical = max(22, int(round(height * 0.035)))
    candidates: list[Candidate] = []

    for y in range(height):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue
        for run in _contiguous_runs(xs):
            if run.size < min_horizontal:
                continue
            x0, x1 = int(run[0]), int(run[-1]) + 1
            pad = max(2, int(round(height * 0.003)))
            bbox = (
                x0 / width,
                max(0, y - pad) / height,
                x1 / width,
                min(height, y + pad + 1) / height,
            )
            if _is_border_frame(bbox):
                continue
            candidates.append(
                Candidate("scratch", bbox, float(residual[y, run].mean()))
            )

    for x in range(width):
        ys = np.flatnonzero(mask[:, x])
        if ys.size == 0:
            continue
        for run in _contiguous_runs(ys):
            if run.size < min_vertical:
                continue
            y0, y1 = int(run[0]), int(run[-1]) + 1
            pad = max(2, int(round(width * 0.003)))
            bbox = (
                max(0, x - pad) / width,
                y0 / height,
                min(width, x + pad + 1) / width,
                y1 / height,
            )
            if _is_border_frame(bbox):
                continue
            candidates.append(
                Candidate("scratch", bbox, float(residual[run, x].mean()))
            )

    return candidates


def _compact_debris_candidates(mask: np.ndarray, residual: np.ndarray) -> list[Candidate]:
    if int(mask.sum()) > 45_000:
        return []

    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    points = np.argwhere(mask)
    candidates: list[Candidate] = []

    for y_raw, x_raw in points:
        y = int(y_raw)
        x = int(x_raw)
        if visited[y, x] or not mask[y, x]:
            continue
        coords = _component(mask, visited, y, x, max_pixels=900)
        if len(coords) < 3 or len(coords) > 850:
            continue
        ys = np.array([coord[0] for coord in coords], dtype=np.int32)
        xs = np.array([coord[1] for coord in coords], dtype=np.int32)
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        box_width = x1 - x0
        box_height = y1 - y0
        if box_width > width * 0.1 or box_height > height * 0.1:
            continue

        aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
        label = "scratch" if aspect >= 8.0 else "dust"
        pad = 3 if label == "scratch" else 2
        bbox = (
            max(0, x0 - pad) / width,
            max(0, y0 - pad) / height,
            min(width, x1 + pad) / width,
            min(height, y1 + pad) / height,
        )
        if _is_tiny_edge_artifact(bbox) or _is_border_frame(bbox):
            continue
        score = float(residual[ys, xs].mean())
        candidates.append(Candidate(label, bbox, score))

    return candidates


def _component(
    mask: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
    *,
    max_pixels: int,
) -> list[tuple[int, int]]:
    height, width = mask.shape
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    coords: list[tuple[int, int]] = []
    while stack and len(coords) < max_pixels:
        y, x = stack.pop()
        coords.append((y, x))
        for next_y in (y - 1, y, y + 1):
            if next_y < 0 or next_y >= height:
                continue
            for next_x in (x - 1, x, x + 1):
                if next_x < 0 or next_x >= width:
                    continue
                if visited[next_y, next_x] or not mask[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))
    return coords


def _contiguous_runs(values: np.ndarray) -> list[np.ndarray]:
    splits = np.where(np.diff(values) > 1)[0] + 1
    return list(np.split(values, splits))


def _dedupe_candidates(
    candidates: list[Candidate],
    *,
    max_defects: int,
) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(kept) >= max_defects:
            break
        bbox = normalize_bbox(candidate.bbox)
        if bbox is None or bbox_area(bbox) <= 0:
            continue
        if any(_overlaps_existing(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _overlaps_existing(candidate: Candidate, existing: Candidate) -> bool:
    if bbox_iou(candidate.bbox, existing.bbox) >= 0.15:
        return True
    if candidate.label != "scratch" or existing.label != "scratch":
        return False
    cx0, cy0, cx1, cy1 = candidate.bbox
    ex0, ey0, ex1, ey1 = existing.bbox
    same_row = abs(((cy0 + cy1) / 2.0) - ((ey0 + ey1) / 2.0)) < 0.025
    x_overlap = max(0.0, min(cx1, ex1) - max(cx0, ex0))
    return same_row and x_overlap > 0.05


def _is_border_frame(bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    near_outer_edge = x0 < 0.012 or y0 < 0.012 or x1 > 0.988 or y1 > 0.988
    return near_outer_edge and (width > 0.2 or height > 0.2)


def _is_tiny_edge_artifact(bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    center_x = (x0 + x1) / 2.0
    return area < 0.0016 and (center_x < 0.12 or center_x > 0.88)


__all__ = ["detect_classical_defects"]
