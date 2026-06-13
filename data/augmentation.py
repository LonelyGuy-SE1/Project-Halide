"""Synthetic defect augmentation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageEnhance, ImageOps

from data.preprocessing import load_image

SYNTHETIC_LABEL_MAP = {
    "dust": "dust",
    "dirt": "dirt",
    "dots": "dust",
    "hair": "long_hair",
    "hair-short": "short_hair",
    "lint": "short_hair",
    "scratches": "scratch",
    "scratch": "scratch",
    "smut": "dirt",
    "spots": "dirt",
    "sprinkles": "dust",
    "stain": "dirt",
}


@dataclass(frozen=True)
class OverlayDefect:
    path: Path
    label: str


def discover_overlays(root: str | Path) -> list[OverlayDefect]:
    """Find transparent PNG overlays and infer labels from folder names."""
    root = Path(root)
    overlays: list[OverlayDefect] = []
    if not root.exists():
        return overlays
    for path in sorted(root.rglob("*.png")):
        label = SYNTHETIC_LABEL_MAP.get(path.parent.name.lower())
        if label:
            overlays.append(OverlayDefect(path=path, label=label))
    return overlays


def _visible_bbox(alpha: Image.Image) -> tuple[int, int, int, int] | None:
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        return None
    return bbox


def _normalized_bbox(
    paste_x: int,
    paste_y: int,
    visible_bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> list[float]:
    x_min, y_min, x_max, y_max = visible_bbox
    return [
        round((paste_x + x_min) / width, 6),
        round((paste_y + y_min) / height, 6),
        round((paste_x + x_max) / width, 6),
        round((paste_y + y_max) / height, 6),
    ]


def apply_overlay(
    base: Image.Image,
    overlay: OverlayDefect,
    *,
    rng: random.Random,
    scale_range: tuple[float, float] = (0.35, 1.4),
    opacity_range: tuple[float, float] = (0.55, 0.95),
) -> tuple[Image.Image, dict] | None:
    """Paste one defect overlay onto a copy of base and return annotation."""
    out = load_image(base).convert("RGBA")
    width, height = out.size

    layer = Image.open(overlay.path).convert("RGBA")
    if rng.random() < 0.5:
        layer = ImageOps.mirror(layer)
    if rng.random() < 0.35:
        layer = layer.rotate(rng.uniform(-22, 22), expand=True, resample=Image.Resampling.BICUBIC)

    scale = rng.uniform(*scale_range)
    new_size = (
        max(2, int(layer.width * scale)),
        max(2, int(layer.height * scale)),
    )
    layer = layer.resize(new_size, Image.Resampling.LANCZOS)
    if layer.width >= width or layer.height >= height:
        layer.thumbnail((width // 2, height // 2), Image.Resampling.LANCZOS)

    alpha = layer.getchannel("A")
    alpha = ImageEnhance.Brightness(alpha).enhance(rng.uniform(*opacity_range))
    layer.putalpha(alpha)
    visible = _visible_bbox(alpha)
    if visible is None:
        return None

    max_x = max(0, width - layer.width)
    max_y = max(0, height - layer.height)
    paste_x = rng.randint(0, max_x) if max_x else 0
    paste_y = rng.randint(0, max_y) if max_y else 0

    out.alpha_composite(layer, (paste_x, paste_y))
    annotation = {
        "label": overlay.label,
        "bbox": _normalized_bbox(paste_x, paste_y, visible, width, height),
    }
    return out.convert("RGB"), annotation


def augment_image(
    base: Image.Image,
    overlays: Iterable[OverlayDefect],
    *,
    seed: int,
    defects_per_image: tuple[int, int] = (3, 9),
) -> tuple[Image.Image, list[dict]]:
    """Create one augmented image and its generated annotations."""
    rng = random.Random(seed)
    overlay_list = list(overlays)
    if not overlay_list:
        return load_image(base), []

    by_label: dict[str, list[OverlayDefect]] = {}
    for overlay in overlay_list:
        by_label.setdefault(overlay.label, []).append(overlay)
    labels = sorted(by_label)

    out = load_image(base)
    annotations: list[dict] = []
    target = rng.randint(*defects_per_image)
    for _ in range(target):
        label = rng.choice(labels)
        overlay = rng.choice(by_label[label])
        result = apply_overlay(out, overlay, rng=rng)
        if result is None:
            continue
        out, annotation = result
        annotations.append(annotation)
    return out, annotations


__all__ = [
    "OverlayDefect",
    "SYNTHETIC_LABEL_MAP",
    "apply_overlay",
    "augment_image",
    "discover_overlays",
]
