"""Generate a larger procedural film-defect dataset for v4 training.

The user-provided negatives are held out for evaluation and are never read by
this script. Bases come from local public/project datasets under data/raw.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing import load_image, resize_for_preview

LabelBox = tuple[str, list[float]]


def _bbox(points: list[tuple[float, float]], width: int, height: int, pad: int = 3) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [
        max(0.0, round((min(xs) - pad) / width, 6)),
        max(0.0, round((min(ys) - pad) / height, 6)),
        min(1.0, round((max(xs) + pad) / width, 6)),
        min(1.0, round((max(ys) + pad) / height, 6)),
    ]


def _line_points(
    rng: random.Random,
    width: int,
    height: int,
    *,
    long: bool,
) -> list[tuple[float, float]]:
    x0 = rng.uniform(-0.05 * width, 0.95 * width)
    y0 = rng.uniform(0, height)
    length = rng.uniform(0.35, 1.15) * (width if long else min(width, height) * 0.35)
    angle = rng.uniform(-math.pi, math.pi)
    curvature = rng.uniform(-0.22, 0.22)
    points: list[tuple[float, float]] = []
    for i in range(5):
        t = i / 4
        x = x0 + math.cos(angle) * length * t
        y = y0 + math.sin(angle) * length * t
        y += math.sin(t * math.pi) * curvature * length
        points.append((x, y))
    return points


def _draw_polyline(
    layer: Image.Image,
    points: list[tuple[float, float]],
    *,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    draw = ImageDraw.Draw(layer)
    draw.line(points, fill=color, width=width, joint="curve")


def _random_size(rng: random.Random, max_side: int) -> tuple[int, int]:
    candidates = [
        (900, 650),
        (960, 720),
        (1000, 760),
        (780, 1040),
        (900, 1200),
        (1200, 900),
    ]
    width, height = rng.choice(candidates)
    scale = min(1.0, max_side / max(width, height))
    return max(320, int(width * scale)), max(320, int(height * scale))


def _gradient_base(width: int, height: int, rng: random.Random) -> Image.Image:
    dark = rng.randint(18, 65)
    light = rng.randint(155, 230)
    if rng.random() < 0.5:
        dark, light = light, dark
    tint = rng.choice(
        [
            (1.0, 1.0, 1.0),
            (1.08, 0.98, 0.88),
            (0.86, 0.96, 1.08),
            (1.05, 0.9, 1.02),
        ]
    )
    vertical = rng.random() < 0.5
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    steps = height if vertical else width
    for i in range(steps):
        t = i / max(1, steps - 1)
        wobble = math.sin(t * math.pi * rng.uniform(1.4, 3.2) + rng.random()) * 10
        value = int(dark * (1 - t) + light * t + wobble)
        color = tuple(max(0, min(255, int(value * c))) for c in tint)
        if vertical:
            draw.line([(0, i), (width, i)], fill=color)
        else:
            draw.line([(i, 0), (i, height)], fill=color)
    return img


def _add_grain(img: Image.Image, rng: random.Random) -> Image.Image:
    sigma = rng.uniform(10, 32)
    noise = Image.effect_noise(img.size, sigma).convert("L")
    if rng.random() < 0.5:
        noise = ImageOps.invert(noise)
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.blend(img, noise_rgb, rng.uniform(0.05, 0.16))


def _draw_portrait_content(img: Image.Image, rng: random.Random) -> None:
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    cx = rng.uniform(0.36, 0.62) * width
    cy = rng.uniform(0.33, 0.5) * height
    face_w = rng.uniform(0.18, 0.28) * width
    face_h = rng.uniform(0.24, 0.38) * height
    tone = rng.choice([(230, 224, 212, 95), (36, 34, 32, 105), (180, 185, 190, 85)])
    hair = rng.choice([(18, 18, 18, 150), (232, 232, 232, 110), (80, 75, 70, 135)])
    draw.ellipse(
        [cx - face_w / 2, cy - face_h / 2, cx + face_w / 2, cy + face_h / 2],
        fill=tone,
    )
    draw.ellipse(
        [cx - face_w * 0.64, cy - face_h * 0.72, cx + face_w * 0.64, cy + face_h * 0.15],
        fill=hair,
    )
    for _ in range(rng.randint(18, 38)):
        start_x = cx + rng.uniform(-0.7, 0.7) * face_w
        start_y = cy - face_h * rng.uniform(0.35, 0.78)
        end_x = start_x + rng.uniform(-0.22, 0.22) * width
        end_y = start_y + rng.uniform(0.04, 0.22) * height
        draw.line([(start_x, start_y), (end_x, end_y)], fill=hair, width=rng.randint(1, 2))
    shoulder_y = cy + face_h * 0.42
    draw.pieslice(
        [cx - face_w * 1.35, shoulder_y, cx + face_w * 1.35, shoulder_y + face_h * 1.15],
        180,
        360,
        fill=rng.choice([(245, 245, 245, 80), (28, 28, 28, 95), (120, 125, 130, 70)]),
    )


def _draw_landscape_content(img: Image.Image, rng: random.Random) -> None:
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    horizon = rng.uniform(0.38, 0.66) * height
    draw.rectangle(
        [0, horizon, width, height],
        fill=rng.choice([(28, 46, 38, 80), (210, 215, 205, 70), (55, 55, 55, 85)]),
    )
    for _ in range(rng.randint(18, 42)):
        x = rng.uniform(0, width)
        y0 = rng.uniform(horizon, height)
        y1 = y0 - rng.uniform(0.04, 0.18) * height
        draw.line(
            [(x, y0), (x + rng.uniform(-0.04, 0.04) * width, y1)],
            fill=rng.choice([(18, 34, 22, 115), (230, 230, 220, 75)]),
            width=rng.randint(1, 3),
        )
    for _ in range(rng.randint(3, 9)):
        x0 = rng.uniform(-0.1, 0.85) * width
        y0 = rng.uniform(0.22, 0.78) * height
        x1 = x0 + rng.uniform(0.12, 0.5) * width
        y1 = y0 + rng.uniform(0.03, 0.18) * height
        draw.line(
            [(x0, y0), (x1, y1)],
            fill=rng.choice([(20, 24, 28, 85), (230, 230, 230, 65)]),
            width=rng.randint(2, 5),
        )


def _draw_structural_content(img: Image.Image, rng: random.Random) -> None:
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(rng.randint(4, 10)):
        x0 = rng.uniform(0, width)
        y0 = rng.uniform(0, height)
        x1 = x0 + rng.uniform(-0.45, 0.45) * width
        y1 = y0 + rng.uniform(-0.45, 0.45) * height
        draw.line(
            [(x0, y0), (x1, y1)],
            fill=rng.choice([(18, 18, 18, 95), (235, 235, 235, 75), (120, 120, 120, 80)]),
            width=rng.randint(2, 7),
        )
    for _ in range(rng.randint(12, 32)):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        radius = rng.uniform(2, 7)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=rng.choice([(25, 25, 25, 75), (235, 235, 235, 70)]),
        )


def make_clean_base(rng: random.Random, *, max_side: int) -> Image.Image:
    width, height = _random_size(rng, max_side)
    img = _gradient_base(width, height, rng)
    scene = rng.choice(["portrait", "landscape", "structure", "minimal"])
    if scene == "portrait":
        _draw_portrait_content(img, rng)
    elif scene == "landscape":
        _draw_landscape_content(img, rng)
    elif scene == "structure":
        _draw_structural_content(img, rng)

    img = _add_grain(img, rng)
    if rng.random() < 0.55:
        img = ImageOps.autocontrast(img, cutoff=rng.uniform(0, 2))
    if rng.random() < 0.35:
        img = ImageOps.invert(img)
    return img.convert("RGB")


def add_scratch(img: Image.Image, rng: random.Random) -> tuple[Image.Image, LabelBox]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    points = _line_points(rng, width, height, long=True)
    color = (245, 245, 245, rng.randint(145, 235)) if rng.random() < 0.55 else (8, 8, 8, rng.randint(130, 220))
    _draw_polyline(layer, points, color=color, width=rng.randint(1, 5))
    if rng.random() < 0.35:
        offset = rng.uniform(2, 7)
        points2 = [(x + offset, y + rng.uniform(-2, 2)) for x, y in points]
        _draw_polyline(layer, points2, color=color, width=max(1, rng.randint(1, 3)))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return out, ("scratch", _bbox(points, width, height, pad=8))


def add_hair(img: Image.Image, rng: random.Random, *, short: bool) -> tuple[Image.Image, LabelBox]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    points = _line_points(rng, width, height, long=not short)
    color = (18, 18, 18, rng.randint(145, 220)) if rng.random() < 0.5 else (235, 235, 235, rng.randint(120, 190))
    _draw_polyline(layer, points, color=color, width=rng.randint(1, 3))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.65)))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return out, ("short_hair" if short else "long_hair", _bbox(points, width, height, pad=6))


def add_dust(img: Image.Image, rng: random.Random) -> tuple[Image.Image, LabelBox]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    radius = rng.uniform(2, max(3, min(width, height) * 0.012))
    x = rng.uniform(radius, width - radius)
    y = rng.uniform(radius, height - radius)
    color_value = rng.choice([25, 235])
    color = (color_value, color_value, color_value, rng.randint(130, 230))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    if rng.random() < 0.4:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.2)))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return out, ("dust", _bbox([(x - radius, y - radius), (x + radius, y + radius)], width, height, pad=2))


def add_dirt(img: Image.Image, rng: random.Random) -> tuple[Image.Image, LabelBox]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = rng.uniform(0.05 * width, 0.95 * width)
    cy = rng.uniform(0.05 * height, 0.95 * height)
    rx = rng.uniform(0.02, 0.12) * width
    ry = rng.uniform(0.02, 0.10) * height
    points = []
    for i in range(rng.randint(7, 14)):
        angle = 2 * math.pi * i / rng.randint(9, 16)
        scale = rng.uniform(0.45, 1.15)
        points.append((cx + math.cos(angle) * rx * scale, cy + math.sin(angle) * ry * scale))
    color = rng.choice([(5, 5, 5, rng.randint(100, 190)), (245, 245, 245, rng.randint(85, 170))])
    draw.polygon(points, fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 2.2)))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return out, ("dirt", _bbox(points, width, height, pad=5))


def add_emulsion_damage(img: Image.Image, rng: random.Random) -> tuple[Image.Image, LabelBox]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = rng.uniform(0.05 * width, 0.95 * width)
    cy = rng.uniform(0.05 * height, 0.95 * height)
    rx = rng.uniform(0.08, 0.32) * width
    ry = rng.uniform(0.06, 0.25) * height
    n = rng.randint(12, 26)
    points = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        wobble = rng.uniform(0.45, 1.35)
        points.append((cx + math.cos(angle) * rx * wobble, cy + math.sin(angle) * ry * wobble))
    color = rng.choice([(240, 240, 240, rng.randint(90, 185)), (0, 0, 0, rng.randint(80, 155))])
    draw.polygon(points, fill=color)
    for _ in range(rng.randint(3, 8)):
        p0 = rng.choice(points)
        p1 = (p0[0] + rng.uniform(-rx, rx), p0[1] + rng.uniform(-ry, ry))
        _draw_polyline(layer, [p0, p1], color=(255, 255, 255, rng.randint(95, 175)), width=rng.randint(1, 4))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.8)))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return out, ("emulsion_damage", _bbox(points, width, height, pad=8))


def add_chemical_stain(img: Image.Image, rng: random.Random) -> tuple[Image.Image, LabelBox]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x0 = rng.uniform(-0.1 * width, 0.8 * width)
    y0 = rng.uniform(-0.1 * height, 0.8 * height)
    x1 = x0 + rng.uniform(0.18, 0.55) * width
    y1 = y0 + rng.uniform(0.16, 0.48) * height
    color = rng.choice([(185, 130, 55, 80), (225, 225, 225, 80), (40, 40, 40, 80)])
    draw.ellipse([x0, y0, x1, y1], fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(8, 24)))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return out, ("chemical_stain", _bbox([(x0, y0), (x1, y1)], width, height, pad=8))


def add_light_leak(img: Image.Image, rng: random.Random) -> tuple[Image.Image, LabelBox]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    edge = rng.choice(["left", "right", "top", "bottom"])
    band = rng.uniform(0.12, 0.42)
    if edge in {"left", "right"}:
        w = int(width * band)
        x0, x1 = (0, w) if edge == "left" else (width - w, width)
        y0, y1 = 0, height
    else:
        h = int(height * band)
        x0, x1 = 0, width
        y0, y1 = (0, h) if edge == "top" else (height - h, height)
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, rng.randint(55, 120)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(20, 60)))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return out, ("light_leak", _bbox([(x0, y0), (x1, y1)], width, height, pad=0))


DEFECT_FNS: dict[str, Callable[[Image.Image, random.Random], tuple[Image.Image, LabelBox]]] = {
    "dust": add_dust,
    "dirt": add_dirt,
    "scratch": add_scratch,
    "long_hair": lambda img, rng: add_hair(img, rng, short=False),
    "short_hair": lambda img, rng: add_hair(img, rng, short=True),
    "emulsion_damage": add_emulsion_damage,
    "chemical_stain": add_chemical_stain,
    "light_leak": add_light_leak,
}


def _candidate_bases(extra_roots: list[Path]) -> list[Path]:
    roots: list[Path] = []
    roots.extend(extra_roots)
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}))
    return out


def _row(image_name: str, annotations: list[dict]) -> dict:
    return {
        "image": f"v4_synthetic/images/{image_name}",
        "annotations": annotations,
        "source": "procedural_v4_synthetic",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/augmented/v4_synthetic")
    parser.add_argument("--count", type=int, default=640)
    parser.add_argument("--clean-count", type=int, default=96)
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument(
        "--base-mode",
        choices=["procedural", "external", "mixed"],
        default="procedural",
        help="Use procedural clean bases, external roots, or a mix of both.",
    )
    parser.add_argument("--extra-root", action="append", default=[])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bases = _candidate_bases([Path(p) for p in args.extra_root])
    base_images = [
        (path, resize_for_preview(load_image(path), max_side=args.max_side))
        for path in bases
    ]
    if args.base_mode == "external" and not base_images:
        raise SystemExit("external base mode requires --extra-root images")

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    counts: dict[str, int] = {}

    label_weights = [
        ("scratch", 24),
        ("dust", 18),
        ("dirt", 16),
        ("emulsion_damage", 18),
        ("chemical_stain", 8),
        ("light_leak", 5),
        ("long_hair", 7),
        ("short_hair", 7),
    ]
    labels = [label for label, weight in label_weights for _ in range(weight)]

    total = args.count + args.clean_count
    for index in range(total):
        use_external = (
            args.base_mode == "external"
            or (args.base_mode == "mixed" and base_images and rng.random() < 0.35)
        )
        if use_external:
            base_path, base_img = rng.choice(base_images)
            img = base_img.copy()
            base_name = str(base_path.relative_to(REPO_ROOT) if base_path.is_relative_to(REPO_ROOT) else base_path)
        else:
            img = make_clean_base(rng, max_side=args.max_side)
            base_name = "procedural_clean_base"
        annotations: list[dict] = []
        if index >= args.clean_count:
            defect_count = rng.randint(3, 12)
            for _ in range(defect_count):
                label = rng.choice(labels)
                fn = DEFECT_FNS[label]
                img, (out_label, bbox) = fn(img, rng)
                annotations.append({"label": out_label, "bbox": bbox})
                counts[out_label] = counts.get(out_label, 0) + 1
        else:
            counts["clean"] = counts.get("clean", 0) + 1

        image_name = f"v4_{index:05d}.jpg"
        img.save(image_dir / image_name, "JPEG", quality=88, optimize=True)
        row = _row(image_name, annotations)
        row["width"] = img.width
        row["height"] = img.height
        row["base_image"] = base_name
        rows.append(row)

    jsonl = out_dir / "v4_synthetic_training.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "rows": len(rows),
        "defect_rows": args.count,
        "clean_rows": args.clean_count,
        "label_counts": dict(sorted(counts.items())),
        "base_mode": args.base_mode,
        "external_base_count": len(bases),
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
