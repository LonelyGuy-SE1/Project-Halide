"""Generate a targeted v5 curriculum for real negative defect detection.

The private user negatives remain held out. This script creates procedural
training images only, with strong analog-negative damage and hard clean
examples that contain subject hair, grass, grain, and texture.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_to_sharegpt import SYSTEM_PROMPT_INT

Point = tuple[float, float]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bbox(points: list[Point], width: int, height: int, pad: int = 8) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [
        clamp((min(xs) - pad) / width, 0.0, 1.0),
        clamp((min(ys) - pad) / height, 0.0, 1.0),
        clamp((max(xs) + pad) / width, 0.0, 1.0),
        clamp((max(ys) + pad) / height, 0.0, 1.0),
    ]


def int_bbox(box: list[float]) -> list[int]:
    return [int(round(clamp(v, 0.0, 1.0) * 999)) for v in box]


def row(image_ref: str, annotations: list[dict]) -> dict:
    defects = [
        {"label": ann["label"], "bbox": int_bbox(ann["bbox"])}
        for ann in annotations
    ]
    return {
        "conversations": [
            {"from": "human", "value": f"<image>{SYSTEM_PROMPT_INT}"},
            {
                "from": "gpt",
                "value": json.dumps({"defects": defects}, separators=(",", ":")),
            },
        ],
        "images": [image_ref],
    }


def make_gradient(width: int, height: int, rng: random.Random) -> Image.Image:
    start = rng.randint(20, 80)
    end = rng.randint(150, 235)
    if rng.random() < 0.45:
        start, end = end, start
    vertical = rng.random() < 0.6
    img = Image.new("L", (width, height))
    draw = ImageDraw.Draw(img)
    steps = height if vertical else width
    for i in range(steps):
        t = i / max(1, steps - 1)
        value = start * (1 - t) + end * t
        value += math.sin(t * math.pi * rng.uniform(2.0, 5.0)) * rng.uniform(6, 18)
        line = int(clamp(value, 0, 255))
        if vertical:
            draw.line([(0, i), (width, i)], fill=line)
        else:
            draw.line([(i, 0), (i, height)], fill=line)
    return img.convert("RGB")


def add_grain(img: Image.Image, rng: random.Random) -> Image.Image:
    noise = Image.effect_noise(img.size, rng.uniform(16, 46)).convert("L")
    if rng.random() < 0.5:
        noise = ImageOps.invert(noise)
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    img = Image.blend(img, noise_rgb, rng.uniform(0.07, 0.22))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))
    return ImageOps.autocontrast(img, cutoff=rng.uniform(0, 2.5))


def draw_portrait(img: Image.Image, rng: random.Random) -> None:
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    cx = rng.uniform(0.38, 0.62) * width
    cy = rng.uniform(0.32, 0.52) * height
    face_w = rng.uniform(0.18, 0.28) * width
    face_h = rng.uniform(0.24, 0.36) * height
    light = rng.choice([(232, 232, 232, 115), (205, 205, 205, 105), (35, 35, 35, 95)])
    dark = rng.choice([(20, 20, 20, 150), (55, 55, 55, 130), (220, 220, 220, 100)])
    draw.ellipse(
        [cx - face_w / 2, cy - face_h / 2, cx + face_w / 2, cy + face_h / 2],
        fill=light,
    )
    draw.ellipse(
        [cx - face_w * 0.7, cy - face_h * 0.74, cx + face_w * 0.7, cy + face_h * 0.16],
        fill=dark,
    )
    for _ in range(rng.randint(20, 48)):
        x0 = cx + rng.uniform(-0.75, 0.75) * face_w
        y0 = cy - face_h * rng.uniform(0.35, 0.82)
        x1 = x0 + rng.uniform(-0.16, 0.16) * width
        y1 = y0 + rng.uniform(0.04, 0.22) * height
        draw.line([(x0, y0), (x1, y1)], fill=dark, width=rng.randint(1, 3))
    shoulder_y = cy + face_h * 0.42
    draw.pieslice(
        [cx - face_w * 1.6, shoulder_y, cx + face_w * 1.6, shoulder_y + face_h * 1.35],
        180,
        360,
        fill=rng.choice([(242, 242, 242, 85), (25, 25, 25, 110), (110, 110, 110, 90)]),
    )


def draw_grass_or_branches(img: Image.Image, rng: random.Random) -> None:
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    horizon = rng.uniform(0.35, 0.68) * height
    draw.rectangle([0, horizon, width, height], fill=rng.choice([(30, 30, 30, 70), (210, 210, 210, 60)]))
    for _ in range(rng.randint(45, 110)):
        x = rng.uniform(0, width)
        y0 = rng.uniform(horizon, height)
        length = rng.uniform(0.06, 0.22) * height
        x1 = x + rng.uniform(-0.045, 0.045) * width
        y1 = y0 - length
        draw.line(
            [(x, y0), (x1, y1)],
            fill=rng.choice([(18, 18, 18, 95), (230, 230, 230, 70)]),
            width=rng.randint(1, 3),
        )
    for _ in range(rng.randint(2, 8)):
        x0 = rng.uniform(-0.1, 0.85) * width
        y0 = rng.uniform(0.15, 0.75) * height
        x1 = x0 + rng.uniform(0.15, 0.58) * width
        y1 = y0 + rng.uniform(-0.12, 0.16) * height
        draw.line([(x0, y0), (x1, y1)], fill=(25, 25, 25, rng.randint(65, 110)), width=rng.randint(2, 6))


def draw_wall_texture(img: Image.Image, rng: random.Random) -> None:
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(rng.randint(80, 180)):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        r = rng.uniform(1, 4)
        v = rng.choice([18, 235, 125])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(v, v, v, rng.randint(18, 55)))
    if rng.random() < 0.45:
        x = rng.uniform(0.62, 0.9) * width
        draw.rectangle([x, 0, width, height], fill=(235, 235, 235, rng.randint(25, 70)))


def make_base(rng: random.Random, max_side: int) -> Image.Image:
    sizes = [(900, 650), (760, 1040), (900, 1200), (1100, 760), (1200, 900)]
    width, height = rng.choice(sizes)
    scale = min(1.0, max_side / max(width, height))
    width, height = int(width * scale), int(height * scale)
    img = make_gradient(width, height, rng)
    scene = rng.choice(["portrait", "portrait", "grass", "wall", "minimal"])
    if scene == "portrait":
        draw_portrait(img, rng)
    elif scene == "grass":
        draw_grass_or_branches(img, rng)
    elif scene == "wall":
        draw_wall_texture(img, rng)
    if rng.random() < 0.55:
        img = ImageOps.invert(img)
    return add_grain(img.convert("RGB"), rng)


def line_points(width: int, height: int, rng: random.Random, *, force_cross: bool = False) -> list[Point]:
    if force_cross:
        x0 = rng.uniform(-0.08, 0.18) * width
        x1 = rng.uniform(0.78, 1.08) * width
        y0 = rng.uniform(0.2, 0.78) * height
        y1 = y0 + rng.uniform(-0.22, 0.22) * height
    else:
        x0 = rng.uniform(-0.08, 0.95) * width
        y0 = rng.uniform(0, height)
        length = rng.uniform(0.25, 1.05) * width
        angle = rng.uniform(-math.pi, math.pi)
        x1 = x0 + math.cos(angle) * length
        y1 = y0 + math.sin(angle) * length
    points: list[Point] = []
    for i in range(6):
        t = i / 5
        x = x0 * (1 - t) + x1 * t
        y = y0 * (1 - t) + y1 * t
        y += math.sin(t * math.pi) * rng.uniform(-0.08, 0.08) * height
        points.append((x, y))
    return points


def add_scratch(img: Image.Image, rng: random.Random, *, force_cross: bool = False) -> tuple[Image.Image, dict]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    points = line_points(width, height, rng, force_cross=force_cross)
    dark = rng.random() < 0.55
    value = rng.randint(0, 28) if dark else rng.randint(222, 255)
    alpha = rng.randint(145, 240)
    ImageDraw.Draw(layer).line(points, fill=(value, value, value, alpha), width=rng.randint(2, 7), joint="curve")
    if rng.random() < 0.38:
        offset = rng.uniform(3, 12)
        points2 = [(x + offset, y + rng.uniform(-3, 3)) for x, y in points]
        ImageDraw.Draw(layer).line(points2, fill=(value, value, value, max(90, alpha - 35)), width=rng.randint(1, 4), joint="curve")
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), {
        "label": "scratch",
        "bbox": bbox(points, width, height, pad=12),
    }


def add_scratch_cluster(img: Image.Image, rng: random.Random) -> tuple[Image.Image, list[dict]]:
    annotations: list[dict] = []
    count = rng.randint(3, 7)
    for _ in range(count):
        img, ann = add_scratch(img, rng, force_cross=True)
        annotations.append(ann)
    return img, annotations


def add_emulsion_damage(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = rng.uniform(0.12, 0.88) * width
    cy = rng.uniform(0.12, 0.88) * height
    rx = rng.uniform(0.10, 0.35) * width
    ry = rng.uniform(0.08, 0.30) * height
    points: list[Point] = []
    for i in range(rng.randint(12, 24)):
        angle = math.tau * i / 20
        scale = rng.uniform(0.45, 1.35)
        points.append((cx + math.cos(angle) * rx * scale, cy + math.sin(angle) * ry * scale))
    v = rng.choice([0, 245, 118])
    draw.polygon(points, fill=(v, v, v, rng.randint(85, 170)))
    for _ in range(rng.randint(3, 8)):
        p0 = rng.choice(points)
        p1 = (p0[0] + rng.uniform(-rx, rx), p0[1] + rng.uniform(-ry, ry))
        draw.line([p0, p1], fill=(255 - v, 255 - v, 255 - v, rng.randint(80, 160)), width=rng.randint(1, 4))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 2.6)))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), {
        "label": "emulsion_damage",
        "bbox": bbox(points, width, height, pad=16),
    }


def add_dirt(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = rng.uniform(0.04, 0.96) * width
    cy = rng.uniform(0.04, 0.96) * height
    rx = rng.uniform(0.015, 0.12) * width
    ry = rng.uniform(0.015, 0.10) * height
    points = [
        (cx + math.cos(math.tau * i / 12) * rx * rng.uniform(0.45, 1.2),
         cy + math.sin(math.tau * i / 12) * ry * rng.uniform(0.45, 1.2))
        for i in range(12)
    ]
    v = rng.choice([0, 245, 85])
    draw.polygon(points, fill=(v, v, v, rng.randint(110, 210)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.6)))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), {
        "label": "dirt",
        "bbox": bbox(points, width, height, pad=6),
    }


def add_chemical_stain(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = rng.uniform(0.08, 0.92) * width
    cy = rng.uniform(0.08, 0.92) * height
    rx = rng.uniform(0.10, 0.36) * width
    ry = rng.uniform(0.05, 0.24) * height
    points: list[Point] = []
    point_count = rng.randint(14, 26)
    for i in range(point_count):
        angle = math.tau * i / point_count
        scale = rng.uniform(0.45, 1.22)
        x = cx + math.cos(angle) * rx * scale
        y = cy + math.sin(angle) * ry * scale
        points.append((x, y))
    tone = rng.choice([18, 65, 188, 236])
    draw.polygon(points, fill=(tone, tone, tone, rng.randint(42, 105)))
    for _ in range(rng.randint(2, 7)):
        offset_x = rng.uniform(-0.18, 0.18) * rx
        offset_y = rng.uniform(-0.18, 0.18) * ry
        small = [
            (x * 0.78 + cx * 0.22 + offset_x, y * 0.78 + cy * 0.22 + offset_y)
            for x, y in points
        ]
        draw.polygon(
            small,
            fill=(255 - tone, 255 - tone, 255 - tone, rng.randint(18, 54)),
        )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3.0, 12.0)))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), {
        "label": "chemical_stain",
        "bbox": bbox(points, width, height, pad=12),
    }


def add_dust(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    r = rng.uniform(2, max(4, min(width, height) * 0.012))
    x = rng.uniform(r, width - r)
    y = rng.uniform(r, height - r)
    v = rng.choice([0, 255])
    draw.ellipse([x - r, y - r, x + r, y + r], fill=(v, v, v, rng.randint(130, 230)))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), {
        "label": "dust",
        "bbox": bbox([(x - r, y - r), (x + r, y + r)], width, height, pad=2),
    }


def add_light_leak(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    width, height = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    edge = rng.choice(["left", "right", "top", "bottom"])
    if edge in {"left", "right"}:
        band = rng.uniform(0.12, 0.34) * width
        x0, x1 = (0, band) if edge == "left" else (width - band, width)
        y0, y1 = 0, height
    else:
        band = rng.uniform(0.12, 0.34) * height
        x0, x1 = 0, width
        y0, y1 = (0, band) if edge == "top" else (height - band, height)
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, rng.randint(50, 115)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(14, 42)))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), {
        "label": "light_leak",
        "bbox": bbox([(x0, y0), (x1, y1)], width, height, pad=0),
    }


def make_example(index: int, rng: random.Random, max_side: int, clean: bool) -> tuple[Image.Image, list[dict]]:
    img = make_base(rng, max_side=max_side)
    annotations: list[dict] = []
    if clean:
        return img, annotations

    if rng.random() < 0.62:
        img, cluster = add_scratch_cluster(img, rng)
        annotations.extend(cluster)
    else:
        for _ in range(rng.randint(1, 4)):
            img, ann = add_scratch(img, rng, force_cross=rng.random() < 0.55)
            annotations.append(ann)

    for _ in range(rng.randint(0, 2)):
        img, ann = add_emulsion_damage(img, rng)
        annotations.append(ann)
    for _ in range(rng.randint(0, 2)):
        img, ann = add_chemical_stain(img, rng)
        annotations.append(ann)
    for _ in range(rng.randint(0, 3)):
        img, ann = add_dirt(img, rng)
        annotations.append(ann)
    for _ in range(rng.randint(0, 5)):
        img, ann = add_dust(img, rng)
        annotations.append(ann)
    if rng.random() < 0.12:
        img, ann = add_light_leak(img, rng)
        annotations.append(ann)
    return img, annotations[:24]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/augmented/v5_negative_curriculum")
    parser.add_argument("--defect-count", type=int, default=416)
    parser.add_argument("--clean-count", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    annotation_rows: list[dict] = []
    total = args.defect_count + args.clean_count
    clean_indices = set(rng.sample(range(total), args.clean_count))
    for index in range(total):
        clean = index in clean_indices
        img, annotations = make_example(index, rng, args.max_side, clean=clean)
        image_name = f"v5_{index:05d}.jpg"
        image_path = image_dir / image_name
        img.save(image_path, "JPEG", quality=90)
        image_ref = f"augmented/v5_negative_curriculum/images/{image_name}"
        records.append(row(image_ref, annotations))
        annotation_rows.append(
            {
                "image": f"v5_negative_curriculum/images/{image_name}",
                "annotations": annotations,
                "width": img.width,
                "height": img.height,
                "source": "procedural_v5_negative_curriculum",
            }
        )

    rng.shuffle(records)
    val_count = max(1, int(round(len(records) * args.val_fraction)))
    val_rows = records[:val_count]
    train_rows = records[val_count:]

    (out_dir / "training_sharegpt_v5.json").write_text(
        json.dumps(train_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "training_sharegpt_val_v5.json").write_text(
        json.dumps(val_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for item in annotation_rows:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

    label_counts: dict[str, int] = {}
    clean_rows = 0
    for item in annotation_rows:
        if not item["annotations"]:
            clean_rows += 1
        for ann in item["annotations"]:
            label_counts[ann["label"]] = label_counts.get(ann["label"], 0) + 1
    summary = {
        "rows": len(records),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "defect_rows": len(records) - clean_rows,
        "clean_rows": clean_rows,
        "label_counts": dict(sorted(label_counts.items())),
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
