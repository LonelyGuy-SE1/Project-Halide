"""Generate augmented film-defect training images from transparent overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.augmentation import augment_image, discover_overlays
from data.datasets import FDS_SCANS_DIR, TRAINING_JSONL, load_jsonl, resolve_image_path
from data.preprocessing import load_image, resize_for_preview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(TRAINING_JSONL))
    parser.add_argument(
        "--overlays",
        default=str(
            FDS_SCANS_DIR.parent / "synthetic"
        ),
    )
    parser.add_argument("--out-dir", default="data/augmented")
    parser.add_argument("--samples-per-image", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entries = load_jsonl(args.input)
    overlays = discover_overlays(args.overlays)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "augmented_training.jsonl"

    rows: list[dict] = []
    for image_idx, entry in enumerate(entries):
        src = resolve_image_path(entry)
        if not src.exists():
            continue
        base = load_image(src)
        for sample_idx in range(args.samples_per_image):
            seed = args.seed + image_idx * 1000 + sample_idx
            img, annotations = augment_image(base, overlays, seed=seed)
            img = resize_for_preview(img, max_side=args.max_side)
            out_name = f"{Path(entry['image']).stem}_aug_{sample_idx:03d}.png"
            img.save(image_dir / out_name, "PNG", optimize=True)
            rows.append(
                {
                    "image": f"images/{out_name}",
                    "width": img.width,
                    "height": img.height,
                    "source": "film_damage_simulator_augmented",
                    "parent_image": entry["image"],
                    "annotations": annotations,
                }
            )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(f"overlays: {len(overlays)}")
    print(f"augmented images: {len(rows)}")
    print(f"wrote: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
