"""
Convert training_data.jsonl to ShareGPT format for LLaMA-Factory.

Each entry becomes:
{
  "conversations": [
    {"from": "human", "value": "<image>Detect all film defects..."},
    {"from": "gpt", "value": "{\"defects\": [...]}"}
  ],
  "images": ["scans/Scan (N).jpg"]
}

Bounding box format options:
  int_0_999: [509, 434, 510, 436]  - aligned with VLM pre-training (MiniCPM-V 4.6
                                     was trained on this 0-1000 grid). Default.
  float_0_1: [0.509, 0.434, 0.510, 0.436]  - normalized [0.0-1.0].

The 0-999 grid is what `models/vision/minicpm_wrapper.py:DETECTION_PROMPT` asks
the model to emit, and what `_parse_defect_json` returns. The downstream
pipeline (`models/vision/inference.py`) converts back to float 0-1.

Cap on defects per scan is required so the response fits within the model's
context window together with the expanded image placeholder (~600 tokens).
Without this cap, scans with 1500+ dust particles produce 25k+ token
responses which get truncated by cutoff_len and break training.

Train/val split: holds out 2 scans by default (Scan (8) and Scan (9)) for
early stopping. The val set is written to a separate file.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from models.vision.prompts import DETECTION_PROMPT_FLOAT, DETECTION_PROMPT_INT


INPUT_JSONL = Path(__file__).parent.parent / "data" / "training_data.jsonl"
OUTPUT_TRAIN = Path(__file__).parent.parent / "data" / "training_sharegpt.json"
OUTPUT_VAL = Path(__file__).parent.parent / "data" / "training_sharegpt_val.json"

# Cap defects per scan so responses fit in cutoff_len=4096.
# 150 defects * ~25 tokens/defect (int 0-999 format) = ~3750 tokens for response
# Plus ~600 tokens for system + image placeholder = ~4350 total
# cutoff_len 4096 with int 0-999 format
MAX_DEFECTS_PER_SCAN = 150
RANDOM_SEED = 42

# Scans held out for early stopping validation.
# Picked: 8 and 9 (538 + 798 defects) - moderate density, distinct from train.
DEFAULT_HOLDOUT = ["Scan (8).jpg", "Scan (9).jpg"]


SYSTEM_PROMPT_INT = DETECTION_PROMPT_INT
SYSTEM_PROMPT_FLOAT = DETECTION_PROMPT_FLOAT


def cap_defects(annotations: list, max_count: int) -> list:
    """
    Cap defects using stratified sampling to keep all classes represented.
    Rare classes are kept in full, common classes are sampled.
    """
    if len(annotations) <= max_count:
        return annotations

    by_class = defaultdict(list)
    for ann in annotations:
        by_class[ann["label"]].append(ann)

    n_classes = len(by_class)
    if n_classes == 0:
        return annotations[:max_count]

    rng = random.Random(RANDOM_SEED)
    for label in by_class:
        rng.shuffle(by_class[label])

    sampled = []
    remaining = max_count
    for label, items in by_class.items():
        share = max(1, int(max_count * len(items) / len(annotations)))
        take = min(share, len(items), remaining)
        sampled.extend(items[:take])
        remaining -= take
        if remaining <= 0:
            break

    if remaining > 0:
        largest_class = max(by_class.keys(), key=lambda k: len(by_class[k]))
        already_taken = sum(1 for x in sampled if x["label"] == largest_class)
        available = by_class[largest_class][already_taken:already_taken + remaining]
        sampled.extend(available)

    return sampled


def convert_bbox(bbox: list, fmt: str) -> list:
    """Convert a float 0-1 bbox to the requested format."""
    x_min, y_min, x_max, y_max = bbox
    if fmt == "int_0_999":
        return [
            int(round(x_min * 999)),
            int(round(y_min * 999)),
            int(round(x_max * 999)),
            int(round(y_max * 999)),
        ]
    elif fmt == "float_0_1":
        return [round(x_min, 6), round(y_min, 6), round(x_max, 6), round(y_max, 6)]
    else:
        raise ValueError(f"unknown bbox format: {fmt}")


def build_image_ref(image_name: str, image_prefix: str) -> str:
    normalized = image_name.replace("\\", "/")
    prefix = image_prefix.strip("/")
    if prefix:
        return f"{prefix}/{normalized}"
    return normalized


def convert_entry(raw: dict, fmt: str, max_defects: int, image_prefix: str) -> dict:
    """Convert one JSONL entry to ShareGPT format."""
    image_name = raw["image"]
    annotations = raw["annotations"]

    capped_annotations = cap_defects(annotations, max_defects)
    if len(annotations) > len(capped_annotations):
        print(
            f"  {image_name}: capped {len(annotations)} -> {len(capped_annotations)} defects"
        )

    defects = []
    for ann in capped_annotations:
        defects.append({
            "label": ann["label"],
            "bbox": convert_bbox(ann["bbox"], fmt),
        })

    response_json = json.dumps({"defects": defects}, separators=(",", ":"))

    system_prompt = SYSTEM_PROMPT_INT if fmt == "int_0_999" else SYSTEM_PROMPT_FLOAT

    return {
        "conversations": [
            {"from": "human", "value": f"<image>{system_prompt}"},
            {"from": "gpt", "value": response_json},
        ],
        "images": [build_image_ref(image_name, image_prefix)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_JSONL),
                        help="Input Halide JSONL annotations")
    parser.add_argument("--output-train", default=str(OUTPUT_TRAIN),
                        help="Output ShareGPT train JSON")
    parser.add_argument("--output-val", default=str(OUTPUT_VAL),
                        help="Output ShareGPT val JSON")
    parser.add_argument("--image-prefix", default="scans",
                        help="Prefix for image paths in LLaMA-Factory data dir")
    parser.add_argument("--max-defects", type=int, default=MAX_DEFECTS_PER_SCAN,
                        help="Cap defects per image")
    parser.add_argument("--no-val", action="store_true",
                        help="Write all rows to train and an empty val file")
    parser.add_argument("--format", choices=["int_0_999", "float_0_1"],
                        default="int_0_999",
                        help="Bounding box format (default: int_0_999)")
    parser.add_argument("--holdout", nargs="*", default=DEFAULT_HOLDOUT,
                        help="Scan filenames to hold out for validation")
    args = parser.parse_args()

    print("=== Converting to ShareGPT Format ===\n")
    print(f"Format: {args.format}")
    print(f"Input: {args.input}")
    print(f"Image prefix: {args.image_prefix!r}")
    print(f"Cap: {args.max_defects} defects per scan")
    print(f"Holdout: {args.holdout if not args.no_val else '(disabled)'}\n")

    with open(args.input) as f:
        raw_data = [json.loads(line) for line in f]

    print(f"Loaded {len(raw_data)} entries from {args.input}")

    holdout_set = set() if args.no_val else set(args.holdout)
    train_entries = [r for r in raw_data if r["image"] not in holdout_set]
    val_entries = [r for r in raw_data if r["image"] in holdout_set]

    print(f"  train: {len(train_entries)} scans")
    print(f"  val:   {len(val_entries)} scans")

    train_sharegpt = [
        convert_entry(e, args.format, args.max_defects, args.image_prefix)
        for e in train_entries
    ]
    val_sharegpt = [
        convert_entry(e, args.format, args.max_defects, args.image_prefix)
        for e in val_entries
    ]

    output_train = Path(args.output_train)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    with open(output_train, "w") as f:
        json.dump(train_sharegpt, f, indent=2)
    print(f"\nWrote {len(train_sharegpt)} train entries to {output_train}")

    output_val = Path(args.output_val)
    output_val.parent.mkdir(parents=True, exist_ok=True)
    with open(output_val, "w") as f:
        json.dump(val_sharegpt, f, indent=2)
    print(f"Wrote {len(val_sharegpt)} val entries to {output_val}")

    def total_defects(entries):
        return sum(e["conversations"][1]["value"].count('"label"') for e in entries)

    print(f"\nTotal defects in train: {total_defects(train_sharegpt)}")
    print(f"Total defects in val:   {total_defects(val_sharegpt)}")

    if train_sharegpt:
        print("\nSample train entry (first conversation only):")
        print(f"  Human: {train_sharegpt[0]['conversations'][0]['value'][:100]}...")
        print(f"  GPT: {train_sharegpt[0]['conversations'][1]['value'][:200]}...")


if __name__ == "__main__":
    main()
