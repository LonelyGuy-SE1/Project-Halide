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

Cap on defects per scan is required so the response fits within the model's
context window together with the expanded image placeholder (~600 tokens).
Without this cap, scans with 1500+ dust particles produce 25k+ token
responses which get truncated by cutoff_len and break training.
"""
import json
import random
from collections import defaultdict
from pathlib import Path


INPUT_JSONL = Path(__file__).parent.parent / "data" / "training_data.jsonl"
OUTPUT_JSON = Path(__file__).parent.parent / "data" / "training_sharegpt.json"
SCANS_DIR = Path("scans")  # Relative path for LLaMA-Factory

# Cap defects per scan so responses fit in cutoff_len=2560.
# 150 defects * ~17 tokens/defect = ~2550 tokens for response
# Plus ~600 tokens for system + image placeholder = ~3150 total
# Set cutoff_len to 3072 in the training config.
MAX_DEFECTS_PER_SCAN = 150
RANDOM_SEED = 42


SYSTEM_PROMPT = (
    "You are a film defect detection engine. Analyze the film scan and detect all visible defects. "
    "Output a JSON object with a 'defects' array. Each defect has: "
    "'label' (dust, dirt, scratch, long_hair, short_hair), "
    "'bbox' (normalized [x_min, y_min, x_max, y_max] from 0.0 to 1.0). "
    "Output JSON only, no explanation."
)


def cap_defects(annotations: list, max_count: int) -> list:
    """
    Cap defects using stratified sampling to keep all classes represented.
    Rare classes are kept in full, common classes are sampled.
    """
    if len(annotations) <= max_count:
        return annotations

    # Group by class
    by_class = defaultdict(list)
    for ann in annotations:
        by_class[ann["label"]].append(ann)

    # Reserve at least N per class, then distribute remainder proportionally
    n_classes = len(by_class)
    if n_classes == 0:
        return annotations[:max_count]

    # Simple approach: shuffle within each class, then take proportionally
    rng = random.Random(RANDOM_SEED)
    for label in by_class:
        rng.shuffle(by_class[label])

    # Each class gets a proportional slice of max_count
    sampled = []
    remaining = max_count
    for label, items in by_class.items():
        share = max(1, int(max_count * len(items) / len(annotations)))
        take = min(share, len(items), remaining)
        sampled.extend(items[:take])
        remaining -= take
        if remaining <= 0:
            break

    # If we still have room, fill from the largest class
    if remaining > 0:
        largest_class = max(by_class.keys(), key=lambda k: len(by_class[k]))
        already_taken = sum(1 for x in sampled if x["label"] == largest_class)
        available = by_class[largest_class][already_taken:already_taken + remaining]
        sampled.extend(available)

    return sampled


def convert_entry(raw: dict) -> dict:
    """Convert one JSONL entry to ShareGPT format."""
    image_name = raw["image"]
    annotations = raw["annotations"]

    # Cap defects to keep response within context window
    capped_annotations = cap_defects(annotations, MAX_DEFECTS_PER_SCAN)
    if len(annotations) > len(capped_annotations):
        print(
            f"  {image_name}: capped {len(annotations)} -> {len(capped_annotations)} defects"
        )

    # Build the defect JSON response
    defects = []
    for ann in capped_annotations:
        defects.append({
            "label": ann["label"],
            "bbox": ann["bbox"],
        })

    response_json = json.dumps({"defects": defects}, separators=(",", ":"))

    return {
        "conversations": [
            {"from": "human", "value": f"<image>{SYSTEM_PROMPT}"},
            {"from": "gpt", "value": response_json},
        ],
        "images": ["scans/" + image_name],
    }


def main():
    print("=== Converting to ShareGPT Format ===\n")
    print(f"Cap: {MAX_DEFECTS_PER_SCAN} defects per scan\n")

    # Load JSONL
    with open(INPUT_JSONL) as f:
        raw_data = [json.loads(line) for line in f]

    print(f"Loaded {len(raw_data)} entries from {INPUT_JSONL}")

    # Convert
    sharegpt_data = [convert_entry(entry) for entry in raw_data]

    # Write JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(sharegpt_data, f, indent=2)

    print(f"\nWrote {len(sharegpt_data)} entries to {OUTPUT_JSON}")

    # Show stats
    total_defects = 0
    for entry in sharegpt_data:
        # Parse response to count defects
        resp = entry["conversations"][1]["value"]
        total_defects += resp.count('"label"')
    print(f"Total defects in training set: {total_defects}")

    # Show sample
    print("\nSample entry (first conversation only):")
    print(f"  Human: {sharegpt_data[0]['conversations'][0]['value'][:100]}...")
    print(f"  GPT: {sharegpt_data[0]['conversations'][1]['value'][:200]}...")


if __name__ == "__main__":
    main()

