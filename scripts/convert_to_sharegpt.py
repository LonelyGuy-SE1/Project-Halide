"""
Convert training_data.jsonl to ShareGPT format for LLaMA-Factory.
Each entry becomes:
{
  "conversations": [
    {"from": "human", "value": "<image>Detect all film defects..."},
    {"from": "gpt", "value": "{\"defects\": [...]}"}
  ],
  "images": ["scans/Scan (1).jpg"]
}
"""
import json
from pathlib import Path


INPUT_JSONL = Path(__file__).parent.parent / "data" / "training_data.jsonl"
OUTPUT_JSON = Path(__file__).parent.parent / "data" / "training_sharegpt.json"
SCANS_DIR = Path("scans")  # Relative path for LLaMA-Factory


SYSTEM_PROMPT = (
    "You are a film defect detection engine. Analyze the film scan and detect all visible defects. "
    "Output a JSON object with a 'defects' array. Each defect has: "
    "'label' (dust, dirt, scratch, long_hair, short_hair), "
    "'bbox' (normalized [x_min, y_min, x_max, y_max] from 0.0 to 1.0). "
    "Output JSON only, no explanation."
)


def convert_entry(raw: dict) -> dict:
    """Convert one JSONL entry to ShareGPT format."""
    image_name = raw["image"]
    annotations = raw["annotations"]

    # Build the defect JSON response
    defects = []
    for ann in annotations:
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

    # Load JSONL
    with open(INPUT_JSONL) as f:
        raw_data = [json.loads(line) for line in f]

    print(f"Loaded {len(raw_data)} entries from {INPUT_JSONL}")

    # Convert
    sharegpt_data = [convert_entry(entry) for entry in raw_data]

    # Write JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(sharegpt_data, f, indent=2)

    print(f"Wrote {len(sharegpt_data)} entries to {OUTPUT_JSON}")

    # Show sample
    print("\nSample entry:")
    print(json.dumps(sharegpt_data[0], indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
