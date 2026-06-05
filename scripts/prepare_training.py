"""
Convert FilmDamageSimulator annotations to normalized JSONL format.
All bounding boxes are normalized [0.0-1.0] (x_min, y_min, x_max, y_max).
"""
import json
from pathlib import Path


SCANS_DIR = Path(__file__).parent.parent / "data" / "raw" / "FilmDamageSimulator" / "FilmDamageSimulator" / "scans"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "training_data.jsonl"


def get_image_dimensions(jpg_path: Path) -> tuple:
    """Get image width and height from file."""
    try:
        from PIL import Image
        with Image.open(jpg_path) as img:
            return img.size  # (width, height)
    except ImportError:
        # Fallback: assume standard film scan dimensions
        return (4944, 3396)


def polygon_to_bbox_norm(points: list, width: int, height: int) -> list:
    """Convert pixel polygon points to normalized bounding box [x_min, y_min, x_max, y_max]."""
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return [
        round(min(xs) / width, 6),
        round(min(ys) / height, 6),
        round(max(xs) / width, 6),
        round(max(ys) / height, 6),
    ]


def convert_scan(json_path: Path, jpg_path: Path) -> dict:
    """Convert a single scan's annotations to our format."""
    with open(json_path) as f:
        raw = json.load(f)

    width, height = get_image_dimensions(jpg_path)

    defects = []
    for key, item in raw.items():
        if "points" in item and "label" in item:
            label = item["label"]["name"].lower().replace(" ", "_")
            bbox = polygon_to_bbox_norm(item["points"], width, height)
            defects.append({
                "label": label,
                "bbox": bbox,
            })

    return {
        "image": str(jpg_path.name),
        "width": width,
        "height": height,
        "source": "film_damage_simulator",
        "annotations": defects,
    }


def main():
    print("=== Converting FilmDamageSimulator Annotations ===\n")

    # Find all scan pairs
    json_files = sorted(SCANS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotation files")

    all_annotations = []
    total_defects = 0

    for json_path in json_files:
        jpg_path = json_path.with_suffix(".jpg")
        if not jpg_path.exists():
            print(f"  Warning: No image for {json_path.name}")
            continue

        annotation = convert_scan(json_path, jpg_path)
        all_annotations.append(annotation)
        total_defects += len(annotation["annotations"])
        print(f"  {json_path.name}: {len(annotation['annotations'])} defects")

    # Write JSONL
    with open(OUTPUT_FILE, "w") as f:
        for item in all_annotations:
            f.write(json.dumps(item) + "\n")

    print(f"\n=== Conversion Complete ===")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total scans: {len(all_annotations)}")
    print(f"Total defects: {total_defects}")

    # Print class distribution
    class_counts = {}
    for item in all_annotations:
        for defect in item["annotations"]:
            label = defect["label"]
            class_counts[label] = class_counts.get(label, 0) + 1

    print("\nClass distribution:")
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
