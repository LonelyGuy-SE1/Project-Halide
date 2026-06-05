"""
Draw bounding boxes on a sample scan to verify annotation conversion.
Creates examples/ directory with annotated images.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
SCANS_DIR = Path(__file__).parent.parent / "data" / "raw" / "FilmDamageSimulator" / "FilmDamageSimulator" / "scans"
JSONL_PATH = Path(__file__).parent.parent / "data" / "training_data.jsonl"


# Defect class colors (RGB)
CLASS_COLORS = {
    "dust": (255, 0, 0),        # Red
    "dirt": (255, 165, 0),      # Orange
    "scratch": (255, 255, 0),   # Yellow
    "long_hair": (0, 255, 0),   # Green
    "short_hair": (0, 255, 255), # Cyan
}


def draw_bounding_boxes(image_path: str, annotations: list, output_path: str):
    """Draw bounding boxes on image and save."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for ann in annotations:
        label = ann["label"]
        bbox = ann["bbox"]  # [x_min, y_min, x_max, y_max] normalized

        # Convert normalized to pixel coordinates
        x_min = bbox[0] * img.width
        y_min = bbox[1] * img.height
        x_max = bbox[2] * img.width
        y_max = bbox[3] * img.height

        # Get color for this class
        color = CLASS_COLORS.get(label, (255, 255, 255))

        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        # Draw label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((x_min, y_min - 25), label, fill=color, font=font)

    # Save
    img.save(output_path)
    print(f"Saved: {output_path}")


def main():
    print("=== Creating Visual Verification Examples ===\n")

    # Create examples directory
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load JSONL
    with open(JSONL_PATH) as f:
        dataset = [json.loads(line) for line in f]

    # Process first 3 scans
    for scan_data in dataset[:3]:
        image_name = scan_data["image"]
        image_path = SCANS_DIR / image_name
        annotations = scan_data["annotations"]

        if not image_path.exists():
            print(f"Warning: {image_path} not found")
            continue

        output_path = EXAMPLES_DIR / f"annotated_{image_name.replace('.jpg', '.png')}"
        draw_bounding_boxes(str(image_path), annotations, str(output_path))
        print(f"  {image_name}: {len(annotations)} bounding boxes drawn")

    print(f"\n=== Done ===")
    print(f"Examples saved to: {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()
