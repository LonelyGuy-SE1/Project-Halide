"""Prepare 3 test images for Modal inference test.

Resizes images to max 1024px, saves as PNG to data/test_images/.
These get uploaded to the Modal volume and read by the inference test.
"""
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "test_images"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("01_indist_scan1.png", REPO / "data" / "raw" / "FilmDamageSimulator" / "FilmDamageSimulator" / "scans" / "Scan (1).jpg", "FilmDamageSimulator Scan (1) - in training"),
    ("02_ood_synthetic_scratch.png", REPO / "data" / "raw" / "FilmDamageSimulator" / "FilmDamageSimulator" / "synthetic" / "scratches" / "scratch-0001.png", "FilmDamageSimulator synthetic scratch - OOD"),
    ("03_ood_blueneg.png", REPO / "data" / "raw" / "BlueNeg" / "negative-preview-8bit" / "blue-corrupted" / "19880215B-30-tama-safari-bogor.preview.png", "BlueNeg real film (Kodak GA 100, 1988) - OOD"),
]

for name, src, desc in SOURCES:
    img = Image.open(src)
    if img.mode == "RGBA":
        # synthetic damage overlays are transparent on white in RGBA;
        # composite onto dark gray (mimicking negative film base) so the scratch is visible
        bg = Image.new("RGB", img.size, (40, 40, 40))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    out_path = OUT / name
    img.save(out_path, "PNG", optimize=True)
    print(f"{name}: {img.size}, {out_path.stat().st_size / 1024:.1f} KB - {desc}")
