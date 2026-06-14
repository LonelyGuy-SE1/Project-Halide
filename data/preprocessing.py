"""Image preprocessing and visualization utilities."""

from __future__ import annotations

import hashlib
import io
import base64
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

from data.schemas import LABEL_DISPLAY_NAMES, bbox_to_pixels

LABEL_STYLE = {
    "dust": ((245, 158, 11), 2),
    "dirt": ((217, 119, 6), 2),
    "scratch": ((220, 38, 38), 3),
    "long_hair": ((124, 58, 237), 2),
    "short_hair": ((8, 145, 178), 2),
    "emulsion_damage": ((226, 232, 240), 3),
    "chemical_stain": ((22, 163, 74), 3),
    "light_leak": ((244, 114, 182), 3),
}
DEFAULT_STYLE = ((255, 255, 255), 2)


def load_image(image: str | Path | Image.Image) -> Image.Image:
    """Load an image-like value and return RGB PIL Image."""
    if isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.open(image)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode == "RGBA":
        background = Image.new("RGB", pil.size, (24, 22, 20))
        background.paste(pil, mask=pil.getchannel("A"))
        return background
    if pil.mode != "RGB":
        return pil.convert("RGB")
    return pil.copy()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    load_image(image).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def image_to_data_uri(
    image: Image.Image,
    *,
    max_side: int = 1800,
    image_format: str = "JPEG",
    quality: int = 92,
) -> str:
    """Return a browser-openable image data URI for review previews."""
    pil = resize_for_preview(load_image(image), max_side=max_side)
    fmt = image_format.upper()
    buf = io.BytesIO()
    if fmt in {"JPG", "JPEG"}:
        pil = pil.convert("RGB")
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    elif fmt == "PNG":
        pil.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        raise ValueError(f"unsupported image_format: {image_format}")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_sha256(image: Image.Image | bytes) -> str:
    if isinstance(image, bytes):
        payload = image
    else:
        payload = image_to_png_bytes(image)
    return hashlib.sha256(payload).hexdigest()


def resize_for_preview(image: Image.Image, max_side: int = 1400) -> Image.Image:
    pil = load_image(image)
    if max(pil.size) <= max_side:
        return pil
    out = pil.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def draw_defects(
    image: Image.Image,
    defects: Iterable[dict[str, Any]],
    *,
    title: str | None = None,
    max_boxes: int = 300,
) -> Image.Image:
    """Draw normalized defect boxes onto an RGB copy of an image."""
    out = load_image(image)
    draw = ImageDraw.Draw(out)
    width, height = out.size
    font = ImageFont.load_default()

    if title:
        draw.rectangle((0, 0, min(width, 440), 24), fill=(12, 10, 9))
        draw.text((8, 6), title, fill=(254, 243, 199), font=font)

    drawn = 0
    for defect in defects:
        if drawn >= max_boxes:
            break
        label = str(defect.get("label", "unknown"))
        pixels = bbox_to_pixels(defect.get("bbox"), width, height)
        if pixels is None:
            continue
        x_min, y_min, x_max, y_max = pixels
        color, line_width = LABEL_STYLE.get(label, DEFAULT_STYLE)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=line_width)

        label_text = LABEL_DISPLAY_NAMES.get(label, label)
        text_bbox = draw.textbbox((x_min, max(0, y_min - 16)), label_text, font=font)
        draw.rectangle(text_bbox, fill=(12, 10, 9))
        draw.text((text_bbox[0] + 1, text_bbox[1]), label_text, fill=color, font=font)
        drawn += 1

    return out


__all__ = [
    "DEFAULT_STYLE",
    "LABEL_STYLE",
    "draw_defects",
    "image_sha256",
    "image_to_data_uri",
    "image_to_png_bytes",
    "load_image",
    "resize_for_preview",
]
