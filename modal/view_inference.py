"""Modal backend for the inference viewer (scripts/view_inference.py).

Two T4 functions that load each model and return raw text + parsed JSON.
Images are passed as paths inside a shared Modal volume (uploaded by the
local viewer, no command-line arg length limits).

Costs:
  - T4: $0.59/hr = $0.000164/sec
  - Model load: ~30s = $0.005
  - Inference: ~5-15s = $0.001-0.003
  - Per test (both models): ~$0.012
"""
import json
import re
import time
from typing import Any

import modal

try:
    from models.vision.prompts import DETECTION_PROMPT_INT
except ModuleNotFoundError:
    # modal run uploads this file without the repository package.
    DETECTION_PROMPT_INT = (
        "You are a film defect detection engine. Analyze the film scan and detect "
        "only physical defects that are on the film, scanner glass, holder, or "
        "scan surface. The image may be a positive film scan of an ordinary scene, "
        "a negative, a slide, a contact sheet, or a film scanner output. Detect "
        "defects that appear as dust spots, dirt blobs, thin abrasion lines, "
        "hair-like overlays, emulsion loss, chemical stains, or light leaks on "
        "top of the photographed content. Return {\"defects\": []} only when no "
        "visible surface artifact is present. Do not return an empty array when "
        "obvious dark or light scratches, cracks, abrasion lines, peeled emulsion, "
        "or opaque dirt cross the subject content. Do not label subject matter as "
        "defects. Do not label grass, tree branches, eyelashes, fabric fibers, "
        "texture, grain, wires, shadows, printed text, or real hair inside the "
        "photographed scene as long_hair or short_hair. Use scratch only for thin "
        "physical abrasion or scan-surface lines, not object edges, stems, "
        "typography, or composition lines. Transparent or semi-transparent crack "
        "networks, lifted film bands, broken coating sheets, and fracture lines "
        "crossing a face or subject are defects, not scene content. Use "
        "emulsion_damage for broad peeled, cracked, torn, lifted, or missing "
        "emulsion regions. Use scratch for fine crack branches and abrasion "
        "lines. Output a JSON object with a "
        "'defects' array. Each defect has: 'label' (dust, dirt, scratch, "
        "long_hair, short_hair, emulsion_damage, chemical_stain, light_leak), "
        "optional 'confidence' from 0.0 to 1.0, 'bbox' as 4 integers in the "
        "[0, 999] grid [x_min, y_min, x_max, y_max] (multiply by image "
        "width/height to get pixels). Return at most 150 defects. Prefer the "
        "clearest defects. Do not repeat the same label and bbox. Output JSON "
        "only, no explanation."
    )

app = modal.App("halide-viewer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers[torch]==5.7.0",
        "torchvision",
        "huggingface_hub",
        "pillow",
    )
)

viewer_volume = modal.Volume.from_name("halide-viewer-uploads", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("halide-checkpoints", create_if_missing=True)

PROMPT = DETECTION_PROMPT_INT
ALLOWED_LABELS = {
    "dust",
    "dirt",
    "scratch",
    "long_hair",
    "short_hair",
    "emulsion_damage",
    "chemical_stain",
    "light_leak",
}
MIN_DEFECT_CONFIDENCE = 0.45

MODELS = {
    "base": "openbmb/MiniCPM-V-4.6",
    # v4 finetune, combined original + hard-negative procedural data.
    "finetuned": "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
    "finetuned_v4_stage1": "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
}


def _run_inference(model, processor, image_pil, device) -> dict:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        downsample_mode="4x",
        max_slice_nums=36,
    ).to(device)

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            downsample_mode="4x",
            max_new_tokens=2048,
            do_sample=False,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {"text": text, "inference_seconds": round(time.time() - t0, 2)}


def _run_inference_with_tiles(model, processor, image_pil, device) -> dict:
    full = _run_inference(model, processor, image_pil, device)
    parsed = _parse(full["text"])
    cleaned = _clean_defects(parsed.get("defects", []) if isinstance(parsed, dict) else [])
    tile_fallback_used = False
    tile_count = 0
    tile_seconds = 0.0

    if len(cleaned) == 0 and max(image_pil.size) >= 900:
        tile_fallback_used = True
        merged_defects: list[dict[str, Any]] = []
        for tile_count, (tile, tile_box) in enumerate(_iter_tiles(image_pil), start=1):
            tile_result = _run_inference(model, processor, tile, device)
            tile_seconds += float(tile_result.get("inference_seconds", 0.0) or 0.0)
            tile_parsed = _parse(tile_result["text"])
            tile_cleaned = _clean_defects(
                tile_parsed.get("defects", []) if isinstance(tile_parsed, dict) else []
            )
            merged_defects.extend(
                _remap_tile_defects(
                    tile_cleaned,
                    tile_box=tile_box,
                    image_size=image_pil.size,
                )
            )
            if tile_count >= 9:
                break
        cleaned = _dedupe_defects(merged_defects)
        parsed = dict(parsed) if isinstance(parsed, dict) else {}
        parsed["defects"] = cleaned
        parsed["_tile_fallback_used"] = True
        parsed["_tile_count"] = tile_count
        parsed["_full_frame_defect_count"] = 0

    full["parsed_json"] = parsed
    full["tile_fallback_used"] = tile_fallback_used
    full["tile_count"] = tile_count
    full["tile_inference_seconds"] = round(tile_seconds, 2)
    full["inference_seconds"] = round(
        float(full.get("inference_seconds", 0.0) or 0.0) + tile_seconds,
        2,
    )
    return full


def _parse(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            fragments = _parse_defect_fragments(text)
            if fragments:
                return {
                    "defects": fragments,
                    "_parse_error": str(exc),
                    "_parse_warning": "salvaged_defect_fragments",
                }
            return {"_parse_error": str(exc), "_raw": text}
    return {"_parse_error": "no_json_object", "_raw": text}


def _parse_defect_fragments(text: str) -> list[dict]:
    fragments = []
    for match in re.finditer(r"\{[^{}]*\"label\"[^{}]*\"bbox\"\s*:\s*\[[^\]]+\][^{}]*\}", text):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            fragments.append(candidate)
    return fragments


def _clean_defects(raw_defects: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_defects, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for raw in raw_defects:
        if not isinstance(raw, dict):
            continue
        label = raw.get("label")
        if label not in ALLOWED_LABELS:
            continue
        bbox = _normalize_bbox(raw.get("bbox"))
        if bbox is None:
            continue
        confidence = raw.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
        if confidence is not None and confidence < MIN_DEFECT_CONFIDENCE:
            continue
        out: dict[str, Any] = {"label": label, "bbox": [round(v, 6) for v in bbox]}
        if confidence is not None:
            out["confidence"] = round(confidence, 4)
        cleaned.append(out)
    return cleaned


def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 1
        and isinstance(bbox[0], (list, tuple))
    ):
        bbox = bbox[0]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    max_val = max(x0, y0, x1, y1)
    all_whole = all(float(v).is_integer() for v in (x0, y0, x1, y1))
    if all_whole and max_val > 1.5:
        x0 /= 999.0
        y0 /= 999.0
        x1 /= 999.0
        y1 /= 999.0
        if not all(-0.001 <= v <= 1.002 for v in (x0, y0, x1, y1)):
            return None
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6))


def _iter_tiles(image_pil) -> list[tuple[Any, tuple[int, int, int, int]]]:
    width, height = image_pil.size
    tile_side = min(960, max(width, height))
    tile_width = min(width, tile_side)
    tile_height = min(height, tile_side)
    xs = _axis_positions(width, tile_width, 0.35)
    ys = _axis_positions(height, tile_height, 0.35)
    center = ((width - tile_width) // 2, (height - tile_height) // 2)
    ordered_positions = [(x, y) for y in ys for x in xs]
    ordered_positions.insert(0, center)
    seen: set[tuple[int, int]] = set()
    tiles = []
    for x, y in ordered_positions:
        x = max(0, min(width - tile_width, x))
        y = max(0, min(height - tile_height, y))
        if (x, y) in seen:
            continue
        seen.add((x, y))
        box = (x, y, x + tile_width, y + tile_height)
        tiles.append((image_pil.crop(box), box))
        if len(tiles) >= 9:
            break
    return tiles


def _axis_positions(length: int, tile_length: int, overlap: float) -> list[int]:
    if length <= tile_length:
        return [0]
    stride = max(1, int(round(tile_length * (1.0 - overlap))))
    limit = length - tile_length
    positions = list(range(0, limit + 1, stride))
    positions.extend([limit, limit // 2])
    return sorted(set(max(0, min(limit, pos)) for pos in positions))


def _remap_tile_defects(
    defects: list[dict[str, Any]],
    *,
    tile_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    x0, y0, x1, y1 = tile_box
    tile_width = max(1, x1 - x0)
    tile_height = max(1, y1 - y0)
    remapped: list[dict[str, Any]] = []
    for defect in defects:
        bbox = _normalize_bbox(defect.get("bbox"))
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        out = {
            "label": defect.get("label"),
            "bbox": [
                round((x0 + bx0 * tile_width) / image_width, 6),
                round((y0 + by0 * tile_height) / image_height, 6),
                round((x0 + bx1 * tile_width) / image_width, 6),
                round((y0 + by1 * tile_height) / image_height, 6),
            ],
        }
        if defect.get("confidence") is not None:
            out["confidence"] = defect.get("confidence")
        remapped.append(out)
    return remapped


def _dedupe_defects(defects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen = set()
    for defect in defects:
        label = defect.get("label")
        bbox = _normalize_bbox(defect.get("bbox"))
        if label not in ALLOWED_LABELS or bbox is None:
            continue
        key = (label, bbox)
        if key in seen:
            continue
        seen.add(key)
        if any(existing.get("label") == label and _bbox_iou(existing.get("bbox"), bbox) >= 0.72 for existing in merged):
            continue
        out = {"label": label, "bbox": [round(v, 6) for v in bbox]}
        if defect.get("confidence") is not None:
            out["confidence"] = defect.get("confidence")
        merged.append(out)
    return merged


def _bbox_iou(a: Any, b: Any) -> float:
    box_a = _normalize_bbox(a)
    box_b = _normalize_bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _inference_for(model_key: str, image_path: str, model_id_override: str | None = None) -> dict:
    from PIL import Image
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = model_id_override or MODELS[model_key]
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    device = str(next(model.parameters()).device)
    load_seconds = round(time.time() - t0, 2)

    pil = Image.open(image_path).convert("RGB")
    result = _run_inference_with_tiles(model, processor, pil, device)
    parsed = result.get("parsed_json", {})

    del model, processor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "device": device,
        "load_seconds": load_seconds,
        "inference_seconds": result["inference_seconds"],
        "tile_fallback_used": result.get("tile_fallback_used", False),
        "tile_count": result.get("tile_count", 0),
        "tile_inference_seconds": result.get("tile_inference_seconds", 0.0),
        "image_size": list(pil.size),
        "raw_output": result["text"],
        "parsed_json": parsed,
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={"/uploads": viewer_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=15 * 60,
)
def run_base(image_path: str) -> dict:
    return _inference_for("base", f"/uploads/{image_path}")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/uploads": viewer_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=15 * 60,
)
def run_finetuned(image_path: str) -> dict:
    return _inference_for("finetuned", f"/uploads/{image_path}")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/uploads": viewer_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=15 * 60,
)
def run_model(image_path: str, model_id: str) -> dict:
    return _inference_for("custom", f"/uploads/{image_path}", model_id_override=model_id)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/uploads": viewer_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=15 * 60,
)
def run_both(image_path: str, finetuned_model: str = MODELS["finetuned"]) -> dict:
    """Run both models in the same container (one cold load, two inferences)."""
    out: dict[str, Any] = {"image_path": image_path, "results": {}}
    out["results"]["base"] = _inference_for("base", f"/uploads/{image_path}")
    out["results"]["finetuned"] = _inference_for(
        "finetuned",
        f"/uploads/{image_path}",
        model_id_override=finetuned_model,
    )
    return out


@app.local_entrypoint()
def main(
    image_path: str = "viewer_smoke_01.png",
    which: str = "both",
    finetuned_model: str = MODELS["finetuned"],
):
    """Local entrypoint: invoke run_both / run_base / run_finetuned and print JSON.

    Usage:
      modal run modal/view_inference.py::main --image-path X.png --which both
      modal run modal/view_inference.py::main --image-path X.png --which finetuned --finetuned-model /checkpoints/minicpm-v-4.6-merged-v4-stage1
    """
    import json as _json
    if which == "both":
        r = run_both.remote(image_path, finetuned_model)
    elif which == "base":
        r = run_base.remote(image_path)
    elif which == "finetuned":
        r = run_model.remote(image_path, finetuned_model)
    else:
        raise SystemExit(f"unknown --which: {which}")
    print("===RESULT_START===")
    print(_json.dumps(r, indent=2))
    print("===RESULT_END===")
