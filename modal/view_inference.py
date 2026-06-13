"""Modal backend for the inference viewer (tests/view_inference.py).

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

PROMPT = (
    "You are a film defect detection engine. Analyze the film scan and detect "
    "only physical defects that are on the film, scanner glass, holder, or "
    "scan surface. The image may be a positive film scan of an ordinary scene, "
    "a negative, a slide, a contact sheet, or a film scanner output. Detect "
    "defects that appear as dust spots, dirt blobs, thin abrasion lines, "
    "hair-like overlays, emulsion loss, chemical stains, or light leaks on "
    "top of the photographed content. If no clear "
    "surface artifact is visible, return {\"defects\": []}. Do not label "
    "subject matter as defects. Do not label grass, tree branches, eyelashes, "
    "fabric fibers, texture, grain, wires, shadows, printed text, or real hair "
    "inside the photographed scene as long_hair or short_hair. Use scratch "
    "only for thin physical abrasion or scan-surface lines, not object edges, "
    "stems, typography, or composition lines. Output a JSON object with a "
    "'defects' array. Each defect has: "
    "'label' (dust, dirt, scratch, long_hair, short_hair, emulsion_damage, "
    "chemical_stain, light_leak), "
    "optional 'confidence' from 0.0 to 1.0, "
    "'bbox' as 4 integers in the [0, 999] grid "
    "[x_min, y_min, x_max, y_max] (multiply by image width/height to get pixels). "
    "Return at most 150 defects. Prefer the clearest defects. Do not repeat "
    "the same label and bbox. If uncertain, return an empty defects array. "
    "Output JSON only, no explanation."
)

MODELS = {
    "base": "openbmb/MiniCPM-V-4.6",
    # v3 finetune, combined original + balanced augmented data.
    "finetuned": "/checkpoints/minicpm-v-4.6-merged-v3",
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


def _inference_for(model_key: str, image_path: str) -> dict:
    from PIL import Image
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = MODELS[model_key]
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    device = str(next(model.parameters()).device)
    load_seconds = round(time.time() - t0, 2)

    pil = Image.open(image_path).convert("RGB")
    result = _run_inference(model, processor, pil, device)
    parsed = _parse(result["text"])

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
def run_both(image_path: str) -> dict:
    """Run both models in the same container (one cold load, two inferences)."""
    out: dict[str, Any] = {"image_path": image_path, "results": {}}
    for key in ("base", "finetuned"):
        out["results"][key] = _inference_for(key, f"/uploads/{image_path}")
    return out


@app.local_entrypoint()
def main(image_path: str = "viewer_smoke_01.png", which: str = "both"):
    """Local entrypoint: invoke run_both / run_base / run_finetuned and print JSON.

    Usage:
      modal run modal/view_inference.py::main --image-path X.png --which both
    """
    import json as _json
    if which == "both":
        r = run_both.remote(image_path)
    elif which == "base":
        r = run_base.remote(image_path)
    elif which == "finetuned":
        r = run_finetuned.remote(image_path)
    else:
        raise SystemExit(f"unknown --which: {which}")
    print("===RESULT_START===")
    print(_json.dumps(r, indent=2))
    print("===RESULT_END===")
