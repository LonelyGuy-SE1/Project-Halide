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

PROMPT = (
    "You are a film defect detection engine. Analyze the film scan and detect "
    "all visible defects. Output a JSON object with a 'defects' array. Each "
    "defect has: 'label' (dust, dirt, scratch, long_hair, short_hair), 'bbox' "
    "(normalized [x_min, y_min, x_max, y_max] from 0.0 to 1.0). Output JSON "
    "only, no explanation."
)

MODELS = {
    "base": "openbmb/MiniCPM-V-4_6",
    "finetuned": "Lonelyguyse1/halide-vision",
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
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception as exc:
            return {"_parse_error": str(exc), "_raw": text}
    return {"_parse_error": "no_json_object", "_raw": text}


def _inference_for(model_key: str, image_path: str) -> dict:
    from PIL import Image
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = MODELS[model_key]
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
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
    volumes={"/uploads": viewer_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=15 * 60,
)
def run_base(image_path: str) -> dict:
    return _inference_for("base", f"/uploads/{image_path}")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/uploads": viewer_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=15 * 60,
)
def run_finetuned(image_path: str) -> dict:
    return _inference_for("finetuned", f"/uploads/{image_path}")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/uploads": viewer_volume},
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
