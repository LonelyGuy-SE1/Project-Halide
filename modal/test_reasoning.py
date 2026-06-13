"""Modal T4 backend for testing the Nemotron reasoning model.

Mirrors the architecture of modal/view_inference.py but for the second stage
of the pipeline. Takes a defect summary dict, builds the production few-shot
message array, and generates a diagnosis.

The test uses the known-broken vision output (default-bbox hallucination)
to verify the Nemotron wrapper correctly ingests the structured messages
array and emits a properly-formatted diagnosis text.

Run:
  modal run modal/test_reasoning.py::main
  modal run modal/test_reasoning.py::main --scenario broken_vision_default_bbox
"""

from __future__ import annotations

import json
import time

import modal

from models.reasoning.prompts import build_messages as build_reasoning_messages

app = modal.App("halide-reasoning-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers[torch]==5.7.0",
        "accelerate>=1.8.0",
        "safetensors",
    )
    .add_local_python_source("models")
)

NEMOTRON_MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"


def build_chat_inputs(tokenizer, messages, device):
    """Return generate kwargs across Transformers chat-template variants."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    if hasattr(encoded, "to"):
        encoded = encoded.to(device)

    if has_input_ids(encoded):
        input_ids = encoded["input_ids"]
        return dict(encoded), input_ids.shape[-1]

    return {"input_ids": encoded}, encoded.shape[-1]


def has_input_ids(encoded) -> bool:
    try:
        return "input_ids" in encoded
    except (TypeError, RuntimeError):
        return False


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_reasoning_all(scenarios: list[dict]) -> list[dict]:
    """Run all scenarios in a single container. Model is loaded once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(NEMOTRON_MODEL_ID)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        NEMOTRON_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    device = str(next(model.parameters()).device)
    load_s = round(time.time() - t0, 2)

    results = []
    for s in scenarios:
        t1 = time.time()
        total = sum(s["defect_summary"].values())
        messages = build_reasoning_messages(
            film_type=s["film_type"],
            film_age_years=s["film_age_years"],
            storage=s["storage"],
            scan_resolution_dpi=s["scan_resolution_dpi"],
            defect_summary=s["defect_summary"],
            total_defects=total,
            spatial_evidence=s.get("spatial_evidence", {}),
        )
        inputs, prompt_length = build_chat_inputs(tokenizer, messages, device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_ids = output[0][prompt_length:]
        text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        gen_s = round(time.time() - t1, 2)
        results.append({
            "scenario_name": s["name"],
            "model_id": NEMOTRON_MODEL_ID,
            "device": device,
            "load_seconds": load_s,
            "generation_seconds": gen_s,
            "num_messages": len(messages),
            "roles": [m["role"] for m in messages],
            "diagnosis_text": text,
            "input_defect_summary": s["defect_summary"],
            "input_film_type": s["film_type"],
        })

    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_reasoning(
    defect_summary: dict,
    film_type: str,
    film_age_years: int,
    storage: str,
    scan_resolution_dpi: int,
) -> dict:
    """Single-scenario convenience wrapper."""
    out = run_reasoning_all.call([{
        "name": "single",
        "defect_summary": defect_summary,
        "film_type": film_type,
        "film_age_years": film_age_years,
        "storage": storage,
        "scan_resolution_dpi": scan_resolution_dpi,
    }])
    return out[0]


SCENARIOS = [
    {
        "name": "broken_vision_default_bbox",
        "defect_summary": {"long_hair": 1},
        "film_type": "Unknown 35mm",
        "film_age_years": 1,
        "storage": "unknown",
        "scan_resolution_dpi": 4000,
    },
    {
        "name": "gt_in_distribution_scan1",
        "defect_summary": {"dust": 868, "dirt": 401, "short_hair": 386, "long_hair": 122, "scratch": 1},
        "film_type": "Kodak Portra 400 (35mm)",
        "film_age_years": 2,
        "storage": "fridge, sealed",
        "scan_resolution_dpi": 4000,
    },
    {
        "name": "minimal_synthetic_scratch",
        "defect_summary": {"scratch": 1, "dust": 5},
        "film_type": "Ilford HP5 (35mm)",
        "film_age_years": 0,
        "storage": "fresh",
        "scan_resolution_dpi": 3200,
    },
]


@app.local_entrypoint()
def main(scenario: str = "all"):
    """Run one or all scenarios through the reasoning model.

    All scenarios in a single invocation share one container, so the model
    is loaded once. Use `--scenario all` to run all 3 scenarios.
    """
    if scenario == "all":
        scenarios = SCENARIOS
    else:
        scenarios = [s for s in SCENARIOS if s["name"] == scenario]
    if not scenarios:
        print(f"unknown scenario: {scenario}. available: {[s['name'] for s in SCENARIOS]}")
        return

    print(f"=== running {len(scenarios)} scenario(s) in one container ===")
    results = run_reasoning_all.remote(scenarios)

    print("===RESULT_START===")
    print(json.dumps(results, indent=2))
    print("===RESULT_END===")
