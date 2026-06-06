"""Modal T4 backend for testing the Nemotron reasoning model.

Mirrors the architecture of modal/view_inference.py but for the second stage
of the pipeline. Takes a defect summary dict, builds the few-shot message
array via inline prompts, and generates a diagnosis.

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

app = modal.App("halide-reasoning-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.0",
        "transformers==4.48.0",
        "accelerate==1.2.0",
        "safetensors",
    )
)

NEMOTRON_MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"


SYSTEM_PROMPT = (
    "You are a senior analog film lab technician with 30 years of experience "
    "in darkroom printing, negative inspection, and equipment repair. You are "
    "diagnosing the physical root cause of degradation in a film scan and "
    "prescribing specific, actionable physical fixes a lab can perform."
)


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "## Defect report\n"
            "{\n"
            '  "film_type": "Kodak Portra 400 (35mm)",\n'
            '  "film_age_years": 2,\n'
            '  "storage": "fridge, sealed",\n'
            '  "defects": [\n'
            '    {"label": "dust", "bbox_count": 87},\n'
            '    {"label": "dirt", "bbox_count": 12}\n'
            '  ],\n'
            '  "scan_resolution_dpi": 4000\n'
            "}\n\n"
            "What is the root cause and what physical fixes do you recommend?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Root cause\n"
            "High dust and dirt count on a recently-shot, properly stored roll "
            "indicates contamination accumulated on the negative during "
            "scanning, not deterioration of the film itself. The scanner's "
            "dust-removal hardware (ICE/DEEP) is either disabled or "
            "ineffective at 4000 dpi.\n\n"
            "## Physical fixes\n"
            "1. Clean the scanner glass and film holder with an anti-static "
            "cloth and PEC*PAD swabs; re-scan.\n"
            "2. Enable hardware-based dust removal in the scanner software "
            "(e.g., Digital ICE for Kodak, DEEP for Noritsu) at a moderate "
            "strength (level 3 of 5).\n"
            "3. If hardware removal is unavailable, apply careful digital "
            "spotting in Photoshop using the healing brush; budget 30-60 "
            "minutes per frame.\n"
            "4. Inspect the original negative under a loupe for any residual "
            "physical particles and gently blow off with a rocket air blower "
            "before re-scanning."
        ),
    },
    {
        "role": "user",
        "content": (
            "## Defect report\n"
            "{\n"
            '  "film_type": "Ilford HP5 (120 medium format)",\n'
            '  "film_age_years": 15,\n'
            '  "storage": "shoe box, attic",\n'
            '  "defects": [\n'
            '    {"label": "long_hair", "bbox_count": 4},\n'
            '    {"label": "scratch", "bbox_count": 1}\n'
            '  ],\n'
            '  "scan_resolution_dpi": 3200\n'
            "}\n\n"
            "What is the root cause and what physical fixes do you recommend?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Root cause\n"
            "Long hair and scratches on an aged, poorly stored roll indicate "
            "physical abrasion from dust and debris inside the storage "
            "container. Attic storage subjects film to temperature and "
            "humidity swings that warp the emulsion and make it brittle.\n\n"
            "## Physical fixes\n"
            "1. Inspect the negative under a 10x loupe to confirm hair is on "
            "the emulsion side (matte) vs. base side (glossy). Emulsion-side "
            "hair cannot be wet-wiped safely.\n"
            "2. If base-side only: gently wipe with a PEC*PAD soaked in "
            "Edwal Negative Cleaner, then PEC*PAD with distilled water, "
            "then dry.\n"
            "3. If emulsion-side: do not wipe. Use Print File archival "
            "sleeves and ship to a lab with wet-mount scanning (e.g., "
            "Northcoast Photographic).\n"
            "4. The single scratch is most likely from a dirty film "
            "transport. Have the scanner cleaned and the film path "
            "inspected; consider wet-gate scanning to mask the scratch.\n"
            "5. Long-term: transfer remaining unexposed rolls to a "
            "freezer at -18C in sealed bags with desiccant."
        ),
    },
    {
        "role": "user",
        "content": (
            "## Defect report\n"
            "{\n"
            '  "film_type": "CineStill 800T (35mm)",\n'
            '  "film_age_years": 0,\n'
            '  "storage": "fresh, refrigerated by manufacturer",\n'
            '  "defects": [\n'
            '    {"label": "dust", "bbox_count": 156},\n'
            '    {"label": "dirt", "bbox_count": 23},\n'
            '    {"label": "short_hair", "bbox_count": 8}\n'
            '  ],\n'
            '  "scan_resolution_dpi": 6000\n'
            "}\n\n"
            "What is the root cause and what physical fixes do you recommend?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Root cause\n"
            "Excessive dust and dirt on a brand-new refrigerated roll at 6000 "
            "dpi scan suggests a combination of (a) the static-prone "
            "CineStill base attracting particles during loading, and (b) "
            "scanner-side contamination. The hair count points to the "
            "scanner's film transport path.\n\n"
            "## Physical fixes\n"
            "1. Anti-static treatment is critical for CineStill: discharge "
            "the negative with a Zerostat gun on low setting 30 cm from the "
            "film before scanning.\n"
            "2. Clean the scanner glass, film holder, and feed rollers with "
            "PEC*PAD swabs and reagent-grade isopropyl alcohol.\n"
            "3. Use a static-discharge ionizing bar (e.g., Simco-Ion) at the "
            "scanner input if available.\n"
            "4. Re-scan with hardware dust removal at level 4 of 5. "
            "CineStill's halated emulsion responds well to Digital ICE.\n"
            "5. For the short hairs, inspect the film path under magnification "
            "and remove any visible lint from the rollers with tweezers."
        ),
    },
]


def build_messages(film_type, film_age_years, storage, scan_resolution_dpi, defect_summary, total_defects):
    payload = {
        "film_type": film_type,
        "film_age_years": film_age_years,
        "storage": storage,
        "defects": [
            {"label": label, "bbox_count": count}
            for label, count in sorted(defect_summary.items())
        ],
        "scan_resolution_dpi": scan_resolution_dpi,
        "total_defect_count": total_defects,
    }
    user_msg = (
        "## Defect report\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "What is the root cause and what physical fixes do you recommend?"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": user_msg})
    return messages


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
    model = AutoModelForCausalLM.from_pretrained(
        NEMOTRON_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    device = str(next(model.parameters()).device)
    load_s = round(time.time() - t0, 2)

    results = []
    for s in scenarios:
        t1 = time.time()
        total = sum(s["defect_summary"].values())
        messages = build_messages(
            film_type=s["film_type"],
            film_age_years=s["film_age_years"],
            storage=s["storage"],
            scan_resolution_dpi=s["scan_resolution_dpi"],
            defect_summary=s["defect_summary"],
            total_defects=total,
        )
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_ids = output[0][input_ids.shape[-1]:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
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
