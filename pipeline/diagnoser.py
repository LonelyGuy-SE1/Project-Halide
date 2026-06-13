"""Diagnoser. Takes defect JSON and user metadata, returns diagnosis and fixes."""

from __future__ import annotations

import logging
import time
from typing import Any

from data.schemas import spatial_summary
from config import DEFAULT_REASONING_MODEL_ID
from models.reasoning.nemotron_wrapper import get_reasoner
from models.reasoning.prompts import SYSTEM_PROMPT, build_messages

logger = logging.getLogger(__name__)


def diagnose(
    defect_result: dict,
    film_type: str,
    film_age_years: int,
    storage: str,
    scan_resolution_dpi: int,
    metadata_confidence: str = "low",
) -> dict:
    """Run Nemotron reasoning over a defect result + user metadata.

    Returns a dict with the raw text response and timing metadata.
    """
    started = time.perf_counter()

    label_counts = defect_result.get("label_counts", {}) or {}
    defects = defect_result.get("defects", []) or []
    total = defect_result.get("defect_count", 0) or sum(label_counts.values())
    spatial = spatial_summary(defects)

    if int(total or 0) <= 0:
        elapsed = time.perf_counter() - started
        return {
            "diagnosis_text": _no_defect_diagnosis_text(),
            "reasoning_seconds": round(elapsed, 3),
            "model_path": f"{DEFAULT_REASONING_MODEL_ID} (skipped, no validated defects)",
            "system_prompt": SYSTEM_PROMPT,
            "skipped_reasoning": "no_validated_defects",
            "input_defect_summary": {
                "label_counts": label_counts,
                "total": total,
                "spatial_evidence": spatial,
                "metadata_confidence": metadata_confidence,
            },
        }

    reasoner = get_reasoner()

    messages = build_messages(
        film_type=film_type,
        film_age_years=film_age_years,
        storage=storage,
        scan_resolution_dpi=scan_resolution_dpi,
        metadata_confidence=metadata_confidence,
        defect_summary=label_counts,
        total_defects=total,
        spatial_evidence=spatial,
    )

    logger.info(
        "Running Nemotron diagnosis "
        "(film=%s, age=%d, storage=%s, metadata_confidence=%s, total_defects=%d)",
        film_type,
        film_age_years,
        storage,
        metadata_confidence,
        total,
    )
    text = reasoner.generate(messages=messages)
    elapsed = time.perf_counter() - started

    return {
        "diagnosis_text": text,
        "reasoning_seconds": round(elapsed, 3),
        "model_path": reasoner.model_path,
        "system_prompt": SYSTEM_PROMPT,
        "input_defect_summary": {
            "label_counts": label_counts,
            "total": total,
            "spatial_evidence": spatial,
            "metadata_confidence": metadata_confidence,
        },
    }


def _no_defect_diagnosis_text() -> str:
    return (
        "## Root cause\n"
        "No physical root cause is diagnosed because the vision stage returned "
        "zero validated film or scanner defects.\n\n"
        "## Evidence\n"
        "- The validated defect count is 0.\n"
        "- No dust, dirt, scratch, long hair, or short hair boxes survived "
        "schema validation.\n"
        "- User metadata is not enough to diagnose a fault without visual "
        "defect evidence.\n\n"
        "## Physical fixes\n"
        "1. Do not clean, wipe, or service the film based on this result alone.\n"
        "2. Inspect the negative or slide under a loupe if the scan looks wrong "
        "to you.\n"
        "3. Re-scan a small crop at the same settings if you suspect a false "
        "negative.\n"
        "4. If this was not a film scan, upload a true scan, negative, slide, "
        "or contact-sheet crop.\n\n"
        "## Confidence\n"
        "High confidence that Halide has no validated defect evidence in this "
        "run. This is not proof that the film is perfect.\n\n"
        "## Next inspection\n"
        "Compare the original scan with a re-scan after a basic blower-only "
        "dust pass. If visible marks persist, treat this run as a false "
        "negative and inspect manually."
    )


__all__ = ["diagnose"]
