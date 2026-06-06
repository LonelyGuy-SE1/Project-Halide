"""Diagnoser. Takes defect JSON and user metadata, returns diagnosis and fixes."""

from __future__ import annotations

import logging
import time
from typing import Any

from models.reasoning.nemotron_wrapper import get_reasoner
from models.reasoning.prompts import SYSTEM_PROMPT, build_messages

logger = logging.getLogger(__name__)


def diagnose(
    defect_result: dict,
    film_type: str,
    film_age_years: int,
    storage: str,
    scan_resolution_dpi: int,
) -> dict:
    """Run Nemotron reasoning over a defect result + user metadata.

    Returns a dict with the raw text response and timing metadata.
    """
    started = time.perf_counter()
    reasoner = get_reasoner()

    label_counts = defect_result.get("label_counts", {}) or {}
    total = defect_result.get("defect_count", 0) or sum(label_counts.values())

    messages = build_messages(
        film_type=film_type,
        film_age_years=film_age_years,
        storage=storage,
        scan_resolution_dpi=scan_resolution_dpi,
        defect_summary=label_counts,
        total_defects=total,
    )

    logger.info(
        "Running Nemotron diagnosis (film=%s, age=%d, storage=%s, total_defects=%d)",
        film_type, film_age_years, storage, total,
    )
    text = reasoner.generate(prompt=messages, system=SYSTEM_PROMPT)
    elapsed = time.perf_counter() - started

    return {
        "diagnosis_text": text,
        "reasoning_seconds": round(elapsed, 3),
        "model_path": reasoner.model_path,
        "system_prompt": SYSTEM_PROMPT,
        "input_defect_summary": {
            "label_counts": label_counts,
            "total": total,
        },
    }


__all__ = ["diagnose"]
