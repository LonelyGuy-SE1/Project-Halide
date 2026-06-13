"""Main pipeline. Orchestrates vision extraction and diagnostic reasoning."""

from __future__ import annotations

import logging
import time
from typing import Any

from pipeline.diagnoser import diagnose
from pipeline.extractor import extract

logger = logging.getLogger(__name__)


def run_diagnosis(
    image: Any,
    film_type: str = "Unknown 35mm",
    film_age_years: int = 1,
    storage: str = "unknown",
    scan_resolution_dpi: int = 4000,
    metadata_confidence: str = "low",
) -> dict:
    """End-to-end: image -> defect JSON -> diagnosis + fixes.

    Returns a single dict with both stages' outputs and timing info.
    """
    started = time.perf_counter()

    logger.info("Stage 1: defect extraction")
    defect_result = extract(image)

    logger.info(
        "Stage 1 complete: %d defects (%s) in %.2fs",
        defect_result.get("defect_count", 0),
        defect_result.get("label_counts", {}),
        defect_result.get("inference_seconds", 0.0),
    )

    logger.info("Stage 2: Nemotron diagnosis")
    diagnosis_result = diagnose(
        defect_result,
        film_type=film_type,
        film_age_years=film_age_years,
        storage=storage,
        scan_resolution_dpi=scan_resolution_dpi,
        metadata_confidence=metadata_confidence,
    )

    total_elapsed = time.perf_counter() - started

    return {
        "film_metadata": {
            "film_type": film_type,
            "film_age_years": film_age_years,
            "storage": storage,
            "scan_resolution_dpi": scan_resolution_dpi,
            "metadata_confidence": metadata_confidence,
        },
        "defects": defect_result,
        "diagnosis": diagnosis_result,
        "total_seconds": round(total_elapsed, 3),
    }


__all__ = ["run_diagnosis"]
