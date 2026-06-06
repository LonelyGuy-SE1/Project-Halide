"""Defect extractor. Takes a film scan and returns structured defect JSON.

This is a thin wrapper that re-exports `extract_defects` from the vision
inference module so the pipeline layer has a stable interface.
"""

from __future__ import annotations

from typing import Any

from models.vision.inference import extract_defects, extract_defects_from_path

__all__ = ["extract_defects", "extract_defects_from_path"]


def extract(image: Any) -> dict:
    """Top-level entry point used by the pipeline orchestrator."""
    return extract_defects(image)
