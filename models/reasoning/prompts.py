"""Few-shot prompt templates for the Nemotron diagnostic reasoner."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a senior analog film lab technician with 30 years of experience "
    "in darkroom printing, negative inspection, and equipment repair. You are "
    "diagnosing the physical root cause of degradation in a film scan and "
    "prescribing specific, actionable physical fixes a lab can perform."
)


FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
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


def build_user_prompt(
    film_type: str,
    film_age_years: int,
    storage: str,
    scan_resolution_dpi: int,
    defect_summary: dict[str, int],
    total_defects: int,
) -> str:
    """Build the user message for the current diagnosis request."""
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
    return (
        "## Defect report\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "What is the root cause and what physical fixes do you recommend?"
    )


def build_messages(
    film_type: str,
    film_age_years: int,
    storage: str,
    scan_resolution_dpi: int,
    defect_summary: dict[str, int],
    total_defects: int,
) -> list[dict[str, str]]:
    """Return full message list (few-shot + current request) for the reasoner."""
    messages: list[dict[str, str]] = []
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append(
        {
            "role": "user",
            "content": build_user_prompt(
                film_type,
                film_age_years,
                storage,
                scan_resolution_dpi,
                defect_summary,
                total_defects,
            ),
        }
    )
    return messages
