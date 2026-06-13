"""Shared prompts for MiniCPM-V film defect extraction."""

from __future__ import annotations

DEFECT_LABELS = (
    "dust, dirt, scratch, long_hair, short_hair, emulsion_damage, "
    "chemical_stain, light_leak"
)

DETECTION_PROMPT_BASE = (
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
    "'defects' array. Each defect has: "
    f"'label' ({DEFECT_LABELS}), "
    "optional 'confidence' from 0.0 to 1.0, "
)

DETECTION_PROMPT_SUFFIX = (
    "Return at most 150 defects. Prefer the clearest defects. Do not repeat "
    "the same label and bbox. Output JSON only, no explanation."
)

DETECTION_PROMPT_INT = (
    DETECTION_PROMPT_BASE
    + "'bbox' as 4 integers in the [0, 999] grid "
    + "[x_min, y_min, x_max, y_max] (multiply by image width/height to get pixels). "
    + DETECTION_PROMPT_SUFFIX
)

DETECTION_PROMPT_FLOAT = (
    DETECTION_PROMPT_BASE
    + "'bbox' (normalized [x_min, y_min, x_max, y_max] from 0.0 to 1.0). "
    + DETECTION_PROMPT_SUFFIX
)
