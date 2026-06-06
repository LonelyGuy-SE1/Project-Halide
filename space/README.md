---
title: Project Halide
emoji: "\U0001F525"
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Edge-native diagnostic engine for analog film scans
---

# Project Halide

An edge-native diagnostic engine for analog film. Upload a film scan, fill in
film stock + storage metadata, and Project Halide runs a two-stage analysis:

1. **Vision Extraction** -- MiniCPM-V 4.6 (1.3B params, fine-tuned with LoRA on
   the FilmDamageSimulator dataset) detects dust, dirt, scratches, and hair
   artifacts as normalized bounding boxes.
2. **Diagnostic Reasoning** -- Nemotron-Mini-4B-Instruct (4B params) with
   3-shot prompting cross-references the defect report against your film stock
   and storage metadata, and prescribes specific physical fixes a lab can
   perform.

The full pipeline runs locally in the Space. No external APIs. Models are
loaded from a private Hugging Face repo at startup.

## How to use

1. Upload a film scan (PNG or JPEG, ideally 35mm or 120 frame).
2. Select your film stock from the dropdown.
3. Adjust the age and storage condition.
4. Pick the scan resolution you used.
5. Click **Diagnose scan**.

Results are stored in a local SQLite database. The "Recent diagnoses" panel
shows the last 10 runs in this Space session.

## Models

- Vision: `Lonelyguyse1/halide-vision` (private), based on
  `openbmb/MiniCPM-V-4_6`.
- Reasoning: `nvidia/Nemotron-Mini-4B-Instruct` (public, 4B params, few-shot
  prompting only, no fine-tuning).

## License

Apache 2.0.
