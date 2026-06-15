"""Prepare the private Hugging Face Space upload bundle.

The bundle is written to `.nottracked/space_upload` so source-control stays
clean while the Space receives the current runtime code and metadata.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / ".nottracked" / "space_upload"

RUNTIME_FILES = [
    "app.py",
    "config.py",
    "LICENSE",
]

RUNTIME_DIRS = [
    "assets",
    "models",
    "pipeline",
    "storage",
    "ui",
]

DATA_FILES = [
    "data/__init__.py",
    "data/preprocessing.py",
    "data/schemas.py",
]

SPACE_README = """---
title: Project Halide
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
license: apache-2.0
models:
  - Lonelyguyse1/halide-vision
  - openbmb/MiniCPM-V-4.6
  - nvidia/Nemotron-Mini-4B-Instruct
tags:
  - gradio
  - film
  - computer-vision
  - diagnostics
  - track:backyard
  - sponsor:openbmb
  - sponsor:nvidia
  - sponsor:modal
  - sponsor:openai
  - badge:offbrand
  - badge:tiny
  - badge:demo
  - badge:quest
  - achievement:offgrid
  - achievement:welltuned
  - achievement:offbrand
  - achievement:fieldnotes
---

# Project Halide

Project Halide is an edge-native diagnostic workbench for analog film scans by
[Lonelyguyse1](https://huggingface.co/Lonelyguyse1).

The runtime uses MiniCPM-V 4.6 for defect extraction and
Nemotron-Mini-4B-Instruct for diagnostic reasoning. The vision pass combines
full-frame inspection, tiled fallback for large scans, a conservative
image-analysis validator for obvious scratches, and geometric filtering for
sprocket or frame-edge artifacts. Model inference runs on the Space GPU runtime
without cloud inference APIs.

Fine-tuned vision model:
<https://huggingface.co/Lonelyguyse1/halide-vision>

Source repository:
<https://github.com/LonelyGuy-SE1/Project-Halide>

Demo video:
<https://huggingface.co/spaces/build-small-hackathon/project-halide/blob/main/assets/demo_walkthrough.mp4>

Public launch post:
<https://huggingface.co/spaces/build-small-hackathon/project-halide/discussions/1>

Modal was used for offline training, held-out GPU evaluation, checkpoint upload,
GGUF conversion, and Space deployment. The runtime app itself does not call
Modal or any hosted inference API.

## How It Works

1. Upload a film scan, negative photo, or contact-sheet crop.
2. MiniCPM-V 4.6 extracts candidate defects as structured JSON.
3. The validator normalizes boxes, filters bad geometry, removes duplicate or
   sprocket-like edge artifacts, and adds high-precision scratch candidates
   when clear linear evidence is visible.
4. Nemotron-Mini-4B-Instruct reads the validated evidence plus user metadata and
   writes a lab-style diagnosis with physical fixes.
5. SQLite stores local diagnostic history so earlier runs can be reopened.

## Sponsor Usage

- OpenBMB: MiniCPM-V 4.6 is the primary vision model, fine-tuned for film defect
  extraction and published at `Lonelyguyse1/halide-vision`.
- NVIDIA: Nemotron-Mini-4B-Instruct produces the diagnostic report and keeps
  uncertain film metadata lower priority than visible evidence.
- Modal: used offline for training, evaluation, checkpoint export, GGUF
  conversion, model upload, and Space deployment support.
- OpenAI Codex: used for implementation, testing, documentation, and
  source-control commits in the linked GitHub repository.

## Field Guide Alignment

- Gradio Space under the official `build-small-hackathon` organization.
- All runtime inference uses open weights on the Space GPU, with no hosted model
  API calls.
- Model sizes stay under the 32B limit, with MiniCPM-V 4.6 at 1.3B parameters
  and Nemotron-Mini-4B-Instruct at 4B parameters.
- Custom autumn-themed UI with a purpose-built compare viewer and diagnostic
  history.
- Fine-tuned vision model and GGUF artifact are published on the author's
  Hugging Face profile.
- Demo video, public launch post, and field notes are linked from this Space.

Held-out validation summary:

- Four visibly damaged private negatives were detected with scratch and
  emulsion-damage evidence.
- One near-clean private negative returned zero defects.
- A broad lifted crack network that failed full-frame inference was recovered by
  the tiled fallback.
"""

SPACE_REQUIREMENTS = """gradio>=6.10.0,<7.0.0
spaces>=0.40.0
torch>=2.4.0
torchvision>=0.19.0
transformers>=5.7.0,<6.0.0
accelerate>=1.0.0
huggingface_hub>=0.20.0
pillow>=10.0.0
numpy>=1.24.0
"""


def _copy_file(relative_path: str) -> None:
    source = REPO_ROOT / relative_path
    target = BUNDLE_DIR / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_dir(relative_path: str) -> None:
    source = REPO_ROOT / relative_path
    target = BUNDLE_DIR / relative_path
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.db",
        ),
    )


def prepare_bundle() -> Path:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    for relative_path in RUNTIME_FILES:
        _copy_file(relative_path)
    for relative_path in RUNTIME_DIRS:
        _copy_dir(relative_path)
    for relative_path in DATA_FILES:
        _copy_file(relative_path)

    (BUNDLE_DIR / "README.md").write_text(SPACE_README, encoding="utf-8")
    (BUNDLE_DIR / "requirements.txt").write_text(
        SPACE_REQUIREMENTS,
        encoding="utf-8",
    )
    return BUNDLE_DIR


def main() -> int:
    bundle = prepare_bundle()
    print(f"Prepared Space bundle at {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
