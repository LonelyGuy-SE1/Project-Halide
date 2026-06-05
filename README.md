# Project Halide: Film Negative & Development Analyzer

## Problem Statement

Analog photographers waste expensive film because they lack an immediate, reliable way to diagnose technical development errors, camera faults, and chemical deterioration directly from their scanned negatives.

## The Context

Shooting on 35mm or medium format film is an expensive and highly technical process. When a roll comes out ruined (e.g., color shifts, light leaks, scratches, or under-development), hobbyists and professionals are forced into a tedious trial-and-error process or rely on internet forums to diagnose the root cause. The issues can stem from mechanical camera faults, expired chemicals, or temperature fluctuations during the development process.

## Proposed Solution

**Halide** is a local, AI-powered diagnostic tool built for the analog photography community.

1. **The Input:** A user uploads a scan of a raw film negative or a contact sheet via a lightweight Gradio interface.
2. **The Engine:** A small, highly-quantized Vision-Language Model (PaliGemma 3B) fine-tuned on specialized film deterioration datasets (such as BlueNeg) analyzes the negative.
3. **The Output:** The VLM instantly detects and highlights analog artifacts-identifying whether the issue is a light seal leak, exhausted developer fluid, or emulsion scratching. It provides actionable fixes for the next roll.
4. **The Legacy:** Halide logs these diagnostics into a local SQLite database, building a private visual history that tracks how specific film stocks perform with specific cameras and development chemicals over time.

**Tech Stack (Proposed):**

* **Frontend:** Gradio
* **Backend:** Modal (Serverless GPU inference)
* **Vision Model:** PaliGemma 3B (Fine-tuned for channel-heterogeneous deterioration and physical defects)
* **Storage:** SQLite (Local)
