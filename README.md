# Project Halide

**An Edge-Native Diagnostic Engine for Analog Film**

## 1. Project Statement

Shooting 35mm and medium format film is an expensive, high-stakes process. When a roll fails—due to light seal degradation, exhausted developer, or incorrect agitation—analog photographers are forced into a tedious trial-and-error process or rely on subjective opinions from online forums. They lack an immediate, reliable way to diagnose technical faults directly from their scanned negatives.

**Halide** solves this by providing an on-device, privacy-first diagnostic tool that runs entirely client-side. By utilizing specialized vision and reasoning models, Halide instantly analyzes film anomalies and prescribes actionable physical fixes, drastically reducing wasted materials and time for the analog community.

## 2. Expected User Flow

1. **Input Generation:** The user captures a photo or high-resolution scan of their raw, developed film negative (or contact sheet).
2. **Context Logging:** The user uploads the scan via the Halide interface and inputs brief contextual metadata (e.g., film stock, development chemical used, camera model).
3. **Automated Extraction:** A specialized vision model analyzes the image, mapping out channel-heterogeneous deterioration, physical emulsion scratches, dust, and light leaks.
4. **Diagnostic Synthesis:** The extracted visual data is passed to a reasoning model, which cross-references the anomalies against the user's provided metadata to deduce the root cause (e.g., a mechanical hinge leak vs. a shutter timing issue).
5. **Output & Legacy:** The interface returns an annotated version of the negative alongside a clear, step-by-step diagnostic fix. The result is logged into a local, edge-resident database to track the photographer's performance across different film stocks over time.

## 3. Technology Stack

Halide utilizes a deterministic, dual-model architecture designed specifically to operate within strict parameter constraints (≤ 32B total parameters) while executing entirely on edge infrastructure.

### Model Architecture

- **Vision Extraction Agent** — `MiniCPM-V 4.6` (1.3B params, OpenBMB). High-density visual extraction and localized defect identification. Fine-tuned for film deterioration profiles.
- **Reasoning & Diagnostic Agent** — `Nemotron-Mini-4B-Instruct` (NVIDIA). Diagnostic logic synthesis and actionable output generation. Ingests vision outputs and user context.

### Infrastructure & Engineering

- **Model Fine-Tuning** — Modal Serverless Infrastructure. Cloud compute for dataset pre-processing and training of the vision agent on specialized visual datasets.
- **Inference Engine** — `llama.cpp` runtime. Zero-bottleneck concurrent model execution on edge constraints.
- **Deployment Runtime** — Hugging Face Spaces. Fully localized hosting with zero external API dependencies during runtime.
- **Frontend UI** — Gradio `gr.Server` with custom HTML/CSS injection. Custom, highly aesthetic diagnostic heads-up display.
- **Local Telemetry** — Embedded SQLite database. On-device historical tracking of user diagnostics.
