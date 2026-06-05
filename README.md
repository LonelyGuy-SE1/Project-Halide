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

## 3. Technology Stack & Track Alignment Matrix

Halide utilizes a deterministic, dual-model architecture designed specifically to operate within strict parameter constraints (≤ 32B) while executing entirely on edge infrastructure.

### Model Architecture

* **Vision Extraction Agent (OpenBMB Track):**
* **Task:** High-density visual extraction and localized defect identification.
* **Technology:** `MiniCPM-V` (Fine-tuned for film deterioration profiles).


* **Reasoning & Diagnostic Agent (NVIDIA Track):**
* **Task:** Diagnostic logic synthesis and actionable output generation.
* **Technology:** `Nemotron-Mini-4B-Instruct` (Ingests vision outputs and user context).



### Infrastructure & Engineering

* **Model Fine-Tuning (Modal Track):**
* **Task:** Dataset pre-processing and training of the vision agent on specialized visual datasets.
* **Technology:** Modal Serverless Infrastructure (Cloud compute for training only).


* **Inference Engine (Llama Champion Badge):**
* **Task:** Zero-bottleneck concurrent model execution on edge constraints.
* **Technology:** `llama.cpp` runtime.


* **Deployment Runtime (Off the Grid Badge):**
* **Task:** Fully localized hosting with zero external API dependencies during runtime.
* **Technology:** Hugging Face Space (Zero-tier hardware allocation).


* **Frontend UI (Off-Brand Badge):**
* **Task:** Custom, highly aesthetic diagnostic heads-up display.
* **Technology:** Gradio `gr.Server` utilizing custom HTML/CSS injection.


* **Local Telemetry:**
* **Task:** On-device historical tracking of user diagnostics.
* **Technology:** Embedded SQLite database.
