<div align="center">

<pre>
    __  __      ___     __
   / / / /___ _/ (_)___/ /__
  / /_/ / __ `/ / / __  / _ \
 / __  / /_/ / / / /_/ /  __/
/_/ /_/\__,_/_/_/\__,_/\___/
</pre>

<h1>Project Halide</h1>

<p><strong>Evidence-first diagnostics for damaged analog film.</strong></p>

<p>
Halide inspects film scans and negative photos, extracts visible defect
evidence, validates it, and turns it into lab-style physical next steps.
It is built for photographers, archivists, and anyone trying to understand
whether a scan shows dust, scratches, lifted emulsion, chemical staining,
light leaks, or scanner artifacts.
</p>

<p>
  <a href="https://huggingface.co/spaces/build-small-hackathon/project-halide"><img alt="Live Space" src="https://img.shields.io/badge/Live%20Space-Hugging%20Face-ff6b35?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white"></a>
  <a href="https://youtube.com/watch?si=apzCiBZcIZWC1nFt&amp;v=DGJ2M1aQCrE&amp;feature=youtu.be"><img alt="Demo Video" src="https://img.shields.io/badge/Demo-YouTube-dc2626?style=for-the-badge&amp;logo=youtube&amp;logoColor=white"></a>
  <a href="https://lonelyguy.vercel.app/articles/2026-06-16-project-halide"><img alt="Technical Blog" src="https://img.shields.io/badge/Technical%20Blog-Field%20Notes-cc8833?style=for-the-badge"></a>
  <a href="https://x.com/lonelyguyse1/status/2066631507956105423?s=20"><img alt="Launch Post" src="https://img.shields.io/badge/Launch%20Post-X-000000?style=for-the-badge&amp;logo=x&amp;logoColor=white"></a>
</p>

<p>
  <a href="https://huggingface.co/Lonelyguyse1/halide-vision"><img alt="Vision Model" src="https://img.shields.io/badge/Vision%20Model-MiniCPM--V%204.6-4a7c59?style=flat-square"></a>
  <a href="https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct"><img alt="Reasoning Model" src="https://img.shields.io/badge/Reasoning-Nemotron--Mini--4B-c4284a?style=flat-square"></a>
  <a href="#runtime-guardrails"><img alt="Runtime" src="https://img.shields.io/badge/Runtime-GPU%20only-e87d2f?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square"></a>
</p>

<img alt="Project Halide app screenshot" src="https://lonelyguy.vercel.app/assets/halide/halide-app.png" width="100%">

</div>

## Try It

| Link | Target |
| --- | --- |
| Live app | <https://huggingface.co/spaces/build-small-hackathon/project-halide> |
| Profile mirror | <https://huggingface.co/spaces/Lonelyguyse1/project-halide> |
| Demo video | <https://youtube.com/watch?si=apzCiBZcIZWC1nFt&v=DGJ2M1aQCrE&feature=youtu.be> |
| Technical blog | <https://lonelyguy.vercel.app/articles/2026-06-16-project-halide> |
| Launch post | <https://x.com/lonelyguyse1/status/2066631507956105423?s=20> |
| Fine-tuned vision model | <https://huggingface.co/Lonelyguyse1/halide-vision> |
| GGUF artifact | <https://huggingface.co/Lonelyguyse1/halide-vision/blob/main/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf> |

## Why Halide Exists

Film damage is physical before it is digital. A white streak may be dust on a
scanner bed, a scratch in the emulsion, a chemical processing mark, or a light
leak from the camera. Those causes imply different fixes.

Halide is designed as a diagnostic workbench rather than a restoration filter.
It keeps the scan visible, marks the evidence it used, and gives practical next
steps such as re-cleaning, rescanning, checking negatives under side light, or
changing development and storage practice. User metadata is useful, but the app
treats uncertain film notes as context rather than truth.

## What It Does

- Accepts scans, negative photos, and contact-sheet crops.
- Extracts candidate defects with MiniCPM-V 4.6 on a GPU runtime.
- Validates the model JSON, normalizes boxes, removes duplicates, and filters
  sprocket or frame-edge artifacts.
- Uses tiled inspection when full-frame inference misses small or transparent
  damage in high-resolution scans.
- Adds conservative scratch candidates when obvious linear evidence is visible.
- Uses Nemotron-Mini-4B-Instruct to write a physical diagnosis and next steps.
- Stores local SQLite history so previous diagnoses can be reopened.

## The Data Story

The hardest part was not connecting the models. It was finding useful training
data. Public examples of damaged film are scattered across forum posts,
restoration articles, scanner screenshots, and phone photos of negatives. Many
are positive scans rather than negatives, many include arrows or annotations,
and many show real damage without a verified physical cause.

Halide therefore uses a layered training curriculum rather than pretending a
clean dataset already existed. The vision set combines FilmDamageSimulator
annotations, procedural film-defect positives, synthetic scratches and stains,
hard clean negatives, and lookalike counterexamples such as grass, subject hair,
sprocket holes, borders, and glare. The five user-supplied negatives stayed
held out for evaluation only.

That data work shaped the product. Fine-tuning improved structured output and
film-defect vocabulary, while the validation layer, tiled fallback, and overlay
UI handle the messy cases that data alone did not solve.

## What Fine-Tuning Changed

The first version could produce plausible boxes, but it was not reliable enough for film diagnostics. It confused sprocket holes, borders, glare, grass, and subject hair with film defects, and it sometimes returned prose or malformed JSON when the pipeline needed structured evidence.

The fine-tuned MiniCPM-V checkpoint improved the parts that matter most for Halide: stable defect JSON, consistent labels, better scratch and emulsion-damage vocabulary, and fewer obvious clean-image false positives. It did not make the model an authority, so the runtime still validates every box, filters geometry, and falls back to tiled inspection when full-frame inference misses damage.

## Visual Evidence

Halide is built around inspectable evidence. These examples use real damaged
negatives and real validation overlays, not generated placeholder art.

| Held-out negative | Validated overlay |
| --- | --- |
| ![Held-out scratched negative](https://lonelyguy.vercel.app/assets/halide/negative1-original.png) | ![Validated overlay for held-out negative](https://lonelyguy.vercel.app/assets/halide/negative1-overlay.png) |

The public demo case uses a real 35mm negative strip with residue, glare,
sprocket holes, and stain-like damage.

| Demo 35mm negative | Validated overlay |
| --- | --- |
| ![Real damaged 35mm negative strip](https://lonelyguy.vercel.app/assets/halide/real-35mm-negative.jpg) | ![Validated Project Halide overlay on 35mm strip](https://lonelyguy.vercel.app/assets/halide/real-35mm-negative-overlay.png) |

In order to view the detailed diagnosis, try the demo by visiting the huggingface space!

## Pipeline

```text
film scan or negative photo
  -> MiniCPM-V 4.6 vision extraction
  -> schema validation and artifact filtering
  -> tiled fallback for missed large-scan damage
  -> conservative scratch assist
  -> validated defect JSON
  -> Nemotron-Mini-4B-Instruct diagnostic reasoning
  -> custom Gradio light-table UI
  -> SQLite diagnosis history
```

| Stage | Model or system | Role |
| --- | --- | --- |
| Vision extraction | `openbmb/MiniCPM-V-4.6` | Finds candidate film defects from the image |
| Fine-tuned checkpoint | `Lonelyguyse1/halide-vision` | Fine-tuned for Halide defect JSON, scratch and emulsion-damage vocabulary, and cleaner evidence proposals |
| Reasoning | `nvidia/Nemotron-Mini-4B-Instruct` | Turns validated evidence into a lab-style report |
| Validation | Local Python pipeline | Filters boxes, handles tiling, and protects against obvious artifacts |
| Interface | Gradio custom UI | Light-table review, compare viewer, raw evidence, and history recall |
| Storage | SQLite | Local diagnostic history and result details |

All runtime model calls use open weights on a GPU-backed runtime. The app does
not call hosted inference APIs.

## Validation Snapshot

The five private negatives used for final smoke testing stayed in `.nottracked`
and were not used for training. The final v7 checkpoint with 960 px tiled
fallback produced this result:

| Image | Visual expectation | Result |
| --- | --- | --- |
| `negative1.png` | Long scratches across portrait | 8 defects |
| `negative2.png` | Abraded emulsion and dirt patches | 9 defects |
| `negative3.png` | Severe emulsion damage and debris | 6 defects |
| `negative4.png` | Near-clean hard negative | 0 defects |
| `negative5.png` | Broad lifted crack network over portrait | 45 defects |

The fifth image is the important scale case. Full-frame inference returned no
validated defects, while tiled inspection recovered the visible crack and
lifted-emulsion network.

For the detailed build report, see [`docs/field-notes.md`](docs/field-notes.md).
For training, conversion, and evaluation commands, see
[`docs/training-and-evaluation.md`](docs/training-and-evaluation.md).

## Run And Test Locally

Local development is CPU-safe for UI checks, storage, schemas, stubs, JSON, and
tests. Actual model inference requires a GPU runtime.

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pytest
python app.py
```

Opening the app locally is useful for browser inspection and stubbed flows.
Running a real diagnosis needs CUDA on a local GPU machine, a GPU-backed
Hugging Face Space, or a Modal GPU job.

## Runtime Guardrails

Halide refuses local CPU model inference. Valid inference targets are:

1. Hugging Face ZeroGPU or another GPU-backed Space.
2. Modal GPU jobs for offline tests, training, and export.
3. A local CUDA machine when CUDA is explicitly visible.

The app uses the `spaces.GPU` decorator when available and exposes an optional
`gr.Server` builder at `ui.server:build_server` with `/healthz`.

GPUs used across the build: Modal A100-80GB for training and export workloads,
Modal T4 for lower-cost GPU checks, Hugging Face ZeroGPU A10G for the official
Space runtime, and a Hugging Face T4 Space mirror for final browser validation
after ZeroGPU quota was exhausted.

## Repository Map

| Path | Purpose |
| --- | --- |
| `app.py` | Gradio launch entrypoint |
| `config.py` | Runtime config, model IDs, GPU guardrails |
| `data/` | Schemas, preprocessing, dataset summaries, augmentation |
| `docs/` | Field notes, training notes, and evaluation reports |
| `models/` | MiniCPM and Nemotron wrappers |
| `pipeline/` | End-to-end orchestration |
| `storage/` | SQLite history and in-memory cache |
| `ui/` | Gradio app, optional server, theme, and HTML renderers |
| `modal/` | GPU training, conversion, upload, and deployment helpers |
| `scripts/` | Dataset conversion, evaluation, and utility scripts |
| `tests/` | CPU-safe unit tests with model stubs |

## Current Status

Implemented:

- Custom autumn-themed Gradio diagnostic workbench.
- Evidence viewer with original image, overlay, confidence summary, and raw JSON.
- MiniCPM-V 4.6 vision wrapper with GPU guardrails.
- Fine-tuned MiniCPM-V checkpoint published on Hugging Face.
- Nemotron-Mini-4B diagnostic report generation.
- Tiled fallback for high-resolution scans.
- Conservative classical scratch assist.
- SQLite history with detail recall.
- Dataset preparation, augmentation, ShareGPT conversion, evaluation, Modal
  training, model upload, Space deployment, and GGUF conversion helpers.
- CPU-safe unit tests with model stubs.

Product scope:

- Halide is an inspection aid, not an archival authority.
- Broad damage can be over-boxed into multiple regions.
- A physical defect may receive more than one label when its appearance overlaps
  scratch, chemical stain, and lifted emulsion cues.
- Film metadata should be trusted only when the user is confident in it.

## License

Apache 2.0.
