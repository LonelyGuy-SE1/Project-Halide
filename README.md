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
  <a href="#runtime-rules"><img alt="Runtime" src="https://img.shields.io/badge/Runtime-GPU%20only-e87d2f?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square"></a>
</p>

<img alt="Project Halide app screenshot" src="https://lonelyguy.vercel.app/assets/halide/halide-app.png" width="100%">

</div>

## Quick Links

| Link | Target |
| --- | --- |
| Live app | <https://huggingface.co/spaces/build-small-hackathon/project-halide> |
| Profile mirror | <https://huggingface.co/spaces/Lonelyguyse1/project-halide> |
| Demo video | <https://youtube.com/watch?si=apzCiBZcIZWC1nFt&v=DGJ2M1aQCrE&feature=youtu.be> |
| Technical blog | <https://lonelyguy.vercel.app/articles/2026-06-16-project-halide> |
| Launch post | <https://x.com/lonelyguyse1/status/2066631507956105423?s=20> |
| Fine-tuned vision model | <https://huggingface.co/Lonelyguyse1/halide-vision> |
| GGUF artifact | <https://huggingface.co/Lonelyguyse1/halide-vision/blob/main/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf> |
| Author | <https://huggingface.co/Lonelyguyse1> |

## What It Does

Halide is not a restoration filter. It is a diagnostic workbench for deciding
what physical problem a film scan is showing and what to try next.

1. Upload a film scan, negative photo, or contact-sheet crop.
2. Add film metadata, or leave it unknown when it is only a guess.
3. Extract candidate defects with MiniCPM-V 4.6 on GPU.
4. Validate boxes, filter sprocket and frame-edge artifacts, merge duplicates,
   and fall back to tiled inspection when full-frame inference misses damage.
5. Add conservative image-analysis scratch candidates when clear linear
   evidence is visible.
6. Use Nemotron-Mini-4B-Instruct to write physical diagnosis and next steps.
7. Review the annotated scan, evidence quality, full JSON, and SQLite history.

Local CPU is used only for safe development tasks: file I/O, image drawing,
JSON parsing, tests, and dataset preparation. Model inference refuses to run
unless CUDA is visible.

## Visual Evidence

Halide is built around inspectable visual evidence. The examples below use real
damaged negatives and real validation overlays, not generated placeholder art.

| Held-out negative | Validated overlay |
| --- | --- |
| ![Held-out scratched negative](https://lonelyguy.vercel.app/assets/halide/negative1-original.png) | ![Validated overlay for held-out negative](https://lonelyguy.vercel.app/assets/halide/negative1-overlay.png) |

The public demo case also uses a real 35mm negative strip with residue, glare,
sprocket holes, and stain-like damage.

| Demo 35mm negative | Validated overlay |
| --- | --- |
| ![Real damaged 35mm negative strip](https://lonelyguy.vercel.app/assets/halide/real-35mm-negative.jpg) | ![Validated Project Halide overlay on 35mm strip](https://lonelyguy.vercel.app/assets/halide/real-35mm-negative-overlay.png) |

## Architecture

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

Default models:

| Stage | Model | Notes |
| --- | --- | --- |
| Vision | `openbmb/MiniCPM-V-4.6` | Uses `downsample_mode="4x"` and JSON-only defect prompts |
| Fine-tuned vision | `Lonelyguyse1/halide-vision` | Optional merged MiniCPM checkpoint for defect schema reliability |
| Reasoning | `nvidia/Nemotron-Mini-4B-Instruct` | Few-shot diagnostic reasoning |

The fine-tuned model is small-data and should be treated as an inspection aid,
not a final archival authority. The UI keeps metadata confidence explicit so a
guessed film stock or storage condition does not override visual evidence.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Gradio launch entrypoint |
| `config.py` | Runtime config, model IDs, GPU guardrails |
| `data/` | Schemas, preprocessing, dataset summaries, augmentation |
| `docs/` | Field notes and evaluation reports |
| `models/` | MiniCPM and Nemotron wrappers |
| `pipeline/` | End-to-end orchestration |
| `storage/` | SQLite history and in-memory cache |
| `ui/` | Gradio Blocks app, optional `gr.Server`, theme, HTML renderers |
| `modal/` | GPU training, conversion, upload, and deployment helpers |
| `scripts/` | Dataset conversion, evaluation, and utility scripts |
| `tests/` | CPU-safe unit tests with model stubs |

## Runtime Rules

No cloud APIs are used at app runtime. Valid inference runtimes are:

1. Hugging Face ZeroGPU or another GPU-backed Space.
2. Modal GPU jobs for offline tests and training.
3. A local GPU machine if CUDA is explicitly available.

If CUDA is not visible, the app returns a clear error instead of loading a
model on CPU.

## Quickstart

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pytest
python app.py
```

Opening the app locally is useful for UI inspection, storage checks, cache
checks, and stubbed tests. Running an actual diagnosis needs a GPU runtime.

## GPU Configuration

For a Hugging Face Space, use a Gradio Space with ZeroGPU or GPU hardware and
set optional variables:

```bash
HALIDE_USE_FINETUNED_VISION=1
HALIDE_VISION_FINETUNED_MODEL_ID=Lonelyguyse1/halide-vision
HALIDE_DOWNSAMPLE_MODE=4x
HALIDE_MAX_SLICE_NUMS=36
HALIDE_MAX_NEW_TOKENS=2048
HALIDE_NEMOTRON_MAX_TOKENS=768
HALIDE_GPU_DURATION_SECONDS=120
HALIDE_ENABLE_TILE_FALLBACK=1
HALIDE_TILE_MAX_SIDE=960
HALIDE_TILE_OVERLAP=0.35
HALIDE_TILE_MAX_TILES=9
HALIDE_ENABLE_CLASSICAL_ASSIST=1
HALIDE_CLASSICAL_ASSIST_MAX_DEFECTS=8
```

The handler is decorated with `spaces.GPU` when the `spaces` package is
available. The app also exposes an optional `gr.Server` builder at
`ui.server:build_server` with `/healthz`.

Modal was used for offline training, held-out GPU evaluation, checkpoint
upload, GGUF conversion, and Space deployment. The runtime app itself does not
call Modal or any hosted inference API.

GPUs used across the build: Modal A100-80GB for training/export workloads,
Modal T4 for lower-cost GPU checks, Hugging Face ZeroGPU A10G for the official
Space runtime, and a Hugging Face T4 Space mirror for final browser validation
after ZeroGPU quota was exhausted.

## Dataset And Training

Current training data combines FilmDamageSimulator annotations with generated
procedural film-defect positives and hard clean negatives. The v7 crack
curriculum adds analog-negative scratch clusters, broad lifted and cracked
emulsion regions, chemical stains, dirt, dust, light leaks, and clean
subject-hair or grass counterexamples. Broad crack trunks are labeled as
`emulsion_damage`, while fine branches and abrasion lines are labeled as
`scratch`. The user-supplied negative samples are held out for evaluation and
are not used for training.

```bash
python scripts/download_datasets.py
python scripts/prepare_training.py
python scripts/convert_to_sharegpt.py --format int_0_999
python scripts/augment_training.py --samples-per-image 3
python scripts/convert_to_sharegpt.py --input data/augmented/augmented_training.jsonl --output-train data/augmented/training_sharegpt_augmented.json --output-val data/augmented/training_sharegpt_augmented_val.json --image-prefix augmented --no-val
python scripts/generate_v4_synthetic_dataset.py --count 720 --clean-count 160 --max-side 1200 --base-mode procedural
python scripts/convert_to_sharegpt.py --input data/augmented/v4_synthetic/v4_synthetic_training.jsonl --output-train data/augmented/training_sharegpt_synthetic_v4.json --output-val data/augmented/training_sharegpt_synthetic_val_v4.json --image-prefix augmented --no-val --format int_0_999
python scripts/combine_sharegpt.py --out data/augmented/training_sharegpt_combined_v4.json data/training_sharegpt_v4.json data/augmented/training_sharegpt_augmented_v4.json data/augmented/training_sharegpt_synthetic_v4.json
python scripts/generate_v5_negative_curriculum.py --dataset-name v7_crack_curriculum --image-prefix v7 --output-stem training_sharegpt_v7 --defect-count 1024 --clean-count 256 --crack-focus-fraction 0.40 --val-fraction 0.08 --max-side 1120 --seed 707
```

Fine-tune on Modal:

```bash
modal run modal/train_vision.py::main --epochs 5 --do-export --learning-rate 0.0001 --train-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_v7.json --val-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_val_v7.json --output-dir /checkpoints/minicpm-v-4.6-lora-v7-crack-curriculum-r1 --merged-output-dir /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625
```

For long runs on an unreliable local connection, deploy the Modal app and
spawn the training function from the deployed app:

```bash
modal deploy modal/train_vision.py --name halide-vision-training-v4
python scripts/spawn_modal_training.py --epochs 5 --learning-rate 0.0001 --train-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_v7.json --val-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_val_v7.json --output-dir /checkpoints/minicpm-v-4.6-lora-v7-crack-curriculum-r1
```

Publish the merged checkpoint:

```bash
modal run modal/upload_model.py::main --model-dir /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625 --repo-id Lonelyguyse1/halide-vision --no-private
```

Convert and publish the llama.cpp GGUF artifact:

```bash
modal run modal/convert_gguf.py::main --model-dir /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625 --outfile /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-f16.gguf --quantized-outfile /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf
modal run modal/upload_model.py::file --local-path /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf --repo-id Lonelyguyse1/halide-vision --path-in-repo minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf
```

## Evaluation

Evaluate saved predictions against local JSONL annotations:

```bash
python scripts/evaluate.py predictions.jsonl --ground-truth data/training_data.jsonl
```

Prediction JSONL format:

```jsonl
{"image":"Scan (1).jpg","predictions":[{"label":"dust","bbox":[0.1,0.1,0.2,0.2]}]}
```

The evaluator reports per-class precision, recall, F1, false positives, false
negatives, and IoU-backed matching.

Run private held-out negatives through Modal GPU inference after a merged
checkpoint exists:

```bash
python scripts/run_private_negative_eval.py --image .nottracked/negative1.png --image .nottracked/negative2.png --image .nottracked/negative3.png --image .nottracked/negative4.png --image .nottracked/negative5.png --model /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625 --out-dir .nottracked/v7_ckpt625_tiled960_private_eval_exact5
```

The five private user negatives stay in `.nottracked` and are not used for
training data. The final v7 checkpoint with 960 px tiled fallback produced this
held-out smoke result:

| Image | Visual expectation | Result |
| --- | --- | --- |
| `negative1.png` | Long scratches across portrait | 8 defects: scratch, emulsion damage, dust, dirt, chemical stain |
| `negative2.png` | Abraded emulsion and dirt patches | 9 defects: scratch, emulsion damage, dust, dirt, chemical stain |
| `negative3.png` | Severe emulsion damage and debris | 6 defects: scratch, emulsion damage, dirt, chemical stain |
| `negative4.png` | Near-clean hard negative | 0 defects |
| `negative5.png` | Broad lifted crack network over portrait | 45 defects: 17 scratch, 14 emulsion damage, 6 chemical stain, 5 dust, 3 dirt |

The fifth sample is the key scale case: the full-frame model returned zero
defects, while tiled inspection recovered the visible crack and lifted-emulsion
network.

## Space Bundle

Prepare the private Space upload directory:

```bash
python scripts/prepare_space_bundle.py
```

The bundle contains only runtime code, selected `data/` helpers, assets, a slim
requirements file, and the Space README metadata.

## Field Notes

The public build and evaluation report is in
[`docs/field-notes.md`](docs/field-notes.md). It documents the final tiled
vision fallback and the five-image held-out smoke test without publishing the
private negatives.

## Current Status

Implemented:

- Gradio diagnostic workbench with custom styling, light-table review, evidence inspector, and history recall.
- MiniCPM-V 4.6 wrapper with GPU guardrails.
- Nemotron-Mini-4B wrapper with metadata-confidence-aware prompting.
- Defect schema validation, low-confidence filtering, bbox normalization, spatial summary, and IoU deduplication.
- Tiled vision fallback for large scans where full-frame inference misses obvious crack networks.
- SQLite diagnosis history with detail recall and full JSON storage.
- Metadata-aware image cache.
- Dataset preparation, augmentation utilities, ShareGPT conversion, evaluation, and Modal training scripts.
- CPU-safe unit tests with model stubs.
- Modal helpers for training, export, GGUF conversion, model upload, Space deployment, and inference comparison.

Verification highlights:

- Local unit tests cover UI handlers, storage, schema validation, pipeline orchestration, prompt construction, and evaluation utilities.
- Python compile checks cover all repository Python files.
- Live Space tests are run from a GPU-backed runtime, not local CPU.
- Public launch materials include a short MP4 demo, live Space screenshots, real negative before/after evidence, a technical blog, and a launch note on Hugging Face.

## License

Apache 2.0.

## Build Small Tags

The official Hugging Face Space README carries these validator tags:

```yaml
tags:
  - track:backyard
  - sponsor:openbmb
  - sponsor:openai
  - sponsor:nvidia
  - sponsor:modal
  - achievement:offgrid
  - achievement:welltuned
  - achievement:offbrand
  - achievement:fieldnotes
```

Prize and badge alignment:

- Backyard AI: practical film-diagnostics workflow for real analog scan problems.
- OpenBMB: MiniCPM-V 4.6 is the vision extraction model.
- OpenAI: source-control history includes the required attributed development work.
- NVIDIA: Nemotron-Mini-4B-Instruct writes the diagnostic report.
- Modal: used for offline training, evaluation, conversion, upload, and Space deployment.
- Off the Grid: runtime uses open weights on the Space GPU with no hosted inference APIs.
- Well-Tuned: the fine-tuned vision model is published on Hugging Face.
- Off-Brand: the app uses a custom autumn Gradio frontend and compare viewer.
- Field Notes: the build report is published in `docs/field-notes.md`.
