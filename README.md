# Project Halide

Edge-native diagnostic engine for analog film scans. Halide analyzes a scan or
contact-sheet crop, extracts visible film and scanner defects, then produces a
lab-style diagnosis with physical fixes.

Codename: Helius

Author: [Lonelyguyse1](https://huggingface.co/Lonelyguyse1)

Source: <https://github.com/LonelyGuy-SE1/Project-Halide>

Live demo: <https://huggingface.co/spaces/build-small-hackathon/project-halide>

Profile mirror: <https://huggingface.co/spaces/Lonelyguyse1/project-halide>

Fine-tuned vision model: <https://huggingface.co/Lonelyguyse1/halide-vision>

Demo video: <https://huggingface.co/spaces/build-small-hackathon/project-halide/blob/main/assets/demo_walkthrough.mp4>

Public launch post: <https://huggingface.co/spaces/build-small-hackathon/project-halide/discussions/1>

llama.cpp GGUF artifact:
<https://huggingface.co/Lonelyguyse1/halide-vision/blob/main/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf>

## What It Does

1. Upload a film scan in the Gradio interface.
2. Add film metadata, or leave it unknown when it is only a guess.
3. Run MiniCPM-V 4.6 on GPU to extract defect JSON.
4. Validate boxes, drop invalid or low-confidence detections, merge near duplicates, and run tiled inspection when a large damaged scan is missed full-frame.
5. Run Nemotron-Mini-4B-Instruct on GPU with lab-style few-shot prompts.
6. Show an annotated scan, evidence quality, diagnosis, fixes, full JSON, and SQLite history.

Local CPU is used only for safe development tasks: file I/O, image drawing,
JSON parsing, tests, and dataset preparation. Model inference refuses to run
unless CUDA is visible.

## Architecture

```text
Film scan
  -> MiniCPM-V 4.6 vision extraction
  -> validated defect JSON
  -> Nemotron-Mini-4B-Instruct diagnostic reasoning
  -> Gradio UI and SQLite history
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
HALIDE_ENABLE_TILE_FALLBACK=1
HALIDE_TILE_MAX_SIDE=960
HALIDE_TILE_OVERLAP=0.35
HALIDE_TILE_MAX_TILES=9
```

The handler is decorated with `spaces.GPU` when the `spaces` package is
available. The app also exposes an optional `gr.Server` builder at
`ui.server:build_server` with `/healthz`.

Modal was used for offline training, held-out GPU evaluation, checkpoint
upload, GGUF conversion, and Space deployment. The runtime app itself does not
call Modal or any hosted inference API.

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
- Public launch materials include a short MP4 demo, live Space screenshots, a public synthetic before/after run, and a launch note on Hugging Face.

## License

Apache 2.0.
