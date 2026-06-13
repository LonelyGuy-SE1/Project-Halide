# Project Halide

Edge-native diagnostic engine for analog film scans. Halide analyzes a scan or
contact-sheet crop, extracts visible film and scanner defects, then produces a
lab-style diagnosis with physical fixes.

Codename: Helius

Author: [Lonelyguyse1](https://huggingface.co/Lonelyguyse1)

Live demo: <https://huggingface.co/spaces/Lonelyguyse1/project-halide>

Fine-tuned vision model: <https://huggingface.co/Lonelyguyse1/halide-vision>

## What It Does

1. Upload a film scan in the Gradio interface.
2. Add film metadata, or leave it unknown when it is only a guess.
3. Run MiniCPM-V 4.6 on GPU to extract defect JSON.
4. Validate boxes, drop invalid or low-confidence detections, and merge near duplicates.
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
python -m pip install -r requirements.txt
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
```

The handler is decorated with `spaces.GPU` when the `spaces` package is
available. The app also exposes an optional `gr.Server` builder at
`ui.server:build_server` with `/healthz`.

## Dataset And Training

Current training data combines FilmDamageSimulator annotations with generated
v4 hard negatives and procedural film-defect positives. The user-supplied
negative samples are held out for evaluation and are not used for training.

```bash
python scripts/download_datasets.py
python scripts/prepare_training.py
python scripts/convert_to_sharegpt.py --format int_0_999
python scripts/augment_training.py --samples-per-image 3
python scripts/convert_to_sharegpt.py --input data/augmented/augmented_training.jsonl --output-train data/augmented/training_sharegpt_augmented.json --output-val data/augmented/training_sharegpt_augmented_val.json --image-prefix augmented --no-val
python scripts/generate_v4_synthetic_dataset.py --count 720 --clean-count 160 --max-side 1200 --base-mode procedural
python scripts/convert_to_sharegpt.py --input data/augmented/v4_synthetic/v4_synthetic_training.jsonl --output-train data/augmented/training_sharegpt_synthetic_v4.json --output-val data/augmented/training_sharegpt_synthetic_val_v4.json --image-prefix augmented --no-val --format int_0_999
python scripts/combine_sharegpt.py --out data/augmented/training_sharegpt_combined_v4.json data/training_sharegpt_v4.json data/augmented/training_sharegpt_augmented_v4.json data/augmented/training_sharegpt_synthetic_v4.json
```

Fine-tune on Modal:

```bash
modal run modal/train_vision.py::main --epochs 8 --do-export --train-json-path /data/augmented/training_sharegpt_combined_v4.json --val-json-path /data/training_sharegpt_val_v4.json --output-dir /checkpoints/minicpm-v-4.6-lora-v4-stage1 --merged-output-dir /checkpoints/minicpm-v-4.6-merged-v4-stage1
```

Publish the merged checkpoint:

```bash
modal run modal/upload_model.py::main --model-dir /checkpoints/minicpm-v-4.6-merged-v3 --repo-id Lonelyguyse1/halide-vision --no-private
```

Convert and publish the llama.cpp GGUF artifact:

```bash
modal run modal/convert_gguf.py::main
modal run modal/upload_model.py::file --local-path /checkpoints/minicpm-v-4.6-merged-v3-q4_k_m.gguf --repo-id Lonelyguyse1/halide-vision --path-in-repo minicpm-v-4.6-merged-v3-q4_k_m.gguf
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

## Current Status

Implemented:

- Gradio diagnostic workbench with custom styling, light-table review, evidence inspector, and history recall.
- MiniCPM-V 4.6 wrapper with GPU guardrails.
- Nemotron-Mini-4B wrapper with metadata-confidence-aware prompting.
- Defect schema validation, low-confidence filtering, bbox normalization, spatial summary, and IoU deduplication.
- SQLite diagnosis history with detail recall and full JSON storage.
- Metadata-aware image cache.
- Dataset preparation, augmentation utilities, ShareGPT conversion, evaluation, and Modal training scripts.
- CPU-safe unit tests with model stubs.
- Modal helpers for training, export, GGUF conversion, model upload, Space deployment, and inference comparison.

Verification highlights:

- Local unit tests cover UI handlers, storage, schema validation, pipeline orchestration, prompt construction, and evaluation utilities.
- Python compile checks cover all repository Python files.
- Live Space tests are run from a GPU-backed runtime, not local CPU.

## License

Apache 2.0.
