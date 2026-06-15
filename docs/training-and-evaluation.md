# Training And Evaluation Notes

This page keeps the longer build commands out of the main README while
preserving the reproducibility trail for Project Halide.

## Dataset Sources

The final training mix combines:

- FilmDamageSimulator annotations.
- Procedural film-defect positives.
- Hard clean negatives.
- Subject-hair and grass counterexamples.
- A v7 crack curriculum focused on analog-negative scratch clusters, lifted
  emulsion, chemical stains, dirt, dust, light leaks, and clean lookalikes.

The private user-supplied negatives are held out for evaluation only. They are
not used for training and stay in `.nottracked`.

## Dataset Preparation

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

## Fine-Tuning On Modal

```bash
modal run modal/train_vision.py::main --epochs 5 --do-export --learning-rate 0.0001 --train-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_v7.json --val-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_val_v7.json --output-dir /checkpoints/minicpm-v-4.6-lora-v7-crack-curriculum-r1 --merged-output-dir /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625
```

For long runs on an unreliable local connection, deploy the Modal app and spawn
the training function from the deployed app:

```bash
modal deploy modal/train_vision.py --name halide-vision-training-v4
python scripts/spawn_modal_training.py --epochs 5 --learning-rate 0.0001 --train-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_v7.json --val-json-path /data/augmented/v7_crack_curriculum/training_sharegpt_val_v7.json --output-dir /checkpoints/minicpm-v-4.6-lora-v7-crack-curriculum-r1
```

## Publish The Merged Checkpoint

```bash
modal run modal/upload_model.py::main --model-dir /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625 --repo-id Lonelyguyse1/halide-vision --no-private
```

## Convert And Publish GGUF

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

Final v7 checkpoint with 960 px tiled fallback:

| Image | Visual expectation | Result |
| --- | --- | --- |
| `negative1.png` | Long scratches across portrait | 8 defects: scratch, emulsion damage, dust, dirt, chemical stain |
| `negative2.png` | Abraded emulsion and dirt patches | 9 defects: scratch, emulsion damage, dust, dirt, chemical stain |
| `negative3.png` | Severe emulsion damage and debris | 6 defects: scratch, emulsion damage, dirt, chemical stain |
| `negative4.png` | Near-clean hard negative | 0 defects |
| `negative5.png` | Broad lifted crack network over portrait | 45 defects: 17 scratch, 14 emulsion damage, 6 chemical stain, 5 dust, 3 dirt |

The fifth sample is the key scale case. Full-frame inference returned zero
defects, while tiled inspection recovered the visible crack and lifted-emulsion
network.

## Space Bundle

Prepare the private Space upload directory:

```bash
python scripts/prepare_space_bundle.py
```

The bundle contains only runtime code, selected `data/` helpers, assets, a slim
requirements file, and the Space README metadata.

