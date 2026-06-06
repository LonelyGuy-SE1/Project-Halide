"""
Modal training script for MiniCPM-V 4.6 using LLaMA-Factory.
Follows official OpenBMB documentation exactly.

Per log.txt directive (2026-06-06):
  - lora_rank: 64
  - lora_alpha: 128
  - lora_target: q_proj,v_proj,k_proj,o_proj (vision tower + language)
  - num_train_epochs: 50
  - early_stopping_patience: 5 (terminates if eval_loss stagnates for 5 evals)
  - bbox format: integer [0, 999] grid (aligned with VLM pre-training)
"""
import modal

app = modal.App("halide-vision-training-v2")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "cmake", "build-essential")
    .run_commands(
        "cd /root && git clone --depth 1 -b v0.9.5 https://github.com/hiyouga/LLaMA-Factory",
        "cd /root/LLaMA-Factory && pip install -e .",
        "cd /root/LLaMA-Factory && pip install -r requirements/metrics.txt",
        "cd /root && git clone --depth 1 https://github.com/ggerganov/llama.cpp",
        "cd /root/llama.cpp && pip install -r requirements/requirements-convert_hf_to_gguf.txt",
        "cd /root/llama.cpp && cmake -B build && cmake --build build --config Release --target llama-quantize --target llama-cli -j4",
    )
    .pip_install(
        "transformers[torch]==5.7.0",
        "torchvision",
        "huggingface_hub",
        "datasets",
        "gguf==0.19.0",
    )
)

# Volumes for data and checkpoints
data_volume = modal.Volume.from_name("halide-training-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("halide-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2 * 3600,
    retries=modal.Retries(initial_delay=0, max_retries=2),
)
def train(
    epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_target: str = "q_proj,v_proj,k_proj,o_proj",
    max_samples: int = 1000,
    max_seq_length: int = 4096,
    output_dir: str = "/checkpoints/minicpm-v-4.6-lora-v2",
    early_stopping_patience: int = 5,
    eval_steps: int = 2,
):
    """
    Run LoRA fine-tuning on MiniCPM-V 4.6 with vision tower targeting.

    Per log.txt (2026-06-06):
      - rank 64, alpha 128
      - target q_proj,v_proj,k_proj,o_proj (covers both SigLIP2-400M vision
        tower and Qwen3.5-0.8B language backbone; both module types share
        these names)
      - 50 epochs, early stop patience 5
      - integer [0, 999] bbox grid format

    Uses A100-80GB because T4 (16GB) is insufficient for MiniCPM-V 4.6 training.
    """
    import subprocess
    import os
    from pathlib import Path

    print("=== Project Halide: Vision Model Training v2 ===")
    print(f"GPU: A100-80GB | Epochs: {epochs} | LR: {learning_rate}")
    print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, target={lora_target}")
    print(f"cutoff_len: {max_seq_length} (fits 150 defects in int 0-999 format)")
    print(f"Early stopping: patience={early_stopping_patience}, eval every {eval_steps} epochs")

    # Step 1: Register both train and val datasets in LLaMA-Factory
    import json
    dataset_dir = Path("/root/LLaMA-Factory/data")
    dataset_info_path = dataset_dir / "dataset_info.json"

    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info["halide_film_defects"] = {
        "file_name": "training_sharegpt.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images",
        },
    }
    dataset_info["halide_film_defects_val"] = {
        "file_name": "training_sharegpt_val.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images",
        },
    }

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("Registered train + val datasets in LLaMA-Factory")

    # Step 2: Copy data to LLaMA-Factory data directory
    import shutil
    shutil.copy("/data/training_sharegpt.json",
                "/root/LLaMA-Factory/data/training_sharegpt.json")
    shutil.copy("/data/training_sharegpt_val.json",
                "/root/LLaMA-Factory/data/training_sharegpt_val.json")

    scans_src = Path("/data/scans")
    scans_dst = Path("/root/LLaMA-Factory/data/scans")
    if scans_src.exists():
        shutil.copytree(scans_src, scans_dst, dirs_exist_ok=True)

    print("Copied data to LLaMA-Factory directory")

    # Step 3: Write YAML config
    config_yaml = f"""
model_name_or_path: openbmb/MiniCPM-V-4_6
image_max_pixels: 262144
trust_remote_code: true

stage: sft
do_train: true

finetuning_type: lora
lora_target: {lora_target}
lora_rank: {lora_rank}
lora_alpha: {lora_alpha}

dataset: halide_film_defects
eval_dataset: halide_film_defects_val
dataset_dir: /root/LLaMA-Factory/data
template: minicpm_v_4_6
cutoff_len: {max_seq_length}
max_samples: {max_samples}
overwrite_cache: true
preprocessing_num_workers: 1

per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: 8
learning_rate: {learning_rate}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

logging_steps: 1
save_strategy: epoch
save_total_limit: 3
report_to: none
save_only_model: false

# Early stopping + best model selection
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
early_stopping_patience: {early_stopping_patience}
early_stopping_threshold: 0.0

eval_strategy: epoch
eval_steps: {eval_steps}

output_dir: {output_dir}
"""

    config_path = Path("/root/LLaMA-Factory/train_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml.strip())

    print(f"Config written to {config_path}")
    print("Starting training...")

    # Step 4: Run training (DISABLE_VERSION_CHECK for transformers 5.7+ compat)
    env = os.environ.copy()
    env["DISABLE_VERSION_CHECK"] = "1"
    result = subprocess.run(
        ["llamafactory-cli", "train", str(config_path)],
        cwd="/root/LLaMA-Factory",
        capture_output=False,
        env=env,
    )

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return None

    print("Training complete!")
    checkpoint_volume.commit()

    print(f"Checkpoint saved to {output_dir}")
    return output_dir


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": checkpoint_volume},
    timeout=3600,
)
def export_model(
    checkpoint_path: str = "/checkpoints/minicpm-v-4.6-lora-v2",
    output_dir: str = "/checkpoints/minicpm-v-4.6-merged-v2",
):
    """Export LoRA checkpoint merged with base model."""
    import os
    import subprocess
    from pathlib import Path

    print("=== Exporting Merged Model ===")

    config_yaml = f"""
model_name_or_path: openbmb/MiniCPM-V-4_6
adapter_name_or_path: {checkpoint_path}
template: minicpm_v_4_6
finetuning_type: lora
export_dir: {output_dir}
export_size: 2
export_legacy_format: false
"""

    config_path = Path("/root/LLaMA-Factory/export_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml.strip())

    env = os.environ.copy()
    env["DISABLE_VERSION_CHECK"] = "1"
    result = subprocess.run(
        ["llamafactory-cli", "export", str(config_path)],
        cwd="/root/LLaMA-Factory",
        capture_output=False,
        env=env,
    )

    if result.returncode != 0:
        print(f"Export failed with return code {result.returncode}")
        return None

    print(f"Exported merged model to {output_dir}")
    checkpoint_volume.commit()
    return output_dir


@app.local_entrypoint()
def main(epochs: int = 50, do_export: bool = True):
    """Run training (and optional export) on Modal A100-80GB.

    Per log.txt (2026-06-06) directive: rank 64, alpha 128, vision_tower
    targets, 50 epochs with early stopping patience 5.
    """
    output = train.remote(epochs=epochs)
    print(f"\nTraining output: {output}\n")
    if do_export:
        merged = export_model.remote()
        print(f"\nMerged model: {merged}\n")
