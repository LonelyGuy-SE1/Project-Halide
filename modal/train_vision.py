"""
Modal training script for MiniCPM-V 4.6 using LLaMA-Factory.
Follows official OpenBMB documentation exactly.
"""
import modal

app = modal.App("halide-vision-training")

# Container image with all dependencies
# NOTE: LLaMA-Factory v0.9.5 pyproject.toml defines NO optional extras (no [torch,metrics,deepspeed]).
# Install is `pip install -e .` plus optional `pip install -r requirements/metrics.txt` etc.
# We also pin the git clone to the v0.9.5 release branch (not main) for reproducibility.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .run_commands(
        "cd /root && git clone --depth 1 -b v0.9.5 https://github.com/hiyouga/LLaMA-Factory",
        "cd /root/LLaMA-Factory && pip install -e .",
        "cd /root/LLaMA-Factory && pip install -r requirements/metrics.txt",
    )
    .pip_install(
        "transformers[torch]==5.7.0",
        "torchvision",
        "huggingface_hub",
        "datasets",
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
    timeout=12 * 3600,
    retries=modal.Retries(initial_delay=0, max_retries=2),
)
def train(
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    max_samples: int = 1000,
    max_seq_length: int = 3072,
    output_dir: str = "/checkpoints/minicpm-v-4.6-lora",
):
    """
    Run LoRA fine-tuning on MiniCPM-V 4.6.
    Follows the official OpenBMB LLaMA-Factory example.
    Uses A100-80GB because T4 (16GB) is insufficient for MiniCPM-V 4.6 training
    (5 prior T4 runs OOMed even with batch_size=1 and max_seq_length=1024).
    """
    import subprocess
    import os
    from pathlib import Path

    print("=== Project Halide: Vision Model Training ===")
    print(f"GPU: A100-80GB | Epochs: {epochs} | LR: {learning_rate} | LoRA rank: {lora_rank}")
    print(f"cutoff_len: 3072 (fits 150 defects * 17 tok/defect + ~600 image+system tokens)")

    # Step 1: Register dataset in LLaMA-Factory
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

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("Registered dataset in LLaMA-Factory")

    # Step 2: Copy data to LLaMA-Factory data directory
    import shutil
    shutil.copy("/data/training_sharegpt.json", "/root/LLaMA-Factory/data/training_sharegpt.json")

    # Copy scan images
    scans_src = Path("/data/scans")
    scans_dst = Path("/root/LLaMA-Factory/data/scans")
    if scans_src.exists():
        shutil.copytree(scans_src, scans_dst, dirs_exist_ok=True)

    print("Copied data to LLaMA-Factory directory")

    # NOTE: The mm_plugin.py patch for image_sizes is NOT needed when using
    # template: minicpm_v_4_6 because MiniCPMV4_6Plugin does NOT use image_bound.
    # It uses _build_v4_6_placeholder and get_placeholder_mask instead.

    # Step 3: Write YAML config (follows official OpenBMB LLaMA-Factory example closely)
    # Use the official registry model name with underscore: openbmb/MiniCPM-V-4_6
    # image_max_pixels limits image size to prevent OOM (matches qwen3_vl example).
    # preprocessing_num_workers: 1 is plenty for a 10-image dataset.
    config_yaml = f"""
model_name_or_path: openbmb/MiniCPM-V-4_6
image_max_pixels: 262144
trust_remote_code: true

stage: sft
do_train: true

finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: {lora_rank}

dataset: halide_film_defects
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

logging_steps: 10
save_steps: 100
save_total_limit: 3
report_to: none
save_only_model: false

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
    checkpoint_path: str = "/checkpoints/minicpm-v-4.6-lora",
    output_dir: str = "/checkpoints/minicpm-v-4.6-merged",
):
    """Export LoRA checkpoint merged with base model."""
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

    result = subprocess.run(
        ["llamafactory-cli", "export", str(config_path)],
        cwd="/root/LLaMA-Factory",
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"Export failed with return code {result.returncode}")
        return None

    print(f"Exported merged model to {output_dir}")
    checkpoint_volume.commit()
    return output_dir


@app.local_entrypoint()
def main():
    """Run training locally or on Modal."""
    train.remote()
