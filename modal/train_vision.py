"""
Modal training script for MiniCPM-V 4.6 using LLaMA-Factory.
Follows official OpenBMB documentation exactly.
"""
import modal

app = modal.App("halide-vision-training")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .run_commands(
        "cd /root && git clone https://github.com/hiyouga/LLaMA-Factory",
        "cd /root/LLaMA-Factory && pip install -e '.[torch,metrics,deepspeed,minicpm_v]'",
    )
    .pip_install("huggingface_hub", "datasets")
)

# Volumes for data and checkpoints
data_volume = modal.Volume.from_name("halide-training-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("halide-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
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
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    max_samples: int = 1000,
    max_seq_length: int = 3072,
    output_dir: str = "/checkpoints/minicpm-v-4.6-lora",
):
    """
    Run LoRA fine-tuning on MiniCPM-V 4.6.
    Follows the official OpenBMB LLaMA-Factory example.
    """
    import subprocess
    from pathlib import Path

    print("=== Project Halide: Vision Model Training ===")
    print(f"GPU: T4 | Epochs: {epochs} | LR: {learning_rate} | LoRA rank: {lora_rank}")

    # Step 1: Register dataset in LLaMA-Factory
    dataset_info_path = Path("/root/LLaMA-Factory/data/dataset_info.json")
    import json
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

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

    # Step 3: Write YAML config (follows official OpenBMB example exactly)
    config_yaml = f"""
model_name_or_path: openbmb/MiniCPM-V-4.6
trust_remote_code: true

stage: sft
do_train: true

finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: {lora_rank}

dataset: halide_film_defects
template: minicpm_v
cutoff_len: {max_seq_length}
max_samples: {max_samples}
overwrite_cache: true
preprocessing_num_workers: 4

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

output_dir: {output_dir}
"""

    config_path = Path("/root/LLaMA-Factory/train_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml.strip())

    print(f"Config written to {config_path}")
    print("Starting training...")

    # Step 4: Run training
    result = subprocess.run(
        ["llamafactory-cli", "train", str(config_path)],
        capture_output=False,
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
    gpu="T4",
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
model_name_or_path: openbmb/MiniCPM-V-4.6
adapter_name_or_path: {checkpoint_path}
template: minicpm_v
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
