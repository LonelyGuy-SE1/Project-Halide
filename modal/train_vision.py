"""
Modal training script for MiniCPM-V 4.6 using LLaMA-Factory.
Runs LoRA fine-tuning on A100-80GB with FilmDamageSimulator + BlueNeg data.
"""
import modal

app = modal.App("halide-vision-training")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .run_commands(
        # Clone and install LLaMA-Factory
        "cd /root && git clone https://github.com/hiyouga/LLaMA-Factory",
        "cd /root/LLaMA-Factory && pip install -e '.[torch,metrics,deepspeed,minicpm_v]'",
    )
    .pip_install(
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
    dataset_path: str = "/data/training_data.json",
    output_dir: str = "/checkpoints/minicpm-v-4.6-lora",
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    max_seq_length: int = 2048,
):
    """
    Run LoRA fine-tuning on MiniCPM-V 4.6.

    Args:
        dataset_path: Path to JSONL training data on volume
        output_dir: Path to save LoRA checkpoint
        epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate for LoRA
        lora_rank: LoRA rank (r parameter)
        max_seq_length: Max sequence length
    """
    import subprocess
    import json
    from pathlib import Path

    print("=== Project Halide: Vision Model Training ===")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, LR: {learning_rate}, LoRA rank: {lora_rank}")

    # Create LLaMA-Factory config
    config = {
        "model_name_or_path": "openbmb/MiniCPM-V-4.6",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "q_proj,v_proj",
        "lora_rank": lora_rank,
        "dataset": "halide_film_defects",
        "template": "minicpm_v",
        "cutoff_len": max_seq_length,
        "preprocessing_num_workers": 4,
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "report_to": "none",
    }

    # Write config file
    config_path = Path("/root/train_config.yaml")
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config written to {config_path}")
    print("Starting training...")

    # Run training
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
    volumes={"/checkpoints": checkpoint_volume},
    timeout=3600,
)
def export_model(
    checkpoint_path: str = "/checkpoints/minicpm-v-4.6-lora",
    output_dir: str = "/checkpoints/minicpm-v-4.6-merged",
):
    """Export LoRA checkpoint merged with base model."""
    import subprocess
    import yaml
    from pathlib import Path

    print("=== Exporting Merged Model ===")

    config = {
        "model_name_or_path": "openbmb/MiniCPM-V-4.6",
        "adapter_name_or_path": checkpoint_path,
        "template": "minicpm_v",
        "finetuning_type": "lora",
        "export_dir": output_dir,
        "export_size": 2,
        "export_legacy_format": False,
    }

    config_path = Path("/root/export_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

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
