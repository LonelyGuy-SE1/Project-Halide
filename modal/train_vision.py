"""
Modal training script for MiniCPM-V 4.6 using LLaMA-Factory.
Follows official OpenBMB documentation exactly.

Current v4 defaults:
  - lora_rank: 64
  - lora_alpha: 128
  - lora_target: q_proj,v_proj,k_proj,o_proj (vision tower + language)
  - num_train_epochs: 8
  - bbox format: integer [0, 999] grid (aligned with VLM pre-training)
"""
import modal

app = modal.App("halide-vision-training-v4")

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
    timeout=8 * 3600,
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
    train_json_path: str = "/data/training_sharegpt.json",
    val_json_path: str = "/data/training_sharegpt_val.json",
    resume_from_checkpoint: str = "",
):
    """
    Run LoRA fine-tuning on MiniCPM-V 4.6 with vision tower targeting.

    Per log.txt (2026-06-06):
      - rank 64, alpha 128
      - target q_proj,v_proj,k_proj,o_proj (covers both SigLIP2-400M vision
        tower and Qwen3.5-0.8B language backbone; both module types share
        these names)
      - 8 epochs by default for stage-one v4
      - integer [0, 999] bbox grid format

    Note on early stopping: LLaMA-Factory v0.9.5's HfArgumentParser rejects
    the standard HuggingFace `early_stopping_patience` /
    `early_stopping_threshold` keys when passed via custom YAML. We use
    `load_best_model_at_end: true` + `metric_for_best_model: eval_loss`
    so the best checkpoint (lowest eval_loss) is kept at termination;
    if eval_loss diverges or hits NaN, kill the modal subprocess
    manually and the best checkpoint so far is preserved.

    Uses A100-80GB because T4 (16GB) is insufficient for MiniCPM-V 4.6 training.
    """
    import subprocess
    import os
    from pathlib import Path

    print("=== Project Halide: Vision Model Training v4 ===")
    print(f"GPU: A100-80GB | Epochs: {epochs} | LR: {learning_rate}")
    print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, target={lora_target}")
    print(f"cutoff_len: {max_seq_length} (fits 150 defects in int 0-999 format)")
    print(f"train_json_path: {train_json_path}")
    print(f"val_json_path: {val_json_path}")
    print(f"Best-model-at-end: enabled, save + eval strategy: epoch")
    if resume_from_checkpoint == "auto":
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        resume_from_checkpoint = str(checkpoints[-1]) if checkpoints else ""
    if resume_from_checkpoint:
        print(f"resume_from_checkpoint: {resume_from_checkpoint}")

    # Step 1: Register both train and val datasets in LLaMA-Factory
    import json
    dataset_dir = Path("/root/LLaMA-Factory/data")
    dataset_info_path = dataset_dir / "dataset_info.json"

    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    train_json_name = Path(train_json_path).name
    val_json_name = Path(val_json_path).name

    dataset_info["halide_film_defects"] = {
        "file_name": train_json_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images",
        },
    }
    dataset_info["halide_film_defects_val"] = {
        "file_name": val_json_name,
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
    shutil.copy(train_json_path, str(dataset_dir / train_json_name))
    shutil.copy(val_json_path, str(dataset_dir / val_json_name))

    scans_src = Path("/data/scans")
    scans_dst = Path("/root/LLaMA-Factory/data/scans")
    if scans_src.exists():
        shutil.copytree(scans_src, scans_dst, dirs_exist_ok=True)

    augmented_src = Path("/data/augmented")
    augmented_dst = Path("/root/LLaMA-Factory/data/augmented")
    if augmented_src.exists():
        shutil.copytree(augmented_src, augmented_dst, dirs_exist_ok=True)

    print("Copied data to LLaMA-Factory directory")

    # Step 3: Write YAML config
    config_yaml = f"""
model_name_or_path: openbmb/MiniCPM-V-4.6
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

# Best-model-at-end: keeps the checkpoint with the lowest eval_loss when
# training terminates. No early stopping keys (LLaMA-Factory v0.9.5's
# HfArgumentParser rejects them). Manual monitor for eval_loss divergence
# or NaN; kill subprocess if needed.
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false

eval_strategy: epoch

output_dir: {output_dir}
"""
    if resume_from_checkpoint:
        config_yaml += f"\nresume_from_checkpoint: {resume_from_checkpoint}\n"

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
model_name_or_path: openbmb/MiniCPM-V-4.6
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
def main(
    epochs: int = 8,
    do_export: bool = True,
    train_json_path: str = "/data/augmented/training_sharegpt_combined_v4.json",
    val_json_path: str = "/data/training_sharegpt_val_v4.json",
    output_dir: str = "/checkpoints/minicpm-v-4.6-lora-v4-stage1",
    merged_output_dir: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
    resume_from_checkpoint: str = "",
):
    """Run training (and optional export) on Modal A100-80GB.

    Current default is the v4 stage-one dataset and eight epochs. Longer runs
    can still be launched by passing --epochs.
    """
    output = train.remote(
        epochs=epochs,
        train_json_path=train_json_path,
        val_json_path=val_json_path,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    print(f"\nTraining output: {output}\n")
    if output is None:
        print("Training did not produce a checkpoint, skipping export.")
        return
    if do_export:
        merged = export_model.remote(
            checkpoint_path=output_dir,
            output_dir=merged_output_dir,
        )
        print(f"\nMerged model: {merged}\n")


@app.local_entrypoint()
def spawn_train(
    epochs: int = 8,
    train_json_path: str = "/data/augmented/training_sharegpt_combined_v4.json",
    val_json_path: str = "/data/training_sharegpt_val_v4.json",
    output_dir: str = "/checkpoints/minicpm-v-4.6-lora-v4-stage1",
    resume_from_checkpoint: str = "auto",
):
    """Start training as a detached Modal function call.

    Use this for long training runs when the local network is unreliable.
    The returned call id can be checked later with `get_train_result`.
    """
    call = train.spawn(
        epochs=epochs,
        train_json_path=train_json_path,
        val_json_path=val_json_path,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    print(f"Spawned training call: {call.object_id}")
    print(f"Dashboard: {call.get_dashboard_url()}")


@app.local_entrypoint()
def get_train_result(call_id: str, timeout_seconds: int = 5):
    """Check a detached training call and print the result when complete."""
    function_call = modal.FunctionCall.from_id(call_id)
    result = function_call.get(timeout=timeout_seconds)
    print(f"Training result: {result}")


@app.local_entrypoint()
def export_only(
    checkpoint_path: str = "/checkpoints/minicpm-v-4.6-lora-v4-stage1",
    output_dir: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
):
    """Export an already-trained LoRA checkpoint without retraining."""
    merged = export_model.remote(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )
    print(f"\nMerged model: {merged}\n")
