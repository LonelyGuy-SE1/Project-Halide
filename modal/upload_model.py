"""Upload a merged Halide vision checkpoint from Modal Volume to Hugging Face."""

from __future__ import annotations

import modal

app = modal.App("halide-model-upload")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.20.0"
)

checkpoint_volume = modal.Volume.from_name("halide-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def upload_merged_model(
    model_dir: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
    repo_id: str = "Lonelyguyse1/halide-vision",
    private: bool = False,
) -> str:
    from pathlib import Path

    from huggingface_hub import HfApi, create_repo

    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Halide MiniCPM-V 4.6 merged checkpoint from {path.name}",
        ignore_patterns=["*.tmp", "optimizer.pt", "scheduler.pt"],
    )
    return repo_id


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def upload_checkpoint_file(
    local_path: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1-q4_k_m.gguf",
    repo_id: str = "Lonelyguyse1/halide-vision",
    path_in_repo: str = "minicpm-v-4.6-merged-v4-stage1-q4_k_m.gguf",
) -> str:
    from pathlib import Path

    from huggingface_hub import HfApi

    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"local_path does not exist: {local_path}")

    HfApi().upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {path_in_repo}",
    )
    return f"{repo_id}/{path_in_repo}"


@app.local_entrypoint()
def main(
    model_dir: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
    repo_id: str = "Lonelyguyse1/halide-vision",
    private: bool = False,
):
    uploaded = upload_merged_model.remote(
        model_dir=model_dir,
        repo_id=repo_id,
        private=private,
    )
    print(f"Uploaded {model_dir} to {uploaded}")


@app.local_entrypoint()
def file(
    local_path: str = "/checkpoints/minicpm-v-4.6-merged-v4-stage1-q4_k_m.gguf",
    repo_id: str = "Lonelyguyse1/halide-vision",
    path_in_repo: str = "minicpm-v-4.6-merged-v4-stage1-q4_k_m.gguf",
):
    uploaded = upload_checkpoint_file.remote(
        local_path=local_path,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
    )
    print(f"Uploaded {uploaded}")
