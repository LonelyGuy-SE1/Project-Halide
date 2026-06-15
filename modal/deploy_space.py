"""Deploy the prepared Halide Space bundle to Hugging Face from Modal.

Uses the Modal `huggingface-secret` token, which has org write permissions.
The local `.nottracked/space_upload` directory is mounted into the container.
"""

from __future__ import annotations

import modal

app = modal.App("halide-space-deploy")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.20.0")
    .add_local_dir(".nottracked/space_upload", remote_path="/space_upload")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def deploy_space(
    repo_id: str = "Lonelyguyse1/project-halide",
    hardware: str = "t4-small",
    model_repo_id: str = "Lonelyguyse1/halide-vision",
) -> str:
    from pathlib import Path

    from huggingface_hub import HfApi, create_repo

    folder = Path("/space_upload")
    if not folder.exists():
        raise FileNotFoundError("mounted Space bundle not found at /space_upload")

    variables = [
        {"key": "HALIDE_USE_FINETUNED_VISION", "value": "1"},
        {
            "key": "HALIDE_VISION_FINETUNED_MODEL_ID",
            "value": model_repo_id,
        },
        {"key": "HALIDE_GPU_DURATION_SECONDS", "value": "120"},
        {"key": "HALIDE_ENABLE_TILE_FALLBACK", "value": "1"},
        {"key": "HALIDE_TILE_MAX_SIDE", "value": "960"},
        {"key": "HALIDE_TILE_OVERLAP", "value": "0.35"},
        {"key": "HALIDE_TILE_MAX_TILES", "value": "9"},
    ]

    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
        space_variables=variables,
    )
    api = HfApi()
    for item in variables:
        api.add_space_variable(repo_id=repo_id, key=item["key"], value=item["value"])
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy Project Halide Gradio Space",
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
    )
    try:
        api.request_space_hardware(repo_id=repo_id, hardware=hardware)
    except Exception as exc:
        return (
            f"https://huggingface.co/spaces/{repo_id} "
            f"(uploaded, hardware request skipped: {exc})"
        )
    return f"https://huggingface.co/spaces/{repo_id}"


@app.local_entrypoint()
def main(
    repo_id: str = "Lonelyguyse1/project-halide",
    hardware: str = "t4-small",
    model_repo_id: str = "Lonelyguyse1/halide-vision",
):
    url = deploy_space.remote(
        repo_id=repo_id,
        hardware=hardware,
        model_repo_id=model_repo_id,
    )
    print(f"Deployed Space: {url}")
