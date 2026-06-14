"""Upload a merged Halide vision checkpoint from Modal Volume to Hugging Face."""

from __future__ import annotations

import modal

app = modal.App("halide-model-upload")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.20.0"
)

checkpoint_volume = modal.Volume.from_name("halide-checkpoints", create_if_missing=True)


MODEL_CARD = """---
license: apache-2.0
base_model: openbmb/MiniCPM-V-4.6
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - film
  - computer-vision
  - defect-detection
  - minicpm-v
---

# Halide Vision

Halide Vision is a MiniCPM-V 4.6 checkpoint fine-tuned for analog film-scan
defect extraction. It is maintained by
[Lonelyguyse1](https://huggingface.co/Lonelyguyse1) for
[Project Halide](https://github.com/Lonelyguyse1/Project-Halide).

The model emits JSON defect proposals for dust, dirt, scratches, hair-like
surface contamination, emulsion damage, chemical stains, and light leaks. The
Project Halide runtime validates the JSON schema, removes low-confidence or
duplicate boxes, and uses tiled inspection when large scans hide thin crack
networks at full-frame scale.

## Training Summary

- Base model: `openbmb/MiniCPM-V-4.6`
- Training method: LoRA fine-tuning with LLaMA-Factory, merged for inference
- Curriculum: FilmDamageSimulator annotations, procedural film-defect positives,
  hard clean negatives, and a v7 crack curriculum
- Held-out private negatives: used only for evaluation, not for training

## Held-Out Smoke Result

Final v7 checkpoint with 960 px tiled fallback:

| Sample | Expected surface condition | Result |
| --- | --- | --- |
| negative1 | Long scratches across portrait | 8 defects |
| negative2 | Abraded emulsion and dirt patches | 9 defects |
| negative3 | Severe emulsion damage and debris | 6 defects |
| negative4 | Near-clean hard negative | 0 defects |
| negative5 | Broad lifted crack network | 45 defects |

## Runtime Notes

This model is intended to run inside Project Halide with GPU inference. The
runtime refuses local CPU model inference and does not call cloud inference APIs.

The repo also includes a llama.cpp Q4_K_M GGUF artifact:
`minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf`.

Use the model as an inspection aid. It can over-box broad damage regions, and
film metadata should be treated as context unless verified by notes or edge
marks.
"""


def _write_model_card(path, repo_id: str) -> None:
    card = MODEL_CARD.replace("Lonelyguyse1/halide-vision", repo_id)
    (path / "README.md").write_text(card, encoding="utf-8")


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

    _write_model_card(path, repo_id)
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


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def upload_model_card(
    repo_id: str = "Lonelyguyse1/halide-vision",
) -> str:
    from huggingface_hub import HfApi

    card = MODEL_CARD.replace("Lonelyguyse1/halide-vision", repo_id)
    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update Halide Vision model card",
    )
    return repo_id


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
def card(repo_id: str = "Lonelyguyse1/halide-vision"):
    uploaded = upload_model_card.remote(repo_id=repo_id)
    print(f"Updated model card for {uploaded}")


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
