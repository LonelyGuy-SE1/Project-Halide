"""Create or update the HF Hub repo for the merged MiniCPM-V 4.6 model.

Idempotent: re-running will not fail if the repo already exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = os.getenv("HALIDE_UPLOAD_REPO_ID", "Lonelyguyse1/halide-vision")
LOCAL_DIR = Path(
    os.getenv(
        "HALIDE_UPLOAD_LOCAL_DIR",
        "checkpoints/minicpm-v-4.6-merged-v4-stage1",
    )
)


def main() -> int:
    api = HfApi()
    print(f"Ensuring repo {REPO_ID} exists")
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=False,
        exist_ok=True,
        token=os.getenv("HF_TOKEN"),
    )

    if not LOCAL_DIR.exists():
        print(f"Local model dir not found: {LOCAL_DIR}", file=sys.stderr)
        return 1

    print(f"Uploading from {LOCAL_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload Halide fine-tuned MiniCPM-V 4.6 (LoRA merged)",
        ignore_patterns=["*.fp16", "Modelfile"],
    )
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
