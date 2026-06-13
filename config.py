"""Runtime configuration for Project Halide.

This module intentionally contains no model imports. It is safe to import in
local CPU-only tooling, tests, and dataset preparation scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
STORAGE_DIR = REPO_ROOT / "storage"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

CANONICAL_VISION_MODEL_ID = "openbmb/MiniCPM-V-4.6"
VISION_MODEL_ALIASES = {
    "openbmb/MiniCPM-V-4_6": CANONICAL_VISION_MODEL_ID,
}
DEFAULT_FINETUNED_MODEL_ID = "Lonelyguyse1/halide-vision"
DEFAULT_REASONING_MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def normalize_model_id(model_id: str) -> str:
    return VISION_MODEL_ALIASES.get(model_id, model_id)


@dataclass(frozen=True)
class VisionConfig:
    base_model_id: str
    finetuned_model_id: str
    local_model_path: Path
    use_finetuned: bool
    downsample_mode: str
    max_slice_nums: int
    max_new_tokens: int
    max_input_pixels: int


@dataclass(frozen=True)
class ReasoningConfig:
    model_id: str
    max_new_tokens: int


@dataclass(frozen=True)
class AppConfig:
    db_path: Path
    cache_size: int
    cache_ttl_seconds: int
    gpu_duration_seconds: int
    max_history_items: int


def get_vision_config() -> VisionConfig:
    return VisionConfig(
        base_model_id=normalize_model_id(
            os.getenv("HALIDE_VISION_BASE_MODEL_ID", CANONICAL_VISION_MODEL_ID)
        ),
        finetuned_model_id=os.getenv(
            "HALIDE_VISION_FINETUNED_MODEL_ID", DEFAULT_FINETUNED_MODEL_ID
        ),
        local_model_path=env_path(
            "HALIDE_VISION_LOCAL_MODEL_PATH",
            CHECKPOINT_DIR / "minicpm-v-4.6-merged-v3",
        ),
        use_finetuned=env_bool("HALIDE_USE_FINETUNED_VISION", False),
        downsample_mode=os.getenv("HALIDE_DOWNSAMPLE_MODE", "4x"),
        max_slice_nums=env_int("HALIDE_MAX_SLICE_NUMS", 36),
        max_new_tokens=env_int("HALIDE_MAX_NEW_TOKENS", 2048),
        max_input_pixels=env_int("HALIDE_MAX_INPUT_PIXELS", 4_000_000),
    )


def get_reasoning_config() -> ReasoningConfig:
    return ReasoningConfig(
        model_id=os.getenv("HALIDE_REASONING_MODEL_ID", DEFAULT_REASONING_MODEL_ID),
        max_new_tokens=env_int("HALIDE_NEMOTRON_MAX_TOKENS", 768),
    )


def get_app_config() -> AppConfig:
    return AppConfig(
        db_path=env_path("HALIDE_DB_PATH", STORAGE_DIR / "halide.db"),
        cache_size=env_int("HALIDE_CACHE_SIZE", 64),
        cache_ttl_seconds=env_int("HALIDE_CACHE_TTL_SECONDS", 3600),
        gpu_duration_seconds=env_int("HALIDE_GPU_DURATION_SECONDS", 300),
        max_history_items=env_int("HALIDE_HISTORY_LIMIT", 10),
    )


def running_on_hugging_face_space() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


def require_gpu_for_inference(stage: str) -> None:
    """Refuse model inference unless a CUDA device is visible.

    Local CPU use is allowed for file I/O, JSON parsing, image resizing, tests,
    and dataset preparation. It is not allowed for loading or running the
    vision or reasoning models.
    """
    import torch

    if torch.cuda.is_available():
        return

    raise RuntimeError(
        f"Halide refused to run {stage} model inference because no CUDA GPU "
        "is visible. Run inference on Modal, Hugging Face ZeroGPU, or another "
        "GPU runtime. Local CPU is reserved for editing, parsing, and tests."
    )


__all__ = [
    "AppConfig",
    "CHECKPOINT_DIR",
    "CANONICAL_VISION_MODEL_ID",
    "DATA_DIR",
    "DEFAULT_FINETUNED_MODEL_ID",
    "DEFAULT_REASONING_MODEL_ID",
    "REPO_ROOT",
    "ReasoningConfig",
    "STORAGE_DIR",
    "VisionConfig",
    "env_bool",
    "env_int",
    "env_path",
    "get_app_config",
    "get_reasoning_config",
    "get_vision_config",
    "normalize_model_id",
    "require_gpu_for_inference",
    "running_on_hugging_face_space",
]
