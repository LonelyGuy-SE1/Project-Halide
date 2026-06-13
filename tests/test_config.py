from __future__ import annotations

import pytest

import config


def test_model_alias_normalizes_to_canonical_id() -> None:
    assert (
        config.normalize_model_id("openbmb/MiniCPM-V-4_6")
        == "openbmb/MiniCPM-V-4.6"
    )


def test_require_gpu_for_inference_raises_on_cpu_runtime(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        config.require_gpu_for_inference("vision")
