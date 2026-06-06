"""Nemotron-Mini-4B wrapper. Loads the model and generates diagnoses.

Per AGENTS.md, this is the second stage of the dual-model pipeline.
Receives defect JSON from the vision model plus user metadata, returns
root cause diagnosis and physical remediation steps.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

NEMOTRON_MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"
MAX_NEW_TOKENS = int(os.getenv("HALIDE_NEMOTRON_MAX_TOKENS", "512"))


class NemotronReasoner:
    """Lazy-loading wrapper around Nemotron-Mini-4B-Instruct."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or NEMOTRON_MODEL_ID
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str = "cpu"
        self._dtype: Any = None

    @property
    def model_path(self) -> str:
        return self._model_path

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Nemotron-Mini-4B from %s", self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._dtype = torch.bfloat16
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=self._dtype,
            device_map="auto",
        )
        self._device = str(next(self._model.parameters()).device)
        logger.info("Nemotron loaded on %s", self._device)

    def generate(self, prompt: str, system: str | None = None) -> str:
        if self._model is None:
            self.load()

        import torch

        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            output = self._model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response_ids = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(response_ids, skip_special_tokens=True)

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None


_default_reasoner: NemotronReasoner | None = None


def get_reasoner() -> NemotronReasoner:
    global _default_reasoner
    if _default_reasoner is None:
        _default_reasoner = NemotronReasoner()
    return _default_reasoner
