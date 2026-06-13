"""Nemotron-Mini-4B wrapper. Loads the model and generates diagnoses.

Per AGENTS.md, this is the second stage of the dual-model pipeline.
Receives defect JSON from the vision model plus user metadata, returns
root cause diagnosis and physical remediation steps.

The `generate` method accepts a fully-formed `messages: list[dict]` array
(system, user, assistant turns). It does NOT pre-process the messages; the
caller is responsible for assembling the full few-shot + system + current
request array. This is the only correct way to use a chat-tuned model with
`tokenizer.apply_chat_template`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from config import get_reasoning_config, require_gpu_for_inference

logger = logging.getLogger(__name__)

_ESCAPED_NEWLINE_PATTERN = re.compile(
    r"(```[\s\S]*?```|`[^`]+`)|(?<!\\)(?:\\r\\n|\\[nr])"
)


class NemotronReasoner:
    """Lazy-loading wrapper around Nemotron-Mini-4B-Instruct."""

    def __init__(self, model_path: str | None = None) -> None:
        cfg = get_reasoning_config()
        self._model_path = model_path or cfg.model_id
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
        require_gpu_for_inference("reasoning")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Nemotron-Mini-4B from %s", self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._dtype = _select_cuda_dtype(torch)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=self._dtype,
            device_map="auto",
        )
        self._device = str(next(self._model.parameters()).device)
        logger.info("Nemotron loaded on %s with dtype %s", self._device, self._dtype)

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Run chat completion on a fully-formed messages array.

        `messages` must be a list of dicts with `role` in
        {"system", "user", "assistant"} and `content` strings. The caller
        is responsible for assembling the full conversation including any
        few-shot examples. This wrapper just tokenizes and generates.
        """
        if self._model is None:
            self.load()

        if not messages:
            raise ValueError("messages must be a non-empty list of {role, content} dicts")

        inputs, prompt_length = _build_chat_inputs(
            self._tokenizer,
            messages,
            self._device,
        )

        import torch
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=get_reasoning_config().max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response_ids = output[0][prompt_length:]
        text = self._tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return normalize_response_text(text).strip()

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


def normalize_response_text(text: str) -> str:
    """Convert literal escaped newlines to display newlines outside code."""
    if not isinstance(text, str) or "\\" not in text:
        return text
    return _ESCAPED_NEWLINE_PATTERN.sub(lambda m: m.group(1) or "\n", text)


def _build_chat_inputs(
    tokenizer: Any,
    messages: list[dict[str, str]],
    device: str,
) -> tuple[dict[str, Any], int]:
    """Return generate kwargs across Transformers chat-template variants."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    if hasattr(encoded, "to"):
        encoded = encoded.to(device)

    if _has_input_ids(encoded):
        input_ids = encoded["input_ids"]
        return dict(encoded), input_ids.shape[-1]

    return {"input_ids": encoded}, encoded.shape[-1]


def _has_input_ids(encoded: Any) -> bool:
    try:
        return "input_ids" in encoded
    except (TypeError, RuntimeError):
        return False


def _select_cuda_dtype(torch_module: Any) -> Any:
    major, _minor = torch_module.cuda.get_device_capability()
    if major >= 8:
        return torch_module.bfloat16
    return torch_module.float16
