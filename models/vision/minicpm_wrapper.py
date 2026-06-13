"""MiniCPM-V 4.6 wrapper. Loads the model and runs inference on film scans."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from config import CHECKPOINT_DIR, get_vision_config, require_gpu_for_inference
from models.vision.prompts import DETECTION_PROMPT_INT

logger = logging.getLogger(__name__)

DETECTION_PROMPT = DETECTION_PROMPT_INT


def _resolve_model_path() -> str:
    """Pick configured fine-tuned model or public base model."""
    cfg = get_vision_config()
    explicit = os.getenv("HALIDE_VISION_MODEL_ID")
    if explicit:
        logger.info("Using explicit vision model %s", explicit)
        return explicit

    if cfg.use_finetuned:
        local_candidates = [
            cfg.local_model_path,
            CHECKPOINT_DIR / "minicpm-v-4.6-merged",
        ]
        seen: set[str] = set()
        for path in local_candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            if path.exists() and (path / "config.json").exists():
                logger.info("Using local fine-tuned vision model at %s", path)
                return str(path)
        logger.info("Using fine-tuned vision model repo %s", cfg.finetuned_model_id)
        return cfg.finetuned_model_id

    logger.info("Using base vision model %s", cfg.base_model_id)
    return cfg.base_model_id


class MiniCPMVDetector:
    """Lazy-loading wrapper around MiniCPM-V 4.6 for film defect detection."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or _resolve_model_path()
        self._model: Any = None
        self._processor: Any = None
        self._dtype: Any = None
        self._device: str = "cpu"

    @property
    def model_path(self) -> str:
        return self._model_path

    def load(self) -> None:
        if self._model is not None:
            return
        require_gpu_for_inference("vision")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading MiniCPM-V 4.6 from %s", self._model_path)
        self._processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self._dtype = _select_cuda_dtype(torch)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_path,
            torch_dtype=self._dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._device = str(next(self._model.parameters()).device)
        logger.info("Model loaded on %s with dtype %s", self._device, self._dtype)

    def detect(self, image: Any) -> dict:
        """Run defect detection on a PIL image. Returns parsed JSON dict."""
        import torch

        if self._model is None:
            self.load()

        cfg = get_vision_config()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }
        ]

        inputs = _apply_chat_template(
            self._processor,
            messages,
            downsample_mode=cfg.downsample_mode,
            max_slice_nums=cfg.max_slice_nums,
        ).to(self._device)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                downsample_mode=cfg.downsample_mode,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return _parse_defect_json(text)

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None


def _parse_defect_json(text: str) -> dict:
    """Extract and parse the first JSON object from model output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {"defects": parsed}
        if isinstance(parsed, dict):
            return parsed
        return {"defects": [], "_raw": text, "_parse_error": "json_not_object"}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        logger.warning("No JSON found in model output: %r", text[:200])
        return {"defects": [], "_raw": text, "_parse_error": "no_json_object"}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
        return {"defects": [], "_raw": text, "_parse_error": "json_not_object"}
    except json.JSONDecodeError as exc:
        fragments = _parse_defect_fragments(text)
        if fragments:
            logger.warning(
                "Salvaged %s defect fragments from malformed JSON: %s",
                len(fragments),
                exc,
            )
            return {
                "defects": fragments,
                "_parse_error": str(exc),
                "_parse_warning": "salvaged_defect_fragments",
            }
        logger.warning("JSON parse error: %s; raw: %r", exc, text[:200])
        return {"defects": [], "_raw": text, "_parse_error": str(exc)}


def _parse_defect_fragments(text: str) -> list[dict[str, Any]]:
    """Recover complete defect objects from truncated JSON arrays."""
    fragments: list[dict[str, Any]] = []
    for match in re.finditer(r"\{[^{}]*\"label\"[^{}]*\"bbox\"\s*:\s*\[[^\]]+\][^{}]*\}", text):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            fragments.append(candidate)
    return fragments


def _apply_chat_template(
    processor: Any,
    messages: list[dict],
    *,
    downsample_mode: str,
    max_slice_nums: int,
) -> Any:
    """Call MiniCPM chat template across Transformers API variants."""
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(
            messages,
            **kwargs,
            downsample_mode=downsample_mode,
            max_slice_nums=max_slice_nums,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            **kwargs,
            processor_kwargs={
                "downsample_mode": downsample_mode,
                "max_slice_nums": max_slice_nums,
            },
        )


def _select_cuda_dtype(torch_module: Any) -> Any:
    major, _minor = torch_module.cuda.get_device_capability()
    if major >= 8:
        return torch_module.bfloat16
    return torch_module.float16


_default_detector: MiniCPMVDetector | None = None


def get_detector() -> MiniCPMVDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = MiniCPMVDetector()
    return _default_detector
