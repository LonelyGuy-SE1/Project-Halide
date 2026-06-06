"""MiniCPM-V 4.6 wrapper. Loads the model and runs inference on film scans."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = REPO_ROOT / "checkpoints" / "minicpm-v-4.6-merged"
HF_MODEL_ID = "Lonelyguyse1/halide-vision"
BASE_MODEL_ID = "openbmb/MiniCPM-V-4_6"

DOWNSAMPLE_MODE = os.getenv("HALIDE_DOWNSAMPLE_MODE", "4x")
MAX_SLICE_NUMS = int(os.getenv("HALIDE_MAX_SLICE_NUMS", "36"))
MAX_NEW_TOKENS = int(os.getenv("HALIDE_MAX_NEW_TOKENS", "3072"))

DETECTION_PROMPT = (
    "You are a film defect detection engine. Analyze the film scan and detect "
    "all visible defects. Output a JSON object with a 'defects' array. Each "
    "defect has: 'label' (dust, dirt, scratch, long_hair, short_hair), "
    "'bbox' as 4 integers in the [0, 999] grid "
    "[x_min, y_min, x_max, y_max] (multiply by image width/height to get pixels). "
    "Output JSON only, no explanation."
)


def _resolve_model_path() -> str:
    """Pick local merged model if present, else HF repo, else base model id."""
    if LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "config.json").exists():
        logger.info("Using local merged model at %s", LOCAL_MODEL_PATH)
        return str(LOCAL_MODEL_PATH)
    if os.getenv("HF_TOKEN"):
        logger.info("Using HF Hub repo %s", HF_MODEL_ID)
        return HF_MODEL_ID
    logger.info("Falling back to base model %s", BASE_MODEL_ID)
    return BASE_MODEL_ID


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
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading MiniCPM-V 4.6 from %s", self._model_path)
        self._processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self._dtype = torch.bfloat16
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_path,
            torch_dtype=self._dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._device = str(next(self._model.parameters()).device)
        logger.info("Model loaded on %s", self._device)

    def detect(self, image: Any) -> dict:
        """Run defect detection on a PIL image. Returns parsed JSON dict."""
        import torch

        if self._model is None:
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=DOWNSAMPLE_MODE,
            max_slice_nums=MAX_SLICE_NUMS,
        ).to(self._device)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                downsample_mode=DOWNSAMPLE_MODE,
                max_new_tokens=MAX_NEW_TOKENS,
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        logger.warning("No JSON found in model output: %r", text[:200])
        return {"defects": [], "_raw": text, "_parse_error": "no_json_object"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s; raw: %r", exc, text[:200])
        return {"defects": [], "_raw": text, "_parse_error": str(exc)}


_default_detector: MiniCPMVDetector | None = None


def get_detector() -> MiniCPMVDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = MiniCPMVDetector()
    return _default_detector
