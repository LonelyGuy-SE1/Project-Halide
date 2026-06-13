from __future__ import annotations

from PIL import Image

from data.schemas import (
    MIN_DEFECT_CONFIDENCE,
    bbox_iou,
    clean_defects,
    dedupe_defects,
    normalize_bbox,
)
from models.vision.prompts import DETECTION_PROMPT_INT
from scripts.convert_to_sharegpt import SYSTEM_PROMPT_INT
from models.vision.minicpm_wrapper import DETECTION_PROMPT
from models.vision import inference
from models.vision.minicpm_wrapper import _parse_defect_json


def test_normalize_bbox_accepts_int_grid_and_nested_float_bbox() -> None:
    assert normalize_bbox([0, 0, 999, 999]) == (0.0, 0.0, 1.0, 1.0)
    assert normalize_bbox([[0.1, 0.2, 0.3, 0.4]]) == (0.1, 0.2, 0.3, 0.4)


def test_clean_defects_drops_unknown_labels_and_bad_boxes() -> None:
    cleaned, dropped = clean_defects(
        [
            {"label": "dust", "bbox": [0, 0, 10, 10]},
            {"label": "unknown_label", "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"label": "scratch", "bbox": [0.9, 0.2, 0.1, 0.4]},
        ]
    )
    assert dropped == 2
    assert cleaned == [{"label": "dust", "bbox": [0.0, 0.0, 0.01001, 0.01001]}]


def test_clean_defects_drops_low_confidence() -> None:
    cleaned, dropped = clean_defects(
        [
            {"label": "dust", "bbox": [0, 0, 10, 10], "confidence": 0.2},
            {"label": "scratch", "bbox": [20, 20, 30, 30], "confidence": 0.9},
        ]
    )
    assert dropped == 1
    assert cleaned[0]["label"] == "scratch"
    assert cleaned[0]["confidence"] == 0.9


def test_dedupe_defects_drops_exact_repeats() -> None:
    unique, duplicate_count = dedupe_defects(
        [
            {"label": "dust", "bbox": [0, 0, 10, 10]},
            {"label": "dust", "bbox": [0, 0, 10, 10]},
            {"label": "scratch", "bbox": [20, 20, 30, 30]},
        ]
    )
    assert len(unique) == 2
    assert duplicate_count == 1


def test_dedupe_defects_merges_high_overlap_and_keeps_confidence() -> None:
    unique, duplicate_count = dedupe_defects(
        [
            {"label": "scratch", "bbox": [0.1, 0.1, 0.4, 0.4], "confidence": 0.4},
            {"label": "scratch", "bbox": [0.11, 0.11, 0.41, 0.41], "confidence": 0.8},
            {"label": "dust", "bbox": [0.11, 0.11, 0.41, 0.41], "confidence": 0.8},
        ],
        iou_threshold=0.7,
    )
    assert duplicate_count == 1
    assert len(unique) == 2
    scratch = next(d for d in unique if d["label"] == "scratch")
    assert scratch["confidence"] == 0.8


def test_detection_prompt_rejects_non_film_content() -> None:
    assert "grass" in DETECTION_PROMPT
    assert "emulsion_damage" in DETECTION_PROMPT
    assert '{"defects": []}' in DETECTION_PROMPT


def test_detection_prompt_has_one_source_of_truth() -> None:
    assert DETECTION_PROMPT == DETECTION_PROMPT_INT
    assert SYSTEM_PROMPT_INT == DETECTION_PROMPT_INT


def test_clean_defects_drops_central_subject_hair_like_false_positive() -> None:
    cleaned, dropped = clean_defects(
        [
            {
                "label": "long_hair",
                "bbox": [250, 400, 750, 460],
            },
            {
                "label": "long_hair",
                "bbox": [0, 400, 900, 430],
            },
        ]
    )
    assert dropped == 1
    assert cleaned == [
        {"label": "long_hair", "bbox": [0.0, 0.4004, 0.900901, 0.43043]}
    ]


def test_min_defect_confidence_default_is_production_strict() -> None:
    assert MIN_DEFECT_CONFIDENCE >= 0.45


def test_bbox_iou() -> None:
    assert bbox_iou([0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]) == 0.142857


def test_parse_defect_json_handles_markdown_fence() -> None:
    parsed = _parse_defect_json(
        '```json\n{"defects":[{"label":"dust","bbox":[0,0,999,999]}]}\n```'
    )
    assert parsed["defects"][0]["label"] == "dust"


def test_parse_defect_json_salvages_truncated_defect_array() -> None:
    parsed = _parse_defect_json(
        '{"defects":[{"label":"dust","bbox":[0,0,10,10]},'
        '{"label":"scratch","bbox":[20,20,30,30]},'
        '{"label":"short_hair","bbox":[40,40'
    )
    assert parsed["_parse_warning"] == "salvaged_defect_fragments"
    assert [d["label"] for d in parsed["defects"]] == ["dust", "scratch"]


def test_extract_defects_uses_detector_stub(monkeypatch) -> None:
    class StubDetector:
        model_path = "stub"

        def detect(self, image):
            assert image.mode == "RGB"
            return {
                "defects": [
                    {"label": "scratch", "bbox": [100, 200, 300, 400]},
                    {"label": "scratch", "bbox": [100, 200, 300, 400]},
                    {"label": "bad", "bbox": [0, 0, 1, 1]},
                ]
            }

    monkeypatch.setattr(inference, "get_detector", lambda: StubDetector())
    img = Image.new("RGB", (16, 16), "black")
    result = inference.extract_defects(img)
    assert result["defect_count"] == 1
    assert result["label_counts"] == {"scratch": 1}
    assert result["dropped_count"] == 1
    assert result["duplicate_count"] == 1


def test_extract_defects_resizes_large_images_for_model(monkeypatch) -> None:
    monkeypatch.setenv("HALIDE_MAX_INPUT_PIXELS", "100")

    class StubDetector:
        model_path = "stub"

        def detect(self, image):
            assert image.size[0] * image.size[1] <= 100
            return {"defects": []}

    monkeypatch.setattr(inference, "get_detector", lambda: StubDetector())
    img = Image.new("RGB", (64, 64), "black")
    result = inference.extract_defects(img)
    assert result["resized_for_model"] is True
    assert result["defect_count"] == 0
