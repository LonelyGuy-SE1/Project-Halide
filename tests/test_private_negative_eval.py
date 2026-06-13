import json
import subprocess

from PIL import Image

from scripts.run_private_negative_eval import (
    RESULT_END,
    RESULT_START,
    _run,
    clean_model_result,
    parse_modal_stdout,
    select_model_result,
    write_image_report,
)


def test_parse_modal_stdout_extracts_marked_json():
    payload = {"image_path": "heldout_negative1.png", "ok": True}

    parsed = parse_modal_stdout(
        f"before\n{RESULT_START}\n{json.dumps(payload)}\n{RESULT_END}\nafter"
    )

    assert parsed == payload


def test_run_timeout_includes_subprocess_output(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=3,
            output="stdout clue",
            stderr="stderr clue",
        )

    monkeypatch.setattr("scripts.run_private_negative_eval.subprocess.run", fake_run)

    try:
        _run(["modal", "run", "broken.py"], timeout=3)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected timeout RuntimeError")

    assert "command timed out after 3s" in message
    assert "stdout clue" in message
    assert "stderr clue" in message


def test_select_model_result_uses_finetuned_result_for_both():
    payload = {
        "results": {
            "base": {"model_id": "base"},
            "finetuned": {"model_id": "finetuned"},
        }
    }

    assert select_model_result(payload, which="both")["model_id"] == "finetuned"


def test_clean_model_result_filters_invalid_and_dedupes():
    result = {
        "parsed_json": {
            "defects": [
                {"label": "scratch", "bbox": [10, 20, 100, 120], "confidence": 0.9},
                {"label": "scratch", "bbox": [10, 20, 100, 120], "confidence": 0.8},
                {"label": "grass", "bbox": [0, 0, 50, 50], "confidence": 0.9},
                {"label": "dust", "bbox": [1, 1, 2, 2], "confidence": 0.1},
            ]
        }
    }

    defects, dropped = clean_model_result(result)

    assert defects == [
        {
            "label": "scratch",
            "bbox": [0.01001, 0.02002, 0.1001, 0.12012],
            "confidence": 0.9,
        }
    ]
    assert dropped == {"invalid": 2, "duplicates": 1}


def test_write_image_report_creates_private_overlay_and_raw_json(tmp_path):
    image_path = tmp_path / "negative_test.png"
    Image.new("RGB", (100, 80), (30, 30, 30)).save(image_path)
    result = {
        "model_id": "/checkpoints/minicpm-v-4.6-merged-v4-stage1",
        "device": "cuda:0",
        "load_seconds": 1.2,
        "inference_seconds": 0.7,
    }
    defects = [{"label": "scratch", "bbox": [0.1, 0.2, 0.4, 0.5], "confidence": 0.9}]

    report = write_image_report(image_path, result, defects, out_dir=tmp_path)

    assert report["defect_count"] == 1
    assert report["label_counts"] == {"scratch": 1}
    assert (tmp_path / "negative_test_overlay.png").exists()
    assert (tmp_path / "negative_test_raw.json").exists()
