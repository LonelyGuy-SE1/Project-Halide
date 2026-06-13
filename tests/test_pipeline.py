from __future__ import annotations

from PIL import Image

from pipeline import pipeline


def test_run_diagnosis_orchestrates_two_stages(monkeypatch) -> None:
    def fake_extract(image):
        assert image.size == (8, 8)
        return {
            "defects": [{"label": "dust", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            "defect_count": 1,
            "label_counts": {"dust": 1},
            "inference_seconds": 0.01,
            "model_path": "vision-stub",
        }

    def fake_diagnose(defect_result, **metadata):
        assert defect_result["defect_count"] == 1
        assert metadata["film_type"] == "Kodak Gold 200"
        assert metadata["metadata_confidence"] == "medium"
        return {
            "diagnosis_text": "Dust on scanner glass.",
            "reasoning_seconds": 0.02,
            "model_path": "reasoning-stub",
        }

    monkeypatch.setattr(pipeline, "extract", fake_extract)
    monkeypatch.setattr(pipeline, "diagnose", fake_diagnose)

    result = pipeline.run_diagnosis(
        Image.new("RGB", (8, 8), "black"),
        film_type="Kodak Gold 200",
        film_age_years=4,
        storage="fridge, sealed",
        scan_resolution_dpi=4000,
        metadata_confidence="medium",
    )

    assert result["film_metadata"]["film_type"] == "Kodak Gold 200"
    assert result["film_metadata"]["metadata_confidence"] == "medium"
    assert result["defects"]["model_path"] == "vision-stub"
    assert result["diagnosis"]["model_path"] == "reasoning-stub"
    assert result["total_seconds"] >= 0
