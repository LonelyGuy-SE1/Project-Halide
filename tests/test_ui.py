from __future__ import annotations

from PIL import Image

from storage.cache import get_cache
from ui import app as ui_app
from ui.components import history_detail_html, diagnosis_html


def test_html_renderers_escape_model_and_user_text() -> None:
    assert "<script>" not in diagnosis_html("<script>alert(1)</script>")
    rendered = diagnosis_html(
        "## Root cause\nTransport scratch.\n\n"
        "## Physical fixes\n1. Re-scan a crop.\n2. Inspect under a loupe."
    )
    assert "## Root cause" not in rendered
    assert "halide-report-section" in rendered
    assert "<li>Re-scan a crop.</li>" in rendered
    detail = history_detail_html(
        {
            "film_type": "<b>bad</b>",
            "film_age_years": 1,
            "storage": "<x>",
            "created_at": 0,
            "total_seconds": 1.2,
            "defect_count": 0,
            "label_counts": {},
            "diagnosis_text": "<script>alert(1)</script>",
        }
    )
    assert "<b>bad</b>" not in detail
    assert "&lt;b&gt;bad&lt;/b&gt;" in detail
    assert "<script>" not in detail


def test_run_pipeline_with_stubbed_diagnosis(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HALIDE_DB_PATH", str(tmp_path / "halide.db"))
    get_cache().clear()

    def fake_run_diagnosis(**kwargs):
        return {
            "film_metadata": {
                "film_type": kwargs["film_type"],
                "film_age_years": kwargs["film_age_years"],
                "storage": kwargs["storage"],
                "scan_resolution_dpi": kwargs["scan_resolution_dpi"],
                "metadata_confidence": kwargs["metadata_confidence"],
            },
            "defects": {
                "defects": [{"label": "scratch", "bbox": [0.1, 0.1, 0.7, 0.2]}],
                "defect_count": 1,
                "label_counts": {"scratch": 1},
                "dropped_count": 0,
                "inference_seconds": 0.01,
                "model_path": "vision-stub",
            },
            "diagnosis": {
                "diagnosis_text": "## Root cause\nTransport scratch.",
                "reasoning_seconds": 0.01,
                "model_path": "reasoning-stub",
            },
            "total_seconds": 0.02,
        }

    monkeypatch.setattr(ui_app, "run_diagnosis", fake_run_diagnosis)
    image = Image.new("RGB", (64, 64), "black")
    (
        lighttable_empty,
        compare_update,
        review_gallery,
        run_state,
        stats,
        notice,
        pills,
        diagnosis,
        metadata,
        raw_json,
        defect_rows,
        history_detail,
        selector_update,
        history_rows,
    ) = ui_app.run_pipeline(
        image,
        "Ilford HP5 Plus (35mm)",
        2,
        "room temp, sealed",
        4000,
        "Low, rough guess",
    )
    assert lighttable_empty["visible"] is False
    assert compare_update["visible"] is True
    image_pair = compare_update["value"]
    original, annotated = image_pair
    assert original.size == (64, 64)
    assert annotated.size == (64, 64)
    assert len(review_gallery["value"]) == 2
    assert review_gallery["visible"] is True
    assert "validated defect" in run_state
    assert pills["visible"] is True
    assert "Scratch" in pills["value"]
    assert "<strong>1</strong>" in pills["value"]
    assert diagnosis["visible"] is True
    assert "Transport scratch" in diagnosis["value"]
    assert "Ilford HP5" in metadata
    assert '"model_path": "vision-stub"' in raw_json
    assert '"metadata_confidence": "low"' in raw_json
    assert defect_rows == [["1", "Scratch", "", "0.100, 0.100, 0.700, 0.200"]]
    assert "Transport scratch" in history_detail
    assert selector_update["value"]
    assert history_rows
    assert "Total defects" in stats
    assert "visible defects" in notice


def test_empty_run_outputs_keep_lighttable_placeholder() -> None:
    (
        lighttable_empty,
        compare_update,
        review_gallery,
        run_state,
        stats,
        notice,
        pills,
        diagnosis,
        metadata,
        raw_json,
        defect_rows,
        history_detail,
        selector_update,
        history_rows,
    ) = ui_app.run_pipeline(
        None,
        "Unknown / Not specified",
        0,
        "unknown",
        4000,
        "low",
    )

    assert lighttable_empty["visible"] is True
    assert compare_update["visible"] is False
    assert review_gallery["visible"] is False
    assert "Load a scan" in run_state
    assert "No image provided" in stats
    assert "No image provided" in notice
    assert pills["visible"] is False
    assert diagnosis["visible"] is False
    assert metadata == ""
    assert raw_json == "{}"
    assert defect_rows == []
    assert "Select a diagnosis" in history_detail or selector_update["value"]
    assert isinstance(history_rows, list)


def test_metadata_confidence_normalization() -> None:
    assert ui_app.normalize_metadata_confidence("High, verified") == "high"
    assert ui_app.normalize_metadata_confidence("Medium, partly verified") == "medium"
    assert ui_app.normalize_metadata_confidence(None) == "low"
