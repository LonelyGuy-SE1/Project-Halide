from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from storage.cache import get_cache
from storage.database import get_diagnosis
from ui import app as ui_app
from ui.components import (
    comparison_viewer_html,
    history_detail_html,
    diagnosis_html,
    raw_json_text,
)


def test_html_renderers_escape_model_and_user_text() -> None:
    assert "<script>" not in diagnosis_html("<script>alert(1)</script>")
    rendered = diagnosis_html(
        "## Root cause\nTransport scratch.\n\n"
        "## Physical fixes\n### Lab bench\n1. **Re-scan** a crop.\n2. Inspect `under a loupe`."
    )
    assert "## Root cause" not in rendered
    assert "halide-report-section" in rendered
    assert "halide-report-subheading" in rendered
    assert "<li><strong>Re-scan</strong> a crop.</li>" in rendered
    assert "<code>under a loupe</code>" in rendered
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


def test_comparison_viewer_html_uses_stable_split_canvas() -> None:
    image = Image.new("RGB", (32, 24), "black")
    html = comparison_viewer_html(image, image)

    assert 'style="--halide-split: 50%;"' in html
    assert 'value="50"' in html
    assert "this.closest('.halide-compare-viewer')" in html
    assert "halide-compare-overlay" in html
    assert html.count("data:image/jpeg;base64") == 2


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
        review_links,
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
    assert review_links["visible"] is True
    assert "Open overlay" in review_links["value"]
    assert "data:image/jpeg;base64" in review_links["value"]
    assert "halide-compare-viewer" in compare_update["value"]
    assert "halide-compare-range" in compare_update["value"]
    assert "data:image/jpeg;base64" in compare_update["value"]
    assert len(review_gallery["value"]) == 2
    original = review_gallery["value"][0][0]
    annotated = review_gallery["value"][1][0]
    assert original.size == (64, 64)
    assert annotated.size == (64, 64)
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
    assert "image data URI omitted" in raw_json
    assert defect_rows == [["1", "Scratch", "not emitted", "0.100, 0.100, 0.700, 0.200"]]
    assert "Transport scratch" in history_detail
    assert "halide-history-preview" in history_detail
    assert selector_update["value"]
    assert history_rows
    assert "Total defects" in stats
    assert "MiniCPM did not emit numeric" in notice


def test_empty_run_outputs_keep_lighttable_placeholder() -> None:
    (
        lighttable_empty,
        compare_update,
        review_gallery,
        review_links,
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
    assert review_links["visible"] is False
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


def test_gpu_duration_cap(monkeypatch) -> None:
    monkeypatch.setenv("HALIDE_GPU_DURATION_SECONDS", "450")
    assert ui_app._gpu_duration_seconds() == ui_app.MAX_ZEROGPU_DURATION_SECONDS


def test_pipeline_error_html_uses_friendly_gpu_message() -> None:
    rendered = ui_app.pipeline_error_html(RuntimeError("no CUDA GPU is visible"))

    assert "GPU unavailable" in rendered
    assert "RuntimeError" not in rendered
    assert "no CUDA GPU" not in rendered


def test_raw_json_text_omits_preview_data_uri() -> None:
    rendered = raw_json_text(
        {
            "preview": {"original": "data:image/jpeg;base64,abc"},
            "defects": {"defect_count": 0},
        }
    )

    assert "data:image/jpeg" not in rendered
    assert "image data URI omitted" in rendered


def test_history_table_selection_opens_saved_diagnosis(monkeypatch, tmp_path) -> None:
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
                "defects": [],
                "defect_count": 0,
                "label_counts": {},
                "dropped_count": 0,
                "inference_seconds": 0.01,
                "model_path": "vision-stub",
            },
            "diagnosis": {
                "diagnosis_text": "## Root cause\nNo visible damage.",
                "reasoning_seconds": 0.01,
                "model_path": "reasoning-stub",
            },
            "total_seconds": 0.02,
        }

    monkeypatch.setattr(ui_app, "run_diagnosis", fake_run_diagnosis)
    image = Image.new("RGB", (64, 64), "black")
    outputs = ui_app.run_pipeline(
        image,
        "Unknown / Not specified",
        0,
        "unknown",
        4000,
        "Low, rough guess",
    )
    history_rows = outputs[-1]

    selector_update, detail, raw = ui_app.open_history_from_table(
        history_rows,
        SimpleNamespace(index=(0, 0)),
    )

    diagnosis_id = selector_update["value"]
    assert selector_update["value"] == diagnosis_id
    assert "No visible damage" in detail
    assert get_diagnosis(diagnosis_id) is not None
    assert '"defect_count": 0' in raw
