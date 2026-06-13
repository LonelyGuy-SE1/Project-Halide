"""Gradio app. Main UI definition and layout."""

from __future__ import annotations

import html
import logging
from typing import Any

import gradio as gr

from config import get_app_config
from data.preprocessing import draw_defects, image_to_png_bytes, load_image
from pipeline.pipeline import run_diagnosis
from storage.cache import get_cache
from storage.database import get_diagnosis, init_db, list_recent, record_diagnosis
from ui.components import (
    EMPTY_STATE,
    HEADER_HTML,
    LIGHTTABLE_EMPTY_STATE,
    LIGHTTABLE_RUNNING_STATE,
    REPORT_EMPTY_STATE,
    confidence_notice_html,
    defect_table_rows,
    defect_pills_html,
    diagnosis_html,
    history_choices,
    history_detail_html,
    history_table_rows,
    metadata_html,
    raw_json_text,
    run_state_html,
    stats_html,
)
from ui.theme import THEME_CSS, build_theme

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _gpu_decorator():
    try:
        import spaces
    except ImportError:
        return lambda fn: fn
    return spaces.GPU(duration=get_app_config().gpu_duration_seconds)


DEFAULT_FILM_TYPES = [
    "Unknown / Not specified",
    "Kodak Portra 400 (35mm)",
    "Kodak Tri-X 400 (35mm)",
    "Kodak Ektar 100 (35mm)",
    "Ilford HP5 Plus (35mm)",
    "Ilford Delta 100 (35mm)",
    "Ilford FP4 Plus (120)",
    "CineStill 800T (35mm)",
    "Fujifilm Pro 400H (35mm)",
    "Fomapan 400 (35mm)",
    "Other / Unknown",
]

STORAGE_OPTIONS = [
    "unknown",
    "fridge, sealed",
    "freezer, sealed",
    "room temp, sealed",
    "room temp, loose",
    "shoe box, attic",
    "shoe box, basement",
]

RESOLUTION_OPTIONS = [2000, 3000, 4000, 5000, 6000, 8000]

METADATA_CONFIDENCE_OPTIONS = [
    "Low, rough guess",
    "Medium, partly verified",
    "High, verified from notes or edge marks",
]


def normalize_metadata_confidence(value: str | None) -> str:
    text = (value or "low").strip().lower()
    if text.startswith("high"):
        return "high"
    if text.startswith("medium"):
        return "medium"
    return "low"


def _history_state(
    selected_id: str | None = None,
) -> tuple[dict | None, Any, list[list[str]]]:
    entries = list_recent(limit=get_app_config().max_history_items)
    choices = history_choices(entries)
    ids = [value for _label, value in choices]
    value = selected_id if selected_id in ids else (ids[0] if ids else None)
    selected = next((entry for entry in entries if entry.get("id") == value), None)
    return selected, gr.update(choices=choices, value=value), history_table_rows(entries)


def _empty_outputs(
    message: str = "Awaiting scan.",
) -> tuple[Any, Any, Any, str, str, str, str, str, str, str, list[list[str]], str, Any, list[list[str]]]:
    selected_entry, selector_update, history_rows = _history_state()
    empty = f'<p class="halide-muted">{html.escape(message)}</p>'
    hidden_html = gr.update(value="", visible=False)
    return (
        gr.update(value=LIGHTTABLE_EMPTY_STATE, visible=True),
        gr.update(value=None, visible=False),
        gr.update(value=[], visible=False),
        run_state_html(None),
        empty,
        empty,
        hidden_html,
        hidden_html,
        "",
        "{}",
        [],
        history_detail_html(selected_entry),
        selector_update,
        history_rows,
    )


def _review_gallery(pil_image: Any, annotated: Any) -> list[tuple[Any, str]]:
    return [
        (pil_image, "Original scan"),
        (annotated, "Validated overlay"),
    ]


def _running_button_state() -> tuple[Any, str, Any]:
    return (
        gr.update(interactive=False, value="Diagnosing..."),
        (
            '<div class="halide-run-state active">'
            '<span class="halide-run-eyebrow">Running</span>'
            "<strong>GPU inspection in progress</strong>"
            "<span>Vision extraction, validation, and report generation are running.</span>"
            "</div>"
        ),
        gr.update(value=LIGHTTABLE_RUNNING_STATE, visible=True),
    )


def pipeline_error_html(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if "no cuda gpu" in lower or "cuda" in lower or "gpu" in lower:
        title = "GPU unavailable"
        body = (
            "The diagnosis needs a live GPU slot. Please retry in a moment, "
            "or run the app on a GPU-backed Space."
        )
    else:
        title = "Pipeline error"
        body = text or "The diagnostic pipeline stopped unexpectedly."
    return (
        '<div class="halide-panel" style="border-color: var(--halide-red);">'
        f'<div class="halide-section-title" style="color: var(--halide-red);">'
        f"{html.escape(title)}</div>"
        f"<p class=\"halide-muted\">{html.escape(body)}</p></div>"
    )


@_gpu_decorator()
def run_pipeline(
    image: Any,
    film_type: str,
    film_age_years: int,
    storage: str,
    scan_dpi: int,
    metadata_confidence: str = "low",
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, Any, Any, str, str, str, str, str, str, str, list[list[str]], str, Any, list[list[str]]]:
    """Gradio handler for the diagnose button."""
    if image is None:
        return _empty_outputs("No image provided.")

    try:
        progress(0.0, "Hashing image for cache lookup...")
        pil_image = load_image(image)
        cache = get_cache()
        image_bytes = image_to_png_bytes(pil_image)
        metadata = {
            "film_type": film_type or "Unknown / Not specified",
            "film_age_years": int(film_age_years or 0),
            "storage": storage or "unknown",
            "scan_resolution_dpi": int(scan_dpi or 4000),
            "metadata_confidence": normalize_metadata_confidence(metadata_confidence),
        }
        cached = cache.get(image_bytes, metadata=metadata)
        if cached is not None:
            logger.info("Returning cached diagnosis")
            result = cached
        else:
            progress(0.05, "Loading GPU models if needed...")
            progress(0.1, "Stage 1/2: running vision defect extraction...")
            result = run_diagnosis(
                image=pil_image,
                film_type=metadata["film_type"],
                film_age_years=metadata["film_age_years"],
                storage=metadata["storage"],
                scan_resolution_dpi=metadata["scan_resolution_dpi"],
                metadata_confidence=metadata["metadata_confidence"],
            )
            progress(0.85, "Stage 2/2: persisting diagnosis...")
            try:
                diagnosis_id = record_diagnosis(result)
                result["diagnosis_id"] = diagnosis_id
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to record diagnosis: %s", exc)
            cache.put(image_bytes, result, metadata=metadata)

        progress(1.0, "Done.")

        counts = result.get("defects", {}).get("label_counts", {}) or {}
        defects = result.get("defects", {}).get("defects", []) or []
        annotated = draw_defects(
            pil_image,
            defects,
            title=f"Halide: {len(defects)} validated defects",
        )
        image_pair = (pil_image, annotated)
        compare = gr.update(value=image_pair, visible=True)
        gallery = gr.update(value=_review_gallery(pil_image, annotated), visible=True)
        run_state = run_state_html(result)
        stats = stats_html(result)
        notice = confidence_notice_html(result)
        pills = gr.update(value=defect_pills_html(counts), visible=True)
        diag = gr.update(
            value=diagnosis_html(result.get("diagnosis", {}).get("diagnosis_text", "")),
            visible=True,
        )
        meta = metadata_html(result)
        raw_json = raw_json_text(result)
        table_rows = defect_table_rows(result)
        selected_entry, selector_update, history_rows = _history_state(result.get("diagnosis_id"))
        return (
            gr.update(value="", visible=False),
            compare,
            gallery,
            run_state,
            stats,
            notice,
            pills,
            diag,
            meta,
            raw_json,
            table_rows,
            history_detail_html(selected_entry),
            selector_update,
            history_rows,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Pipeline failed")
        err = pipeline_error_html(exc)
        selected_entry, selector_update, history_rows = _history_state()
        hidden_html = gr.update(value="", visible=False)
        return (
            gr.update(value=LIGHTTABLE_EMPTY_STATE, visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            err,
            err,
            "",
            hidden_html,
            hidden_html,
            "",
            "{}",
            [],
            history_detail_html(selected_entry),
            selector_update,
            history_rows,
        )


def refresh_history(selected_id: str | None = None) -> tuple[Any, str, str, list[list[str]]]:
    selected_entry, selector_update, history_rows = _history_state(selected_id)
    return (
        selector_update,
        history_detail_html(selected_entry),
        raw_json_text(selected_entry),
        history_rows,
    )


def open_history(diagnosis_id: str | None) -> tuple[str, str]:
    entry = get_diagnosis(diagnosis_id or "") if diagnosis_id else None
    return history_detail_html(entry), raw_json_text(entry)


def build_app() -> gr.Blocks:
    init_db()
    selected_entry, selector_update, initial_history_rows = _history_state()
    initial_choices = selector_update["choices"] if isinstance(selector_update, dict) else []
    initial_value = selector_update["value"] if isinstance(selector_update, dict) else None

    with gr.Blocks(
        title="Project Halide",
        fill_width=True,
        fill_height=True,
        elem_id="halide-app",
    ) as app:
        gr.HTML(HEADER_HTML)

        with gr.Row(elem_classes="halide-workbench", equal_height=False):
            with gr.Column(scale=2, min_width=300, elem_classes="halide-intake-panel"):
                gr.HTML(
                    '<div class="halide-panel-title">Scan intake</div>'
                    '<p class="halide-rail-copy">Metadata is context, visible evidence is primary.</p>'
                )
                image_input = gr.Image(
                    label="Film scan",
                    type="pil",
                    height=300,
                    sources=["upload", "clipboard"],
                    buttons=["download", "fullscreen"],
                    elem_classes="halide-upload",
                )
                film_type = gr.Dropdown(
                    choices=DEFAULT_FILM_TYPES,
                    value=DEFAULT_FILM_TYPES[0],
                    label="Film stock",
                    allow_custom_value=True,
                )
                film_age = gr.Slider(
                    minimum=0,
                    maximum=80,
                    step=1,
                    value=0,
                    label="Age (years)",
                    buttons=["reset"],
                )
                scan_dpi = gr.Dropdown(
                    choices=RESOLUTION_OPTIONS,
                    value=4000,
                    label="DPI",
                    allow_custom_value=True,
                )
                storage = gr.Radio(
                    choices=STORAGE_OPTIONS,
                    value=STORAGE_OPTIONS[0],
                    label="Storage",
                )
                metadata_confidence = gr.Dropdown(
                    choices=METADATA_CONFIDENCE_OPTIONS,
                    value=METADATA_CONFIDENCE_OPTIONS[0],
                    label="Metadata confidence",
                    interactive=True,
                )
                run_btn = gr.Button(
                    "Diagnose scan",
                    variant="primary",
                    size="lg",
                    elem_id="halide-run-button",
                )
                gr.HTML(
                    '<div class="halide-model-card">'
                    '<span>Runtime</span>'
                    '<strong>Open weights, GPU only</strong>'
                    '<p>MiniCPM-V extracts evidence. Nemotron writes the lab report.</p>'
                    "</div>"
                )

            with gr.Column(scale=6, min_width=560, elem_classes="halide-main-stage"):
                run_state_output = gr.HTML(value=run_state_html(None))

                with gr.Group(elem_classes="halide-lighttable"):
                    gr.HTML(
                        '<div class="halide-section-header">'
                        '<div><span>Light table</span><strong>Original versus validated overlay</strong></div>'
                        '<small>Review</small>'
                        "</div>"
                    )
                    lighttable_empty = gr.HTML(value=LIGHTTABLE_EMPTY_STATE)
                    compare_output = gr.ImageSlider(
                        value=None,
                        label="Original / overlay",
                        type="pil",
                        height="54vh",
                        max_height=760,
                        slider_position=52,
                        interactive=False,
                        buttons=["download", "fullscreen"],
                        elem_id="halide-compare",
                        visible=False,
                    )
                    review_gallery = gr.Gallery(
                        value=[],
                        label="Review frames",
                        columns=2,
                        rows=1,
                        height=220,
                        allow_preview=True,
                        object_fit="contain",
                        buttons=["download", "fullscreen"],
                        elem_classes="halide-review-gallery",
                        visible=False,
                    )

            with gr.Column(scale=3, min_width=390, elem_classes="halide-inspector"):
                with gr.Tabs(selected="report", elem_classes="halide-inspector-tabs"):
                    with gr.Tab("Report", id="report"):
                        notice_output = gr.HTML(value=REPORT_EMPTY_STATE)
                        defect_summary = gr.HTML(value="", visible=False)
                        diagnosis_output = gr.HTML(value="", visible=False)
                    with gr.Tab("Evidence", id="evidence"):
                        stats_output = gr.HTML(value=EMPTY_STATE)
                        metadata_output = gr.HTML(value=EMPTY_STATE)
                        defect_table = gr.Dataframe(
                            value=[],
                            headers=["#", "Label", "Confidence", "Box"],
                            datatype=["str", "str", "str", "str"],
                            type="array",
                            label="Validated boxes",
                            interactive=False,
                            wrap=True,
                            max_height=300,
                        )
                    with gr.Tab("History", id="history"):
                        history_select = gr.Dropdown(
                            choices=initial_choices,
                            value=initial_value,
                            label="Saved diagnosis",
                            interactive=True,
                        )
                        refresh_btn = gr.Button("Refresh", size="sm")
                        history_table = gr.Dataframe(
                            value=initial_history_rows,
                            headers=["Saved", "Film stock", "Defects", "Labels", "ID"],
                            datatype=["str", "str", "str", "str", "str"],
                            type="array",
                            label="Recent diagnoses",
                            interactive=False,
                            wrap=True,
                            max_height=260,
                        )
                        history_detail = gr.HTML(
                            value=history_detail_html(selected_entry)
                        )
                    with gr.Tab("JSON", id="json"):
                        raw_output = gr.Code(
                            value="{}",
                            language="json",
                            label="Pipeline JSON",
                            lines=18,
                            max_lines=30,
                            wrap_lines=True,
                            buttons=["copy", "download"],
                        )

        gr.HTML(
            "<footer>Project Halide, open-weight film diagnostics. "
            "Vision: MiniCPM-V 4.6. Reasoning: Nemotron-Mini-4B-Instruct.</footer>"
        )

        run_event = run_btn.click(
            fn=_running_button_state,
            outputs=[run_btn, run_state_output, lighttable_empty],
            queue=False,
        )
        run_event.then(
            fn=run_pipeline,
            inputs=[
                image_input,
                film_type,
                film_age,
                storage,
                scan_dpi,
                metadata_confidence,
            ],
            outputs=[
                lighttable_empty,
                compare_output,
                review_gallery,
                run_state_output,
                stats_output,
                notice_output,
                defect_summary,
                diagnosis_output,
                metadata_output,
                raw_output,
                defect_table,
                history_detail,
                history_select,
                history_table,
            ],
        ).then(
            fn=lambda: gr.update(interactive=True, value="Diagnose scan"),
            outputs=[run_btn],
            queue=False,
        )
        refresh_btn.click(
            fn=refresh_history,
            inputs=[history_select],
            outputs=[history_select, history_detail, raw_output, history_table],
        )
        history_select.change(
            fn=open_history,
            inputs=[history_select],
            outputs=[history_detail, raw_output],
        )

    return app


def main() -> None:
    app = build_app()
    app.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=build_theme(),
        css=THEME_CSS,
        allowed_paths=["assets"],
        favicon_path="assets/logo.jpg",
        max_file_size="60mb",
    )


if __name__ == "__main__":
    main()
