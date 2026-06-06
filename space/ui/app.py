"""Gradio app. Main UI definition and layout."""

from __future__ import annotations

import html
import io
import logging
from typing import Any

import gradio as gr

from pipeline.pipeline import run_diagnosis
from storage.cache import get_cache
from storage.database import init_db, list_recent, record_diagnosis
from ui.components import (
    HEADER_HTML,
    THEME_CSS,
    defect_pills_html,
    diagnosis_html,
    render_history,
    stats_html,
)
from ui.theme import build_theme

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_FILM_TYPES = [
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
    "fridge, sealed",
    "freezer, sealed",
    "room temp, sealed",
    "room temp, loose",
    "shoe box, attic",
    "shoe box, basement",
    "unknown",
]

RESOLUTION_OPTIONS = [2000, 3000, 4000, 5000, 6000, 8000]


def _image_to_bytes(pil_image: Any) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def run_pipeline(
    image: Any,
    film_type: str,
    film_age_years: int,
    storage: str,
    scan_dpi: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str]:
    """Gradio handler for the diagnose button."""
    if image is None:
        empty = '<p style="color: var(--halide-crimson);">No image provided.</p>'
        return empty, empty, empty, render_history(list_recent(limit=10))

    try:
        progress(0.0, "Hashing image for cache lookup...")
        cache = get_cache()
        image_bytes = _image_to_bytes(image)
        cached = cache.get(image_bytes)
        if cached is not None:
            logger.info("Returning cached diagnosis")
            result = cached
        else:
            progress(0.1, "Stage 1/2: running vision defect extraction...")
            result = run_diagnosis(
                image=image,
                film_type=film_type or "Unknown 35mm",
                film_age_years=int(film_age_years or 0),
                storage=storage or "unknown",
                scan_resolution_dpi=int(scan_dpi or 4000),
            )
            progress(0.85, "Stage 2/2: persisting diagnosis...")
            try:
                record_diagnosis(result)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to record diagnosis: %s", exc)
            cache.put(image_bytes, result)

        progress(1.0, "Done.")

        counts = result.get("defects", {}).get("label_counts", {}) or {}
        stats = stats_html(result)
        pills = defect_pills_html(counts)
        diag = diagnosis_html(result.get("diagnosis", {}).get("diagnosis_text", ""))
        history = render_history(list_recent(limit=10))
        return stats, pills, diag, history
    except Exception as exc:  # pragma: no cover
        logger.exception("Pipeline failed")
        err = (
            '<div class="halide-card" style="border-color: var(--halide-crimson);">'
            f'<div class="halide-section-title" style="color: var(--halide-red);">'
            f"Pipeline error</div>"
            f"<pre style=\"color: var(--halide-parchment); white-space: pre-wrap;\">"
            f"{html.escape(str(exc))}</pre></div>"
        )
        return err, "", "", render_history(list_recent(limit=10))


def refresh_history() -> str:
    return render_history(list_recent(limit=10))


def build_app() -> gr.Blocks:
    init_db()
    theme = build_theme()

    with gr.Blocks(title="Project Halide") as app:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="halide-card"):
                    gr.Markdown('<div class="halide-section-title">Scan upload</div>')
                    image_input = gr.Image(
                        label="Film scan",
                        type="pil",
                        height=380,
                        sources=["upload", "clipboard"],
                    )

                with gr.Group(elem_classes="halide-card"):
                    gr.Markdown('<div class="halide-section-title">Film metadata</div>')
                    film_type = gr.Dropdown(
                        choices=DEFAULT_FILM_TYPES,
                        value=DEFAULT_FILM_TYPES[0],
                        label="Film stock",
                    )
                    with gr.Row():
                        film_age = gr.Slider(
                            minimum=0,
                            maximum=80,
                            step=1,
                            value=2,
                            label="Age (years)",
                        )
                        scan_dpi = gr.Dropdown(
                            choices=RESOLUTION_OPTIONS,
                            value=4000,
                            label="Scan resolution (dpi)",
                        )
                    storage = gr.Radio(
                        choices=STORAGE_OPTIONS,
                        value=STORAGE_OPTIONS[0],
                        label="Storage condition",
                    )

                run_btn = gr.Button("Diagnose scan", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Group(elem_classes="halide-card"):
                    gr.Markdown('<div class="halide-section-title">Defect summary</div>')
                    defect_summary = gr.HTML(
                        value='<p style="color: var(--halide-slate);">Awaiting scan.</p>'
                    )

                with gr.Group(elem_classes="halide-card"):
                    gr.Markdown('<div class="halide-section-title">Diagnosis & fixes</div>')
                    diagnosis_output = gr.HTML(
                        value='<p style="color: var(--halide-slate);">Awaiting scan.</p>'
                    )

                with gr.Group(elem_classes="halide-card"):
                    gr.Markdown('<div class="halide-section-title">Session stats</div>')
                    stats_output = gr.HTML(
                        value='<p style="color: var(--halide-slate);">Awaiting scan.</p>'
                    )

            with gr.Column(scale=1):
                with gr.Group(elem_classes="halide-card"):
                    gr.Markdown('<div class="halide-section-title">Recent diagnoses</div>')
                    history_output = gr.HTML(value=render_history(list_recent(limit=10)))
                    refresh_btn = gr.Button("Refresh history", size="sm")

        gr.HTML(
            "<footer>Project Halide. Edge-native, no cloud APIs. "
            "Vision: MiniCPM-V 4.6 (1.3B). Reasoning: Nemotron-Mini-4B-Instruct (few-shot).</footer>"
        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[image_input, film_type, film_age, storage, scan_dpi],
            outputs=[stats_output, defect_summary, diagnosis_output, history_output],
        )
        refresh_btn.click(fn=refresh_history, outputs=[history_output])

    return app


def main() -> None:
    app = build_app()
    app.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=build_theme(),
        css=THEME_CSS,
    )


if __name__ == "__main__":
    main()
