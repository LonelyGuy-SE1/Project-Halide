"""Gradio Server wrapper for Project Halide."""

from __future__ import annotations

import gradio as gr

from config import CANONICAL_VISION_MODEL_ID, DEFAULT_REASONING_MODEL_ID
from ui.app import build_app
from ui.theme import THEME_CSS, build_theme


def build_server(blocks: gr.Blocks | None = None) -> gr.Server:
    """Build a gr.Server with the UI mounted at root and health metadata."""
    server = gr.Server(
        title="Project Halide",
        version="0.1.0",
        description="Edge-native analog film diagnostic engine.",
    )

    @server.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "vision_model": CANONICAL_VISION_MODEL_ID,
            "reasoning_model": DEFAULT_REASONING_MODEL_ID,
        }

    gr.mount_gradio_app(
        server,
        blocks or build_app(),
        path="/",
        theme=build_theme(),
        css=THEME_CSS,
        show_error=True,
        allowed_paths=["assets"],
        favicon_path="assets/logo.jpg",
        max_file_size="60mb",
    )
    return server


__all__ = ["build_server"]
