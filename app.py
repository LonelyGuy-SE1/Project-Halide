"""Main entry point. Launches the Gradio app."""

from __future__ import annotations

import logging

from ui.app import build_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    from ui.theme import THEME_CSS, build_theme
    app = build_app()
    app.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=build_theme(),
        css=THEME_CSS,
    )


if __name__ == "__main__":
    main()
