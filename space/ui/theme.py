"""Autumn theme. Colors derived from the project logo (orange-to-red on black)."""

from __future__ import annotations

import gradio as gr

AMBER = "#d97706"
AMBER_DEEP = "#b45309"
ORANGE = "#ea580c"
RED = "#dc2626"
CRIMSON = "#991b1b"
EMBER = "#f59e0b"

INK = "#0c0a09"
INK_SOFT = "#1c1917"
PARCHMENT = "#fef3c7"
PARCHMENT_DEEP = "#fde68a"
SLATE = "#44403c"

THEME_CSS = f"""
:root {{
  --halide-amber: {AMBER};
  --halide-amber-deep: {AMBER_DEEP};
  --halide-orange: {ORANGE};
  --halide-red: {RED};
  --halide-crimson: {CRIMSON};
  --halide-ember: {EMBER};
  --halide-ink: {INK};
  --halide-ink-soft: {INK_SOFT};
  --halide-parchment: {PARCHMENT};
  --halide-parchment-deep: {PARCHMENT_DEEP};
  --halide-slate: {SLATE};
}}

body, .gradio-container {{
  background: linear-gradient(180deg, #0c0a09 0%, #1c1917 50%, #0c0a09 100%);
  color: var(--halide-parchment);
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
}}

#halide-header {{
  background: linear-gradient(90deg, var(--halide-crimson) 0%, var(--halide-orange) 50%, var(--halide-amber) 100%);
  padding: 1.4rem 2rem;
  border-radius: 0 0 18px 18px;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 32px rgba(217, 119, 6, 0.25);
  border-bottom: 1px solid var(--halide-amber);
}}

#halide-header h1 {{
  color: var(--halide-parchment);
  font-size: 2.4rem;
  margin: 0;
  letter-spacing: 0.02em;
  text-shadow: 0 2px 4px rgba(0,0,0,0.4);
}}

#halide-header p {{
  color: var(--halide-parchment-deep);
  margin: 0.4rem 0 0 0;
  font-size: 1.05rem;
  font-style: italic;
}}

.halide-card {{
  background: rgba(28, 25, 23, 0.85);
  border: 1px solid var(--halide-amber-deep);
  border-radius: 12px;
  padding: 1.2rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}}

.halide-section-title {{
  color: var(--halide-amber);
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
  border-bottom: 1px solid var(--halide-amber-deep);
  padding-bottom: 0.3rem;
}}

.halide-stat {{
  display: flex;
  justify-content: space-between;
  padding: 0.4rem 0;
  border-bottom: 1px dotted var(--halide-slate);
  color: var(--halide-parchment);
}}

.halide-stat-label {{
  color: var(--halide-amber);
  font-weight: 600;
}}

.halide-diagnosis {{
  background: rgba(217, 119, 6, 0.08);
  border-left: 4px solid var(--halide-amber);
  padding: 1rem 1.2rem;
  border-radius: 6px;
  white-space: pre-wrap;
  font-size: 0.98rem;
  line-height: 1.6;
  color: var(--halide-parchment);
}}

.halide-defect-pill {{
  display: inline-block;
  background: var(--halide-amber);
  color: var(--halide-ink);
  padding: 0.2rem 0.7rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  margin: 0 0.3rem 0.3rem 0;
}}

.halide-defect-pill.dust {{ background: var(--halide-amber); color: var(--halide-ink); }}
.halide-defect-pill.dirt {{ background: var(--halide-orange); color: var(--halide-parchment); }}
.halide-defect-pill.scratch {{ background: var(--halide-red); color: var(--halide-parchment); }}
.halide-defect-pill.long_hair {{ background: var(--halide-crimson); color: var(--halide-parchment); }}
.halide-defect-pill.short_hair {{ background: var(--halide-ember); color: var(--halide-ink); }}

button.primary, .primary button {{
  background: linear-gradient(135deg, var(--halide-orange), var(--halide-red)) !important;
  color: var(--halide-parchment) !important;
  border: 1px solid var(--halide-amber) !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 2px 12px rgba(234, 88, 12, 0.4) !important;
}}

button.primary:hover, .primary button:hover {{
  background: linear-gradient(135deg, var(--halide-red), var(--halide-crimson)) !important;
}}

input, textarea, select {{
  background: var(--halide-ink-soft) !important;
  color: var(--halide-parchment) !important;
  border: 1px solid var(--halide-amber-deep) !important;
}}

input:focus, textarea:focus, select:focus {{
  border-color: var(--halide-amber) !important;
  box-shadow: 0 0 0 2px rgba(217, 119, 6, 0.3) !important;
}}

label, .label, .gradio-radio label, .gradio-checkbox label {{
  color: var(--halide-parchment-deep) !important;
  font-weight: 500 !important;
}}

footer {{
  color: var(--halide-slate) !important;
  text-align: center;
  padding: 1rem;
  font-size: 0.85rem;
}}
"""


def build_theme() -> gr.Theme:
    """Build the autumn-themed Gradio theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#fef3c7",
            c100="#fde68a",
            c200="#fcd34d",
            c300="#fbbf24",
            c400="#f59e0b",
            c500=AMBER,
            c600=AMBER_DEEP,
            c700="#92400e",
            c800="#78350f",
            c900=CRIMSON,
            c950="#7c2d12",
        ),
        secondary_hue=gr.themes.Color(
            c50="#fef3c7",
            c100="#fde68a",
            c200="#fcd34d",
            c300="#fbbf24",
            c400=EMBER,
            c500=AMBER,
            c600=ORANGE,
            c700=RED,
            c800=CRIMSON,
            c900="#7c2d12",
            c950="#431407",
        ),
        neutral_hue=gr.themes.Color(
            c50="#fafaf9",
            c100="#f5f5f4",
            c200="#e7e5e4",
            c300="#d6d3d1",
            c400=SLATE,
            c500="#57534e",
            c600="#44403c",
            c700="#292524",
            c800=INK_SOFT,
            c900=INK,
            c950="#0c0a09",
        ),
        font=gr.themes.GoogleFont("Iowan Old Style"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill=INK,
        body_background_fill_dark=INK,
        body_text_color=PARCHMENT,
        body_text_color_dark=PARCHMENT,
        button_primary_background_fill=ORANGE,
        button_primary_background_fill_dark=ORANGE,
        button_primary_text_color=PARCHMENT,
        button_primary_text_color_dark=PARCHMENT,
        block_background_fill=INK_SOFT,
        block_background_fill_dark=INK_SOFT,
        block_border_color=AMBER_DEEP,
        block_border_color_dark=AMBER_DEEP,
        input_background_fill=INK,
        input_background_fill_dark=INK,
        input_border_color=AMBER_DEEP,
        input_border_color_dark=AMBER_DEEP,
    )
