"""Custom Gradio theme for the Project Halide workbench."""

from __future__ import annotations

import gradio as gr

OXIDE = "#c96f36"
OXIDE_DEEP = "#8f3f27"
AMBER = "#f6c85f"
GREEN = "#6ee7b7"
VIOLET = "#a78bfa"
RED = "#ef4444"

INK = "#0b0a08"
SURFACE = "#151311"
SURFACE_SOFT = "#1d1916"
SURFACE_LIFT = "#27211c"
PAPER = "#f6efe2"
MUTED = "#b9afa2"
BORDER = "#3b332b"
BLACK = "#050403"

THEME_CSS = f"""
:root {{
  --halide-oxide: {OXIDE};
  --halide-oxide-deep: {OXIDE_DEEP};
  --halide-amber: {AMBER};
  --halide-green: {GREEN};
  --halide-violet: {VIOLET};
  --halide-red: {RED};
  --halide-ink: {INK};
  --halide-surface: {SURFACE};
  --halide-surface-soft: {SURFACE_SOFT};
  --halide-surface-lift: {SURFACE_LIFT};
  --halide-paper: {PAPER};
  --halide-muted: {MUTED};
  --halide-border: {BORDER};
  --halide-black: {BLACK};
}}

body,
.gradio-container {{
  background:
    linear-gradient(180deg, #17110d 0%, #0d0a08 48%, #070605 100%) !important;
  color: var(--halide-paper) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif !important;
}}

.gradio-container {{
  max-width: none !important;
  padding: 0 1rem 1rem !important;
}}

#halide-app {{
  min-height: 100vh;
}}

#halide-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.95rem 0.25rem 0.85rem;
  margin: 0 0 0.75rem;
  border-bottom: 1px solid rgba(246, 200, 95, 0.25);
}}

.halide-brand-lockup {{
  min-width: 0;
}}

#halide-header h1 {{
  color: var(--halide-paper);
  font-size: clamp(1.35rem, 2.1vw, 2.15rem);
  line-height: 1.08;
  margin: 0.15rem 0 0;
  letter-spacing: 0;
}}

.halide-kicker {{
  color: var(--halide-green);
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 850;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}}

.halide-model-strip {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.45rem;
  min-width: min(40vw, 34rem);
}}

.halide-model-strip span,
.halide-model-strip a {{
  border: 1px solid rgba(110, 231, 183, 0.34);
  color: #ddfff3;
  background: rgba(110, 231, 183, 0.08);
  border-radius: 8px;
  padding: 0.36rem 0.56rem;
  font-size: 0.76rem;
  font-weight: 760;
  line-height: 1;
  text-decoration: none;
  white-space: nowrap;
}}

.halide-workbench {{
  gap: 0.85rem !important;
  align-items: flex-start;
}}

.halide-main-stage,
.halide-inspector,
.halide-intake-panel {{
  min-width: 0 !important;
}}

.halide-status-band {{
  gap: 0.7rem !important;
  margin-bottom: 0.75rem;
}}

.halide-status-cell {{
  min-width: 0;
}}

.halide-lighttable,
.halide-diagnosis-panel,
.halide-panel {{
  background: rgba(21, 19, 17, 0.96) !important;
  border: 1px solid var(--halide-border) !important;
  border-radius: 8px !important;
  padding: 0.9rem !important;
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.34) !important;
}}

.halide-lighttable {{
  border-color: rgba(246, 200, 95, 0.24) !important;
  padding-bottom: 0.7rem !important;
}}

.halide-diagnosis-panel {{
  margin-top: 0.85rem;
  border-left: 3px solid var(--halide-oxide) !important;
}}

.halide-intake-panel .block,
.halide-inspector .block,
.halide-lighttable .block,
.halide-diagnosis-panel .block {{
  background: rgba(29, 25, 22, 0.88) !important;
  border-color: rgba(59, 51, 43, 0.92) !important;
  border-radius: 8px !important;
}}

.halide-upload img,
.halide-lighttable img {{
  background: #050403 !important;
}}

.halide-lighttable .wrap,
.halide-lighttable .image-container,
.halide-upload .wrap,
.halide-upload .image-container {{
  border-color: rgba(246, 239, 226, 0.34) !important;
}}

.halide-inline-controls {{
  gap: 0.55rem !important;
}}

#halide-run-button,
button.primary,
.primary button {{
  background: linear-gradient(135deg, var(--halide-oxide), var(--halide-red)) !important;
  color: white !important;
  border: 1px solid rgba(255, 255, 255, 0.14) !important;
  border-radius: 8px !important;
  font-weight: 850 !important;
  letter-spacing: 0 !important;
  box-shadow: 0 12px 28px rgba(201, 111, 54, 0.28) !important;
}}

#halide-run-button:hover,
button.primary:hover,
.primary button:hover {{
  filter: brightness(1.06);
}}

button.secondary,
.secondary button,
button {{
  border-radius: 8px !important;
  font-weight: 740 !important;
}}

.halide-section-title {{
  color: var(--halide-paper);
  font-size: 0.78rem;
  font-weight: 850;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  margin: 0 0 0.7rem;
}}

.halide-subsection {{
  color: var(--halide-muted);
  font-size: 0.72rem;
  font-weight: 850;
  letter-spacing: 0.08em;
  margin: 0.95rem 0 0.45rem;
  text-transform: uppercase;
}}

.halide-muted {{
  color: var(--halide-muted);
  margin: 0;
}}

.halide-notice {{
  border-radius: 8px;
  padding: 0.74rem 0.82rem;
  border: 1px solid var(--halide-border);
  color: var(--halide-paper);
  background: rgba(39, 33, 28, 0.86);
  font-size: 0.88rem;
  line-height: 1.38;
  min-height: 3rem;
}}

.halide-notice.good {{
  border-color: rgba(110, 231, 183, 0.38);
  background: rgba(110, 231, 183, 0.08);
}}

.halide-notice.caution {{
  border-color: rgba(246, 200, 95, 0.42);
  background: rgba(246, 200, 95, 0.09);
}}

.halide-notice.neutral {{
  border-color: rgba(167, 139, 250, 0.36);
  background: rgba(167, 139, 250, 0.08);
}}

.halide-stats {{
  display: grid;
  gap: 0.42rem;
}}

.halide-stats.compact {{
  gap: 0.32rem;
}}

.halide-stat {{
  display: grid;
  grid-template-columns: minmax(7.2rem, 0.42fr) minmax(0, 1fr);
  gap: 0.7rem;
  padding: 0.42rem 0;
  border-bottom: 1px solid rgba(59, 51, 43, 0.88);
  color: var(--halide-paper);
  min-width: 0;
}}

.halide-stat:last-child {{
  border-bottom: 0;
}}

.halide-stat-label {{
  color: var(--halide-muted);
  font-weight: 760;
}}

.halide-stat span:last-child {{
  overflow-wrap: anywhere;
  text-align: right;
}}

.halide-diagnosis {{
  background: rgba(39, 33, 28, 0.78);
  border: 1px solid rgba(201, 111, 54, 0.28);
  padding: 1rem 1.05rem;
  border-radius: 8px;
  white-space: pre-wrap;
  font-size: 0.95rem;
  line-height: 1.58;
  color: var(--halide-paper);
}}

.halide-pill-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.42rem;
}}

.halide-defect-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: rgba(201, 111, 54, 0.15);
  color: #ffe6cf;
  border: 1px solid rgba(201, 111, 54, 0.36);
  padding: 0.33rem 0.58rem;
  border-radius: 8px;
  font-size: 0.8rem;
  font-weight: 760;
  white-space: nowrap;
}}

.halide-defect-pill strong {{
  color: var(--halide-paper);
}}

.halide-defect-pill.dust {{
  background: rgba(246, 200, 95, 0.13);
  border-color: rgba(246, 200, 95, 0.35);
}}

.halide-defect-pill.dirt {{
  background: rgba(201, 111, 54, 0.15);
  border-color: rgba(201, 111, 54, 0.36);
}}

.halide-defect-pill.scratch {{
  background: rgba(239, 68, 68, 0.14);
  border-color: rgba(239, 68, 68, 0.4);
}}

.halide-defect-pill.long_hair {{
  background: rgba(167, 139, 250, 0.13);
  border-color: rgba(167, 139, 250, 0.36);
}}

.halide-defect-pill.short_hair {{
  background: rgba(110, 231, 183, 0.11);
  border-color: rgba(110, 231, 183, 0.34);
}}

.halide-defect-pill.emulsion_damage {{
  background: rgba(226, 232, 240, 0.13);
  border-color: rgba(226, 232, 240, 0.34);
}}

.halide-defect-pill.chemical_stain {{
  background: rgba(34, 197, 94, 0.12);
  border-color: rgba(34, 197, 94, 0.34);
}}

.halide-defect-pill.light_leak {{
  background: rgba(244, 114, 182, 0.13);
  border-color: rgba(244, 114, 182, 0.36);
}}

.halide-history-item {{
  background: rgba(29, 25, 22, 0.78);
  border: 1px solid rgba(59, 51, 43, 0.92);
  border-radius: 8px;
  margin-bottom: 0.65rem;
  padding: 0.75rem;
}}

.halide-history-title {{
  color: var(--halide-paper);
  font-weight: 850;
  margin-bottom: 0.25rem;
  overflow-wrap: anywhere;
}}

.halide-history-meta {{
  color: var(--halide-muted);
  font-size: 0.78rem;
  line-height: 1.35;
  margin: 0.3rem 0;
}}

.halide-history-detail {{
  display: grid;
  gap: 0.4rem;
}}

.halide-history-feed {{
  max-height: 18rem;
  overflow: auto;
}}

input,
textarea,
select,
.wrap,
.input-container,
.tokenizer,
.prose {{
  background-color: var(--halide-surface-soft) !important;
  color: var(--halide-paper) !important;
}}

label,
.label,
.gradio-radio label,
.gradio-checkbox label {{
  color: var(--halide-muted) !important;
  font-weight: 760 !important;
}}

.label-wrap,
.label-wrap span {{
  background: rgba(39, 33, 28, 0.96) !important;
  border-color: rgba(246, 200, 95, 0.22) !important;
  color: var(--halide-paper) !important;
  border-radius: 6px !important;
  font-weight: 760 !important;
}}

label[data-testid="block-label"] {{
  background: rgba(39, 33, 28, 0.96) !important;
  border: 1px solid rgba(246, 200, 95, 0.22) !important;
  color: var(--halide-paper) !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}}

label[data-testid="block-label"] span {{
  color: var(--halide-paper) !important;
}}

label[data-testid$="-radio-label"] {{
  background: rgba(39, 33, 28, 0.98) !important;
  border: 1px solid rgba(246, 239, 226, 0.18) !important;
  border-radius: 7px !important;
  box-shadow: none !important;
  opacity: 1 !important;
}}

label[data-testid$="-radio-label"].selected {{
  background: rgba(201, 111, 54, 0.26) !important;
  border-color: rgba(246, 200, 95, 0.45) !important;
}}

label[data-testid$="-radio-label"] span,
label[data-testid$="-radio-label"] input,
.gradio-checkbox label span {{
  color: var(--halide-paper) !important;
  opacity: 1 !important;
}}

label[data-testid$="-radio-label"] input {{
  accent-color: var(--halide-oxide);
}}

.tabs button {{
  color: var(--halide-muted) !important;
  font-weight: 820 !important;
}}

.tabs button.selected {{
  color: var(--halide-paper) !important;
  border-color: var(--halide-oxide) !important;
}}

table,
thead,
tbody,
tr,
td,
th {{
  color: var(--halide-paper) !important;
}}

pre,
code {{
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Consolas, monospace !important;
}}

footer {{
  color: var(--halide-muted) !important;
  text-align: center;
  padding: 0.9rem;
  font-size: 0.82rem;
}}

@media (max-width: 980px) {{
  #halide-header {{
    align-items: flex-start;
    flex-direction: column;
  }}

  .halide-model-strip {{
    justify-content: flex-start;
    min-width: 0;
  }}

  .halide-status-band {{
    flex-direction: column;
  }}

  .halide-workbench {{
    flex-direction: column !important;
  }}

  .halide-intake-panel,
  .halide-main-stage,
  .halide-inspector {{
    width: 100% !important;
  }}

  .halide-stat {{
    grid-template-columns: 1fr;
    gap: 0.18rem;
  }}

  .halide-stat span:last-child {{
    text-align: left;
  }}
}}
"""


def build_theme() -> gr.Theme:
    """Build a compact custom workbench theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#fff7ed",
            c100="#ffead5",
            c200="#fed1a8",
            c300="#f6aa6a",
            c400="#e88b45",
            c500=OXIDE,
            c600=OXIDE_DEEP,
            c700="#74311f",
            c800="#5c281d",
            c900="#3f1b14",
            c950="#24100b",
        ),
        secondary_hue=gr.themes.Color(
            c50="#ecfdf5",
            c100="#d1fae5",
            c200="#a7f3d0",
            c300=GREEN,
            c400="#34d399",
            c500="#10b981",
            c600="#059669",
            c700="#047857",
            c800="#065f46",
            c900="#064e3b",
            c950="#022c22",
        ),
        neutral_hue=gr.themes.Color(
            c50="#faf7f0",
            c100=PAPER,
            c200="#d8cec0",
            c300=MUTED,
            c400="#918477",
            c500="#75695e",
            c600="#5f5248",
            c700=BORDER,
            c800=SURFACE_SOFT,
            c900=SURFACE,
            c950=BLACK,
        ),
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill=INK,
        body_background_fill_dark=INK,
        body_text_color=PAPER,
        body_text_color_dark=PAPER,
        button_primary_background_fill=OXIDE,
        button_primary_background_fill_dark=OXIDE,
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        block_background_fill=SURFACE,
        block_background_fill_dark=SURFACE,
        block_border_color=BORDER,
        block_border_color_dark=BORDER,
        input_background_fill=SURFACE_SOFT,
        input_background_fill_dark=SURFACE_SOFT,
        input_border_color=BORDER,
        input_border_color_dark=BORDER,
    )
