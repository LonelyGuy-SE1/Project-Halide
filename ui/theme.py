"""Custom Gradio theme for the Project Halide workbench."""

from __future__ import annotations

import gradio as gr

BRASS = "#c59a52"
BRASS_DARK = "#8a6431"
COPPER = "#b85f3f"
TEAL = "#66d4c1"
VIOLET = "#9d8cff"
RED = "#ef5d52"

INK = "#0a0a0a"
CARBON = "#111111"
SURFACE = "#181715"
SURFACE_SOFT = "#211f1c"
SURFACE_LIFT = "#2c2924"
PAPER = "#f3eadb"
PAPER_SOFT = "#d7cbb8"
MUTED = "#a99b88"
BORDER = "#3a352e"
BLACK = "#050505"

THEME_CSS = f"""
:root {{
  --halide-brass: {BRASS};
  --halide-brass-dark: {BRASS_DARK};
  --halide-copper: {COPPER};
  --halide-teal: {TEAL};
  --halide-violet: {VIOLET};
  --halide-red: {RED};
  --halide-ink: {INK};
  --halide-carbon: {CARBON};
  --halide-surface: {SURFACE};
  --halide-surface-soft: {SURFACE_SOFT};
  --halide-surface-lift: {SURFACE_LIFT};
  --halide-paper: {PAPER};
  --halide-paper-soft: {PAPER_SOFT};
  --halide-muted: {MUTED};
  --halide-border: {BORDER};
  --halide-black: {BLACK};
}}

html,
body,
.gradio-container {{
  background: var(--halide-ink) !important;
  color: var(--halide-paper) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif !important;
  letter-spacing: 0 !important;
}}

body::before {{
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    repeating-linear-gradient(
      90deg,
      rgba(255, 255, 255, 0.018) 0,
      rgba(255, 255, 255, 0.018) 1px,
      transparent 1px,
      transparent 16px
    );
  opacity: 0.45;
}}

.gradio-container {{
  max-width: none !important;
  min-height: 100vh !important;
  padding: 0 18px 18px !important;
}}

#halide-app {{
  min-height: 100vh;
}}

#halide-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  max-width: 1720px;
  margin: 0 auto;
  padding: 18px 0 14px;
  border-bottom: 1px solid rgba(197, 154, 82, 0.34);
}}

.halide-brand-lockup {{
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}}

.halide-brand-mark {{
  width: 46px;
  height: 46px;
  border-radius: 8px;
  object-fit: cover;
  border: 1px solid rgba(197, 154, 82, 0.45);
  background: var(--halide-black);
}}

#halide-header h1 {{
  color: var(--halide-paper);
  font-size: clamp(1.5rem, 2.2vw, 2.35rem);
  line-height: 1.04;
  margin: 2px 0 0;
  font-weight: 860;
  letter-spacing: 0 !important;
}}

.halide-kicker {{
  color: var(--halide-teal);
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 840;
  letter-spacing: 0.12em !important;
  text-transform: uppercase;
}}

.halide-model-strip {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  min-width: min(42vw, 36rem);
}}

.halide-model-strip span,
.halide-model-strip a {{
  border: 1px solid rgba(197, 154, 82, 0.32);
  color: var(--halide-paper);
  background: rgba(24, 23, 21, 0.92);
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 0.76rem;
  font-weight: 780;
  line-height: 1;
  text-decoration: none;
  white-space: nowrap;
}}

.halide-model-strip a {{
  color: #dffcf6;
  border-color: rgba(102, 212, 193, 0.38);
}}

.halide-workbench {{
  max-width: 1720px;
  margin: 14px auto 0 !important;
  gap: 14px !important;
  align-items: flex-start;
}}

.halide-main-stage,
.halide-inspector,
.halide-intake-panel {{
  min-width: 0 !important;
}}

.halide-intake-panel,
.halide-inspector {{
  background: rgba(17, 17, 17, 0.98);
  border: 1px solid rgba(58, 53, 46, 0.95);
  border-radius: 8px;
  padding: 12px;
}}

.halide-panel-title {{
  color: var(--halide-paper);
  font-size: 0.76rem;
  font-weight: 880;
  letter-spacing: 0.1em !important;
  text-transform: uppercase;
  margin: 0 0 8px;
}}

.halide-rail-copy {{
  color: var(--halide-muted);
  font-size: 0.84rem;
  line-height: 1.42;
  margin: 0 0 12px;
}}

.halide-model-card {{
  border: 1px solid rgba(102, 212, 193, 0.28);
  background: rgba(102, 212, 193, 0.055);
  border-radius: 8px;
  padding: 12px;
  margin-top: 12px;
}}

.halide-model-card span {{
  color: var(--halide-teal);
  display: block;
  font-size: 0.68rem;
  font-weight: 860;
  letter-spacing: 0.11em !important;
  text-transform: uppercase;
  margin-bottom: 5px;
}}

.halide-model-card strong {{
  color: var(--halide-paper);
  display: block;
  font-size: 0.98rem;
  line-height: 1.2;
}}

.halide-model-card p {{
  color: var(--halide-muted);
  margin: 6px 0 0;
  font-size: 0.82rem;
  line-height: 1.35;
}}

.halide-run-state {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 5px;
  border-radius: 8px;
  padding: 13px 15px;
  margin-bottom: 12px;
  background: var(--halide-surface);
  border: 1px solid rgba(197, 154, 82, 0.30);
  min-height: 76px;
}}

.halide-run-state strong {{
  color: var(--halide-paper);
  font-size: clamp(1.08rem, 1.5vw, 1.45rem);
  line-height: 1.1;
}}

.halide-run-state span:last-child {{
  color: var(--halide-muted);
  font-size: 0.84rem;
}}

.halide-run-state.active {{
  border-color: rgba(102, 212, 193, 0.46);
}}

.halide-run-state.quiet {{
  border-color: rgba(157, 140, 255, 0.35);
}}

.halide-run-eyebrow {{
  color: var(--halide-brass);
  font-size: 0.68rem;
  font-weight: 860;
  letter-spacing: 0.12em !important;
  text-transform: uppercase;
}}

.halide-lighttable {{
  background: #0f0f0e !important;
  border: 1px solid rgba(197, 154, 82, 0.34) !important;
  border-radius: 8px !important;
  padding: 13px !important;
  box-shadow: 0 22px 52px rgba(0, 0, 0, 0.42) !important;
}}

.halide-section-header {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}}

.halide-section-header span {{
  color: var(--halide-brass);
  display: block;
  font-size: 0.68rem;
  font-weight: 880;
  letter-spacing: 0.12em !important;
  text-transform: uppercase;
}}

.halide-section-header strong {{
  color: var(--halide-paper);
  display: block;
  font-size: 1rem;
  line-height: 1.2;
  margin-top: 3px;
}}

.halide-section-header small {{
  color: var(--halide-muted);
  border: 1px solid rgba(197, 154, 82, 0.26);
  border-radius: 999px;
  padding: 5px 9px;
  font-size: 0.72rem;
  line-height: 1;
}}

#halide-compare,
.halide-review-gallery {{
  background: var(--halide-black) !important;
}}

.halide-review-gallery {{
  margin-top: 12px !important;
}}

.halide-upload img,
#halide-compare img,
.halide-review-gallery img {{
  background: var(--halide-black) !important;
  object-fit: contain !important;
}}

.halide-inline-controls {{
  gap: 8px !important;
}}

#halide-run-button,
button.primary,
.primary button {{
  background: var(--halide-copper) !important;
  color: white !important;
  border: 1px solid rgba(255, 255, 255, 0.16) !important;
  border-radius: 8px !important;
  font-weight: 860 !important;
  letter-spacing: 0 !important;
  min-height: 46px !important;
  box-shadow: 0 16px 30px rgba(184, 95, 63, 0.24) !important;
}}

#halide-run-button:hover,
button.primary:hover,
.primary button:hover {{
  background: #ca6f4c !important;
}}

button.secondary,
.secondary button,
button {{
  border-radius: 8px !important;
  font-weight: 760 !important;
}}

.halide-section-title {{
  color: var(--halide-paper);
  font-size: 0.76rem;
  font-weight: 880;
  letter-spacing: 0.1em !important;
  text-transform: uppercase;
  margin: 0 0 10px;
}}

.halide-subsection {{
  color: var(--halide-muted);
  font-size: 0.72rem;
  font-weight: 860;
  letter-spacing: 0.08em !important;
  margin: 14px 0 7px;
  text-transform: uppercase;
}}

.halide-muted {{
  color: var(--halide-muted);
  margin: 0;
}}

.halide-empty-card {{
  background: rgba(33, 31, 28, 0.82);
  border: 1px solid rgba(197, 154, 82, 0.28);
  border-radius: 8px;
  padding: 13px;
}}

.halide-empty-card span {{
  color: var(--halide-brass);
  display: block;
  font-size: 0.68rem;
  font-weight: 880;
  letter-spacing: 0.12em !important;
  text-transform: uppercase;
  margin-bottom: 6px;
}}

.halide-empty-card strong {{
  color: var(--halide-paper);
  display: block;
  font-size: 1rem;
  line-height: 1.2;
}}

.halide-empty-card p {{
  color: var(--halide-muted);
  font-size: 0.84rem;
  line-height: 1.4;
  margin: 7px 0 0;
}}

.halide-notice {{
  border-radius: 8px;
  padding: 11px 12px;
  border: 1px solid var(--halide-border);
  color: var(--halide-paper);
  background: rgba(33, 31, 28, 0.9);
  font-size: 0.88rem;
  line-height: 1.42;
  margin-bottom: 10px;
}}

.halide-notice.good {{
  border-color: rgba(102, 212, 193, 0.38);
  background: rgba(102, 212, 193, 0.075);
}}

.halide-notice.caution {{
  border-color: rgba(197, 154, 82, 0.48);
  background: rgba(197, 154, 82, 0.08);
}}

.halide-notice.neutral {{
  border-color: rgba(157, 140, 255, 0.36);
  background: rgba(157, 140, 255, 0.075);
}}

.halide-stats {{
  display: grid;
  gap: 0;
}}

.halide-stats.compact {{
  gap: 0;
}}

.halide-stat {{
  display: grid;
  grid-template-columns: minmax(7rem, 0.42fr) minmax(0, 1fr);
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(58, 53, 46, 0.85);
  color: var(--halide-paper);
  min-width: 0;
}}

.halide-stat:last-child {{
  border-bottom: 0;
}}

.halide-stat-label {{
  color: var(--halide-muted);
  font-weight: 780;
}}

.halide-stat span:last-child {{
  overflow-wrap: anywhere;
  text-align: right;
}}

.halide-report {{
  display: grid;
  gap: 10px;
}}

.halide-report-section {{
  background: rgba(33, 31, 28, 0.82);
  border: 1px solid rgba(58, 53, 46, 0.94);
  border-left: 3px solid var(--halide-brass);
  border-radius: 8px;
  padding: 12px 13px;
}}

.halide-report-heading {{
  color: var(--halide-paper);
  font-size: 0.78rem;
  font-weight: 880;
  letter-spacing: 0.1em !important;
  margin-bottom: 8px;
  text-transform: uppercase;
}}

.halide-report-body {{
  color: var(--halide-paper-soft);
  font-size: 0.93rem;
  line-height: 1.56;
}}

.halide-report-body p {{
  margin: 0 0 8px;
}}

.halide-report-body p:last-child {{
  margin-bottom: 0;
}}

.halide-report-body ul,
.halide-report-body ol {{
  margin: 0;
  padding-left: 1.15rem;
}}

.halide-report-body li {{
  margin: 5px 0;
  padding-left: 2px;
}}

.halide-pill-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: 10px;
}}

.halide-defect-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(184, 95, 63, 0.13);
  color: #ffe4d8;
  border: 1px solid rgba(184, 95, 63, 0.38);
  padding: 6px 9px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 780;
  white-space: nowrap;
}}

.halide-defect-pill strong {{
  color: var(--halide-paper);
}}

.halide-defect-pill.dust {{
  background: rgba(197, 154, 82, 0.13);
  border-color: rgba(197, 154, 82, 0.38);
}}

.halide-defect-pill.dirt {{
  background: rgba(184, 95, 63, 0.13);
  border-color: rgba(184, 95, 63, 0.38);
}}

.halide-defect-pill.scratch {{
  background: rgba(239, 93, 82, 0.14);
  border-color: rgba(239, 93, 82, 0.42);
}}

.halide-defect-pill.long_hair {{
  background: rgba(157, 140, 255, 0.13);
  border-color: rgba(157, 140, 255, 0.38);
}}

.halide-defect-pill.short_hair {{
  background: rgba(102, 212, 193, 0.11);
  border-color: rgba(102, 212, 193, 0.36);
}}

.halide-defect-pill.emulsion_damage {{
  background: rgba(215, 203, 184, 0.12);
  border-color: rgba(215, 203, 184, 0.34);
}}

.halide-defect-pill.chemical_stain {{
  background: rgba(74, 222, 128, 0.12);
  border-color: rgba(74, 222, 128, 0.34);
}}

.halide-defect-pill.light_leak {{
  background: rgba(244, 114, 182, 0.13);
  border-color: rgba(244, 114, 182, 0.36);
}}

.halide-history-item {{
  background: rgba(33, 31, 28, 0.82);
  border: 1px solid rgba(58, 53, 46, 0.94);
  border-radius: 8px;
  margin-bottom: 9px;
  padding: 11px;
}}

.halide-history-title {{
  color: var(--halide-paper);
  font-weight: 860;
  margin-bottom: 4px;
  overflow-wrap: anywhere;
}}

.halide-history-meta {{
  color: var(--halide-muted);
  font-size: 0.78rem;
  line-height: 1.36;
  margin: 4px 0;
}}

.halide-history-detail {{
  display: grid;
  gap: 8px;
}}

.halide-history-feed {{
  max-height: 18rem;
  overflow: auto;
}}

.halide-intake-panel .block,
.halide-inspector .block,
.halide-lighttable .block {{
  background: rgba(24, 23, 21, 0.95) !important;
  border-color: rgba(58, 53, 46, 0.95) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
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
.label-wrap span,
label[data-testid="block-label"] {{
  background: rgba(33, 31, 28, 0.98) !important;
  border-color: rgba(197, 154, 82, 0.24) !important;
  color: var(--halide-paper) !important;
  border-radius: 6px !important;
  font-weight: 780 !important;
  box-shadow: none !important;
}}

label[data-testid="block-label"] span {{
  color: var(--halide-paper) !important;
}}

label[data-testid$="-radio-label"] {{
  background: rgba(33, 31, 28, 0.98) !important;
  border: 1px solid rgba(243, 234, 219, 0.18) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  opacity: 1 !important;
}}

label[data-testid$="-radio-label"].selected {{
  background: rgba(197, 154, 82, 0.16) !important;
  border-color: rgba(197, 154, 82, 0.5) !important;
}}

label[data-testid$="-radio-label"] span,
label[data-testid$="-radio-label"] input,
.gradio-checkbox label span {{
  color: var(--halide-paper) !important;
  opacity: 1 !important;
}}

label[data-testid$="-radio-label"] input {{
  accent-color: var(--halide-brass);
}}

.tabs {{
  gap: 4px !important;
}}

.tabs button {{
  color: var(--halide-muted) !important;
  font-weight: 820 !important;
  border-radius: 8px !important;
}}

.tabs button.selected {{
  color: var(--halide-paper) !important;
  border-color: var(--halide-brass) !important;
  background: rgba(197, 154, 82, 0.10) !important;
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
  max-width: 1720px;
  margin: 0 auto;
  color: var(--halide-muted) !important;
  text-align: center;
  padding: 14px 8px 4px;
  font-size: 0.82rem;
}}

@media (max-width: 1180px) {{
  .halide-workbench {{
    flex-direction: column !important;
  }}

  .halide-intake-panel,
  .halide-main-stage,
  .halide-inspector {{
    width: 100% !important;
  }}
}}

@media (max-width: 760px) {{
  .gradio-container {{
    padding: 0 10px 12px !important;
  }}

  #halide-header {{
    align-items: flex-start;
    flex-direction: column;
  }}

  .halide-model-strip {{
    justify-content: flex-start;
    min-width: 0;
  }}

  .halide-brand-mark {{
    width: 40px;
    height: 40px;
  }}

  .halide-stat {{
    grid-template-columns: 1fr;
    gap: 3px;
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
            c50="#fff8ed",
            c100="#f8ead2",
            c200="#ebd1a3",
            c300="#d9b36d",
            c400=BRASS,
            c500=BRASS,
            c600=BRASS_DARK,
            c700="#694a24",
            c800="#4d361d",
            c900="#332313",
            c950="#1c1209",
        ),
        secondary_hue=gr.themes.Color(
            c50="#effffb",
            c100="#d5fff7",
            c200="#a8f5ea",
            c300=TEAL,
            c400="#3fbfae",
            c500="#259d90",
            c600="#1d7d74",
            c700="#1b645e",
            c800="#174d49",
            c900="#123c39",
            c950="#071f1d",
        ),
        neutral_hue=gr.themes.Color(
            c50="#fbf7ef",
            c100=PAPER,
            c200=PAPER_SOFT,
            c300=MUTED,
            c400="#85796a",
            c500="#6e6357",
            c600="#544d44",
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
        button_primary_background_fill=COPPER,
        button_primary_background_fill_dark=COPPER,
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
