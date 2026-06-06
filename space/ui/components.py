"""UI components. Defect list rendering and shared visual helpers."""

from __future__ import annotations

from typing import Iterable

from ui.theme import THEME_CSS


HEADER_HTML = """
<div id="halide-header">
  <h1>Project Halide</h1>
  <p>Edge-native diagnostic engine for analog film scans</p>
</div>
"""


def defect_pills_html(label_counts: dict[str, int]) -> str:
    """Render defect counts as colored pills."""
    if not label_counts:
        return '<p style="color: var(--halide-slate);">No defects detected.</p>'
    pills: list[str] = []
    for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        pills.append(
            f'<span class="halide-defect-pill {label}">{label}: {count}</span>'
        )
    return '<div>' + "".join(pills) + "</div>"


def stats_html(result: dict) -> str:
    """Render a stats card with defect counts and timing."""
    defects = result.get("defects", {}) or {}
    diagnosis = result.get("diagnosis", {}) or {}
    total = result.get("total_seconds", 0.0) or 0.0
    vision_s = defects.get("inference_seconds", 0.0) or 0.0
    reasoning_s = diagnosis.get("reasoning_seconds", 0.0) or 0.0

    rows: list[str] = []
    rows.append(_stat_row("Total defects", str(defects.get("defect_count", 0))))
    rows.append(_stat_row("Dropped (invalid)", str(defects.get("dropped_count", 0))))
    rows.append(_stat_row("Vision inference", f"{vision_s:.2f}s"))
    rows.append(_stat_row("Reasoning", f"{reasoning_s:.2f}s"))
    rows.append(_stat_row("Total", f"{total:.2f}s"))
    rows.append(_stat_row("Vision model", _truncate(defects.get("model_path", ""), 50)))
    rows.append(_stat_row("Reasoning model", _truncate(diagnosis.get("model_path", ""), 50)))
    return f'<div class="halide-card">{"".join(rows)}</div>'


def _stat_row(label: str, value: str) -> str:
    return (
        '<div class="halide-stat">'
        f'<span class="halide-stat-label">{label}</span>'
        f'<span>{value}</span>'
        "</div>"
    )


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return "..." + s[-(n - 3):]


def diagnosis_html(text: str) -> str:
    """Wrap diagnosis text in the styled card."""
    safe = (text or "(no diagnosis produced)").replace("\n", "<br>")
    return f'<div class="halide-diagnosis">{safe}</div>'


def history_row_html(entry: dict) -> str:
    """Render a single row in the recent-diagnoses sidebar."""
    counts = entry.get("label_counts", {}) or {}
    total = entry.get("defect_count", 0) or 0
    film = entry.get("film_type", "Unknown")
    age = entry.get("film_age_years", "?")
    storage = entry.get("storage", "?")
    ts = entry.get("created_at", 0)
    seconds = entry.get("total_seconds", 0.0) or 0.0
    return (
        f'<div class="halide-card" style="margin-bottom: 0.6rem;">'
        f'<div class="halide-section-title" style="font-size: 0.95rem;">'
        f"{film} (age {age}y, {storage})</div>"
        f"{defect_pills_html(counts)}"
        f'<div style="color: var(--halide-slate); font-size: 0.8rem; margin-top: 0.4rem;">'
        f"defects: {total} | {seconds:.2f}s | {ts:.0f}"
        f"</div></div>"
    )


def render_history(entries: Iterable[dict]) -> str:
    items = "".join(history_row_html(e) for e in entries)
    if not items:
        return '<p style="color: var(--halide-slate);">No diagnoses yet.</p>'
    return items


__all__ = [
    "HEADER_HTML",
    "THEME_CSS",
    "defect_pills_html",
    "stats_html",
    "diagnosis_html",
    "render_history",
]
