"""UI components. Defect list rendering and shared visual helpers."""

from __future__ import annotations

import base64
import html
import json
import time
from pathlib import Path
import re
from typing import Iterable

from data.schemas import LABEL_DISPLAY_NAMES


def _logo_html() -> str:
    path = Path(__file__).resolve().parents[1] / "assets" / "logo.jpg"
    if not path.exists():
        return '<span class="halide-brand-mark halide-brand-mark-text">H</span>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<img class="halide-brand-mark" '
        f'src="data:image/jpeg;base64,{encoded}" alt="" />'
    )


HEADER_HTML = f"""
<div id="halide-header">
  <div class="halide-brand-lockup">
    {_logo_html()}
    <div>
      <span class="halide-kicker">Project Halide</span>
      <h1>Analog Film Diagnostic Workbench</h1>
    </div>
  </div>
  <div class="halide-model-strip">
    <a href="https://huggingface.co/Lonelyguyse1" target="_blank" rel="noreferrer">Lonelyguyse1</a>
    <a href="https://huggingface.co/Lonelyguyse1/halide-vision" target="_blank" rel="noreferrer">halide-vision</a>
    <span>MiniCPM-V 4.6</span>
    <span>Nemotron Mini 4B</span>
  </div>
</div>
"""

EMPTY_STATE = '<p class="halide-muted">Awaiting scan.</p>'
REPORT_EMPTY_STATE = (
    '<div class="halide-empty-card">'
    '<span>Report</span>'
    '<strong>No scan analyzed yet</strong>'
    '<p>Results, evidence counts, and physical fixes will appear here after a GPU run.</p>'
    "</div>"
)
LIGHTTABLE_EMPTY_STATE = (
    '<div class="halide-empty-lighttable">'
    '<div class="halide-empty-frame-grid">'
    '<div><span>Original</span></div>'
    '<div><span>Validated overlay</span></div>'
    "</div>"
    '<div class="halide-empty-center">'
    '<span>Ready</span>'
    '<strong>No scan loaded</strong>'
    "</div>"
    "</div>"
)
LIGHTTABLE_RUNNING_STATE = (
    '<div class="halide-empty-lighttable active">'
    '<div class="halide-empty-frame-grid">'
    '<div><span>Vision extraction</span></div>'
    '<div><span>Diagnostic report</span></div>'
    "</div>"
    '<div class="halide-empty-center">'
    '<span>Running</span>'
    '<strong>GPU inspection in progress</strong>'
    "</div>"
    "</div>"
)

REPORT_SECTIONS = {
    "root cause": "Root cause",
    "evidence": "Evidence",
    "physical fixes": "Physical fixes",
    "confidence": "Confidence",
    "next inspection": "Next inspection",
}


def defect_pills_html(label_counts: dict[str, int]) -> str:
    """Render defect counts as colored pills."""
    if not label_counts:
        return '<p class="halide-muted">No validated defects detected.</p>'
    pills: list[str] = []
    for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        display = LABEL_DISPLAY_NAMES.get(label, label.replace("_", " ").title())
        pills.append(
            f'<span class="halide-defect-pill {html.escape(label)}">'
            f"{html.escape(display)} <strong>{int(count)}</strong></span>"
        )
    return '<div class="halide-pill-row">' + "".join(pills) + "</div>"


def compact_label_counts(label_counts: dict[str, int]) -> str:
    if not label_counts:
        return "none"
    parts = []
    for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        display = LABEL_DISPLAY_NAMES.get(label, label.replace("_", " ").title())
        parts.append(f"{display}: {int(count)}")
    return ", ".join(parts)


def defect_table_rows(result: dict | None) -> list[list[str]]:
    """Return rows for the evidence dataframe."""
    if not result:
        return []
    defects = (result.get("defects", {}) or {}).get("defects", []) or []
    rows: list[list[str]] = []
    for index, defect in enumerate(defects, start=1):
        label = str(defect.get("label", ""))
        display = LABEL_DISPLAY_NAMES.get(label, label.replace("_", " ").title())
        bbox = defect.get("bbox", []) or []
        if len(bbox) == 4:
            box_text = ", ".join(f"{float(v):.3f}" for v in bbox)
        else:
            box_text = "invalid"
        confidence = defect.get("confidence")
        confidence_text = "" if confidence is None else f"{float(confidence):.2f}"
        rows.append([str(index), display, confidence_text, box_text])
    return rows


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
    rows.append(_stat_row("Duplicates removed", str(defects.get("duplicate_count", 0))))
    rows.append(_stat_row("Resized for model", "yes" if defects.get("resized_for_model") else "no"))
    rows.append(_stat_row("Vision inference", f"{vision_s:.2f}s"))
    rows.append(_stat_row("Reasoning", f"{reasoning_s:.2f}s"))
    rows.append(_stat_row("Total", f"{total:.2f}s"))
    rows.append(_stat_row("Vision model", _truncate(defects.get("model_path", ""), 46)))
    rows.append(_stat_row("Reasoning model", _truncate(diagnosis.get("model_path", ""), 46)))
    return (
        '<div class="halide-panel-title">Run telemetry</div>'
        f'<div class="halide-stats">{"".join(rows)}</div>'
    )


def _stat_row(label: str, value: str) -> str:
    return (
        '<div class="halide-stat">'
        f'<span class="halide-stat-label">{html.escape(label)}</span>'
        f'<span>{html.escape(value)}</span>'
        "</div>"
    )


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return "..." + s[-(n - 3):]


def diagnosis_html(text: str) -> str:
    """Render the Nemotron Markdown report into structured HTML."""
    return render_markdown_report(text or "(no diagnosis produced)")


def render_markdown_report(text: str) -> str:
    """Render the constrained diagnosis Markdown used by Nemotron.

    This intentionally supports only the report shapes we request from the
    model: section headings, paragraphs, bullets, and numbered fixes.
    """
    sections: list[dict[str, list[str] | str]] = []
    current: dict[str, list[str] | str] = {
        "title": "Report",
        "lines": [],
    }

    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if line.startswith("## "):
            if current["lines"]:
                sections.append(current)
            title_key = line[3:].strip().lower()
            current = {
                "title": REPORT_SECTIONS.get(title_key, line[3:].strip() or "Report"),
                "lines": [],
            }
            continue
        current["lines"].append(line)
    if current["lines"] or not sections:
        sections.append(current)

    rendered = []
    for section in sections:
        title = html.escape(str(section["title"]))
        body = _render_report_lines(section["lines"])  # type: ignore[arg-type]
        rendered.append(
            '<section class="halide-report-section">'
            f'<div class="halide-report-heading">{title}</div>'
            f'<div class="halide-report-body">{body}</div>'
            "</section>"
        )
    return '<div class="halide-report">' + "".join(rendered) + "</div>"


def _render_report_lines(lines: list[str]) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []
    bullet_items: list[str] = []
    ordered_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(
                "<p>"
                + " ".join(html.escape(part) for part in paragraph if part)
                + "</p>"
            )
            paragraph.clear()

    def flush_bullets() -> None:
        if bullet_items:
            items = "".join(f"<li>{item}</li>" for item in bullet_items)
            blocks.append(f"<ul>{items}</ul>")
            bullet_items.clear()

    def flush_ordered() -> None:
        if ordered_items:
            items = "".join(f"<li>{item}</li>" for item in ordered_items)
            blocks.append(f"<ol>{items}</ol>")
            ordered_items.clear()

    for line in lines:
        if not line:
            flush_paragraph()
            flush_bullets()
            flush_ordered()
            continue
        numbered = re.match(r"^\d+\.\s+(.*)$", line)
        if line.startswith("- "):
            flush_paragraph()
            flush_ordered()
            bullet_items.append(html.escape(line[2:].strip()))
        elif numbered:
            flush_paragraph()
            flush_bullets()
            ordered_items.append(html.escape(numbered.group(1).strip()))
        else:
            flush_bullets()
            flush_ordered()
            paragraph.append(line)

    flush_paragraph()
    flush_bullets()
    flush_ordered()
    return "".join(blocks) or '<p class="halide-muted">No report text.</p>'


def metadata_html(result: dict) -> str:
    """Render a compact metadata strip for the current run."""
    meta = result.get("film_metadata", {}) or {}
    confidence = str(meta.get("metadata_confidence", "low") or "low")
    rows = [
        _stat_row("Film stock", str(meta.get("film_type", "Unknown"))),
        _stat_row("Age", f"{meta.get('film_age_years', 0)} years"),
        _stat_row("Storage", str(meta.get("storage", "unknown"))),
        _stat_row("Scan DPI", str(meta.get("scan_resolution_dpi", "unknown"))),
        _stat_row("Metadata confidence", confidence.title()),
    ]
    return (
        '<div class="halide-panel-title">Film dossier</div>'
        f'<div class="halide-stats compact">{"".join(rows)}</div>'
    )


def confidence_notice_html(result: dict) -> str:
    """Render a small evidence-quality notice from validation metadata."""
    defects = result.get("defects", {}) or {}
    total = int(defects.get("defect_count", 0) or 0)
    dropped = int(defects.get("dropped_count", 0) or 0)
    duplicate = int(defects.get("duplicate_count", 0) or 0)
    meta = result.get("film_metadata", {}) or {}
    confidence = str(meta.get("metadata_confidence", "low") or "low").lower()

    if total == 0:
        message = "No validated boxes were returned. Inspect the scan before assuming a film fault."
        tone = "neutral"
    elif confidence == "low":
        message = "Metadata is marked low confidence, so the diagnosis is driven mainly by visible defects."
        tone = "caution"
    elif dropped or duplicate:
        message = f"Schema checks removed {dropped} invalid boxes and {duplicate} duplicate boxes."
        tone = "caution"
    else:
        message = "Schema validation passed for the visible defect evidence."
        tone = "good"
    return f'<div class="halide-notice {tone}">{html.escape(message)}</div>'


def run_state_html(result: dict | None) -> str:
    """Render a top-level status block for the current diagnosis."""
    if not result:
        return (
            '<div class="halide-run-state idle">'
            '<span class="halide-run-eyebrow">Ready</span>'
            "<strong>Load a scan to begin inspection.</strong>"
            "</div>"
        )

    defects = result.get("defects", {}) or {}
    count = int(defects.get("defect_count", 0) or 0)
    dropped = int(defects.get("dropped_count", 0) or 0)
    duplicates = int(defects.get("duplicate_count", 0) or 0)
    if count:
        title = f"{count} validated defect{'s' if count != 1 else ''}"
        tone = "active"
    else:
        title = "No validated defects"
        tone = "quiet"

    detail = (
        f"{dropped} invalid removed, {duplicates} duplicate removed, "
        f"{float(result.get('total_seconds', 0.0) or 0.0):.2f}s total"
    )
    return (
        f'<div class="halide-run-state {tone}">'
        '<span class="halide-run-eyebrow">Current scan</span>'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(detail)}</span>"
        "</div>"
    )


def history_label(entry: dict) -> str:
    """Return a compact plain-text label for the history dropdown."""
    stamp = time.strftime(
        "%Y-%m-%d %H:%M",
        time.localtime(float(entry.get("created_at", 0) or 0)),
    )
    film = str(entry.get("film_type", "Unknown"))
    count = int(entry.get("defect_count", 0) or 0)
    return f"{stamp} | {film} | {count} defects"


def history_table_rows(entries: Iterable[dict]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in entries:
        stamp = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(float(entry.get("created_at", 0) or 0)),
        )
        rows.append(
            [
                stamp,
                str(entry.get("film_type", "Unknown")),
                str(int(entry.get("defect_count", 0) or 0)),
                compact_label_counts(entry.get("label_counts", {}) or {}),
                str(entry.get("id", "")),
            ]
        )
    return rows


def history_choices(entries: Iterable[dict]) -> list[tuple[str, str]]:
    return [(history_label(e), str(e.get("id", ""))) for e in entries if e.get("id")]


def history_detail_html(entry: dict | None) -> str:
    if not entry:
        return '<p class="halide-muted">Select a diagnosis to review details.</p>'

    counts = entry.get("label_counts", {}) or {}
    meta = entry.get("raw_json", {}).get("film_metadata", {}) if entry.get("raw_json") else {}
    confidence = meta.get("metadata_confidence", entry.get("metadata_confidence", "low"))
    stamp = time.strftime(
        "%Y-%m-%d %H:%M",
        time.localtime(float(entry.get("created_at", 0) or 0)),
    )
    header_rows = [
        _stat_row("Saved", stamp),
        _stat_row("Film stock", str(entry.get("film_type", "Unknown"))),
        _stat_row("Age", f"{entry.get('film_age_years', '?')} years"),
        _stat_row("Storage", str(entry.get("storage", "?"))),
        _stat_row("Scan DPI", str(entry.get("scan_dpi", "?"))),
        _stat_row("Metadata confidence", str(confidence).title()),
        _stat_row("Vision model", _truncate(str(entry.get("vision_model", "")), 60)),
        _stat_row("Reasoning model", _truncate(str(entry.get("reasoning_model", "")), 60)),
    ]
    return (
        '<div class="halide-history-detail">'
        f'<div class="halide-stats compact">{"".join(header_rows)}</div>'
        '<div class="halide-subsection">Defects</div>'
        f"{defect_pills_html(counts)}"
        '<div class="halide-subsection">Diagnosis</div>'
        f"{diagnosis_html(str(entry.get('diagnosis_text', '')))}"
        "</div>"
    )


def raw_json_text(result_or_entry: dict | None) -> str:
    if not result_or_entry:
        return "{}"
    if "raw_json" in result_or_entry:
        payload = result_or_entry.get("raw_json") or {}
    else:
        payload = result_or_entry
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "HEADER_HTML",
    "EMPTY_STATE",
    "LIGHTTABLE_EMPTY_STATE",
    "LIGHTTABLE_RUNNING_STATE",
    "REPORT_EMPTY_STATE",
    "compact_label_counts",
    "confidence_notice_html",
    "defect_table_rows",
    "defect_pills_html",
    "diagnosis_html",
    "history_choices",
    "history_detail_html",
    "history_label",
    "history_table_rows",
    "metadata_html",
    "render_markdown_report",
    "run_state_html",
    "raw_json_text",
    "stats_html",
]
