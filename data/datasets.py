"""Dataset loading and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config import DATA_DIR, REPO_ROOT
from data.schemas import ALLOWED_LABELS, clean_defects, label_counts

TRAINING_JSONL = DATA_DIR / "training_data.jsonl"
FDS_SCANS_DIR = (
    DATA_DIR
    / "raw"
    / "FilmDamageSimulator"
    / "FilmDamageSimulator"
    / "scans"
)


@dataclass(frozen=True)
class DatasetIssue:
    image: str
    message: str


def load_jsonl(path: str | Path = TRAINING_JSONL) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def resolve_image_path(entry: dict[str, Any], scans_dir: Path = FDS_SCANS_DIR) -> Path:
    image = str(entry.get("image", ""))
    path = Path(image)
    if path.is_absolute():
        return path
    candidate = scans_dir / image
    if candidate.exists():
        return candidate
    return REPO_ROOT / image


def validate_entries(
    entries: Iterable[dict[str, Any]],
    *,
    require_images: bool = False,
    scans_dir: Path = FDS_SCANS_DIR,
) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    for entry in entries:
        image = str(entry.get("image", ""))
        if not image:
            issues.append(DatasetIssue(image="(missing)", message="missing image field"))
        if require_images and image and not resolve_image_path(entry, scans_dir).exists():
            issues.append(DatasetIssue(image=image, message="image file does not exist"))

        annotations = entry.get("annotations", [])
        cleaned, dropped = clean_defects(annotations)
        if dropped:
            issues.append(
                DatasetIssue(
                    image=image or "(missing)",
                    message=f"{dropped} invalid annotations",
                )
            )
        for defect in cleaned:
            if defect["label"] not in ALLOWED_LABELS:
                issues.append(
                    DatasetIssue(
                        image=image or "(missing)",
                        message=f"unknown label {defect['label']}",
                    )
                )
    return issues


def dataset_summary(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    entries_list = list(entries)
    all_defects: list[dict[str, Any]] = []
    dropped = 0
    sources: dict[str, int] = {}
    for entry in entries_list:
        source = str(entry.get("source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        cleaned, bad = clean_defects(entry.get("annotations", []))
        all_defects.extend(cleaned)
        dropped += bad
    counts = label_counts(all_defects)
    return {
        "images": len(entries_list),
        "defects": len(all_defects),
        "dropped_annotations": dropped,
        "label_counts": counts,
        "sources": dict(sorted(sources.items())),
    }


def load_training_summary(path: str | Path = TRAINING_JSONL) -> dict[str, Any]:
    entries = load_jsonl(path)
    summary = dataset_summary(entries)
    summary["issues"] = [
        issue.__dict__ for issue in validate_entries(entries, require_images=False)
    ]
    return summary


__all__ = [
    "DatasetIssue",
    "FDS_SCANS_DIR",
    "TRAINING_JSONL",
    "dataset_summary",
    "load_jsonl",
    "load_training_summary",
    "resolve_image_path",
    "validate_entries",
]
