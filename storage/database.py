"""SQLite database. Stores diagnostic history and user sessions.

Schema:
  sessions(id, started_at, film_type, film_age_years, storage, scan_dpi)
  diagnoses(id, session_id, created_at, defect_count, label_counts_json,
            diagnosis_text, vision_seconds, reasoning_seconds, total_seconds,
            vision_model, reasoning_model, raw_json)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import get_app_config

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "storage" / "halide.db"
_INITIALIZED_DB_PATHS: set[Path] = set()


def get_db_path() -> Path:
    db_path = get_app_config().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    film_type TEXT NOT NULL,
    film_age_years INTEGER NOT NULL,
    storage TEXT NOT NULL,
    scan_dpi INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    defect_count INTEGER NOT NULL,
    label_counts_json TEXT NOT NULL,
    diagnosis_text TEXT NOT NULL,
    vision_seconds REAL NOT NULL,
    reasoning_seconds REAL NOT NULL,
    total_seconds REAL NOT NULL,
    vision_model TEXT NOT NULL,
    reasoning_model TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_session ON diagnoses(session_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_created ON diagnoses(created_at);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
    db_path = get_db_path()
    _INITIALIZED_DB_PATHS.add(db_path)
    logger.info("DB initialized at %s", db_path)


def _ensure_db_initialized() -> None:
    db_path = get_db_path()
    if db_path in _INITIALIZED_DB_PATHS and db_path.exists():
        return
    init_db()


def record_diagnosis(result: dict) -> str:
    """Persist a full pipeline result. Returns the diagnosis id."""
    _ensure_db_initialized()
    diagnosis_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    now = time.time()

    meta = result.get("film_metadata", {}) or {}
    defects = result.get("defects", {}) or {}
    diagnosis = result.get("diagnosis", {}) or {}

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, started_at, film_type, film_age_years,
                                  storage, scan_dpi)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                now,
                meta.get("film_type", "Unknown"),
                int(meta.get("film_age_years", 0) or 0),
                meta.get("storage", "unknown"),
                int(meta.get("scan_resolution_dpi", 0) or 0),
            ),
        )
        conn.execute(
            """
            INSERT INTO diagnoses (id, session_id, created_at, defect_count,
                                   label_counts_json, diagnosis_text,
                                   vision_seconds, reasoning_seconds,
                                   total_seconds, vision_model, reasoning_model,
                                   raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                diagnosis_id,
                session_id,
                now,
                int(defects.get("defect_count", 0) or 0),
                json.dumps(defects.get("label_counts", {}) or {}),
                diagnosis.get("diagnosis_text", ""),
                float(defects.get("inference_seconds", 0.0) or 0.0),
                float(diagnosis.get("reasoning_seconds", 0.0) or 0.0),
                float(result.get("total_seconds", 0.0) or 0.0),
                defects.get("model_path", ""),
                diagnosis.get("model_path", ""),
                json.dumps(result),
            ),
        )
    logger.info("Recorded diagnosis %s (session %s)", diagnosis_id, session_id)
    return diagnosis_id


def _decode_raw_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_to_diagnosis(row: sqlite3.Row) -> dict[str, Any]:
    raw = _decode_raw_json(row["raw_json"])
    meta = raw.get("film_metadata", {}) or {}
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "film_type": row["film_type"],
        "film_age_years": row["film_age_years"],
        "storage": row["storage"],
        "scan_dpi": row["scan_dpi"],
        "metadata_confidence": meta.get("metadata_confidence", "low"),
        "defect_count": row["defect_count"],
        "label_counts": json.loads(row["label_counts_json"]),
        "diagnosis_text": row["diagnosis_text"],
        "vision_seconds": row["vision_seconds"],
        "reasoning_seconds": row["reasoning_seconds"],
        "total_seconds": row["total_seconds"],
        "vision_model": row["vision_model"],
        "reasoning_model": row["reasoning_model"],
        "raw_json": raw,
    }


def list_recent(limit: int = 20) -> list[dict]:
    _ensure_db_initialized()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.created_at, s.film_type, s.film_age_years,
                   s.storage, s.scan_dpi, d.defect_count, d.label_counts_json,
                   d.diagnosis_text, d.vision_seconds, d.reasoning_seconds,
                   d.total_seconds, d.vision_model, d.reasoning_model,
                   d.raw_json
            FROM diagnoses d
            JOIN sessions s ON s.id = d.session_id
            ORDER BY d.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_diagnosis(r) for r in rows]


def get_diagnosis(diagnosis_id: str) -> dict[str, Any] | None:
    """Return one persisted diagnosis, including its full pipeline JSON."""
    if not diagnosis_id:
        return None
    _ensure_db_initialized()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT d.id, d.created_at, s.film_type, s.film_age_years,
                   s.storage, s.scan_dpi, d.defect_count, d.label_counts_json,
                   d.diagnosis_text, d.vision_seconds, d.reasoning_seconds,
                   d.total_seconds, d.vision_model, d.reasoning_model,
                   d.raw_json
            FROM diagnoses d
            JOIN sessions s ON s.id = d.session_id
            WHERE d.id = ?
            """,
            (diagnosis_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_diagnosis(row)


__all__ = [
    "init_db",
    "record_diagnosis",
    "list_recent",
    "get_diagnosis",
    "get_db_path",
]
