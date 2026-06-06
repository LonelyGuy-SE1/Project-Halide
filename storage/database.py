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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "storage" / "halide.db"


def get_db_path() -> Path:
    import os
    custom = os.getenv("HALIDE_DB_PATH")
    if custom:
        return Path(custom)
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DB_PATH


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
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
    logger.info("DB initialized at %s", get_db_path())


def record_diagnosis(result: dict) -> str:
    """Persist a full pipeline result. Returns the diagnosis id."""
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


def list_recent(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.created_at, s.film_type, s.film_age_years,
                   s.storage, s.scan_dpi, d.defect_count, d.label_counts_json,
                   d.diagnosis_text, d.total_seconds
            FROM diagnoses d
            JOIN sessions s ON s.id = d.session_id
            ORDER BY d.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "film_type": r["film_type"],
                "film_age_years": r["film_age_years"],
                "storage": r["storage"],
                "scan_dpi": r["scan_dpi"],
                "defect_count": r["defect_count"],
                "label_counts": json.loads(r["label_counts_json"]),
                "diagnosis_text": r["diagnosis_text"],
                "total_seconds": r["total_seconds"],
            }
        )
    return out


__all__ = [
    "init_db",
    "record_diagnosis",
    "list_recent",
    "get_db_path",
]
