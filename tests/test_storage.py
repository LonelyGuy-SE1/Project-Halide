from __future__ import annotations

from storage.cache import DiagnosisCache
from storage import database


def test_record_and_list_recent_uses_configured_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HALIDE_DB_PATH", str(tmp_path / "halide.db"))
    database.init_db()
    diagnosis_id = database.record_diagnosis(
        {
            "film_metadata": {
                "film_type": "Kodak Portra 400",
                "film_age_years": 1,
                "storage": "fridge, sealed",
                "scan_resolution_dpi": 4000,
                "metadata_confidence": "high",
            },
            "defects": {
                "defect_count": 2,
                "label_counts": {"dust": 2},
                "inference_seconds": 0.1,
                "model_path": "vision",
            },
            "diagnosis": {
                "diagnosis_text": "Clean scanner glass.",
                "reasoning_seconds": 0.2,
                "model_path": "reasoning",
            },
            "total_seconds": 0.3,
        }
    )
    rows = database.list_recent(limit=5)
    assert rows[0]["id"] == diagnosis_id
    assert rows[0]["metadata_confidence"] == "high"
    assert rows[0]["label_counts"] == {"dust": 2}
    assert rows[0]["diagnosis_text"] == "Clean scanner glass."
    detail = database.get_diagnosis(diagnosis_id)
    assert detail is not None
    assert detail["raw_json"]["film_metadata"]["metadata_confidence"] == "high"
    assert detail["reasoning_model"] == "reasoning"


def test_cache_key_includes_metadata() -> None:
    cache = DiagnosisCache(max_size=4, ttl_seconds=60)
    image_bytes = b"scan"
    cache.put(image_bytes, {"film": "portra"}, metadata={"film_type": "Portra"})

    assert cache.get(image_bytes, metadata={"film_type": "Portra"}) == {"film": "portra"}
    assert cache.get(image_bytes, metadata={"film_type": "Tri-X"}) is None
