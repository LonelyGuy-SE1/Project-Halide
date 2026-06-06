"""Caching. In-process LRU cache for diagnosis results keyed by image hash.

For privacy, we hash the image bytes; the image itself is never persisted
in the cache. Identical scans produce identical hashes, giving us a simple
content-addressed cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class DiagnosisCache:
    """Thread-safe LRU cache for diagnosis results."""

    def __init__(self, max_size: int = 64, ttl_seconds: int = 3600) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def hash_image(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image_bytes: bytes) -> dict | None:
        key = self.hash_image(image_bytes)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if now - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            logger.info("Cache hit for %s", key[:12])
            return value

    def put(self, image_bytes: bytes, value: dict) -> None:
        key = self.hash_image(image_bytes)
        now = time.time()
        with self._lock:
            self._store[key] = (now, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


_default_cache: DiagnosisCache | None = None


def get_cache() -> DiagnosisCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = DiagnosisCache()
    return _default_cache


__all__ = ["DiagnosisCache", "get_cache"]
