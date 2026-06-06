"""Smoke test for the OpenBMB free MiniCPM-V 4.6 API.

Endpoint documented in .nottracked/reference.md (Section 6):
  http://35.203.155.71:8003
  Auth: Bearer sk-minicpm-V8bcD-YTAMxECagaKOnbwTCN69IN2LhSeqGiOgq2Ues

Tests in order:
  1. health: GET /
  2. models: GET /v1/models
  3. chat text-only: POST /v1/chat/completions with a tiny text message
  4. chat with image: same endpoint with a base64-encoded test image
  5. completion: POST /v1/completions (alt endpoint)

For each: print HTTP status, response time, response body (truncated).
On success: record working combination for later use.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests

BASE = "http://35.203.155.71:8003"
AUTH = "Bearer sk-minicpm-V8bcD-YTAMxECagaKOnbwTCN69IN2LhSeqGiOgq2Ues"
ALT_AUTH = "sk-minicpm-V8bcD-YTAMxECagaKOnbwTCN69IN2LhSeqGiOgq2Ues"  # no Bearer

REPO = Path(__file__).resolve().parents[1]
TEST_IMG = REPO / "data" / "test_images" / "02_ood_synthetic_scratch.png"
TEST_MODEL = "MiniCPM-V-4.6"


def call(method: str, path: str, *, headers: dict | None = None, json_body: dict | None = None,
         auth: str = AUTH, timeout: int = 60) -> tuple[int, float, str]:
    url = f"{BASE}{path}"
    h = {"Content-Type": "application/json", "Authorization": auth}
    if headers:
        h.update(headers)
    t0 = time.time()
    try:
        if method == "GET":
            r = requests.get(url, headers=h, timeout=timeout)
        else:
            r = requests.post(url, headers=h, json=json_body, timeout=timeout)
        elapsed = time.time() - t0
        body = r.text[:600]
        return r.status_code, elapsed, body
    except Exception as exc:
        elapsed = time.time() - t0
        return -1, elapsed, f"EXC: {type(exc).__name__}: {exc}"


def header(label: str) -> None:
    print(f"\n=== {label} ===")


def main() -> None:
    print(f"Endpoint: {BASE}")
    print(f"Image:    {TEST_IMG}  exists={TEST_IMG.exists()}")

    header("1. GET /  (root)")
    code, dt, body = call("GET", "/")
    print(f"  HTTP {code} in {dt:.2f}s")
    print(f"  body: {body}")

    header("2. GET /v1/models")
    code, dt, body = call("GET", "/v1/models")
    print(f"  HTTP {code} in {dt:.2f}s")
    print(f"  body: {body}")

    header("3. POST /v1/chat/completions  text-only")
    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    code, dt, body = call("POST", "/v1/chat/completions", json_body=payload)
    print(f"  HTTP {code} in {dt:.2f}s")
    print(f"  body: {body}")

    header("3b. POST /v1/chat/completions  text-only  (no Bearer)")
    code, dt, body = call("POST", "/v1/chat/completions", json_body=payload, auth=ALT_AUTH)
    print(f"  HTTP {code} in {dt:.2f}s")
    print(f"  body: {body}")

    header("4. POST /v1/chat/completions  text + image")
    if TEST_IMG.exists():
        img_b64 = base64.b64encode(TEST_IMG.read_bytes()).decode()
        payload_img = {
            "model": TEST_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with the single word: ok"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        code, dt, body = call("POST", "/v1/chat/completions", json_body=payload_img, timeout=120)
        print(f"  HTTP {code} in {dt:.2f}s")
        print(f"  body: {body}")
    else:
        print(f"  SKIP: {TEST_IMG} not found")

    header("5. POST /v1/completions  (alt endpoint)")
    payload_c = {
        "model": TEST_MODEL,
        "prompt": "pong",
        "max_tokens": 8,
        "temperature": 0.0,
    }
    code, dt, body = call("POST", "/v1/completions", json_body=payload_c)
    print(f"  HTTP {code} in {dt:.2f}s")
    print(f"  body: {body}")

    header("6. GET /health")
    code, dt, body = call("GET", "/health")
    print(f"  HTTP {code} in {dt:.2f}s")
    print(f"  body: {body}")

    print("\n=== Done. Look for HTTP 200 responses above. ===")


if __name__ == "__main__":
    main()
