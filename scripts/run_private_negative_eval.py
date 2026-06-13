"""Run private held-out negatives through Modal GPU inference.

This script performs only local file I/O, Modal CLI calls, JSON cleanup, and
overlay drawing. It does not load a model locally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing import draw_defects, load_image
from data.schemas import clean_defects, dedupe_defects, label_counts

DEFAULT_MODEL = "/checkpoints/minicpm-v-4.6-merged-v4-stage1"
DEFAULT_VOLUME = "halide-viewer-uploads"
RESULT_START = "===RESULT_START==="
RESULT_END = "===RESULT_END==="


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    home = env.get("HOME") or env.get("USERPROFILE") or str(Path.home())
    env["HOME"] = home
    env["USERPROFILE"] = env.get("USERPROFILE") or home
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=_subprocess_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_subprocess_text(exc.stdout)
        stderr = _coerce_subprocess_text(exc.stderr)
        raise RuntimeError(
            f"command timed out after {timeout}s: {' '.join(cmd)}\n"
            f"stdout tail:\n{stdout[-2000:]}\n"
            f"stderr tail:\n{stderr[-4000:]}"
        ) from exc


def _coerce_subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def upload_to_modal(local_path: Path, *, volume: str, prefix: str) -> str:
    remote_name = f"{prefix}_{local_path.stem}_{int(time.time())}{local_path.suffix.lower()}"
    proc = _run(
        ["modal", "volume", "put", volume, str(local_path), remote_name],
        timeout=10 * 60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"modal volume put failed for {local_path}: {proc.stderr[-2000:]}"
        )
    return remote_name


def parse_modal_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find(RESULT_START)
    end = stdout.find(RESULT_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"result markers missing from Modal stdout: {stdout[-1200:]}")
    payload = stdout[start + len(RESULT_START) : end].strip()
    return json.loads(payload)


def run_modal_inference(
    remote_name: str,
    *,
    model: str,
    which: str,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        "modal",
        "run",
        "modal/view_inference.py::main",
        "--image-path",
        remote_name,
        "--which",
        which,
    ]
    if which in {"both", "finetuned"}:
        cmd.extend(["--finetuned-model", model])
    proc = _run(cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"modal run failed for {remote_name}: {proc.stderr[-4000:]}")
    return parse_modal_stdout(proc.stdout)


def select_model_result(payload: dict[str, Any], *, which: str) -> dict[str, Any]:
    if which == "both":
        return payload.get("results", {}).get("finetuned", {})
    return payload


def clean_model_result(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    parsed = result.get("parsed_json", {})
    raw_defects = parsed.get("defects", []) if isinstance(parsed, dict) else []
    cleaned, dropped_invalid = clean_defects(raw_defects)
    deduped, dropped_duplicates = dedupe_defects(cleaned)
    dropped = {
        "invalid": dropped_invalid,
        "duplicates": dropped_duplicates,
    }
    return deduped, dropped


def write_image_report(
    image_path: Path,
    result: dict[str, Any],
    defects: list[dict[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    pil = load_image(image_path)
    overlay = draw_defects(
        pil,
        defects,
        title=f"{image_path.name}: {len(defects)} defects",
    )
    overlay_path = out_dir / f"{image_path.stem}_overlay.png"
    overlay.save(overlay_path)

    raw_path = out_dir / f"{image_path.stem}_raw.json"
    raw_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = label_counts(defects)
    report = {
        "image": image_path.name,
        "size": list(pil.size),
        "model_id": result.get("model_id"),
        "device": result.get("device"),
        "load_seconds": result.get("load_seconds"),
        "inference_seconds": result.get("inference_seconds"),
        "defect_count": len(defects),
        "label_counts": counts,
        "overlay": str(overlay_path),
        "raw_result": str(raw_path),
    }
    return report


def resolve_images(input_dir: Path, pattern: str, explicit: list[str]) -> list[Path]:
    if explicit:
        images = [Path(p) for p in explicit]
    else:
        images = sorted(input_dir.glob(pattern))
    missing = [str(p) for p in images if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing input images: {missing}")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run private held-out negatives through Modal GPU inference."
    )
    parser.add_argument("--input-dir", default=".nottracked")
    parser.add_argument("--pattern", default="negative*.png")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--out-dir", default=".nottracked/user_negative_results_v4_stage1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--which", choices=["finetuned", "base", "both"], default="finetuned")
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--remote-prefix", default="heldout")
    parser.add_argument("--timeout", type=int, default=20 * 60)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = resolve_images(input_dir, args.pattern, args.image)
    if not images:
        raise SystemExit(f"no images matched {input_dir / args.pattern}")

    summary: dict[str, Any] = {
        "model": args.model,
        "which": args.which,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "images": [],
    }
    predictions_path = out_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as pred_file:
        for image_path in images:
            print(f"[halide] uploading {image_path}")
            remote_name = upload_to_modal(
                image_path,
                volume=args.volume,
                prefix=args.remote_prefix,
            )
            print(f"[halide] running {args.which} on {remote_name}")
            payload = run_modal_inference(
                remote_name,
                model=args.model,
                which=args.which,
                timeout=args.timeout,
            )
            result = select_model_result(payload, which=args.which)
            defects, dropped = clean_model_result(result)
            report = write_image_report(image_path, result, defects, out_dir=out_dir)
            report["remote_name"] = remote_name
            report["dropped"] = dropped
            summary["images"].append(report)
            pred_file.write(
                json.dumps(
                    {
                        "image": image_path.name,
                        "predictions": defects,
                        "label_counts": report["label_counts"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                f"[halide] {image_path.name}: {len(defects)} defects, "
                f"{report['label_counts']}"
            )

    summary_path = out_dir / "summary.json"
    summary["predictions_jsonl"] = str(predictions_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[halide] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
