"""Summarize LLaMA-Factory metric dictionaries from a training log."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def parse_metrics(log_text: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("{") or "epoch" not in line:
            continue
        try:
            parsed = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            metrics.append(parsed)
    return metrics


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    train = [m for m in metrics if "loss" in m]
    evals = [m for m in metrics if "eval_loss" in m]
    summary: dict[str, Any] = {
        "train_points": len(train),
        "eval_points": len(evals),
        "latest_train": train[-1] if train else None,
        "latest_eval": evals[-1] if evals else None,
    }
    if evals:
        best = min(evals, key=lambda m: float(m["eval_loss"]))
        summary["best_eval"] = best
    if train:
        summary["first_train_loss"] = float(train[0]["loss"])
        summary["latest_train_loss"] = float(train[-1]["loss"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    summary = summarize_metrics(parse_metrics(text))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"train points: {summary['train_points']}")
        print(f"eval points:  {summary['eval_points']}")
        print(f"latest train: {summary['latest_train']}")
        print(f"latest eval:  {summary['latest_eval']}")
        print(f"best eval:    {summary.get('best_eval')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
