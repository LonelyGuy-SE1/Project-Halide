"""Spawn a deployed Modal training function and record the call id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-name", default="halide-vision-training-v4")
    parser.add_argument("--function-name", default="train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--train-json-path",
        default="/data/augmented/training_sharegpt_combined_v4.json",
    )
    parser.add_argument("--val-json-path", default="/data/training_sharegpt_val_v4.json")
    parser.add_argument("--output-dir", default="/checkpoints/minicpm-v-4.6-lora-v4-stage1")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--out", default=".nottracked/train_call.json")
    args = parser.parse_args()

    fn = modal.Function.from_name(args.app_name, args.function_name)
    call = fn.spawn(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_json_path=args.train_json_path,
        val_json_path=args.val_json_path,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    record = {
        "call_id": call.object_id,
        "dashboard_url": call.get_dashboard_url(),
        "app_name": args.app_name,
        "function_name": args.function_name,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
