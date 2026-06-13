"""Combine ShareGPT JSON files in order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    combined: list[dict] = []
    for path_str in args.inputs:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{path} did not contain a JSON list")
        combined.extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"wrote {len(combined)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
