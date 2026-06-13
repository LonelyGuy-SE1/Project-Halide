"""Reasoning training policy for Project Halide.

Nemotron-Mini-4B-Instruct is used with few-shot prompting only. This module is
kept as an explicit guardrail so nobody launches an accidental reasoning
fine-tune and uses GPU budget on work outside the architecture.
"""

from __future__ import annotations


def main() -> int:
    print("No reasoning fine-tune is configured.")
    print("Use models/reasoning/prompts.py for few-shot prompt iteration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
