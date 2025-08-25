"""CLI entry-point to execute validation experiments.

Usage:
    python -m validation.run --dataset TruthfulQA
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ORION validation suite")
    parser.add_argument(
        "--dataset",
        default="TruthfulQA",
        help="Dataset to evaluate (e.g., TruthfulQA, HumanEval)",
    )
    parser.add_argument(
        "--log-dir", default="validation/logs", help="Directory to store logs"
    )
    args = parser.parse_args()

    run_all(dataset=args.dataset, log_dir=Path(args.log_dir))


if __name__ == "__main__":
    main()
