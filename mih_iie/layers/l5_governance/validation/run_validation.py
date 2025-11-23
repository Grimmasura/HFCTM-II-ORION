"""CLI entrypoint for running validation experiments S1-S6.

The script exposes a thin wrapper around :mod:`validation.experiments`
that allows callers to specify datasets for each experiment and produces a
CSV decision matrix with the recorded KPIs.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

from .experiments import run_experiments, write_decision_matrix


def parse_dataset_map(entries: list[str]) -> Dict[str, str]:
    """Parse ``EXP=DATASET`` assignments from the CLI."""

    dataset_map: Dict[str, str] = {}
    for item in entries:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid dataset mapping '{item}'. Expected EXP=DATASET"
            )
        exp, dataset = item.split("=", 1)
        dataset_map[exp] = dataset
    return dataset_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run validation experiments")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset mapping in the form EXP=NAME. Use 'default=NAME' to set a default",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("validation", "results", "decision_matrix.csv"),
        help="Path for the decision matrix CSV",
    )
    args = parser.parse_args(argv)

    dataset_map = parse_dataset_map(args.dataset)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = run_experiments(dataset_map)
    write_decision_matrix(results, args.output)

    logging.info("Decision matrix written to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
