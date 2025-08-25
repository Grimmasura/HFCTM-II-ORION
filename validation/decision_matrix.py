"""Utilities to build decision matrices and reports from experiment logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DecisionRow:
    experiment: str
    status: str
    notes: str
    next_steps: str


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "overhead": 0.5,
}


def load_results(path: Path) -> List[Dict]:
    """Load experiment results previously written by the runner."""
    results: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            results.append(json.loads(line))
    return results


def build_decision_matrix(
    results: List[Dict], thresholds: Dict[str, float] | None = None
) -> List[DecisionRow]:
    """Derive pass/fail decisions from experiment results."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    matrix: List[DecisionRow] = []
    for item in results:
        status = "pass"
        notes: List[str] = []
        if item["overhead"] > thresholds["overhead"]:
            status = "fail"
            notes.append("overhead above threshold")
        matrix.append(
            DecisionRow(
                experiment=item["config"]["name"],
                status=status,
                notes="; ".join(notes) or "n/a",
                next_steps="investigate" if status == "fail" else "none",
            )
        )
    return matrix


def render_matrix(matrix: List[DecisionRow]) -> str:
    lines = [
        "| Experiment | Status | Notes | Next Steps |",
        "|------------|--------|-------|------------|",
    ]
    for row in matrix:
        lines.append(
            f"| {row.experiment} | {row.status} | {row.notes} | {row.next_steps} |"
        )
    return "\n".join(lines) + "\n"


def save_matrix(matrix: List[DecisionRow], path: Path) -> None:
    """Persist the decision matrix in markdown format."""
    path.write_text(render_matrix(matrix), encoding="utf-8")


def generate_report_template(matrix: List[DecisionRow], path: Path) -> None:
    """Create a full report template including the decision matrix."""
    content = [
        "# Validation Report",
        "",
        "## Decision Matrix",
        "",
        render_matrix(matrix),
        "## Summary",
        "",
        "- Overall status: ",
        "- Notes: ",
        "- Next steps: ",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
