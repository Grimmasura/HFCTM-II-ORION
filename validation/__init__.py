"""Validation experiment package for ORION.

This package contains scripts to execute post-training validation experiments
for different model configurations and datasets.

For convenience, the key entry points are re-exported at the package level:

``run_experiments``
    Execute all configured experiments and return their results.
``write_decision_matrix``
    Persist the collected results to a CSV file for reporting.
"""

from .experiments import run_experiments, write_decision_matrix

__all__ = ["run_experiments", "write_decision_matrix"]
