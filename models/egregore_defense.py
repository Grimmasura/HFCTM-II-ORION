"""Simple threat scoring utilities for the egregore defense subsystem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DefenseResult:
    """Result returned after evaluating a potential threat."""

    threat_score: float
    action: str


def evaluate_threat(threat_score: float) -> DefenseResult:
    """Return a quarantine or allow action based on the threat score.

    Parameters
    ----------
    threat_score:
        Numeric score from ``0`` (safe) to ``1`` (malicious).

    Returns
    -------
    DefenseResult
        The recommended action and threat score.
    """

    action = "quarantine" if threat_score > 0.7 else "allow"
    return DefenseResult(threat_score=threat_score, action=action)
