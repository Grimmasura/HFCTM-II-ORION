"""Lightweight trust scoring utilities for recursive agents."""

from __future__ import annotations


def assess_score(score: int) -> dict[str, int | str]:
    """Return a label describing how trustworthy a score is."""

    if score >= 80:
        level = "high"
    elif score >= 50:
        level = "medium"
    else:
        level = "low"
    return {"score": score, "level": level}
