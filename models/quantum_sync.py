"""Utilities for simulating quantum synchronization events."""

from __future__ import annotations

import random


def get_sync_status() -> dict[str, float | str]:
    """Return a mocked status of the quantum sync subsystem."""

    return {
        "status": "operational",
        "coherence_level": round(random.uniform(0.8, 1.0), 3),
    }
