"""Chiral inversion utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel


class ChiralConfig(BaseModel):
    """Configuration for :class:`ChiralInversion`."""

    enabled: bool = True


class ChiralInversion:
    """Flips the sign of every other component to simulate chirality."""

    def __init__(self, config: ChiralConfig | None = None):
        self.config = config or ChiralConfig()

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        arr = np.asarray(state, dtype=float)
        if not self.config.enabled:
            return arr, {"chirality": False}
        flipped = arr.copy()
        flipped[..., 1::2] *= -1
        return flipped, {"chirality": True}
