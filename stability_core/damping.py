"""Adaptive damping mechanisms."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel


class DampingConfig(BaseModel):
    """Configuration for :class:`AdaptiveDamping`."""

    base: float = 0.1
    gain: float = 1.0


class AdaptiveDamping:
    """Applies simple adaptive damping based on a Lyapunov metric."""

    def __init__(self, config: DampingConfig | None = None):
        self.config = config or DampingConfig()

    def __call__(self, state: np.ndarray, lyapunov: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """Dampen the state proportionally to the Lyapunov metric."""

        arr = np.asarray(state, dtype=float)
        factor = self.config.base / (1.0 + self.config.gain * abs(lyapunov))
        damped = arr * (1.0 - factor)
        return damped, {"damping_factor": factor}
