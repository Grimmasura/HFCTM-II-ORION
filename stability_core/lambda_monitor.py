"""Lyapunov exponent monitoring."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel


class LyapunovConfig(BaseModel):
    """Configuration for :class:`LyapunovMonitor`."""

    epsilon: float = 1e-8


class LyapunovMonitor:
    """Tracks a crude Lyapunov metric of state divergence.

    The monitor stores the previous state and computes the L2 distance to the
    new state each time it is called.  This value can be interpreted as a
    proxy for the Lyapunov exponent in simple simulations.
    """

    def __init__(self, config: LyapunovConfig | None = None):
        self.config = config or LyapunovConfig()
        self._prev_state: np.ndarray | None = None

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Return the input state and Lyapunov metric.

        Parameters
        ----------
        state:
            Current state vector.

        Returns
        -------
        Tuple containing the original state and a metrics dictionary with the
        key ``"lyapunov"``.
        """

        arr = np.asarray(state, dtype=float)
        if self._prev_state is None:
            metric = 0.0
        else:
            metric = float(np.linalg.norm(arr - self._prev_state))
        self._prev_state = arr.copy()
        return arr, {"lyapunov": metric}
