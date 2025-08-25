from __future__ import annotations

from .config import LambdaConfig


class LambdaMonitor:
    """Simple Lyapunov-like monitor.

    Computes a distance between the current state and latent target. This is a
    placeholder for more sophisticated Lyapunov exponent calculations.
    """

    def __init__(self, config: LambdaConfig) -> None:
        self.config = config

    def compute(self, state: float, latents: float) -> float:
        """Return a Lyapunov metric between ``state`` and ``latents``."""

        return abs(state - latents)
