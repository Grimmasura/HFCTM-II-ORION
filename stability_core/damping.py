from __future__ import annotations

from .config import DampingConfig


class AdaptiveDamping:
    """Apply adaptive damping based on a Lyapunov metric."""

    def __init__(self, config: DampingConfig) -> None:
        self.config = config

    def apply(self, state: float, lambda_val: float) -> tuple[float, float]:
        """Dampen ``state`` using ``lambda_val`` and return (new_state, factor)."""

        factor = 1.0 / (1.0 + self.config.base * lambda_val)
        return state * factor, factor
