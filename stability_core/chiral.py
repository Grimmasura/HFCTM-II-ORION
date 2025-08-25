from __future__ import annotations

from .config import ChiralConfig


class ChiralInversion:
    """Optional sign inversion to simulate chirality effects."""

    def __init__(self, config: ChiralConfig) -> None:
        self.config = config

    def apply(self, state: float) -> float:
        """Return ``-state`` if inversion is enabled."""

        return -state if self.config.invert else state
