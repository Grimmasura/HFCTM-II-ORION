from __future__ import annotations

from .config import E8Config


class E8Anchor:
    """Optional projection into an E8-like lattice.

    This is a placeholder; the projection simply wraps the value within a small
    range when enabled.
    """

    def __init__(self, config: E8Config) -> None:
        self.config = config

    def project(self, state: float) -> float:
        """Project ``state`` if enabled."""

        if not self.config.enabled:
            return state
        return state % 8
