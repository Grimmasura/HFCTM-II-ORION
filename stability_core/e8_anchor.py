"""Optional projection onto the E8 lattice."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel


class E8Config(BaseModel):
    """Configuration for :class:`E8Anchor`."""

    enabled: bool = False


class E8Anchor:
    """Projects the first eight components onto the E8 root lattice."""

    def __init__(self, config: E8Config | None = None):
        self.config = config or E8Config()

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, list]]:
        arr = np.asarray(state, dtype=float)
        if not self.config.enabled:
            return arr, {"e8_projection": []}

        vec = np.zeros(8)
        vec[: min(8, arr.size)] = arr[:8]
        proj = np.round(vec)
        if proj.sum() % 2 != 0:
            proj[0] += 1
        new_state = arr.copy()
        new_state[:8] = proj[: arr.size]
        return new_state, {"e8_projection": proj.tolist()}
