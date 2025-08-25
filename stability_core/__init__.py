"""Stability core package with modular heads."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel

from .lambda_monitor import LyapunovConfig, LyapunovMonitor
from .damping import DampingConfig, AdaptiveDamping
from .chiral import ChiralConfig, ChiralInversion
from .wavelet import WaveletConfig, WaveletScanner
from .e8_anchor import E8Config, E8Anchor

try:  # Optional dependency
    import yaml
except ImportError:  # pragma: no cover - handled in tests
    yaml = None


class StabilityConfig(BaseModel):
    """Pydantic configuration for :class:`StabilityCore`."""

    lyapunov: LyapunovConfig = LyapunovConfig()
    damping: DampingConfig = DampingConfig()
    chiral: ChiralConfig = ChiralConfig()
    wavelet: WaveletConfig = WaveletConfig()
    e8: E8Config = E8Config()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StabilityConfig":
        """Load configuration from a YAML file."""
        if yaml is None:  # pragma: no cover - environment dependent
            raise ImportError("PyYAML is required to load YAML configs")
        with open(path, "r", encoding="utf8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


class StabilityCore:
    """Coordinates all stability heads in a sequential step."""

    def __init__(self, config: StabilityConfig | None = None):
        self.config = config or StabilityConfig()
        self.lyapunov = LyapunovMonitor(self.config.lyapunov)
        self.damping = AdaptiveDamping(self.config.damping)
        self.chiral = ChiralInversion(self.config.chiral)
        self.wavelet = WaveletScanner(self.config.wavelet)
        self.e8 = E8Anchor(self.config.e8)

    def step(self, state: np.ndarray, latents: np.ndarray | None = None) -> Tuple[np.ndarray, Dict[str, object]]:
        """Run a single update step.

        Parameters
        ----------
        state:
            Current state vector.
        latents:
            Unused placeholder for compatibility with larger systems.
        """

        metrics: Dict[str, object] = {}
        state, m = self.lyapunov(state)
        metrics.update(m)
        state, m = self.damping(state, metrics.get("lyapunov", 0.0))
        metrics.update(m)
        state, m = self.chiral(state)
        metrics.update(m)
        state, m = self.wavelet(state)
        metrics.update(m)
        state, m = self.e8(state)
        metrics.update(m)
        return state, metrics


__all__ = [
    "LyapunovConfig",
    "LyapunovMonitor",
    "DampingConfig",
    "AdaptiveDamping",
    "ChiralConfig",
    "ChiralInversion",
    "WaveletConfig",
    "WaveletScanner",
    "E8Config",
    "E8Anchor",
    "StabilityConfig",
    "StabilityCore",
]
