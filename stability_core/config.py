from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel


class LambdaConfig(BaseModel):
    """Settings for Lyapunov monitoring."""

    threshold: float = 1.0


class DampingConfig(BaseModel):
    """Settings for adaptive damping."""

    base: float = 0.1


class ChiralConfig(BaseModel):
    """Settings for chiral inversion."""

    invert: bool = True


class WaveletConfig(BaseModel):
    """Settings for wavelet anomaly scanner."""

    threshold: float = 0.5


class E8Config(BaseModel):
    """Settings for optional E8 projection."""

    enabled: bool = False


class StabilityConfig(BaseModel):
    """Aggregate configuration for :class:`StabilityCore`."""

    lambda_monitor: LambdaConfig = LambdaConfig()
    damping: DampingConfig = DampingConfig()
    chiral: ChiralConfig = ChiralConfig()
    wavelet: WaveletConfig = WaveletConfig()
    e8_anchor: E8Config = E8Config()


def load_config(path: Union[str, Path]) -> StabilityConfig:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML file containing a ``StabilityConfig`` specification.
    """

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return StabilityConfig(**data)
