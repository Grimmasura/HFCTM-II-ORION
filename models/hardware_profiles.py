"""Utilities for loading accelerator configs and selecting runtime profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List

try:  # pragma: no cover - handled in tests
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML missing
    yaml = None


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


@dataclass
class HardwareConfig:
    """Configuration for a hardware accelerator."""

    precision_modes: List[str]
    intervals: Dict[str, int]
    sentinel_count: int
    latency_budget_ms: int
    tpu_mesh: Dict[str, int] | None = None
    optional: bool | None = False


def _load_raw(path: str) -> Dict[str, Any]:
    """Load a YAML (or JSON) file into a dictionary."""

    with open(path, "r", encoding="utf-8") as fh:
        if yaml is not None:
            return yaml.safe_load(fh)
        return json.load(fh)


def load_hardware_config(name: str) -> HardwareConfig:
    """Load the configuration for ``name`` from :mod:`configs`.

    Parameters
    ----------
    name:
        Name of the accelerator configuration without extension.
    """

    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    data = _load_raw(path)
    return HardwareConfig(**data)


def select_profile(accelerators: List[str]) -> Dict[str, Any]:
    """Return profile information based on available accelerators.

    GPU-only environments fall back to the ``Aspen`` profile which
    provides reduced metrics and wider hysteresis with limited actions.
    """

    profiles: Dict[str, Any] = {}
    for acc in accelerators:
        if acc in ("ironwood_tpu", "majorana1_qpu"):
            profiles[acc] = load_hardware_config(acc)
    if not profiles:
        profiles["profile"] = "aspen"
        profiles["metrics"] = "reduced"
        profiles["hysteresis"] = "wide"
        profiles["actions"] = ["warn", "stabilize"]
    return profiles


def compute(values: List[int], use_accelerator: bool = False) -> List[int]:
    """Simple computation path used for parity testing."""

    if use_accelerator:
        import numpy as np
        arr = np.array(values)
        return np.square(arr).tolist()
    return [v * v for v in values]


def parity_check(values: List[int]) -> bool:
    """Ensure host and accelerator paths yield identical results."""

    return compute(values, use_accelerator=False) == compute(values, use_accelerator=True)
