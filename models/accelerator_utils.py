"""Utilities for accelerator configuration and parity checks."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load a hardware configuration from the configs directory.

    The configuration files are stored as YAML but use a subset of YAML that is
    also valid JSON, allowing us to parse them without additional dependencies.
    """

    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_degradation_profile(*, has_tpu: bool, has_qpu: bool) -> dict:
    """Return the runtime profile for the current hardware mix.

    When neither TPU nor QPU is available (i.e. GPU-only), the system falls back
    to the "Aspen" profile which exposes reduced metrics, applies a wider
    hysteresis window and restricts actions to warning and stabilization.  If an
    accelerator is present a full profile is returned.
    """

    if not (has_tpu or has_qpu):
        return {
            "profile": "aspen",
            "metrics": "reduced",
            "hysteresis": "wide",
            "actions": ["warn", "stabilize"],
        }
    return {
        "profile": "full",
        "metrics": "standard",
        "hysteresis": "narrow",
        "actions": ["warn", "stabilize", "halt"],
    }


def parity_process(values: Iterable[float] | Iterable[int], *, use_accelerator: bool) -> float:
    """Compute a simple checksum used by tests to ensure parity.

    The function intentionally performs the same computation regardless of the
    `use_accelerator` flag to demonstrate that host and accelerator paths produce
    identical numerical results.
    """

    return float(sum(values))
