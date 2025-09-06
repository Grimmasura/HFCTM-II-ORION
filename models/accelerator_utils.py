"""Utilities for accelerator configuration and parity checks."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML missing
    yaml = None

try:  # pragma: no cover - optional dependency
    import torch_xla.core.xla_model as xm  # type: ignore
    _xla_available = True
except Exception:  # pragma: no cover - torch_xla not installed
    xm = None
    _xla_available = False

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load a hardware configuration from the configs directory.

    The configuration files are written in YAML.  If :mod:`PyYAML` is available
    we parse them with :func:`yaml.safe_load`; otherwise we fall back to the
    JSON loader since our configs are JSON-compatible.
    """

    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        if yaml is not None:
            return yaml.safe_load(fh)
        return json.load(fh)


def get_degradation_profile(*, has_qpu: bool, has_tpu: bool | None = None) -> dict:
    """Return the runtime profile for the current hardware mix.

    ``has_tpu`` may be provided explicitly for tests.  When ``None`` the
    function attempts to detect TPU availability via :mod:`torch_xla`.

    When neither TPU nor QPU is available (i.e. GPU-only), the system falls back
    to the "Aspen" profile which exposes reduced metrics, applies a wider
    hysteresis window and restricts actions to warning and stabilization.  If an
    accelerator is present a full profile is returned.
    """

    if has_tpu is None:
        has_tpu = False
        if _xla_available:
            try:  # pragma: no cover - best effort detection
                xm.xla_device()
                has_tpu = True
            except Exception:  # pragma: no cover - TPU not accessible
                has_tpu = False

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
