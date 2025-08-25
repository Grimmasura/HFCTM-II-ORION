from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .metrics import (
    phase_order,
    spectral_capacity,
    mutual_information,
    wavelet_burst_energy,
    entropy_slope,
)
from .detector import CompositeDetector

# Default thresholds for metrics: (warn, escalate)
_THRESHOLDS = {
    "phase_order": (0.7, 0.9),
    "spectral_capacity": (1.5, 2.0),
    "mutual_info": (0.3, 0.5),
    "wavelet_energy": (10.0, 20.0),
    "entropy_slope": (0.01, 0.02),
}

_detector = CompositeDetector(_THRESHOLDS, dwell=3)

_log_file = Path(__file__).resolve().with_name("egregore.log")
if _log_file.exists():
    try:
        last_line = _log_file.read_text().splitlines()[-1]
        _last_hash = last_line.split()[0]
    except Exception:  # pragma: no cover
        _last_hash = "0"
else:
    _last_hash = "0"


def EgAudit(message: str) -> str:
    """Append message to hash-chained log and return digest."""
    global _last_hash
    record = f"{message}|prev={_last_hash}"
    digest = hashlib.sha256(record.encode()).hexdigest()
    with _log_file.open("a") as fh:
        fh.write(f"{digest} {record}\n")
    _last_hash = digest
    return digest


def EgMitigate(level: str) -> Dict[str, Any]:
    """Perform mitigation action (warn, stabilize, escalate) and audit it."""
    EgAudit(f"mitigation:{level}")
    return {"action": level}


def EgDetect(
    signal: np.ndarray, matrix: np.ndarray, other: np.ndarray
) -> Dict[str, Any]:
    """Compute metrics, run detector, and return metrics and action."""
    metrics = {
        "phase_order": phase_order(signal),
        "spectral_capacity": spectral_capacity(matrix),
        "mutual_info": mutual_information(signal, other),
        "wavelet_energy": wavelet_burst_energy(signal),
        "entropy_slope": entropy_slope(signal),
    }
    action = _detector.check(metrics)
    if action:
        EgMitigate(action)
    return {"metrics": metrics, "action": action}


def EgState() -> str:
    """Return current detector state."""
    return _detector.state


__all__ = [
    "EgDetect",
    "EgState",
    "EgMitigate",
    "EgAudit",
    "CompositeDetector",
    "phase_order",
    "spectral_capacity",
    "mutual_information",
    "wavelet_burst_energy",
    "entropy_slope",
]
