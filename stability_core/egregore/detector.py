import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional

from . import metrics as m


@dataclass
class MetricSpec:
    func: Callable[..., float]
    low: float
    high: float
    dwell: float = 0.0
    state: str = "normal"
    timer: float = 0.0
    value: float = 0.0


class CompositeDetector:
    """Hysteresis detector with dwell timers for multiple metrics."""

    def __init__(self, metrics: Dict[str, MetricSpec]):
        self.metrics = metrics
        self.last_check = time.time()

    def update(self, **kwargs) -> Dict[str, float]:
        now = time.time()
        dt = now - self.last_check
        self.last_check = now
        results = {}
        for name, spec in self.metrics.items():
            # Only pass arguments required by the metric function
            sig = getattr(spec.func, "__signature", None)
            if sig is None:
                import inspect

                sig = inspect.signature(spec.func)
            params = {k: kwargs[k] for k in sig.parameters if k in kwargs}
            spec.value = spec.func(**params)
            if spec.state == "normal":
                if spec.value > spec.high:
                    spec.timer += dt
                    if spec.timer >= spec.dwell:
                        spec.state = "alert"
                        spec.timer = 0.0
                else:
                    spec.timer = 0.0
            else:  # alert
                if spec.value < spec.low:
                    spec.timer += dt
                    if spec.timer >= spec.dwell:
                        spec.state = "normal"
                        spec.timer = 0.0
                else:
                    spec.timer = 0.0
            results[name] = spec.value
        return results

    def state(self) -> Dict[str, str]:
        return {n: s.state for n, s in self.metrics.items()}


class HashChainLogger:
    def __init__(self, path: str = "egregore.log"):
        self.path = path
        self.last_hash = "0"
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    h = line.strip().split(" ", 1)[0]
                    if h:
                        self.last_hash = h

    def log(self, message: str) -> str:
        data = f"{self.last_hash}{message}".encode()
        h = hashlib.sha256(data).hexdigest()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"{h} {message}\n")
        self.last_hash = h
        return h

    def read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh]


METRICS = {
    "phase_order": MetricSpec(m.phase_order, low=0.7, high=0.9),
    "spectral_capacity": MetricSpec(m.spectral_capacity, low=0.5, high=0.7),
    "mutual_information": MetricSpec(m.mutual_information, low=0.1, high=0.2),
    "wavelet_burst_energy": MetricSpec(m.wavelet_burst_energy, low=0.2, high=0.4),
    "entropy_slope": MetricSpec(m.entropy_slope, low=0.0, high=0.1),
}

DETECTOR = CompositeDetector(METRICS)
LOGGER = HashChainLogger()


def EgDetect(**data) -> Dict[str, float]:
    """Evaluate metrics and update detector; returns metric values."""
    values = DETECTOR.update(**data)
    if any(state == "alert" for state in DETECTOR.state().values()):
        EgMitigate()
    return values


def EgState() -> Dict[str, str]:
    """Current state of each metric."""
    return DETECTOR.state()


def EgMitigate() -> Optional[str]:
    """Apply mitigation based on detector state."""
    states = DETECTOR.state()
    alerts = sum(s == "alert" for s in states.values())
    action = None
    if alerts >= 3:
        action = "escalate"
    elif alerts == 2:
        action = "stabilize"
    elif alerts == 1:
        action = "warn"
    if action:
        LOGGER.log(action)
    return action


def EgAudit():
    """Return hash-chained log entries."""
    return LOGGER.read()

