from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class CompositeDetector:
    """Composite detector with hysteresis and dwell timers."""

    thresholds: Dict[str, Tuple[float, float]]
    dwell: int = 3
    counters: Dict[str, int] = field(default_factory=dict)
    state: str = "normal"

    def __post_init__(self) -> None:
        self.counters = {k: 0 for k in self.thresholds}

    def check(self, metrics: Dict[str, float]) -> str | None:
        """Evaluate metrics and return mitigation level."""
        action = None
        for key, value in metrics.items():
            warn, escalate = self.thresholds.get(key, (None, None))
            if escalate is not None and value > escalate:
                self.counters[key] += 1
                if self.counters[key] >= self.dwell:
                    action = "escalate"
                    self.counters[key] = 0
                    self.state = "escalate"
            elif warn is not None and value > warn:
                self.counters[key] += 1
                if self.counters[key] >= self.dwell and action != "escalate":
                    action = "stabilize"
                    self.counters[key] = 0
                    self.state = "stabilize"
            else:
                self.counters[key] = 0
        return action
