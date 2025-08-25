from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class CompositeDetector:
    """Composite detector with hysteresis and dwell timers."""

    thresholds: Dict[str, Tuple[float, float]]
    dwell: int = 3
    warn_counters: Dict[str, int] = field(default_factory=dict)
    esc_counters: Dict[str, int] = field(default_factory=dict)
    safe_counter: int = 0
    state: str = "normal"

    def __post_init__(self) -> None:
        self.warn_counters = {k: 0 for k in self.thresholds}
        self.esc_counters = {k: 0 for k in self.thresholds}

    def check(self, metrics: Dict[str, float]) -> str | None:
        """Evaluate metrics and return mitigation level."""
        action = None
        for key, value in metrics.items():
            warn, escalate = self.thresholds.get(key, (None, None))
            if escalate is not None and value > escalate:
                self.esc_counters[key] += 1
            else:
                self.esc_counters[key] = 0

            if warn is not None and value > warn:
                self.warn_counters[key] += 1
            else:
                self.warn_counters[key] = 0

        if any(c >= self.dwell for c in self.esc_counters.values()):
            action = "escalate"
            self.state = "escalate"
            self.safe_counter = 0
            for k in self.warn_counters:
                self.warn_counters[k] = 0
                self.esc_counters[k] = 0
        elif any(c >= self.dwell for c in self.warn_counters.values()):
            action = "warn"
            self.state = "warn"
            self.safe_counter = 0
        else:
            if all(
                metrics[k] < self.thresholds[k][0]
                for k in metrics
                if self.thresholds[k][0] is not None
            ):
                self.safe_counter += 1
                if self.safe_counter >= self.dwell and self.state != "normal":
                    action = "stabilize"
                    self.state = "normal"
                    self.safe_counter = 0
                    for k in self.warn_counters:
                        self.warn_counters[k] = 0
                        self.esc_counters[k] = 0
            else:
                self.safe_counter = 0
        return action
