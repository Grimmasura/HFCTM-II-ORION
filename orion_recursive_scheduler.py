"""Recursive scheduler implementing a simple leaky-bucket credit system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RecursiveScheduler:
    c0: float = 1.0
    gamma: float = 0.9
    credits: Dict[int, float] = field(default_factory=dict)

    def credit(self, agent: int = 0, task_cost: float = 1.0) -> float:
        """Update and return remaining credits for an agent."""
        prev = self.credits.get(agent, self.c0)
        new = self.gamma * prev - task_cost
        self.credits[agent] = new
        return new


__all__ = ["RecursiveScheduler"]
