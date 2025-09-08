"""Recursive loop prevention scheduler with optional Prometheus metrics."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Gauge, Histogram, REGISTRY
except Exception:  # pragma: no cover - optional dependency
    class _NoopMetric:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def inc(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set(self, *args: Any, **kwargs: Any) -> None:
            return None

        def observe(self, *args: Any, **kwargs: Any) -> None:
            return None

    Counter = Gauge = Histogram = _NoopMetric  # type: ignore[misc,assignment]
    REGISTRY = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional config import
    from src.orion.config import SchedulerConfig  # type: ignore
except Exception:  # pragma: no cover
    try:
        from orion.config import SchedulerConfig  # type: ignore
    except Exception:  # pragma: no cover
        @dataclass
        class SchedulerConfig:  # type: ignore[misc]
            depth_cost_gamma: float = 1.5
            base_cost: float = 1.0
            beam_size: int = 10
            max_depth: int = 20
            wall_time_limit: float = 300.0
            memory_limit_mb: float = 1024.0
            token_limit: float = 100000
            c_explore: float = 1.41
            cost_penalty: float = 0.1
            marginal_gain_epsilon: float = 0.01
            convergence_window: int = 3
            min_entropy: float = 0.1
            entropy_convergence_window: int = 5

def _get_metric(metric_cls: Any, name: str, documentation: str):
    if REGISTRY is None:  # type: ignore[truthy-function]
        return metric_cls(name, documentation)
    try:
        return metric_cls(name, documentation)
    except ValueError:
        existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)  # type: ignore[attr-defined]
        if existing is not None:
            return existing
        raise


RECURSION_DEPTH = _get_metric(Gauge, "orion_recursion_depth_sched", "Current recursion depth")
CREDITS_REMAINING = _get_metric(Gauge, "orion_credits_remaining_sched", "Remaining recursion credits")
LOOP_DETECTIONS = _get_metric(Counter, "orion_loop_detections_total_sched", "Total loop detections")
BEAM_SIZE_CURRENT = _get_metric(Gauge, "orion_beam_size_current_sched", "Current beam frontier size")
RESOURCE_EFFICIENCY = _get_metric(Histogram, "orion_resource_efficiency_sched", "Value per cost ratio")


@dataclass
class RecursionBudget:
    base_cost: float = 1.0
    depth_cost_gamma: float = 1.5
    remaining_credits: float = 100.0

    def cost(self, depth: int) -> float:
        return self.base_cost * (self.depth_cost_gamma ** depth)

    def can_afford(self, depth: int) -> bool:
        return self.remaining_credits >= self.cost(depth)

    def spend(self, depth: int) -> float:
        cost = self.cost(depth)
        self.remaining_credits -= cost
        return self.remaining_credits


@dataclass(order=True)
class RecursionNode:
    value: float
    state: Any = field(compare=False)
    cost: float = field(default=0.0, compare=False)
    depth: int = field(default=0, compare=False)
    parent: Optional["RecursionNode"] = field(default=None, compare=False)

    def path(self) -> List[Any]:
        node: Optional["RecursionNode"] = self
        out: List[Any] = []
        while node is not None:
            out.append(node.state)
            node = node.parent
        return list(reversed(out))


class CycleDetector:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    @staticmethod
    def _hash_state(state: Any) -> str:
        norm = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(norm.encode()).hexdigest()

    def is_cycle(self, state: Any) -> bool:
        key = self._hash_state(state)
        if key in self.seen:
            return True
        self.seen.add(key)
        return False


class BeamSearchFrontier:
    def __init__(self, beam_size: int) -> None:
        self.beam_size = beam_size
        self._heap: List[Tuple[float, RecursionNode]] = []

    def add(self, node: RecursionNode) -> None:
        heapq.heappush(self._heap, (-node.value, node))
        if len(self._heap) > self.beam_size:
            heapq.heappop(self._heap)

    def pop(self) -> RecursionNode:
        return heapq.heappop(self._heap)[1]

    def best(self) -> Optional[RecursionNode]:
        if not self._heap:
            return None
        return max(self._heap, key=lambda x: x[1].value)[1]

    def __len__(self) -> int:
        return len(self._heap)


class EntropyGate:
    def __init__(self, min_entropy: float, convergence_window: int) -> None:
        self.min_entropy = min_entropy
        self.history: Deque[float] = deque(maxlen=convergence_window)

    def should_continue(self, entropy: float) -> bool:
        self.history.append(entropy)
        if len(self.history) < self.history.maxlen:
            return True
        if max(self.history) < self.min_entropy:
            if all(
                self.history[i] >= self.history[i - 1] for i in range(1, len(self.history))
            ):
                return False
        return True


class RecursiveScheduler:
    def __init__(
        self,
        beam_size: int = 10,
        max_depth: int = 20,
        budget_config: Optional[RecursionBudget] = None,
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        self.config = config or SchedulerConfig(beam_size=beam_size, max_depth=max_depth)
        self.beam_size = self.config.beam_size
        self.max_depth = self.config.max_depth
        self.budget = budget_config or RecursionBudget(
            base_cost=self.config.base_cost,
            depth_cost_gamma=self.config.depth_cost_gamma,
            remaining_credits=self.config.token_limit,
        )
        self.expansion_count = 0
        self.loop_detections = 0

    def schedule_recursion(
        self,
        initial_state: Any,
        expand_function: Callable[[RecursionNode], Iterable[RecursionNode]],
        epsilon: float | None = None,
    ) -> Optional[RecursionNode]:
        frontier = BeamSearchFrontier(self.beam_size)
        root = RecursionNode(state=initial_state, value=0.0, cost=0.0, depth=0)
        frontier.add(root)
        best = root
        cycle_detector = CycleDetector()
        entropy_gate = EntropyGate(
            self.config.min_entropy, self.config.entropy_convergence_window
        )
        start_time = time.time()
        while len(frontier) > 0:
            if time.time() - start_time > self.config.wall_time_limit:
                logger.info("wall time exceeded; stopping search")
                break
            node = frontier.pop()
            RECURSION_DEPTH.set(node.depth)
            CREDITS_REMAINING.set(self.budget.remaining_credits)
            if node.value > best.value:
                best = node
            if node.depth >= self.max_depth:
                continue
            if not self.budget.can_afford(node.depth + 1):
                continue
            children = list(expand_function(node))
            self.expansion_count += 1
            entropy = math.log2(len(children)) if children else 0.0
            for child in children:
                if cycle_detector.is_cycle(child.state):
                    LOOP_DETECTIONS.inc()
                    self.loop_detections += 1
                    continue
                if not self.budget.can_afford(child.depth):
                    continue
                efficiency = child.value / child.cost if child.cost else child.value
                RESOURCE_EFFICIENCY.observe(efficiency)
                frontier.add(child)
            BEAM_SIZE_CURRENT.set(len(frontier))
            self.budget.spend(node.depth + 1)
            if not entropy_gate.should_continue(entropy):
                break
        return best

    def get_metrics(self) -> dict[str, float]:
        return {
            "expansions": self.expansion_count,
            "loops": self.loop_detections,
            "remaining_credits": self.budget.remaining_credits,
        }


def example_expand_function(node: RecursionNode) -> List[RecursionNode]:
    children: List[RecursionNode] = []
    for i in range(2):
        child = RecursionNode(
            state=f"{node.state}-{i}",
            value=node.value + i + 1,
            cost=1.0,
            depth=node.depth + 1,
            parent=node,
        )
        children.append(child)
    return children


if __name__ == "__main__":  # pragma: no cover - smoke test
    scheduler = RecursiveScheduler()
    best = scheduler.schedule_recursion("root", example_expand_function)
    logger.info("Best path: %s value=%s", best.path(), best.value if best else None)


__all__ = [
    "RecursionBudget",
    "RecursionNode",
    "CycleDetector",
    "BeamSearchFrontier",
    "EntropyGate",
    "RecursiveScheduler",
    "SchedulerConfig",
    "example_expand_function",
]
