import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orion.schedule.recursive_scheduler import (
    RecursionBudget,
    CycleDetector,
    EntropyGate,
    RecursiveScheduler,
    RecursionNode,
    SchedulerConfig,
)

import pytest


def test_budget_cost_curve_and_spend() -> None:
    budget = RecursionBudget(base_cost=1.0, depth_cost_gamma=1.5, remaining_credits=5.0)
    assert budget.cost(0) == pytest.approx(1.0)
    assert budget.cost(2) == pytest.approx(1.0 * 1.5 ** 2)
    remaining = budget.spend(1)
    assert remaining == pytest.approx(3.5)
    assert budget.can_afford(2)
    budget.spend(2)
    assert not budget.can_afford(3)


def test_cycle_detector_hash_and_detection() -> None:
    detector = CycleDetector()
    state = {"a": 1}
    assert detector.is_cycle(state) is False
    assert detector.is_cycle(state) is True


def test_entropy_gate_convergence_stop() -> None:
    gate = EntropyGate(min_entropy=0.2, convergence_window=4)
    for _ in range(3):
        assert gate.should_continue(0.1)
    assert gate.should_continue(0.1) is False


def test_scheduler_runs_and_returns_best_node() -> None:
    config = SchedulerConfig(beam_size=2, max_depth=4)
    scheduler = RecursiveScheduler(config=config)

    def expand(node: RecursionNode) -> list[RecursionNode]:
        if node.depth >= 4:
            return []
        child1 = RecursionNode(
            state=(node.depth, 0),
            value=node.value + 1,
            cost=1.0,
            depth=node.depth + 1,
            parent=node,
        )
        child2 = RecursionNode(
            state=(node.depth, 1),
            value=node.value + 2,
            cost=1.0,
            depth=node.depth + 1,
            parent=node,
        )
        return [child1, child2]

    best = scheduler.schedule_recursion("root", expand)
    assert best is not None
    assert best.value >= 2
    metrics = scheduler.get_metrics()
    assert metrics["expansions"] > 0
