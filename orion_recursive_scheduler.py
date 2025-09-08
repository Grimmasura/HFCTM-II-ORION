"""ORION Recursive Loop Prevention Scheduler
Implements budget management, structural pruning, and cycle detection
to prevent exponential resource consumption in recursive AI systems.
"""

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from prometheus_client import Counter, Gauge, Histogram
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
RECURSION_DEPTH = Gauge('orion_recursion_depth', 'Current recursion depth')
CREDITS_REMAINING = Gauge('orion_credits_remaining', 'Remaining recursion credits')
LOOP_DETECTIONS = Counter('orion_loop_detections_total', 'Total loop detections')
BEAM_SIZE_CURRENT = Gauge('orion_beam_size_current', 'Current beam frontier size')
RESOURCE_EFFICIENCY = Histogram('orion_resource_efficiency', 'Value per cost ratio')


@dataclass
class RecursionBudget:
    """Budget management for recursive operations"""

    initial_credits: float = 1000.0
    remaining_credits: float = field(default_factory=lambda: 1000.0)
    depth_cost_gamma: float = 1.5  # c(n) = c₀ * γⁿ
    base_cost: float = 1.0
    wall_time_limit: float = 300.0  # 5 minutes
    memory_limit_mb: float = 1024.0
    token_limit: int = 100000

    # SLA tracking
    start_time: float = field(default_factory=time.time)
    memory_used_mb: float = 0.0
    tokens_used: int = 0

    def cost_at_depth(self, depth: int) -> float:
        """Compute cost c(n) = c₀ * γⁿ"""
        return self.base_cost * (self.depth_cost_gamma ** depth)

    def can_afford(self, depth: int) -> bool:
        """Check if we can afford operation at given depth"""
        cost = self.cost_at_depth(depth)
        return (
            self.remaining_credits >= cost
            and self.within_time_limit()
            and self.within_memory_limit()
            and self.within_token_limit()
        )

    def spend(self, depth: int) -> bool:
        """Spend credits for operation at depth"""
        if not self.can_afford(depth):
            return False

        cost = self.cost_at_depth(depth)
        self.remaining_credits -= cost
        CREDITS_REMAINING.set(self.remaining_credits)
        return True

    def within_time_limit(self) -> bool:
        """Check wall time constraint"""
        return (time.time() - self.start_time) < self.wall_time_limit

    def within_memory_limit(self) -> bool:
        """Check memory constraint"""
        return self.memory_used_mb < self.memory_limit_mb

    def within_token_limit(self) -> bool:
        """Check token constraint"""
        return self.tokens_used < self.token_limit

    def update_resource_usage(self, memory_mb: float, tokens: int):
        """Update resource usage tracking"""
        self.memory_used_mb = memory_mb
        self.tokens_used = tokens


@dataclass
class RecursionNode:
    """Node in the recursion tree"""

    state_hash: str
    goal: str
    assumptions: List[str]
    plan: str
    depth: int
    value: float = 0.0
    cost: float = 1.0
    visits: int = 0
    parent: Optional["RecursionNode"] = None
    children: List["RecursionNode"] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)

    def value_per_cost(self) -> float:
        """Compute value per cost ratio"""
        return self.value / max(self.cost, 1e-6)

    def uct_score(self, c_explore: float = 1.41, cost_penalty: float = 0.1) -> float:
        """UCT score with cost penalty: Q + c√(ln(N)/n) - λ·cost"""
        if self.visits == 0:
            return float('inf')

        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_explore * np.sqrt(np.log(parent_visits) / self.visits)
        cost_term = cost_penalty * self.cost

        return (self.value / self.visits) + exploration - cost_term


class CycleDetector:
    """Detects cycles and loops in recursion"""

    def __init__(self, max_cache_size: int = 10000):
        self.seen_states: Set[str] = set()
        self.state_history: deque = deque(maxlen=max_cache_size)
        self.cycle_cache: Dict[str, int] = {}

    def normalize_state(
        self,
        goal: str,
        assumptions: List[str],
        plan: str,
        tools_used: List[str],
    ) -> str:
        """Create normalized state representation"""
        assumptions_sorted = sorted(assumptions)
        tools_sorted = sorted(tools_used)

        state_tuple = (
            goal.strip().lower(),
            tuple(assumptions_sorted),
            plan.strip().lower(),
            tuple(tools_sorted),
        )

        state_str = str(state_tuple)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def is_cycle(self, state_hash: str, new_info_threshold: float = 0.1) -> bool:
        """Check if state represents a cycle"""
        if state_hash in self.seen_states:
            LOOP_DETECTIONS.inc()
            logger.warning(f"Cycle detected: {state_hash[:16]}...")
            return True

        self.seen_states.add(state_hash)
        self.state_history.append(state_hash)
        return False

    def compute_novelty(self, current_plan: str, best_plans: List[str]) -> float:
        """Compute plan novelty/entropy"""
        if not best_plans:
            return 1.0

        current_words = set(current_plan.lower().split())

        max_similarity = 0.0
        for plan in best_plans:
            plan_words = set(plan.lower().split())
            if len(current_words) == 0 and len(plan_words) == 0:
                similarity = 1.0
            elif len(current_words) == 0 or len(plan_words) == 0:
                similarity = 0.0
            else:
                intersection = len(current_words & plan_words)
                union = len(current_words | plan_words)
                similarity = intersection / union

            max_similarity = max(max_similarity, similarity)

        return 1.0 - max_similarity


class BeamSearchFrontier:
    """Manages frontier for beam search with value-per-cost ranking"""

    def __init__(self, beam_size: int = 10):
        self.beam_size = beam_size
        self.frontier: List[RecursionNode] = []
        self.memoization_cache: Dict[str, RecursionNode] = {}

    def add_node(self, node: RecursionNode) -> bool:
        """Add node to frontier with beam pruning"""
        if node.state_hash in self.memoization_cache:
            cached_node = self.memoization_cache[node.state_hash]
            if cached_node.value >= node.value:
                return False

        self.memoization_cache[node.state_hash] = node

        self.frontier.append(node)
        self.frontier.sort(key=lambda n: n.value_per_cost(), reverse=True)

        if len(self.frontier) > self.beam_size:
            self.frontier = self.frontier[: self.beam_size]

        BEAM_SIZE_CURRENT.set(len(self.frontier))
        return True

    def get_best_node(self) -> Optional[RecursionNode]:
        """Get best node from frontier"""
        if not self.frontier:
            return None

        best_node = max(self.frontier, key=lambda n: n.uct_score())
        self.frontier.remove(best_node)

        BEAM_SIZE_CURRENT.set(len(self.frontier))
        return best_node

    def is_empty(self) -> bool:
        """Check if frontier is empty"""
        return len(self.frontier) == 0

    def get_best_plans(self, top_k: int = 3) -> List[str]:
        """Get top-k plans for novelty comparison"""
        sorted_nodes = sorted(
            self.frontier, key=lambda n: n.value_per_cost(), reverse=True
        )
        return [node.plan for node in sorted_nodes[:top_k]]


class EntropyGate:
    """Gates recursion based on plan entropy and convergence"""

    def __init__(self, min_entropy: float = 0.1, convergence_window: int = 5):
        self.min_entropy = min_entropy
        self.convergence_window = convergence_window
        self.entropy_history: deque = deque(maxlen=convergence_window)
        self.plan_history: deque = deque(maxlen=convergence_window)

    def should_continue(self, current_plan: str, best_plans: List[str]) -> bool:
        """Determine if recursion should continue based on entropy"""
        plan_words = current_plan.lower().split()
        if len(plan_words) <= 1:
            entropy = 0.0
        else:
            word_counts = defaultdict(int)
            for word in plan_words:
                word_counts[word] += 1

            total_words = len(plan_words)
            entropy = 0.0
            for count in word_counts.values():
                p = count / total_words
                entropy -= p * np.log2(p)

        self.entropy_history.append(entropy)
        self.plan_history.append(current_plan)

        if len(self.entropy_history) >= self.convergence_window:
            recent_entropies = list(self.entropy_history)
            entropy_trend = np.polyfit(
                range(len(recent_entropies)), recent_entropies, 1
            )[0]

            if entropy_trend >= 0 and entropy < self.min_entropy:
                logger.info(
                    f"Entropy gate triggered: entropy={entropy:.3f}, trend={entropy_trend:.3f}"
                )
                return False

        if best_plans:
            best_plan = best_plans[0]
            current_words = set(current_plan.lower().split())
            best_words = set(best_plan.lower().split())

            if len(current_words) > 0 and len(best_words) > 0:
                overlap = len(current_words & best_words)
                union_size = len(current_words | best_words)
                similarity = overlap / union_size

                if similarity > 0.9:
                    logger.info(
                        f"Converged to best plan: similarity={similarity:.3f}"
                    )
                    return False

        return True


class RecursiveScheduler:
    """Main recursive scheduler with loop prevention"""

    def __init__(
        self,
        beam_size: int = 10,
        max_depth: int = 20,
        budget_config: Optional[RecursionBudget] = None,
    ):
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.budget = budget_config or RecursionBudget()

        self.cycle_detector = CycleDetector()
        self.frontier = BeamSearchFrontier(beam_size)
        self.entropy_gate = EntropyGate()

        self.expansion_count = 0
        self.total_value_gained = 0.0
        self.marginal_gains = deque(maxlen=10)

    def schedule_recursion(
        self,
        initial_goal: str,
        initial_assumptions: List[str],
        expand_function: Callable[[RecursionNode], List[RecursionNode]],
        epsilon: float = 0.01,
    ) -> Optional[RecursionNode]:
        """Main scheduling loop with all prevention mechanisms"""

        initial_state_hash = self.cycle_detector.normalize_state(
            initial_goal, initial_assumptions, "", []
        )

        root_node = RecursionNode(
            state_hash=initial_state_hash,
            goal=initial_goal,
            assumptions=initial_assumptions,
            plan="",
            depth=0,
            value=0.0,
            cost=self.budget.cost_at_depth(0),
        )

        self.frontier.add_node(root_node)
        best_node = root_node

        while (
            self.budget.remaining_credits > 0
            and not self.frontier.is_empty()
            and self.budget.within_time_limit()
        ):
            current_node = self.frontier.get_best_node()
            if current_node is None:
                break

            RECURSION_DEPTH.set(current_node.depth)

            if current_node.depth >= self.max_depth:
                continue

            if not self.budget.can_afford(current_node.depth + 1):
                logger.info("Budget exhausted, stopping recursion")
                break

            if self.cycle_detector.is_cycle(current_node.state_hash):
                continue

            best_plans = self.frontier.get_best_plans()
            if not self.entropy_gate.should_continue(
                current_node.plan, best_plans
            ):
                continue

            try:
                child_nodes = expand_function(current_node)
                self.expansion_count += 1

                for child in child_nodes:
                    child.parent = current_node
                    child.depth = current_node.depth + 1

                    if self.budget.spend(child.depth):
                        if self.frontier.add_node(child):
                            value_gain = child.value - current_node.value
                            cost_increase = child.cost - current_node.cost

                            if cost_increase > 0:
                                efficiency = value_gain / cost_increase
                                RESOURCE_EFFICIENCY.observe(efficiency)

                                self.marginal_gains.append(efficiency)
                                window = self.entropy_gate.convergence_window
                                if len(self.marginal_gains) >= window:
                                    recent_gains = list(self.marginal_gains)[-window:]
                                    avg_recent_gain = np.mean(recent_gains)

                                    if len(self.marginal_gains) >= 2 * window:
                                        prev_gains = list(self.marginal_gains)[
                                            -(2 * window) : -window
                                        ]
                                        prev_avg = np.mean(prev_gains)

                                        if (
                                            avg_recent_gain < epsilon
                                            and prev_avg < epsilon
                                        ):
                                            logger.info(
                                                "Anytime stop: consecutive low gains "
                                                f"{avg_recent_gain:.4f}, {prev_avg:.4f}"
                                            )
                                            break
                                    elif avg_recent_gain < epsilon:
                                        logger.info(
                                            f"Marginal gain below threshold: {avg_recent_gain:.4f}"
                                        )
                                        break

                            if (
                                child.value_per_cost()
                                > best_node.value_per_cost()
                            ):
                                best_node = child

            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Error expanding node: {e}")
                continue

        logger.info(
            f"Recursion completed: {self.expansion_count} expansions, "
            f"best value/cost: {best_node.value_per_cost():.3f}"
        )

        return best_node

    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics for monitoring"""
        return {
            'expansions_total': self.expansion_count,
            'credits_remaining': self.budget.remaining_credits,
            'frontier_size': len(self.frontier.frontier),
            'avg_marginal_gain': np.mean(list(self.marginal_gains))
            if self.marginal_gains
            else 0.0,
            'cycles_detected': LOOP_DETECTIONS._value._value,
            'memory_usage_mb': self.budget.memory_used_mb,
            'tokens_used': self.budget.tokens_used,
            'time_elapsed': time.time() - self.budget.start_time,
        }

    def reset(self):
        """Reset scheduler state"""
        self.budget.remaining_credits = self.budget.initial_credits
        self.budget.start_time = time.time()
        self.budget.memory_used_mb = 0.0
        self.budget.tokens_used = 0

        self.cycle_detector.seen_states.clear()
        self.cycle_detector.state_history.clear()

        self.frontier = BeamSearchFrontier(self.beam_size)
        self.entropy_gate = EntropyGate()

        self.expansion_count = 0
        self.marginal_gains.clear()


def example_expand_function(node: RecursionNode) -> List[RecursionNode]:
    """Example expansion function for testing"""
    child_nodes = []

    for i in range(2):
        child_hash = hashlib.sha256(
            f"{node.state_hash}_child_{i}".encode()
        ).hexdigest()
        child_plan = f"{node.plan} -> step_{i}"

        child_value = node.value + np.random.exponential(1.0)
        child_cost = node.cost * 1.2 + np.random.exponential(0.5)

        child = RecursionNode(
            state_hash=child_hash,
            goal=node.goal,
            assumptions=node.assumptions
            + [f"assumption_depth_{node.depth + 1}"],
            plan=child_plan,
            depth=node.depth + 1,
            value=child_value,
            cost=child_cost,
        )

        child_nodes.append(child)

    return child_nodes


if __name__ == "__main__":  # pragma: no cover - manual run
    scheduler = RecursiveScheduler(beam_size=5, max_depth=10)

    result = scheduler.schedule_recursion(
        initial_goal="Solve complex problem",
        initial_assumptions=["Initial context"],
        expand_function=example_expand_function,
        epsilon=0.01,
    )

    print("Final metrics:", scheduler.get_metrics())
    if result:
        print(f"Best solution: {result.plan}")
        print(f"Value/Cost: {result.value_per_cost():.3f}")

