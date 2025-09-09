from dataclasses import dataclass
import time

@dataclass
class Budget:
    max_depth: int = 8
    max_nodes: int = 512
    wall_time_s: float = 20.0
    min_marginal_gain: float = 0.02

class BudgetedRecursionController:
    def __init__(self, budget: Budget):
        self.budget = budget
        self.start = time.time()
        self.nodes_visited = 0

    def should_expand(self, depth: int, est_marginal_gain: float) -> bool:
        if depth >= self.budget.max_depth:
            return False
        if (time.time() - self.start) > self.budget.wall_time_s:
            return False
        if self.nodes_visited >= self.budget.max_nodes:
            return False
        if est_marginal_gain < self.budget.min_marginal_gain:
            return False
        self.nodes_visited += 1
        return True
