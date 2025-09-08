import numpy as np
import pytest

from orion_config import OrionConfig, QuantumConfig

torch = __import__("pytest").importorskip("torch")

from orion_chiral_trainer import ChiralTrainer
from orion_quantum_stabilizer import QuantumStabilizer
from orion_recursive_scheduler import (
    RecursiveScheduler,
    example_expand_function,
)
from orion_egregore_defense import EgregoreDefense


def test_subsystems_smoke():
    cfg = OrionConfig()
    trainer = ChiralTrainer()
    stabilizer = QuantumStabilizer(cfg.quantum)
    scheduler = RecursiveScheduler(beam_size=2, max_depth=2)
    defense = EgregoreDefense()

    policy_orig = torch.randn(2, 3)
    policy_chiral_mapped = torch.randn(2, 3)
    js = trainer.js_divergence(policy_orig, policy_chiral_mapped)
    assert js >= 0

    rho = torch.eye(2, dtype=torch.complex64)
    rho2 = stabilizer.evolve(rho)
    assert torch.allclose(torch.trace(rho2).real, torch.tensor(1.0), atol=1e-6)

    result = scheduler.schedule_recursion(
        initial_goal="Solve",
        initial_assumptions=["ctx"],
        expand_function=example_expand_function,
        epsilon=0.01,
    )
    assert result is not None

    data = np.random.rand(10, 3)
    metrics = defense.analyze(data)
    assert "mutual_info" in metrics
