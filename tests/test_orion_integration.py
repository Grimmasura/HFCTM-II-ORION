import numpy as np
import pytest

from orion_config import OrionConfig, QuantumConfig

torch = __import__("pytest").importorskip("torch")

from orion_chiral_trainer import ChiralTrainer
from orion_quantum_stabilizer import QuantumStabilizer
from orion_recursive_scheduler import RecursiveScheduler
from orion_egregore_defense import EgregoreDefense


def test_subsystems_smoke():
    cfg = OrionConfig(quantum=QuantumConfig(n_qubits=1, lyapunov_kappa=[0.1]))
    trainer = ChiralTrainer()
    stabilizer = QuantumStabilizer(cfg.quantum)
    scheduler = RecursiveScheduler()
    defense = EgregoreDefense()

    policy_orig = torch.randn(2, 3)
    policy_chiral_mapped = torch.randn(2, 3)
    js = trainer.js_divergence(policy_orig, policy_chiral_mapped)
    assert js >= 0

    remaining = scheduler.credit(agent=1, task_cost=0.5)
    assert isinstance(remaining, float)

    data = np.random.rand(10, 3)
    metrics = defense.analyze(data)
    assert "mutual_info" in metrics
