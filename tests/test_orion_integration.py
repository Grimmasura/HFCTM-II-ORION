import pytest
pytest.importorskip("numpy", reason="numpy not installed")

import numpy as np

from orion_config import OrionConfig, QuantumConfig

torch = pytest.importorskip("torch")

from orion_chiral_trainer import ChiralTrainer, ChiralConfig
from orion_quantum_stabilizer import QuantumStabilizer
from orion_recursive_scheduler import RecursiveScheduler
from orion_egregore_defense import EgregoreDefense


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_dim = 3
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


def test_subsystems_smoke():
    cfg = OrionConfig()
    model = DummyModel()
    trainer = ChiralTrainer(model, ChiralConfig())
    stabilizer = QuantumStabilizer(cfg.quantum)
    scheduler = RecursiveScheduler()
    defense = EgregoreDefense()

    x_batch = torch.randn(2, 3)
    y_batch = torch.tensor([0, 1])
    loss_dict = trainer.compute_chiral_loss(x_batch, y_batch)
    assert "total_loss" in loss_dict

    rho = torch.eye(2, dtype=torch.complex64)
    rho2 = stabilizer.evolve(rho)
    assert torch.allclose(torch.trace(rho2).real, torch.tensor(1.0), atol=1e-6)

    remaining = scheduler.credit(agent=1, task_cost=0.5)
    assert isinstance(remaining, float)

    data = np.random.rand(10, 3)
    metrics = defense.analyze(data)
    assert "mutual_info" in metrics
