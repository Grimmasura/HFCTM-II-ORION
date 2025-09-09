import pytest
pytest.importorskip("numpy", reason="numpy not installed")

import importlib
from fastapi.testclient import TestClient
import orion_api.main as main


def test_safety_status_without_torch():
    importlib.reload(main)
    client = TestClient(main.app)
    response = client.get("/api/safety/status")
    assert response.status_code == 200
    assert response.json()["error"].lower().startswith("pytorch not installed")


def test_safety_status_with_stubbed_torch(monkeypatch):
    importlib.reload(main)
    client = TestClient(main.app)

    class TorchStub:
        def randn(self, *shape):
            return [[0] * (shape[1] if len(shape) > 1 else 1) for _ in range(shape[0] if shape else 1)]

    monkeypatch.setattr(main, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(main, "torch", TorchStub())

    class SafetyCoreStub:
        intervention_count = 1

        async def safety_check(self, state):  # pragma: no cover - simple stub
            return {"safe": True}

    monkeypatch.setattr(main, "safety_core", SafetyCoreStub())

    response = client.get("/api/safety/status")
    assert response.status_code == 200
    data = response.json()
    assert data["safety_active"] is True
    assert data["interventions_total"] == 1
    assert data["last_check"] == {"safe": True}

