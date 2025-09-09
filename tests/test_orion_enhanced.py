import pytest
from fastapi.testclient import TestClient
from orion_enhanced.orion_complete import create_complete_orion_app


def test_status_and_inference():
    app = create_complete_orion_app()
    c = TestClient(app)
    r = c.get("/system/status")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    payload = {"text": "test query", "baseline": "test", "depth": 1}
    r = c.post("/system/inference", json=payload)
    assert r.status_code == 200
    assert any(k in r.json() for k in ("expanded", "stopped", "quarantined"))


def test_health_endpoint():
    app = create_complete_orion_app()
    c = TestClient(app)
    r = c.get("/system/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
