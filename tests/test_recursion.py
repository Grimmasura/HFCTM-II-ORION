import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from orion_enhanced.orion_complete import create_complete_orion_app


def test_budget_stops_depth():
    app = create_complete_orion_app()
    c = TestClient(app)
    r = c.post("/system/inference", json={"text": "x", "baseline": "x", "depth": 99})
    assert r.status_code == 200
    assert r.json().get("stopped") is True
