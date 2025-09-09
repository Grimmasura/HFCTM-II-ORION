import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from orion_enhanced.orion_complete import create_complete_orion_app


def test_metrics_endpoint():
    app = create_complete_orion_app()
    c = TestClient(app)
    r = c.get("/metrics")
    assert r.status_code == 200
    body = r.text.lower()
    assert any(x in body for x in ("prometheus", "_total", "_histogram", "unavailable"))
