from fastapi.testclient import TestClient
import pytest

torch = pytest.importorskip("torch")
from orion_api.main import app

client = TestClient(app)

def test_health_smoke():
    response = client.get("/health")
    assert response.status_code in (200, 204)
    if response.status_code == 200:
        assert response.json().get("status") == "ok"


def test_version_smoke():
    response = client.get("/version")
    if response.status_code == 404:
        pytest.skip("/version endpoint not available")
    assert response.status_code == 200
    assert "version" in response.json()
