from fastapi.testclient import TestClient
from orion_api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_telemetry_endpoint():
    response = client.get("/telemetry")
    assert response.status_code == 200
    assert "telemetry" in response.json()
