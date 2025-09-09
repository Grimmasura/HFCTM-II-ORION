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


def test_quantum_sync_status_endpoint():
    response = client.get("/quantum-sync/status")
    assert response.status_code == 200
    assert "status" in response.json()


def test_recursive_trust_assess_endpoint():
    response = client.post("/trust/assess", params={"score": 80})
    assert response.status_code == 200
    assert response.json().get("score") == 80


def test_egregore_shield_endpoint():
    response = client.get("/egregore/shield", params={"threat": 0.8})
    assert response.status_code == 200
    assert "action" in response.json()


def test_manifold_distribute_task_endpoint():
    response = client.post(
        "/manifold/distribute_task", params={"task": "test", "depth": 1}
    )
    assert response.status_code == 200
    assert "message" in response.json()
