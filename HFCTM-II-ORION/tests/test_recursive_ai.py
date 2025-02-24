import pytest
from fastapi.testclient import TestClient
from orion_api.main import app

client = TestClient(app)

def test_recursive_ai():
    response = client.post("/api/v1/recursive_ai/infer", json={"query": "Test", "depth": 3})
    assert response.status_code == 200
    assert "response" in response.json()