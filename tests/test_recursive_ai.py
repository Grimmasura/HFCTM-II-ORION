import os
import sys
import pytest

try:
    from fastapi.testclient import TestClient
except Exception as e:  # e.g., httpx missing
    pytest.skip(f"FastAPI TestClient unavailable: {e}", allow_module_level=True)

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from orion_api.main import app

client = TestClient(app)

def test_recursive_ai():
    response = client.post("/api/v1/recursive_ai/infer", json={"query": "Test", "depth": 3})
    assert response.status_code == 200
    assert "response" in response.json()
