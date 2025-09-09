import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

torch = pytest.importorskip("torch")

from orion_api.main import app

client = TestClient(app)


def test_recursive_infer():
    data = {"query": "Expand recursive intelligence.", "depth": 0}
    response = client.post("/api/v1/recursive_ai/infer", json=data)
    assert response.status_code == 200

