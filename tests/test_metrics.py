"""Tests for Prometheus metrics on recursive endpoints."""

from fastapi.testclient import TestClient
import pytest

pytest.importorskip("prometheus_client")
pytest.importorskip("torch")

from orion_api.main import app


client = TestClient(app)


def test_recursive_ai_depth_metric() -> None:
    """Ensure recursive AI endpoint records depth metric."""
    response = client.post(
        "/api/v1/recursive_ai/infer", json={"query": "test", "depth": 2}
    )
    assert response.status_code == 200
    metrics_resp = client.get("/metrics")
    assert "orion_recursive_ai_depth_sum" in metrics_resp.text


def test_manifold_depth_metric() -> None:
    """Ensure manifold router records recursion depth metric."""
    response = client.post(
        "/manifold/distribute_task", params={"task": "t", "depth": 3}
    )
    assert response.status_code == 200
    metrics_resp = client.get("/metrics")
    assert "orion_manifold_depth_sum" in metrics_resp.text

