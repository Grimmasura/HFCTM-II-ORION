"""Tests for Prometheus metrics on recursive endpoints."""

from fastapi.testclient import TestClient
import pytest

pytest.importorskip("prometheus_client")
pytest.importorskip("torch")

from prometheus_client.parser import text_string_to_metric_families


def _metric_count_and_sum(client: TestClient, name: str) -> tuple[float, float]:
    """Return the count and sum for a Prometheus histogram."""
    metrics_resp = client.get("/metrics")
    count = total = 0.0
    for family in text_string_to_metric_families(metrics_resp.text):
        if family.name == name:
            for sample in family.samples:
                if sample.name == f"{name}_count":
                    count = sample.value
                elif sample.name == f"{name}_sum":
                    total = sample.value
            break
    return count, total


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Provide a TestClient with heavy modules stubbed out."""
    import sys
    import types

    dummy_module = types.ModuleType("models.recursive_ai_model")
    dummy_module.recursive_model_live = lambda query, depth: "ok"
    monkeypatch.setitem(sys.modules, "models.recursive_ai_model", dummy_module)

    from orion_api.main import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics(client: TestClient) -> None:  # noqa: PT004
    """Reset histogram metrics before each test to avoid interference."""
    from orion_api.routers.recursive_ai import recursive_depth_metric
    from orion_api.routers.manifold_router import manifold_depth_metric

    for hist in (recursive_depth_metric, manifold_depth_metric):
        if hasattr(hist, "_sum"):
            hist._sum.set(0)  # type: ignore[attr-defined]
        for bucket in getattr(hist, "_buckets", []):  # type: ignore[attr-defined]
            bucket.set(0)


def test_recursive_ai_depth_metric(client: TestClient) -> None:
    """Ensure recursive AI endpoint records depth metric with correct values."""
    depth = 2
    response = client.post(
        "/api/v1/recursive_ai/infer", json={"query": "test", "depth": depth}
    )
    assert response.status_code == 200
    count, total = _metric_count_and_sum(client, "orion_recursive_ai_depth")
    assert count == 1.0
    assert total == depth


def test_manifold_depth_metric(client: TestClient) -> None:
    """Ensure manifold router records recursion depth metric with correct values."""
    depth = 3
    response = client.post(
        "/manifold/distribute_task", params={"task": "t", "depth": depth}
    )
    assert response.status_code == 200
    count, total = _metric_count_and_sum(client, "orion_manifold_depth")
    assert count == 1.0
    assert total == depth

