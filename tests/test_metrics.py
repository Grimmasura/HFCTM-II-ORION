"""Prometheus metrics tests for scheduler and API endpoints."""

from __future__ import annotations

import importlib
import re
from typing import Optional

import pytest


def _has_prom() -> bool:
    try:
        import prometheus_client  # noqa: F401
        return True
    except Exception:
        return False


def _load_sched():
    for name in ("orion.schedule.recursive_scheduler", "src.orion.schedule.recursive_scheduler"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise ImportError("Could not import scheduler module")


def _maybe_client() -> Optional["TestClient"]:
    try:
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
    except Exception:
        return None

    app = None
    try:
        m = importlib.import_module("orion_api.main")
        if hasattr(m, "app"):
            app = getattr(m, "app")
    except Exception:
        pass
    if app is None:
        for name in ("orion.integrator", "src.orion.integrator"):
            try:
                m = importlib.import_module(name)
                if hasattr(m, "create_orion_api"):
                    app = getattr(m, "create_orion_api")(config_path=None)
                    break
            except Exception:
                continue
    if app is None:
        return None
    return TestClient(app)


def _metric_sum(client: "TestClient", name: str) -> float:
    text = client.get("/metrics").text
    m = re.search(rf"{name}_sum\s+(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else 0.0


def test_scheduler_imports_without_prometheus() -> None:
    mod = _load_sched()
    assert hasattr(mod, "RecursionBudget")
    assert hasattr(mod, "RecursionNode")
    assert hasattr(mod, "RecursiveScheduler")


def test_scheduler_metrics_when_prometheus_available() -> None:
    if not _has_prom():
        pytest.skip("prometheus_client not installed")
    from prometheus_client import REGISTRY, generate_latest

    _ = _load_sched()
    text = generate_latest(REGISTRY).decode("utf-8", errors="ignore")
    expected = [
        "orion_recursion_depth_sched",
        "orion_credits_remaining_sched",
        "orion_loop_detections_total_sched",
        "orion_beam_size_current_sched",
        "orion_resource_efficiency_sched",
    ]
    found = sum(1 for name in expected if name in text)
    assert found >= 2, "Expected at least two scheduler metrics in registry scrape"


def test_metrics_endpoint_plaintext_if_api_present() -> None:
    client = _maybe_client()
    if client is None:
        pytest.skip("No API app/factory found")
    r = client.get("/metrics")
    if r.status_code == 404:
        pytest.skip("/metrics not implemented in this branch")
    assert r.status_code == 200
    assert isinstance(r.text, str)


def test_recursive_ai_depth_metric_value() -> None:
    if not _has_prom():
        pytest.skip("prometheus_client not installed")
    client = _maybe_client()
    if client is None:
        pytest.skip("No API app/factory found")
    before = _metric_sum(client, "orion_recursive_ai_depth")
    resp = client.post(
        "/api/v1/recursive_ai/infer", json={"query": "test", "depth": 2}
    )
    assert resp.status_code == 200
    after = _metric_sum(client, "orion_recursive_ai_depth")
    assert after >= before + 2


def test_manifold_depth_metric_value() -> None:
    if not _has_prom():
        pytest.skip("prometheus_client not installed")
    client = _maybe_client()
    if client is None:
        pytest.skip("No API app/factory found")
    before = _metric_sum(client, "orion_manifold_depth")
    resp = client.post(
        "/manifold/distribute_task", params={"task": "t", "depth": 3}
    )
    assert resp.status_code == 200
    after = _metric_sum(client, "orion_manifold_depth")
    assert after >= before + 3


def test_metrics_endpoint_without_prometheus() -> None:
    if _has_prom():
        pytest.skip("prometheus_client installed")
    client = _maybe_client()
    if client is None:
        pytest.skip("No API app/factory found")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.text == ""

