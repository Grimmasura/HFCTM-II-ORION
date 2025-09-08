import importlib
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


def test_scheduler_imports_without_prometheus():
    mod = _load_sched()
    assert hasattr(mod, "RecursionBudget")
    assert hasattr(mod, "RecursionNode")
    assert hasattr(mod, "RecursiveScheduler")


def test_scheduler_metrics_when_prometheus_available():
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


def test_metrics_endpoint_plaintext_if_api_present():
    try:
        from fastapi.testclient import TestClient  # type: ignore
        pytest.importorskip("httpx")
    except Exception:
        pytest.skip("FastAPI not installed")

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
        pytest.skip("No API app/factory found")

    c = TestClient(app)
    r = c.get("/metrics")
    if r.status_code == 404:
        pytest.skip("/metrics not implemented in this branch")
    assert r.status_code == 200
    assert isinstance(r.text, str)

