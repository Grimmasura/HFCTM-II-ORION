import importlib
import pytest


def _has_prometheus() -> bool:
    try:
        import prometheus_client  # noqa: F401
        return True
    except Exception:
        return False


def _load_scheduler_module():
    """Import scheduler module with optional namespace fallbacks."""
    candidates = [
        "orion.schedule.recursive_scheduler",
        "src.orion.schedule.recursive_scheduler",
    ]
    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise last_err or ImportError("Could not import recursive_scheduler module")


def test_scheduler_module_imports_without_prometheus() -> None:
    if _has_prometheus():
        pytest.skip("prometheus_client installed; run no-op import test only when absent")
    mod = _load_scheduler_module()
    assert hasattr(mod, "RecursionBudget")
    assert hasattr(mod, "RecursionNode")
    assert hasattr(mod, "RecursiveScheduler")


def test_metrics_names_when_prometheus_available() -> None:
    if not _has_prometheus():
        pytest.skip("prometheus_client not installed")

    from prometheus_client import REGISTRY, generate_latest

    _ = _load_scheduler_module()
    text = generate_latest(REGISTRY).decode("utf-8", errors="ignore")
    expected = [
        "orion_recursion_depth_sched",
        "orion_credits_remaining_sched",
        "orion_loop_detections_total_sched",
        "orion_beam_size_current_sched",
        "orion_resource_efficiency_sched",
    ]
    missing = [name for name in expected if name not in text]
    assert len(expected) - len(missing) >= 2, f"Too many missing metrics: {missing}"


def test_metrics_noop_behavior_without_prometheus() -> None:
    if _has_prometheus():
        pytest.skip("prometheus_client installed; skip no-op behavior test")

    mod = _load_scheduler_module()
    scheduler = mod.RecursiveScheduler()
    best = scheduler.schedule_recursion("root", mod.example_expand_function)
    assert best is not None


def _maybe_fastapi_client():
    try:
        fastapi = importlib.import_module("fastapi")  # noqa: F401
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
    except Exception:
        return None

    try:
        mod = importlib.import_module("orion_api.main")
        if hasattr(mod, "app"):
            return TestClient(getattr(mod, "app"))
    except Exception:
        pass

    for modname in ("orion.integrator", "src.orion.integrator"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "create_orion_api"):
                app = getattr(mod, "create_orion_api")(config_path=None)
                return TestClient(app)
        except Exception:
            continue
    return None


def test_metrics_endpoint_plaintext_if_available() -> None:
    client = _maybe_fastapi_client()
    if client is None:
        pytest.skip("FastAPI app not available")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.text, str)

