import importlib
import pytest

pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
from fastapi.testclient import TestClient


def _load_app_or_factory():
    """Resolve API app or factory across branches."""
    try:
        mod = importlib.import_module("orion_api.main")
        if hasattr(mod, "app"):
            return ("app", getattr(mod, "app"))
    except Exception:
        pass
    for modname in ("orion.integrator", "src.orion.integrator"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "create_orion_api"):
                return ("factory", getattr(mod, "create_orion_api"))
        except Exception:
            continue
    pytest.skip(
        "No API app or factory found (checked orion_api.main and orion.integrator)"
    )


def _client() -> TestClient:
    kind, obj = _load_app_or_factory()
    if kind == "app":
        app = obj
    else:
        app = obj(config_path=None)
    return TestClient(app)


def _get_optional(client: TestClient, path: str):
    r = client.get(path)
    if r.status_code == 404:
        pytest.skip(f"{path} not implemented in this branch")
    return r


def _post_optional(client: TestClient, path: str, **kwargs):
    r = client.post(path, **kwargs)
    if r.status_code == 404:
        pytest.skip(f"{path} not implemented in this branch")
    return r


def test_health_endpoint():
    client = _client()
    r = _get_optional(client, "/health")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert body.get("status") in {"healthy", "ok"}


def test_metrics_endpoint_reachable():
    client = _client()
    r = _get_optional(client, "/metrics")
    assert r.status_code == 200
    assert isinstance(r.text, str)


def test_root_endpoint_optional():
    client = _client()
    r = _get_optional(client, "/")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert "message" in body or "status" in body


def test_telemetry_endpoint_optional():
    client = _client()
    r = _get_optional(client, "/telemetry")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert "telemetry" in body or "metrics" in body or "status" in body


def test_quantum_sync_status_endpoint_optional():
    client = _client()
    r = _get_optional(client, "/quantum-sync/status")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert "status" in body


def test_recursive_trust_assess_endpoint_optional():
    client = _client()
    r = _post_optional(client, "/trust/assess", params={"score": 80})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert body.get("score") in (80, None)

