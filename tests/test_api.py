import importlib
import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient


def _import_api_factory():
    """Try common module paths for the API factory."""
    candidates = [
        "orion.integrator",
        "src.orion.integrator",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "create_orion_api"):
                return mod.create_orion_api
        except Exception:
            continue
    pytest.skip("create_orion_api factory not found")


def test_health_endpoint_smoke() -> None:
    create_orion_api = _import_api_factory()
    app = create_orion_api(config_path=None)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["status"] in ("healthy", "ok")


def test_metrics_endpoint_plaintext() -> None:
    """Ensure /metrics endpoint returns text and is reachable."""
    create_orion_api = _import_api_factory()
    app = create_orion_api(config_path=None)
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.text, str)

