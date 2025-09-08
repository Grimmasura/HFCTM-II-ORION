import importlib
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient


def _get_app():
    candidates = [
        ("orion.integrator", "create_orion_api"),
        ("src.orion.integrator", "create_orion_api"),
        ("orion_api.main", "app"),
    ]
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, attr)
            if callable(obj):
                return obj(config_path=None)
            return obj
        except Exception:
            continue
    pytest.skip("FastAPI application not available")


def test_health_endpoint_smoke() -> None:
    app = _get_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    status = response.json().get("status")
    assert status in {"ok", "healthy"}


def test_metrics_endpoint_plaintext() -> None:
    app = _get_app()
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.text, str)
