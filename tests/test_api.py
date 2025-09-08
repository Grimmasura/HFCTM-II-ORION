import importlib
import pytest

# Skip the whole file if FastAPI is not available
pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
from fastapi.testclient import TestClient


def _resolve_app():
    """
    Try the two common entry points:
    1) legacy main branch: orion_api.main:app
    2) integrator factory: orion.integrator:create_orion_api (or src.orion.integrator)
    """
    try:
        mod = importlib.import_module("orion_api.main")
        if hasattr(mod, "app"):
            return getattr(mod, "app")
    except Exception:
        pass
    for name in ("orion.integrator", "src.orion.integrator"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "create_orion_api"):
                return getattr(mod, "create_orion_api")(config_path=None)
        except Exception:
            continue
    pytest.skip(
        "No API app or factory found (checked orion_api.main and orion.integrator)"
    )


def _client() -> TestClient:
    return TestClient(_resolve_app())


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


def test_health() -> None:
    c = _client()
    r = _get_optional(c, "/health")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert body.get("status") in {"healthy", "ok"}


def test_metrics_reachable() -> None:
    c = _client()
    r = _get_optional(c, "/metrics")
    assert r.status_code == 200
    assert isinstance(r.text, str)


def test_root_optional() -> None:
    c = _client()
    r = _get_optional(c, "/")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_telemetry_optional() -> None:
    c = _client()
    r = _get_optional(c, "/telemetry")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_trust_assess_optional() -> None:
    c = _client()
    r = _post_optional(c, "/trust/assess", params={"score": 80})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)

