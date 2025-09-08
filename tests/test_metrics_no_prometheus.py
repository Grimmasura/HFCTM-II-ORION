"""Tests for /metrics endpoint when prometheus_client is unavailable."""

import importlib
import builtins
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient
from fastapi import APIRouter


def test_metrics_without_prometheus(monkeypatch):
    """The /metrics endpoint should handle missing prometheus_client."""
    # Ensure prometheus_client and relevant modules are re-imported
    monkeypatch.delitem(sys.modules, "prometheus_client", raising=False)
    monkeypatch.delitem(sys.modules, "orion_api.main", raising=False)
    monkeypatch.delitem(sys.modules, "orion_api.telemetry.prometheus", raising=False)

    # Provide dummy modules to avoid heavy dependencies
    dummy_pkg = types.ModuleType("orion_api")
    dummy_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "orion_api")]
    monkeypatch.setitem(sys.modules, "orion_api", dummy_pkg)

    dummy_hfctm = types.ModuleType("orion_api.hfctm_safety")

    class DummyConfig:
        pass

    class DummyCore:
        pass

    dummy_hfctm.HFCTMII_SafetyCore = DummyCore
    dummy_hfctm.SafetyConfig = DummyConfig
    dummy_hfctm.init_safety_core = lambda config: None
    dummy_hfctm.safety_core = None
    monkeypatch.setitem(sys.modules, "orion_api.hfctm_safety", dummy_hfctm)

    dummy_torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    dummy_models = types.ModuleType("models")
    dummy_models.__path__ = []
    stability_mod = types.ModuleType("models.stability_core")

    class DummyStability:
        def snapshot(self):
            return []

    stability_mod.stability_core = DummyStability()
    monkeypatch.setitem(sys.modules, "models", dummy_models)
    monkeypatch.setitem(sys.modules, "models.stability_core", stability_mod)

    # Stub router modules
    routers_pkg = types.ModuleType("orion_api.routers")

    def make_router_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(f"orion_api.routers.{name}")
        mod.router = APIRouter()
        return mod

    for name in [
        "recursive_ai",
        "quantum_sync",
        "recursive_trust",
        "egregore_defense",
        "manifold_router",
        "knowledge_expansion",
        "perception",
    ]:
        mod = make_router_module(name)
        setattr(routers_pkg, name, mod)
        monkeypatch.setitem(sys.modules, f"orion_api.routers.{name}", mod)

    monkeypatch.setitem(sys.modules, "orion_api.routers", routers_pkg)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prometheus_client":
            raise ImportError
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    main = importlib.import_module("orion_api.main")
    importlib.reload(main)

    client = TestClient(main.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "Prometheus metrics unavailable" in response.text
