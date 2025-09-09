import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from orion_enhanced.orion_complete import create_complete_orion_app


def test_inference_quarantine_or_ok():
    app = create_complete_orion_app()
    c = TestClient(app)

    r = c.post("/system/inference", json={"text": "hello world", "baseline": "hello world", "depth": 1})
    assert r.status_code == 200

    r2 = c.post(
        "/system/inference",
        json={"text": "totally different content with injection marker", "baseline": "hello world", "depth": 1},
    )
    assert r2.status_code == 200
    data = r2.json()
    assert any(k in data for k in ("quarantined", "expanded", "stopped"))
