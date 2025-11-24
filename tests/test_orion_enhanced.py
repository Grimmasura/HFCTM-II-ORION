import pytest
from httpx import AsyncClient, ASGITransport
from orion_api.enhanced.orion_complete import create_complete_orion_app

@pytest.mark.asyncio
async def test_status_and_inference():
    app = create_complete_orion_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/system/status")
        assert r.status_code == 200
        assert r.json()["system_status"] == "operational"

        payload = {"query": "test query", "concepts": ["recursion", "time"]}
        r = await client.post("/system/inference", json=payload)
        assert r.status_code == 200
        assert "system_coherence" in r.json()

@pytest.mark.asyncio
async def test_health_endpoint():
    app = create_complete_orion_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/system/health")
        assert r.status_code == 200
        assert r.json()["health"] in ["excellent", "good", "fair", "poor"]
