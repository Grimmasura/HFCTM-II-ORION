import pytest
from httpx import AsyncClient
from orion_enhanced.orion_complete import create_complete_orion_app

@pytest.mark.asyncio
async def test_status_and_inference():
    app = create_complete_orion_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.get("/system/status")
        assert r.status_code == 200
        assert r.json()["system_status"] == "operational"

        payload = {"query": "test query", "concepts": ["recursion", "time"]}
        r = await client.post("/system/inference", params=payload)
        assert r.status_code == 200
        assert "system_coherence" in r.json()

@pytest.mark.asyncio
async def test_health_endpoint():
    app = create_complete_orion_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.get("/system/health")
        assert r.status_code == 200
        assert r.json()["health"] in ["excellent", "good", "fair", "poor"]
