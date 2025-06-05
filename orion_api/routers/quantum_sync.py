from fastapi import APIRouter

from models.quantum_sync import get_sync_status

router = APIRouter(prefix="/api/v1/quantum_sync", tags=["Quantum Sync"])

@router.get("/status")
async def get_status():
    """Return status of quantum synchronization subsystem."""

    return get_sync_status()

