from fastapi import APIRouter

from models.quantum_sync import get_sync_status

router = APIRouter()


@router.get("/status")
async def get_status():
    """Expose real backend latency and coherence statistics when available."""

    status = get_sync_status()
    return {
        "status": status.get("status"),
        "latency": status.get("latency"),
        "coherence_time": status.get("coherence_time"),
        "message": status.get("message"),
    }

