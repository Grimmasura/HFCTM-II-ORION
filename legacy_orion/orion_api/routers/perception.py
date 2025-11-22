from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
async def ping():
    """Perception heartbeat."""
    return {"perception": "online"}

