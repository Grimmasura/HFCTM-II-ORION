from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/perception", tags=["Perception"])

@router.get("/ping")
async def ping():
    """Perception heartbeat."""
    return {"perception": "online"}

