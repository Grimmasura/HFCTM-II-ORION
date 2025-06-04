from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Expansion"])

@router.post("/learn")
async def learn(info: str):
    """Pretend to expand knowledge."""
    return {"learned": info}

