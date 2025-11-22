from fastapi import APIRouter

router = APIRouter()

@router.post("/learn")
async def learn(info: str):
    """Pretend to expand knowledge."""
    return {"learned": info}

