from fastapi import APIRouter

from models.recursive_trust import assess_score

router = APIRouter()

@router.post("/assess")
async def assess_trust(score: int):
    """Assess trust score using a simple heuristic."""

    return assess_score(score)

