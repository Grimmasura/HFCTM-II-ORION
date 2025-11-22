from fastapi import APIRouter

from models.egregore_defense import evaluate_threat

router = APIRouter()

@router.get("/shield")
async def activate_shield(threat: float = 0.0):
    """Activate egregore defense based on a threat score."""

    result = evaluate_threat(threat)
    return {"action": result.action, "threat_score": result.threat_score}

