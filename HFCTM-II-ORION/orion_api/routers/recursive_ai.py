from fastapi import APIRouter
from models.recursive_ai_model import recursive_model_live

router = APIRouter(prefix="/api/v1/recursive_ai", tags=["Recursive AI"])

@router.post("/infer")
async def recursive_infer(query: str, depth: int = 1):
    response = recursive_model_live(query, depth)
    return {"response": response, "depth": depth}
