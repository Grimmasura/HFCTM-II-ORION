from fastapi import APIRouter
from pydantic import BaseModel
from models.recursive_ai_model import recursive_model_live

router = APIRouter()

class RecursiveRequest(BaseModel):
    query: str
    depth: int = 1

@router.post("/infer")
async def recursive_infer(request: RecursiveRequest):
    response = recursive_model_live(request.query, request.depth)
    return {"response": response, "depth": request.depth}

