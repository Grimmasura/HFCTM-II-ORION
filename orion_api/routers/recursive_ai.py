from fastapi import APIRouter
from pydantic import BaseModel
from models.recursive_ai_model import recursive_model_live

router = APIRouter()


class RecursiveRequest(BaseModel):
    query: str
    depth: int = 1
    chi_Eg: int = 0
    lambda_: float = 0.0


@router.post("/infer")
async def recursive_infer(request: RecursiveRequest):
    """Run the recursive model with telemetry and safety guards."""
    response = recursive_model_live(
        request.query,
        request.depth,
        chi_Eg=request.chi_Eg,
        lambda_=request.lambda_,
    )
    return {"response": response, "depth": request.depth}

