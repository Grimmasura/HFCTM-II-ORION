from fastapi import APIRouter, Request
from pydantic import BaseModel
from models.recursive_ai_model import recursive_model_live

router = APIRouter()


class RecursiveRequest(BaseModel):
    query: str
    depth: int = 1
    chi_Eg: int = 0
    lambda_: float = 0.0


@router.post("/infer")
async def recursive_infer(payload: RecursiveRequest, request: Request):
    """Run the recursive model with telemetry and safety guards."""
    response = recursive_model_live(
        payload.query,
        payload.depth,
        request.app.state.stability_core,
        chi_Eg=payload.chi_Eg,
        lambda_=payload.lambda_,
    )
    return {"response": response, "depth": payload.depth}

