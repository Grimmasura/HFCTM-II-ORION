"""Recursive AI router with depth metric instrumentation."""

from fastapi import APIRouter
from pydantic import BaseModel
from models.recursive_ai_model import recursive_model_live

from orion_api.telemetry.prometheus import Histogram

recursive_depth_metric = Histogram(
    "orion_recursive_ai_depth",
    "Depth of recursion requested for the recursive AI endpoint",
)

router = APIRouter()


class RecursiveRequest(BaseModel):
    """Request model for recursive inference calls."""

    query: str
    depth: int = 1


@router.post("/infer")
async def recursive_infer(request: RecursiveRequest) -> dict:
    """Perform recursive inference and track recursion depth."""
    recursive_depth_metric.observe(request.depth)
    response = recursive_model_live(request.query, request.depth)
    return {"response": response, "depth": request.depth}

