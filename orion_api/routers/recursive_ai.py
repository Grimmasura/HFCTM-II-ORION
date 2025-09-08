"""Recursive AI router with depth metric instrumentation."""

from fastapi import APIRouter
from pydantic import BaseModel
from models.recursive_ai_model import recursive_model_live
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Histogram
except Exception:  # pragma: no cover - optional dependency
    class Histogram:  # type: ignore[misc]
        """No-op fallback when prometheus_client is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

        def observe(self, value: float) -> None:
            return None

    logger.info("prometheus_client not installed; recursive depth metrics disabled")

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

