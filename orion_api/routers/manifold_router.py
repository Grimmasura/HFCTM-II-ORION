"""Manifold router for distributing tasks with depth metrics."""

from fastapi import APIRouter
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

    logger.info("prometheus_client not installed; manifold depth metrics disabled")

manifold_depth_metric = Histogram(
    "orion_manifold_depth",
    "Recursive depth used when distributing tasks",
)

router = APIRouter()


@router.post("/distribute_task")
async def distribute_task(task: str, depth: int) -> dict:
    """Distribute tasks among recursive agents and record depth."""
    manifold_depth_metric.observe(depth)
    return {"message": f"Task '{task}' distributed with recursion depth {depth}."}

