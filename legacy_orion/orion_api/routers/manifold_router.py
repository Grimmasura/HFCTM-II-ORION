"""Manifold router for distributing tasks with depth metrics."""

from fastapi import APIRouter

from orion_api.telemetry.prometheus import Histogram

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

