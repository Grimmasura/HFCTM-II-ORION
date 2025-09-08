from fastapi import FastAPI, Response
from orion_api.routers import (
    recursive_ai,
    quantum_sync,
    recursive_trust,
    egregore_defense,
    manifold_router,
    knowledge_expansion,
    perception,
)
from models.stability_core import stability_core
from orion_api.config import settings
from .hfctm_safety import init_safety_core, safety_core, SafetyConfig
from pathlib import Path
import subprocess

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest() -> bytes:
        """Fallback generator when prometheus_client is missing."""
        return b""

app = FastAPI(title="O.R.I.O.N. ∞ API")

# Initialize safety core on startup
@app.on_event("startup")
async def startup_event():
    config = SafetyConfig()
    init_safety_core(config)

# Safety middleware
@app.middleware("http")
async def safety_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-HFCTM-Active"] = "true"
    if safety_core:
        response.headers["X-Interventions"] = str(safety_core.intervention_count)
    return response

# Include routers from orion_api
app.include_router(recursive_ai.router, prefix="/api/v1/recursive_ai", tags=["Recursive AI"])
app.include_router(quantum_sync.router, prefix="/quantum-sync", tags=["Quantum Sync"])
app.include_router(recursive_trust.router, prefix="/trust", tags=["Recursive Trust"])
app.include_router(egregore_defense.router, prefix="/egregore", tags=["Egregore Defense"])
app.include_router(manifold_router.router, prefix="/manifold", tags=["Manifold Routing"])
app.include_router(knowledge_expansion.router, prefix="/api/v1/knowledge", tags=["Knowledge Expansion"])
app.include_router(perception.router, prefix="/api/v1/perception", tags=["Perception"])

@app.get("/")
async def root():
    return {
        "message": f"Welcome to O.R.I.O.N. ∞ API",
        "host": settings.host,
        "port": settings.port,
    }


@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/telemetry")
async def telemetry() -> dict:
    """Expose a snapshot of the StabilityCore telemetry."""
    return {"telemetry": stability_core.snapshot()}


@app.get("/version")
async def version() -> dict:
    repo_dir = Path(__file__).resolve().parent.parent
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir)
            .decode()
            .strip()
        )
    except Exception:
        commit_hash = "unknown"
    return {"version": commit_hash}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Add safety endpoint
@app.get("/api/safety/status")
async def safety_status():
    if not safety_core:
        return {"error": "Safety core not initialized"}
    if torch is None:
        return {"error": "Torch not installed"}

    mock_state = torch.randn(10, 10)
    result = await safety_core.safety_check(mock_state)

    return {
        "safety_active": True,
        "interventions_total": safety_core.intervention_count,
        "last_check": result,
    }
