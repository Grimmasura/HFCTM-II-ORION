from fastapi import FastAPI, Request
from orion_api.routers import (
    recursive_ai,
    quantum_sync,
    recursive_trust,
    egregore_defense,
    manifold_router,
)
from models.stability_core import StabilityCore

app = FastAPI(title="O.R.I.O.N. ∞ API")
app.state.stability_core = StabilityCore()

# Include routers from orion_api
app.include_router(recursive_ai.router, prefix="/api/v1/recursive_ai", tags=["Recursive AI"])
app.include_router(quantum_sync.router, prefix="/quantum-sync", tags=["Quantum Sync"])
app.include_router(recursive_trust.router, prefix="/trust", tags=["Recursive Trust"])
app.include_router(egregore_defense.router, prefix="/egregore", tags=["Egregore Defense"])
app.include_router(manifold_router.router, prefix="/manifold", tags=["Manifold Routing"])

@app.get("/")
async def root():
    return {"message": "Welcome to O.R.I.O.N. ∞ API"}


@app.get("/health")
async def health(request: Request) -> dict:
    """Simple service liveness endpoint."""
    return request.app.state.stability_core.health()


@app.get("/telemetry")
async def telemetry(request: Request) -> dict:
    """Expose a snapshot of recent inference telemetry."""
    return request.app.state.stability_core.snapshot()
