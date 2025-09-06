from fastapi import FastAPI
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

app = FastAPI(title="O.R.I.O.N. ∞ API")

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
    return {"message": "Welcome to O.R.I.O.N. ∞ API"}


@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/telemetry")
async def telemetry() -> dict:
    """Expose a snapshot of the StabilityCore telemetry."""
    return {"telemetry": stability_core.snapshot()}
