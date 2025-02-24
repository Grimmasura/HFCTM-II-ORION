from fastapi import FastAPI
from orion_api.routers import recursive_ai, quantum_sync, recursive_trust, egregore_defense, manifold_router

app = FastAPI(title="O.R.I.O.N. ∞ API")

app.include_router(recursive_ai.router)
app.include_router(quantum_sync.router)
app.include_router(recursive_trust.router)
app.include_router(egregore_defense.router)
app.include_router(manifold_router.router)

@app.get("/")
async def root():
    return {"message": "Welcome to O.R.I.O.N. ∞"}
