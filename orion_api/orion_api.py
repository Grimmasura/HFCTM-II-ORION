from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
from orion_api.config import settings

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - import error handling
    TORCH_AVAILABLE = False

    class _TorchStub:
        def __getattr__(self, name):  # pragma: no cover - defensive
            raise RuntimeError("PyTorch is not installed")

    torch = _TorchStub()  # type: ignore

app = FastAPI(title="O.R.I.O.N. ∞ API", version="1.0")

class RecursiveRequest(BaseModel):
    query: str
    depth: int = 0

@app.post("/api/v1/recursive_infer")
async def recursive_infer(request: RecursiveRequest):
    return {"response": f"Recursive Expansion → {request.query}", "depth": request.depth + 1}

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
