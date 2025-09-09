from .backend import AbstractQuantumBackend, NullBackend, HeuristicNormBackend
import math
import numpy as np

class QuantumStabilizer:
    def __init__(self, client=None, metrics=None, backend: AbstractQuantumBackend | None = None):
        self.client = client
        self.metrics = metrics
        self.backend = backend or (HeuristicNormBackend() if client else NullBackend())

    def available(self) -> bool:
        ok = bool(self.client)
        try:
            if self.metrics:
                self.metrics.quantum_unavailable.set(0 if ok else 1)
        except Exception:
            pass
        return ok

    def status(self) -> dict:
        return {
            "available": self.available(),
            "backend": getattr(self.client, "name", None),
            "mode": self.backend.__class__.__name__,
            "experimental": not isinstance(self.backend, NullBackend) and not self.available(),
        }

    def solve(self, payload: dict) -> dict:
        try:
            return self.backend.solve(payload)
        except Exception as e:
            if self.metrics:
                try:
                    self.metrics.quarantine_events.inc()
                except Exception:
                    pass
            return {"result": None, "degraded": True, "reason": f"quantum_error:{type(e).__name__}"}
