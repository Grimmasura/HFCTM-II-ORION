from __future__ import annotations

class QuantumStabilizer:
    def __init__(self, client=None, metrics=None):
        self.client = client
        self.metrics = metrics

    def available(self) -> bool:
        ok = bool(self.client)
        try:
            if self.metrics:
                self.metrics.quantum_unavailable.set(0 if ok else 1)
        except Exception:
            pass
        return ok

    def status(self) -> dict:
        return {"available": self.available(), "backend": getattr(self.client, "name", None)}

    def solve(self, payload: dict) -> dict:
        if not self.available():
            # graceful degradation path: return echo with a flag
            return {"result": None, "degraded": True, "reason": "quantum_unavailable"}
        # integration point to the real quantum backend
        return {"result": {"objective": 0.0}, "degraded": False}
