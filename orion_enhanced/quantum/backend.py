from abc import ABC, abstractmethod
from typing import Any, Dict

class AbstractQuantumBackend(ABC):
    @abstractmethod
    def solve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

class NullBackend(AbstractQuantumBackend):
    """Production-safe: declares no quantum capability."""
    def solve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": None, "degraded": True, "reason": "quantum_unavailable"}

class HeuristicNormBackend(AbstractQuantumBackend):
    """Experimental: placeholder objective (L2 norm of normalized rho)."""
    def _normalize_density(self, state):
        import math, numpy as np
        arr = np.asarray(state).reshape(-1)
        n = int(math.isqrt(arr.size))
        if n * n != arr.size:
            need = (n + 1) ** 2
            if arr.size < need:
                arr = np.pad(arr, (0, need - arr.size))
            else:
                arr = arr[:need]
            n += 1
        return arr.reshape(n, n)

    def solve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        state = payload.get("rho")
        if state is None:
            return {"result": {"objective": 0.0}, "degraded": False}
        rho = self._normalize_density(state)
        return {"result": {"objective": float(np.linalg.norm(rho))}, "degraded": False}
