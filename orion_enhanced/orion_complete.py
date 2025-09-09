from functools import lru_cache
import os, time
from fastapi import FastAPI, Response
from orion_enhanced.metrics import OrionMetrics
from orion_enhanced.egregore.defense import EgregoreDefense
from orion_enhanced.recursion.budget_controller import Budget, BudgetedRecursionController
from orion_enhanced.quantum.stabilizer import QuantumStabilizer


@lru_cache(maxsize=1)
def create_complete_orion_app() -> FastAPI:
    app = FastAPI(title="ORION Complete")

    metrics = OrionMetrics()
    defense = EgregoreDefense(metrics=metrics)  # EXPERIMENTAL: content-drift guard
    def _default_gain(text: str) -> float:
        # Default proxy remains simple but pluggable (replace with coherence estimator later)
        return min(1.0, 0.5 + 0.05 * len(text.split()))
    controller = BudgetedRecursionController(Budget())
    qstab = QuantumStabilizer(client=None, metrics=metrics)
    app.state.quantum_stabilizer = qstab

    @app.get("/system/status")
    def status():
        return {"ok": True}

    @app.get("/system/health")
    def health():
        return {
            "status": "healthy",
            "labels": {
                "egregore_defense": "experimental",
                "quantum_backend": "experimental" if not qstab.available() else "integrated",
                "recursion_budget": "production"
            }
        }

    @app.post("/system/test")
    def test(payload: dict):
        return {"echo": payload}

    @app.post("/system/warmup")
    def warmup():
        t0 = time.time()
        try:
            defense._embedding_shift("a", "b")  # model lazy-init if available
        except Exception:
            pass
        return {"ok": True, "t_ms": int((time.time() - t0) * 1000)}

    @app.get("/metrics")
    def metrics_endpoint():
        content_type, body = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/quantum/status")
    def quantum_status():
        return qstab.status()

    @app.post("/system/inference")
    def inference(payload: dict):
        text = str(payload.get("text", ""))
        baseline = str(payload.get("baseline", ""))
        if defense.should_quarantine(text, baseline, err_rate_delta=0.1, tool_mix_delta=0.1):
            return {"quarantined": True, "reason": "egregore_anomaly"}
        depth = int(payload.get("depth", 1))
        est_gain = _default_gain(text)
        if not controller.should_expand(depth, est_gain):
            return {"stopped": True, "depth": depth}
        return {"expanded": True, "next_depth": depth + 1}

    return app
