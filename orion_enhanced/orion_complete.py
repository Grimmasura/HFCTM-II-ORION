from fastapi import FastAPI, Response
from orion_enhanced.metrics import OrionMetrics
from orion_enhanced.egregore.defense import EgregoreDefense, Quarantine
from orion_enhanced.recursion.budget_controller import Budget, BudgetedRecursionController
from orion_enhanced.quantum.stabilizer import QuantumStabilizer


def create_complete_orion_app() -> FastAPI:
    app = FastAPI(title="ORION Complete")

    metrics = OrionMetrics()
    defense = EgregoreDefense(metrics=metrics)
    budget = Budget()
    controller = BudgetedRecursionController(budget)
    qstab = QuantumStabilizer(client=None, metrics=metrics)

    @app.get("/system/status")
    def status():
        return {"ok": True}

    @app.get("/system/health")
    def health():
        return {"status": "healthy"}

    @app.post("/system/test")
    def test(payload: dict):
        return {"echo": payload}

    @app.get("/metrics")
    def metrics_endpoint():
        content_type, body = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/quantum/status")
    def quantum_status():
        return qstab.status()

    @app.post("/system/inference")
    def inference(payload: dict):
        # toy demo: examine payload complexity as marginal gain proxy
        text = str(payload.get("text", ""))
        baseline = str(payload.get("baseline", ""))
        try:
            defense.guard(text, baseline, err_rate_delta=0.1, tool_mix_delta=0.1)
        except Quarantine as e:
            return {"quarantined": True, "reason": str(e)}

        depth = int(payload.get("depth", 1))
        est_gain = min(1.0, 0.5 + 0.05 * len(text.split()))
        if not controller.should_expand(depth, est_gain):
            return {"stopped": True, "depth": depth}
        # pretend we expanded once
        return {"expanded": True, "next_depth": depth + 1}

    return app
