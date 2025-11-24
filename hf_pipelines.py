"""
Hugging Face pipeline stubs for MIH-IIE.

These wrappers provide a familiar interface around E8 verification and
EDS frame evaluation. They call the local library; swap the call sites to
hit a deployed FastAPI/MCP endpoint if desired.
"""

from typing import Any, Dict, List

from mih_iie.layers.l2_majorana_array import compute_e8_invariants, verify_e8_invariants
from models.v2_1 import ConvergenceEvaluator, get_default_frames


class E8VerificationPipeline:
    task = "e8-verification"

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        report = compute_e8_invariants()
        checks = verify_e8_invariants(report)
        return {"summary": report.summary(), "checks": checks}


class EDSFramePipeline:
    task = "eds-eval"

    def __call__(self, proposition: str) -> Dict[str, Any]:
        frames = get_default_frames()
        ev = ConvergenceEvaluator()
        results = [f.evaluate(proposition) for f in frames]
        return {
            "convergence": ev.convergence(results),
            "frames": [
                {"name": r.name, "confidence": r.confidence, "vector": r.vector.tolist()}
                for r in results
            ],
        }
