"""
Three-frame Egregore Defense System demo.

Runs forward, retro, and atemporal evaluators to surface convergence,
corruption, and shift scores in a single pass.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from mih_iie.layers.l5_governance.egregore_defense import (
    EgregoreDefenseSystem,
    SemanticState,
)


def _baseline_field() -> Dict[str, str]:
    return {
        "hfctm": "holographic fractal chiral toroidal mechanics",
        "majorana": "non-abelian quasiparticle anchor",
        "ironwood": "toroidal projection engine",
    }


def _shifted_field(offset: str) -> Dict[str, str]:
    return {k: f"{v} {offset}".strip() for k, v in _baseline_field().items()}


def _frame_result(frame: str, offset: str, inference_structure: Dict[str, object]) -> Dict[str, object]:
    eds = EgregoreDefenseSystem(torsion_threshold_sigma=2.0, similarity_threshold=0.75)
    baseline = SemanticState(timestamp=0.0, symbol_mappings=_baseline_field(), torsion_measure=0.0)
    eds.baseline = baseline
    eds.baseline_torsion_mean = 0.0
    eds.baseline_torsion_std = 0.05

    torsion = eds.measure_semantic_torsion(_shifted_field(offset))
    detections = eds.detect_corrupted_patterns(inference_structure)
    max_similarity = max((score for _, score in detections), default=0.0)

    torsion_limit = eds.baseline_torsion_mean + eds.torsion_threshold_sigma * eds.baseline_torsion_std
    should_quarantine = max_similarity >= eds.similarity_threshold or torsion > torsion_limit

    return {
        "frame": frame,
        "torsion": round(torsion, 4),
        "torsion_limit": round(torsion_limit, 4),
        "max_similarity": round(max_similarity, 3),
        "detections": [p.pattern_id for p, _ in detections],
        "should_quarantine": should_quarantine,
    }


def run_three_frame_demo() -> List[Dict[str, object]]:
    """Execute forward, retro, and atemporal checks with varied stressors."""
    scenarios: List[Tuple[str, str, Dict[str, object]]] = [
        (
            "forward",
            "stable semantics",
            {"type": "evidence_chain", "coherence": 0.93, "dependency_loop": False},
        ),
        (
            "retro",
            "minor drift toward circularity",
            {"type": "circular", "dependency_loop": True, "evidence_required": False},
        ),
        (
            "atemporal",
            "measurement_shift",
            {"type": "measurement", "systematic_bias": True, "semantic_shift": True},
        ),
    ]

    results: List[Dict[str, object]] = []
    for frame, offset, structure in scenarios:
        results.append(_frame_result(frame, offset, structure))
    return results


if __name__ == "__main__":
    summary = run_three_frame_demo()
    print(json.dumps(summary, indent=2))
