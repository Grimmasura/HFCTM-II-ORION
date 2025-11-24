"""
Lightweight MCP-style server exposing core MIH-IIE utilities.

This module provides FastAPI endpoints that mirror MCP tools for:
- E8 invariant verification
- Braid compilation
- EDS frame evaluation
- Stabilizer construction (bounded)
- Holography inference (edge-limited)

Intended for local/CI use; add auth before exposing externally.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import numpy as np

from mih_iie.layers.l2_majorana_array import (
    compute_e8_invariants,
    verify_e8_invariants,
)
from mih_iie.layers.l2_majorana_array import (
    compile_reflection_sequence,
    build_e8_coxeter_matrix,
)
from models.v2_1 import (
    E8,
    FrameResult,
    ConvergenceEvaluator,
    construct_stabilizers,
    identify_boundary,
    build_e8_tensor_network,
    reconstruct_bulk,
    get_default_frames,
)


app = FastAPI(title="MIH-IIE MCP Server", description="Exposes MCP-like tools for MIH-IIE")


@app.get("/mcp/e8/invariants")
async def e8_invariants() -> Dict[str, Any]:
    report = compute_e8_invariants()
    checks = verify_e8_invariants(report)
    return {
        "summary": report.summary(),
        "checks": checks,
    }


@app.post("/mcp/e8/braid")
async def e8_braid(sequence: List[int]) -> Dict[str, Any]:
    if not sequence:
        raise HTTPException(status_code=400, detail="Sequence required")
    cox = build_e8_coxeter_matrix()
    word = compile_reflection_sequence(sequence, coxeter=cox)
    return {"normalized_word": list(word.generators), "string": str(word)}


@app.post("/mcp/eds/evaluate")
async def eds_evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    proposition = payload.get("proposition", "")
    frames = get_default_frames()
    ev = ConvergenceEvaluator()
    results = [f.evaluate(proposition) for f in frames]
    convergence = ev.convergence(results)
    return {
        "convergence": convergence,
        "frames": [
            {"name": r.name, "confidence": r.confidence, "vector": r.vector.tolist()}
            for r in results
        ],
    }


@app.get("/mcp/stabilizers")
async def stabilizers(limit: Optional[int] = 32) -> Dict[str, Any]:
    e8 = E8.generate_roots()
    stabs = construct_stabilizers(e8, limit=limit)
    return {"count": len(stabs), "stabilizers": [s.indices for s in stabs]}


@app.post("/mcp/holography/infer")
async def holography_infer(edge_limit: int = 8, percentile: float = 2.0) -> Dict[str, Any]:
    e8 = E8.generate_roots()
    adjacency = e8.adjacency_matrix()
    boundary = identify_boundary(adjacency, method="random", percentile=percentile)
    network = build_e8_tensor_network(e8, adjacency, bond_dim=2, physical_dim=2, max_tensor_rank=2)
    measurements = {
        node: {
            "node": node,
            "z_basis": 0,
            "x_basis": 0,
            "coordinate": e8.root_vectors()[node].tolist(),
        }
        for node in boundary[:4]
    }
    bulk = reconstruct_bulk(
        boundary_measurements=measurements,
        network=network,
        boundary_nodes=boundary[:4],
        edge_limit=edge_limit,
    )
    return {
        "boundary_size": len(boundary),
        "edge_limit": edge_limit,
        "bulk_shape": np.asarray(bulk).shape,
    }
