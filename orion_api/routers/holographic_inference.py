"""
Holographic Inference Router for MIH-IIE v2.0

Exposes holographic state readout and inference via REST API.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from models.holographic_readout import create_holographic_readout, HolographicReadoutProtocol
    from models.e8_topology import E8RootSystem
    HOLOGRAPHIC_AVAILABLE = True
except ImportError:
    HOLOGRAPHIC_AVAILABLE = False

router = APIRouter(prefix="/api/v2/holographic", tags=["Holographic Inference"])

# Global state
_readout: Optional[Any] = None
_root_system: Optional[Any] = None


class InferenceRequest(BaseModel):
    """Request for holographic inference"""
    query: List[float] = Field(..., description="8D query vector in E8 space")


class InferenceResponse(BaseModel):
    """Holographic inference result"""
    inference_vector: List[float]
    inference_magnitude: float
    boundary_measurements: int
    bulk_state_reconstructed: bool


class BoundaryMeasurementResponse(BaseModel):
    """Boundary measurement result"""
    num_boundary_nodes: int
    boundary_fraction: float
    measurements: List[Dict[str, Any]]


@router.post("/initialize")
async def initialize_holographic_readout():
    """Initialize holographic readout protocol"""
    if not HOLOGRAPHIC_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holographic modules not available"
        )

    global _readout, _root_system

    try:
        _root_system = E8RootSystem()
        _readout = create_holographic_readout(_root_system)

        stats = _readout.get_statistics()

        return {
            "status": "initialized",
            "num_roots": stats["num_roots"],
            "num_boundary_nodes": stats["num_boundary_nodes"],
            "boundary_fraction": stats["boundary_fraction"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize holographic readout: {str(e)}"
        )


@router.post("/inference", response_model=InferenceResponse)
async def execute_inference(request: InferenceRequest):
    """
    Execute holographic inference.

    Process:
    1. Encode query into E8 network
    2. Measure boundary nodes
    3. Reconstruct bulk state
    4. Compute inference vector
    """
    if not HOLOGRAPHIC_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holographic modules not available"
        )

    if _readout is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Holographic readout not initialized. Call /initialize first."
        )

    if len(request.query) != 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be 8-dimensional (E8 space)"
        )

    try:
        query_vector = np.array(request.query)
        result = _readout.execute_inference(query_vector)

        return InferenceResponse(
            inference_vector=result["inference_vector"].tolist(),
            inference_magnitude=result["inference_magnitude"],
            boundary_measurements=result["boundary_measurements"],
            bulk_state_reconstructed=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.get("/boundary/measure", response_model=BoundaryMeasurementResponse)
async def measure_boundary():
    """
    Measure holographic boundary nodes.

    Per holographic principle: boundary measurements encode bulk state.
    """
    if not HOLOGRAPHIC_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holographic modules not available"
        )

    if _readout is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Holographic readout not initialized"
        )

    try:
        measurements = _readout.measure_boundary()

        measurement_data = []
        for node_idx, meas in list(measurements.items())[:10]:  # First 10 for brevity
            measurement_data.append({
                "node_index": node_idx,
                "z_basis": meas.z_basis,
                "x_basis": meas.x_basis,
                "e8_coordinate": meas.e8_coordinate.tolist()
            })

        stats = _readout.get_statistics()

        return BoundaryMeasurementResponse(
            num_boundary_nodes=stats["num_boundary_nodes"],
            boundary_fraction=stats["boundary_fraction"],
            measurements=measurement_data
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Boundary measurement failed: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_holographic_status():
    """Get holographic readout status"""
    if not HOLOGRAPHIC_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holographic modules not available"
        )

    if _readout is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Holographic readout not initialized"
        )

    stats = _readout.get_statistics()

    return {
        "initialized": True,
        "num_roots": stats["num_roots"],
        "num_boundary_nodes": stats["num_boundary_nodes"],
        "boundary_fraction": stats["boundary_fraction"],
        "adjacency_constructed": stats["adjacency_constructed"]
    }


@router.post("/reset")
async def reset_holographic():
    """Reset holographic readout"""
    global _readout, _root_system

    _readout = None
    _root_system = None

    return {"status": "reset", "message": "Holographic readout reset successfully"}
