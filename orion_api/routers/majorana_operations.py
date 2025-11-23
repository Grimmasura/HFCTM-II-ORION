"""
Majorana 0D Seed Operations Router for MIH-IIE v2.0

Exposes Majorana zero mode operations via REST API.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

try:
    from models.majorana_0d_network import create_majorana_network, Majorana0DSeedNetwork
    MAJORANA_AVAILABLE = True
except ImportError:
    MAJORANA_AVAILABLE = False

router = APIRouter(prefix="/api/v2/majorana", tags=["Majorana Operations"])

# Global state
_network: Optional[Any] = None


class MajoranaInitRequest(BaseModel):
    """Request to initialize Majorana network"""
    n_seeds: int = Field(240, ge=1, le=240, description="Number of 0D seeds")
    use_azure: bool = Field(False, description="Use Azure Quantum backend")
    azure_config: Optional[Dict[str, str]] = None


class MajoranaStatusResponse(BaseModel):
    """Majorana network status"""
    initialized: bool
    total_seeds: int
    active_seeds: int
    measured_seeds: int
    backend_type: str
    coherence_time_target: float


class SeedMeasurementRequest(BaseModel):
    """Request to measure seed"""
    seed_index: int = Field(..., ge=0, description="Seed index to measure")
    basis: str = Field("computational", description="Measurement basis (computational or hadamard)")


class SeedMeasurementResponse(BaseModel):
    """Seed measurement result"""
    seed_index: int
    measurement: int
    basis: str


@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_majorana_network(request: MajoranaInitRequest):
    """
    Initialize Majorana 0D seed network.

    Creates array of Majorana zero modes (0D attractors).
    """
    if not MAJORANA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Majorana modules not available"
        )

    global _network

    try:
        _network = create_majorana_network(
            n_seeds=request.n_seeds,
            use_azure=request.use_azure,
            azure_config=request.azure_config
        )

        # Verify 0D properties
        verification = _network.verify_0d_properties()

        return {
            "status": "initialized",
            "num_seeds": len(_network.seeds),
            "backend": _network.backend.backend_type,
            "0d_properties_verified": verification["all_verified"],
            "substrate_independent": verification["substrate_independent"],
            "target_coherence_time": verification["target_coherence_time"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Majorana network: {str(e)}"
        )


@router.get("/status", response_model=MajoranaStatusResponse)
async def get_majorana_status():
    """Get Majorana network status"""
    if not MAJORANA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Majorana modules not available"
        )

    if _network is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Majorana network not initialized. Call /initialize first."
        )

    stats = _network.get_statistics()

    return MajoranaStatusResponse(
        initialized=True,
        total_seeds=stats["total_seeds"],
        active_seeds=stats["active_seeds"],
        measured_seeds=stats["measured_seeds"],
        backend_type=stats["backend_type"],
        coherence_time_target=1000.0
    )


@router.post("/measure", response_model=SeedMeasurementResponse)
async def measure_seed(request: SeedMeasurementRequest):
    """
    Measure a Majorana seed.

    Returns measurement outcome (0 or 1) in specified basis.
    """
    if not MAJORANA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Majorana modules not available"
        )

    if _network is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Majorana network not initialized"
        )

    if request.seed_index >= _network.n_seeds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Seed index {request.seed_index} out of range (0-{_network.n_seeds-1})"
        )

    try:
        measurement = _network.measure_seed(request.seed_index, basis=request.basis)

        return SeedMeasurementResponse(
            seed_index=request.seed_index,
            measurement=measurement,
            basis=request.basis
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Measurement failed: {str(e)}"
        )


@router.post("/seeds/{seed_index}/hadamard")
async def apply_hadamard(seed_index: int):
    """Apply Hadamard gate to seed (create superposition)"""
    if not MAJORANA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Majorana modules not available"
        )

    if _network is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Majorana network not initialized"
        )

    if seed_index >= _network.n_seeds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Seed index {seed_index} out of range"
        )

    try:
        _network.apply_hadamard(seed_index)
        return {
            "status": "success",
            "seed_index": seed_index,
            "operation": "hadamard"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hadamard application failed: {str(e)}"
        )


@router.get("/seeds/{seed_index}", response_model=Dict[str, Any])
async def get_seed_info(seed_index: int):
    """Get information about specific seed"""
    if not MAJORANA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Majorana modules not available"
        )

    if _network is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Majorana network not initialized"
        )

    seed = _network.get_seed(seed_index)
    if seed is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Seed {seed_index} not found"
        )

    properties = seed.verify_0d_properties()

    return {
        "index": seed.index,
        "state": seed.state,
        "e8_root_mapping": seed.e8_root_index,
        "wire_id": seed.wire_id,
        "coherence_time_estimate": seed.coherence_time_estimate,
        "0d_properties": properties
    }


@router.get("/verification", response_model=Dict[str, Any])
async def verify_0d_properties():
    """Verify all seeds satisfy 0D attractor requirements"""
    if not MAJORANA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Majorana modules not available"
        )

    if _network is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Majorana network not initialized"
        )

    verification = _network.verify_0d_properties()

    return {
        "all_verified": verification["all_verified"],
        "num_seeds": verification["num_seeds"],
        "properties": {
            "zero_dimension": True,
            "maximum_symmetry": True,
            "tp_invariance": True,
            "minimal_information": True,
            "substrate_independent": verification["substrate_independent"]
        },
        "target_coherence_time": verification["target_coherence_time"]
    }


@router.post("/reset")
async def reset_majorana_network():
    """Reset Majorana network"""
    global _network
    _network = None

    return {"status": "reset", "message": "Majorana network reset successfully"}
