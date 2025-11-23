"""
E8 Operations Router for MIH-IIE v2.0

Exposes E8 network topology operations via REST API.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from models.e8_topology import E8RootSystem, E8QuantumNetwork
    from models.e8_coordination import create_e8_coordination
    E8_AVAILABLE = True
except ImportError:
    E8_AVAILABLE = False

router = APIRouter(prefix="/api/v2/e8", tags=["E8 Operations"])

# Global state
_root_system: Optional[Any] = None
_coordination: Optional[Any] = None


class E8InitRequest(BaseModel):
    """Request to initialize E8 system"""
    n_nodes: int = Field(240, ge=8, le=240, description="Number of E8 nodes (8-240)")


class E8StatusResponse(BaseModel):
    """E8 system status"""
    initialized: bool
    num_roots: int
    adjacency_56_regular: bool
    coordination_active: bool
    diameter: int


class E8NetworkStatsResponse(BaseModel):
    """E8 network statistics"""
    num_roots: int
    num_entanglements: int
    expected_entanglements: int
    all_nodes_connected: bool


class OperationScheduleRequest(BaseModel):
    """Request to schedule parallel operations"""
    operations: List[List[int]] = Field(..., description="List of [i, j] operation pairs")


class OperationScheduleResponse(BaseModel):
    """Scheduled operation groups"""
    num_groups: int
    groups: List[List[List[int]]]
    total_operations: int


@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_e8_system(request: E8InitRequest):
    """
    Initialize E8 root system and coordination.

    Creates complete E8 topology with specified number of nodes.
    """
    if not E8_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E8 modules not available"
        )

    global _root_system, _coordination

    try:
        # Create root system
        _root_system = E8RootSystem()

        # Extract substructure if less than 240 nodes
        if request.n_nodes < 240:
            selected, _ = _root_system.extract_substructure(request.n_nodes)
            active_nodes = selected
        else:
            active_nodes = list(range(240))

        # Create coordination
        _coordination = create_e8_coordination(_root_system)

        # Verify structure
        verification = _root_system.verify_structure()

        return {
            "status": "initialized",
            "num_roots": len(_root_system.roots),
            "num_active_nodes": len(active_nodes),
            "structure_valid": verification["valid"],
            "coordination_active": True,
            "properties": verification
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize E8 system: {str(e)}"
        )


@router.get("/status", response_model=E8StatusResponse)
async def get_e8_status():
    """Get current E8 system status"""
    if not E8_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E8 modules not available"
        )

    if _root_system is None or _coordination is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="E8 system not initialized. Call /initialize first."
        )

    verification = _root_system.verify_structure()
    coord_status = _coordination.verify_coordination()

    adjacency = _root_system.adjacency_matrix
    is_56_regular = bool(np.all(adjacency.sum(axis=1) == 56)) if adjacency is not None else False

    return E8StatusResponse(
        initialized=True,
        num_roots=len(_root_system.roots),
        adjacency_56_regular=is_56_regular,
        coordination_active=coord_status["coordination_active"],
        diameter=verification["diameter"]
    )


@router.get("/network/stats", response_model=E8NetworkStatsResponse)
async def get_network_stats():
    """Get E8 quantum network statistics"""
    if not E8_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E8 modules not available"
        )

    if _root_system is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="E8 system not initialized"
        )

    # Create network
    network = E8QuantumNetwork(_root_system)
    entanglements = network.establish_network()
    topology = network.verify_topology()

    return E8NetworkStatsResponse(
        num_roots=len(_root_system.roots),
        num_entanglements=len(entanglements),
        expected_entanglements=topology["expected_bell_pairs"],
        all_nodes_connected=topology["all_nodes_56_connected"]
    )


@router.post("/operations/schedule", response_model=OperationScheduleResponse)
async def schedule_operations(request: OperationScheduleRequest):
    """
    Schedule operations for parallel execution preserving E8 symmetry.

    Groups operations that can run in parallel without node conflicts.
    """
    if not E8_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E8 modules not available"
        )

    if _coordination is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coordination not initialized"
        )

    try:
        # Convert to tuples
        operations = [tuple(op) for op in request.operations]

        # Schedule
        groups = _coordination.schedule_operations(operations)

        return OperationScheduleResponse(
            num_groups=len(groups),
            groups=[[list(op) for op in group] for group in groups],
            total_operations=len(operations)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule operations: {str(e)}"
        )


@router.get("/roots", response_model=Dict[str, Any])
async def get_root_vectors():
    """Get E8 root vectors"""
    if not E8_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E8 modules not available"
        )

    if _root_system is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="E8 system not initialized"
        )

    roots_data = []
    for root in _root_system.roots[:10]:  # Return first 10 for brevity
        roots_data.append({
            "index": root.index,
            "type": root.root_type,
            "vector": root.vector.tolist()
        })

    return {
        "total_roots": len(_root_system.roots),
        "sample_size": len(roots_data),
        "roots": roots_data
    }


@router.get("/coordination/sync", response_model=Dict[str, Any])
async def get_synchronization_status():
    """Get polychronic synchronization status"""
    if not E8_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E8 modules not available"
        )

    if _coordination is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coordination not initialized"
        )

    sync_status = _coordination.synchronizer.check_synchronization()

    return {
        "synchronized": sync_status["synchronized"],
        "num_anchors": sync_status.get("num_anchors", 0),
        "ghz_coherence": sync_status.get("ghz_coherence", 0.0),
        "temporal_frames": ["linear", "circular", "atemporal", "meta"]
    }


@router.post("/reset")
async def reset_e8_system():
    """Reset E8 system"""
    global _root_system, _coordination

    _root_system = None
    _coordination = None

    return {"status": "reset", "message": "E8 system reset successfully"}
