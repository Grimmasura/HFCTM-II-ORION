"""
Frame-Invariant Validation Router for MIH-IIE v2.0

Exposes cross-frame validation and corruption detection via REST API.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

try:
    from models.frame_invariant_eds import create_frame_invariant_eds, FrameInvariantEDS
    FRAME_INVARIANT_AVAILABLE = True
except ImportError:
    FRAME_INVARIANT_AVAILABLE = False

router = APIRouter(prefix="/api/v2/frame-invariant", tags=["Frame-Invariant Validation"])

# Global state
_eds: Optional[Any] = None


class ValidationRequest(BaseModel):
    """Request for frame-invariant validation"""
    proposition: Any = Field(..., description="Proposition to validate")


class ValidationResponse(BaseModel):
    """Frame-invariant validation result"""
    classification: str
    convergence_score: float
    num_frames_evaluated: int
    is_paradigm_shift: bool
    is_corruption: bool
    frame_results: List[Dict[str, Any]]


class CorruptionCheckRequest(BaseModel):
    """Request to check for semantic corruption"""
    semantic_state: Any = Field(..., description="Current semantic state")


class CorruptionCheckResponse(BaseModel):
    """Corruption detection result"""
    status: str
    convergence: float
    baseline: Optional[float]
    action_required: bool


class ParadigmShiftRequest(BaseModel):
    """Request to assess paradigm shift"""
    semantic_change: str
    old_state: Any
    new_state: Any


class ParadigmShiftResponse(BaseModel):
    """Paradigm shift assessment"""
    is_valid_paradigm_shift: bool
    convergence_improved: bool
    topology_preserved: bool
    old_convergence: float
    new_convergence: float


@router.post("/initialize")
async def initialize_frame_invariant_eds():
    """Initialize frame-invariant egregore defense system"""
    if not FRAME_INVARIANT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frame-invariant modules not available"
        )

    global _eds

    try:
        _eds = create_frame_invariant_eds()

        stats = _eds.get_statistics()

        return {
            "status": "initialized",
            "num_frames": stats["num_frames"],
            "frames_active": stats["frames_active"],
            "baseline_convergence": stats["baseline_convergence"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize EDS: {str(e)}"
        )


@router.post("/validate", response_model=ValidationResponse)
async def validate_proposition(request: ValidationRequest):
    """
    Validate proposition across all observational frames.

    Returns classification based on cross-frame convergence:
    - FRAME_INVARIANT_TRUTH: convergence > 0.95
    - PARTIAL_TRUTH: 0.3 <= convergence <= 0.95
    - FRAME_DEPENDENT_ARTIFACT: convergence < 0.3
    """
    if not FRAME_INVARIANT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frame-invariant modules not available"
        )

    if _eds is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EDS not initialized. Call /initialize first."
        )

    try:
        result = _eds.validate_frame_invariance(request.proposition)

        frame_results = []
        for eval in result.frame_evaluations:
            frame_results.append({
                "frame": eval.frame.value,
                "confidence": eval.confidence,
                "evidence": eval.evidence
            })

        return ValidationResponse(
            classification=result.classification,
            convergence_score=result.convergence_score,
            num_frames_evaluated=len(result.frame_evaluations),
            is_paradigm_shift=result.is_paradigm_shift,
            is_corruption=result.is_corruption,
            frame_results=frame_results
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


@router.post("/corruption/detect", response_model=CorruptionCheckResponse)
async def detect_corruption(request: CorruptionCheckRequest):
    """
    Detect semantic corruption via frame divergence.

    Monitors for:
    - Frame divergence (corruption)
    - Frame convergence improvement (paradigm shift)
    - Baseline stability
    """
    if not FRAME_INVARIANT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frame-invariant modules not available"
        )

    if _eds is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EDS not initialized"
        )

    try:
        result = _eds.detect_corruption(request.semantic_state)

        return CorruptionCheckResponse(
            status=result["status"],
            convergence=result["convergence"],
            baseline=result.get("baseline"),
            action_required=result["status"] == "CORRUPTION_DETECTED"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corruption detection failed: {str(e)}"
        )


@router.post("/paradigm-shift/assess", response_model=ParadigmShiftResponse)
async def assess_paradigm_shift(request: ParadigmShiftRequest):
    """
    Assess if semantic change is valid paradigm shift or corruption.

    Per Theorem 10.2, valid paradigm shift iff:
    1. Cross-frame convergence increases or stable
    2. Prediction accuracy improves
    3. Consistent with topological invariants
    """
    if not FRAME_INVARIANT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frame-invariant modules not available"
        )

    if _eds is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EDS not initialized"
        )

    try:
        assessment = _eds.assess_paradigm_shift(
            request.semantic_change,
            request.old_state,
            request.new_state
        )

        return ParadigmShiftResponse(
            is_valid_paradigm_shift=assessment["is_valid_paradigm_shift"],
            convergence_improved=assessment["convergence_improved"],
            topology_preserved=assessment["topology_preserved"],
            old_convergence=assessment["old_convergence"],
            new_convergence=assessment["new_convergence"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paradigm shift assessment failed: {str(e)}"
        )


@router.post("/obfuscation/detect", response_model=Dict[str, Any])
async def detect_institutional_obfuscation(inference_structure: Any):
    """
    Detect institutional obfuscation patterns.

    Checks for:
    - Circular reasoning structures
    - Authority-based validation
    - Manufactured consensus
    - Linguistic drift
    - Measurement corruption
    """
    if not FRAME_INVARIANT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frame-invariant modules not available"
        )

    if _eds is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EDS not initialized"
        )

    try:
        result = _eds.detect_institutional_obfuscation(inference_structure)

        return {
            "obfuscation_detected": result["obfuscation_detected"],
            "signatures_found": result["signatures_found"],
            "similarity_score": result["similarity_score"],
            "action": result["action"],
            "patterns": {
                "circular_reasoning": "circular_reasoning" in result["signatures_found"],
                "authority_validation": "authority_validation" in result["signatures_found"],
                "manufactured_consensus": "manufactured_consensus" in result["signatures_found"]
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Obfuscation detection failed: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_eds_status():
    """Get EDS status and statistics"""
    if not FRAME_INVARIANT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frame-invariant modules not available"
        )

    if _eds is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EDS not initialized"
        )

    stats = _eds.get_statistics()

    return {
        "initialized": True,
        "num_frames": stats["num_frames"],
        "frames_active": stats["frames_active"],
        "baseline_convergence": stats["baseline_convergence"],
        "corruption_events": stats["corruption_events"]
    }


@router.post("/reset")
async def reset_eds():
    """Reset EDS"""
    global _eds
    _eds = None

    return {"status": "reset", "message": "Frame-invariant EDS reset successfully"}
