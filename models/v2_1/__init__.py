"""
MIH-IIE v2.1 Reference Implementations

This module contains the official v2.1 reference implementations
from the MIH-IIE specification.

Key improvements over v2.0:
- Integer-scaled E8 representation for numerical stability
- Immutable dataclasses with frozen=True
- Abstract backend interfaces
- Explicit type hints
- Pure functional algorithms where possible

Reference: spec/mih_iie_v2_1_spec.pdf
"""

__version__ = "2.1.0"

# Fix relative imports in reference implementations
import sys
from pathlib import Path

# Make imports work correctly
_module_path = Path(__file__).parent
sys.path.insert(0, str(_module_path))

try:
    from .e8 import E8, Vector, RootTuple
    from .coordination import (
        BellPair,
        MajoranaZeroMode,
        EntanglementRegistry,
        QuantumBackend,
    )
    from .eds import (
        FrameResult,
        ValidationState,
        SystemState,
        ConvergenceEvaluator,
    )
    from .holography import (
        BoundaryMeasurement,
        TensorNode,
        TensorNetwork,
        identify_boundary,
    )
    from .error_correction import (
        Stabilizer,
        Syndrome,
        ErrorLocation,
        CorrectionResult,
        construct_stabilizers,
    )

    V2_1_AVAILABLE = True
except ImportError as e:
    print(f"Warning: v2.1 imports failed: {e}")
    V2_1_AVAILABLE = False

__all__ = [
    "E8",
    "Vector",
    "RootTuple",
    "BellPair",
    "MajoranaZeroMode",
    "EntanglementRegistry",
    "QuantumBackend",
    "FrameResult",
    "ValidationState",
    "SystemState",
    "ConvergenceEvaluator",
    "BoundaryMeasurement",
    "TensorNode",
    "TensorNetwork",
    "identify_boundary",
    "Stabilizer",
    "Syndrome",
    "ErrorLocation",
    "CorrectionResult",
    "construct_stabilizers",
    "V2_1_AVAILABLE",
]
