"""
L3: Quantum-Classical Interface Layer

Manages decoherence, error correction, measurement, and state projection
between L2 (Majorana array) and L4 (Ironwood tensor processing).

Components:
- Decoherence management
- Surface code error correction for non-Abelian anyons
- Measurement protocol
- State projection operators

Status: Phase 1 target
Reference: Section 4.5 of MIH-IIE specification
"""

from .quantum_classical_bridge import (
    QuantumClassicalBridge,
    QuantumState,
    ErrorSyndrome,
    DecoherenceModel,
    ErrorCorrectionCode,
    MeasurementBasis
)

__all__ = [
    "QuantumClassicalBridge",
    "QuantumState",
    "ErrorSyndrome",
    "DecoherenceModel",
    "ErrorCorrectionCode",
    "MeasurementBasis"
]
