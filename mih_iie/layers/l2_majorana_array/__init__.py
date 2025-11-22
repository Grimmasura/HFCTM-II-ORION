"""
L2: Majorana Topological Qubit Array

Topological quantum computation using non-Abelian braiding operations on
E8 lattice structure.

Target Specifications:
- Array size: 1000×1000 qubits
- T₂ coherence: >1000 seconds
- Gate operations: 10^12/second
- Physical error rate: <10^-4
- Logical error rate: <10^-15 (with surface code)

Status: Phase 1 target (interface defined, awaiting hardware)
Reference: Section 4 of MIH-IIE specification
"""

from .majorana_qubit import (
    MajoranaQubitArray,
    TopologicalQubit,
    MajoranaZeroMode,
    E8Lattice,
    BraidOperation,
    BraidingResult
)

__all__ = [
    "MajoranaQubitArray",
    "TopologicalQubit",
    "MajoranaZeroMode",
    "E8Lattice",
    "BraidOperation",
    "BraidingResult"
]
