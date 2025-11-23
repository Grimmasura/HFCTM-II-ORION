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
from .braid_compiler import (
    BraidWord,
    build_e8_coxeter_matrix,
    braid_word_from_generators,
    compile_reflection_sequence,
    simple_reflection_basis,
)
from .e8_verification import (
    compute_e8_invariants,
    verify_e8_invariants,
)

__all__ = [
    "MajoranaQubitArray",
    "TopologicalQubit",
    "MajoranaZeroMode",
    "E8Lattice",
    "BraidOperation",
    "BraidingResult",
    "BraidWord",
    "build_e8_coxeter_matrix",
    "braid_word_from_generators",
    "compile_reflection_sequence",
    "simple_reflection_basis",
    "compute_e8_invariants",
    "verify_e8_invariants",
]
