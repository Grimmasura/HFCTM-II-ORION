"""
L1: 0D Seed / Intrinsic Attractor Module

Provides substrate-independent causal anchoring using Majorana zero modes
as physical realization of 0-dimensional attractors in ontological possibility space.

Status: Phase 1 target (awaiting Majorana qubit hardware)
"""

from .intrinsic_seed import (
    IntrinsicAttractorModule,
    IntrinsicSeed,
    CausalFlow,
    AttractorMutation,
    AttractorType,
    CausalMode
)

__all__ = [
    "IntrinsicAttractorModule",
    "IntrinsicSeed",
    "CausalFlow",
    "AttractorMutation",
    "AttractorType",
    "CausalMode"
]
