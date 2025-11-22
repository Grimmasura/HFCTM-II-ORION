"""
L5: Recursive Governance Layer

This layer implements HFCTM-II compliance monitoring, chiral inversion,
polychronic synchronization, and egregore defense as specified in the
MIH-IIE architecture.

Components:
- HFCTM-II Compliance Monitor (chiral symmetry, fractal self-consistency, toroidal closure)
- Chiral Inversion Controller (time-reversal validation)
- Polychronic Synchronization Protocol (4 temporal reference frames)
- Egregore Defense System (semantic drift protection)
"""

from .hfctm_safety import (
    HFCTMII_SafetyCore,
    SafetyConfig,
    init_safety_core,
    safety_core,
)

__all__ = [
    "HFCTMII_SafetyCore",
    "SafetyConfig",
    "init_safety_core",
    "safety_core",
]
