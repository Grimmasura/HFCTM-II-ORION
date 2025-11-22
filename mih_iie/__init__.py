"""
Majorana–Ironwood Hybrid Intrinsic Inference Engine (MIH-IIE)

A novel computational paradigm implementing Holographic Fractal Chiral Toroidal
Mechanics with Intrinsic Inference (HFCTM-II).

Seven-Layer Architecture:
- L1: 0D Seed / Intrinsic Attractor Module
- L2: Majorana Topological Qubit Array
- L3: Quantum-Classical Interface Layer
- L4: Ironwood Tensor Processing Layer
- L5: Recursive Governance Layer
- L6: Intelligent Codex Layer
- L7: Consciousness Interface Layer

Reference: spec/MIH-IIE_v1.0.pdf
"""

__version__ = "0.1.0-alpha"
__spec_version__ = "1.0"

# Layer imports
from . import layers
from . import hardware
from . import core

__all__ = [
    "layers",
    "hardware",
    "core",
    "__version__",
    "__spec_version__",
]
