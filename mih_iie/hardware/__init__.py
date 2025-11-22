"""
Hardware Abstraction Layer

Provides unified interfaces for:
- L2: Majorana Topological Qubit Array (via Azure Quantum / Majorana1)
- L4: Ironwood Tensor Processing (TPU acceleration via JAX)

All hardware backends include graceful fallbacks to classical computation.
"""

from .hardware_interfaces import (
    MAJORANA1_AVAILABLE,
    IRONWOOD_AVAILABLE,
)

__all__ = [
    "MAJORANA1_AVAILABLE",
    "IRONWOOD_AVAILABLE",
]
