"""
MIH-IIE Seven-Layer Architecture

Implements the complete recursive stack with bidirectional causal channels
between adjacent layers and toroidal closure ensuring no information loss.
"""

from . import l5_governance
from . import l4_ironwood

__all__ = [
    "l5_governance",
    "l4_ironwood",
]
