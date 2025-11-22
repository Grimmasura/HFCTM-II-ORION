"""
L4: Ironwood Tensor Processing Layer

Implements holographic state projection, manifold expansion, and multi-agent
inference coordination as specified in Section 5 of the MIH-IIE architecture.

Components:
- Holographic State Projector: Maps quantum states to tensor representations
- Manifold Expansion Engine: Dynamic fractal tensor growth
- Multi-Agent Inference Coordinator: Orchestrates parallel inference across
  four temporal modes (forward causal, retrocausal, atemporal, metacognitive)

Target Performance: 10^24 tensor operations/second (full-scale deployment)
"""

from .multi_agent_coordinator import (
    MultiAgentInferenceCoordinator,
    TemporalMode,
    InferenceAgent,
    SynchronizationPulse
)

from .holographic_projector import (
    HolographicProjector,
    HolographicState,
    ProjectionMode,
    ProjectionMetrics
)

from .manifold_expansion import (
    ManifoldExpansionEngine,
    ManifoldNode,
    ExpansionStrategy,
    ExpansionMetrics
)

__all__ = [
    # Multi-Agent Coordinator
    "MultiAgentInferenceCoordinator",
    "TemporalMode",
    "InferenceAgent",
    "SynchronizationPulse",
    # Holographic Projector
    "HolographicProjector",
    "HolographicState",
    "ProjectionMode",
    "ProjectionMetrics",
    # Manifold Expansion
    "ManifoldExpansionEngine",
    "ManifoldNode",
    "ExpansionStrategy",
    "ExpansionMetrics"
]
