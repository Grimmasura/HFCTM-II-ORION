"""
L1: 0D Seed / Intrinsic Attractor Module

Implements substrate-independent causal anchoring using dimensionless
attractors in ontological possibility space.

Conceptual Foundation:
- Computation emerges from intrinsic mathematical structures
- Attractors exist in possibility space independent of physical substrate
- Provides causal anchoring without coordinate dependence
- Bridges pure mathematics and physical instantiation

Reference: Section 3 of MIH-IIE specification

This layer represents the deepest theoretical foundation of the MIH-IIE:
the idea that computational primitives can be grounded in invariant
mathematical structures rather than arbitrary symbol manipulation.
"""

from typing import Dict, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AttractorType(Enum):
    """Types of intrinsic attractors."""
    FIXED_POINT = "fixed_point"  # Stable equilibrium
    LIMIT_CYCLE = "limit_cycle"  # Periodic orbit
    STRANGE = "strange"  # Chaotic attractor
    FRACTAL = "fractal"  # Self-similar fractal
    TOROIDAL = "toroidal"  # Toroidal manifold


class CausalMode(Enum):
    """Causal flow modes."""
    FORWARD = "forward"  # Future-directed
    BACKWARD = "backward"  # Past-directed
    BIDIRECTIONAL = "bidirectional"  # Both directions
    ACAUSAL = "acausal"  # Outside causal structure


@dataclass
class IntrinsicSeed:
    """
    0D seed - dimensionless causal anchor.

    Represents a point in ontological possibility space from which
    computational structures emerge through recursive application
    of generative rules.
    """
    id: str
    attractor_type: AttractorType
    basin_of_attraction: np.ndarray  # Characteristic function of basin
    stability_eigenvalues: np.ndarray  # Lyapunov spectrum
    dimension: float  # Intrinsic dimension (can be non-integer)
    entropy: float  # Information content
    generative_potential: float  # Capacity to spawn structures


@dataclass
class CausalFlow:
    """Causal flow from attractor."""
    source_attractor: str
    target_attractor: Optional[str]
    flow_vector: np.ndarray
    strength: float  # Flow intensity
    mode: CausalMode


@dataclass
class AttractorMutation:
    """Record of attractor transformations."""
    parent_id: str
    child_id: str
    transformation: str  # Description of mutation
    timestamp: float
    preserved_invariants: List[str]


class IntrinsicAttractorModule:
    """
    Manages intrinsic attractors in ontological possibility space.

    This module operates at the deepest level of the MIH-IIE stack,
    providing substrate-independent computational primitives.

    Key Principles:
    1. Attractors exist independently of physical substrate
    2. Computation emerges from attractor dynamics
    3. Causal flow defines information processing
    4. Dimensional reduction preserves essential structure
    """

    def __init__(
        self,
        possibility_space_dimension: int = 8,
        stability_threshold: float = 1e-6,
        entropy_target: float = np.log(2)  # 1 bit
    ):
        """
        Initialize intrinsic attractor module.

        Args:
            possibility_space_dimension: Dimension of ambient possibility space
            stability_threshold: Threshold for attractor stability
            entropy_target: Target entropy for seed attractors
        """
        self.possibility_space_dimension = possibility_space_dimension
        self.stability_threshold = stability_threshold
        self.entropy_target = entropy_target

        # Attractor registry
        self.attractors: Dict[str, IntrinsicSeed] = {}

        # Causal flow network
        self.causal_flows: List[CausalFlow] = []

        # Mutation history
        self.mutation_history: List[AttractorMutation] = []

        # Initialize primordial seed
        self._initialize_primordial_seed()

    def _initialize_primordial_seed(self):
        """
        Initialize the primordial 0D seed - the root attractor.

        The primordial seed represents the foundational mathematical
        structure from which all computation emerges.
        """
        # Fixed point at origin of possibility space
        primordial = IntrinsicSeed(
            id="primordial",
            attractor_type=AttractorType.FIXED_POINT,
            basin_of_attraction=np.zeros(self.possibility_space_dimension),
            stability_eigenvalues=np.ones(self.possibility_space_dimension) * -1.0,  # Stable
            dimension=0.0,  # True 0D point
            entropy=0.0,  # Minimal entropy
            generative_potential=1.0  # Maximum potential
        )

        self.attractors["primordial"] = primordial

    def create_attractor(
        self,
        attractor_type: AttractorType,
        parent_id: Optional[str] = None,
        generative_function: Optional[Callable] = None
    ) -> str:
        """
        Create new attractor from parent (or de novo).

        Args:
            attractor_type: Type of attractor to create
            parent_id: Parent attractor ID (None for de novo creation)
            generative_function: Custom function defining attractor dynamics

        Returns:
            New attractor ID
        """
        attractor_id = f"attractor_{len(self.attractors)}"

        if parent_id and parent_id in self.attractors:
            parent = self.attractors[parent_id]

            # Inherit properties from parent with variation
            basin = parent.basin_of_attraction + np.random.randn(self.possibility_space_dimension) * 0.1
            dimension = parent.dimension + 0.5  # Emergent structure increases dimension
            generative_pot = parent.generative_potential * 0.9  # Decreases with complexity

        else:
            # De novo creation
            basin = np.random.randn(self.possibility_space_dimension)
            dimension = 0.5
            generative_pot = 0.8

        # Generate stability spectrum based on attractor type
        if attractor_type == AttractorType.FIXED_POINT:
            # All negative eigenvalues → stable
            eigenvalues = -np.abs(np.random.randn(self.possibility_space_dimension))
        elif attractor_type == AttractorType.LIMIT_CYCLE:
            # One zero eigenvalue (oscillation), rest negative
            eigenvalues = -np.abs(np.random.randn(self.possibility_space_dimension))
            eigenvalues[0] = 0.0
        elif attractor_type == AttractorType.STRANGE:
            # Mixed positive/negative → chaotic
            eigenvalues = np.random.randn(self.possibility_space_dimension)
        elif attractor_type == AttractorType.FRACTAL:
            # Non-integer Lyapunov dimension
            eigenvalues = -np.abs(np.random.randn(self.possibility_space_dimension))
            eigenvalues[0] = 0.5  # Marginal stability
        else:  # TOROIDAL
            # Multiple zero eigenvalues (periodic structure)
            eigenvalues = -np.abs(np.random.randn(self.possibility_space_dimension))
            eigenvalues[:2] = 0.0

        # Compute entropy (von Neumann-like)
        entropy = self._compute_attractor_entropy(basin)

        attractor = IntrinsicSeed(
            id=attractor_id,
            attractor_type=attractor_type,
            basin_of_attraction=basin,
            stability_eigenvalues=eigenvalues,
            dimension=dimension,
            entropy=entropy,
            generative_potential=generative_pot
        )

        self.attractors[attractor_id] = attractor

        # Record mutation if from parent
        if parent_id:
            mutation = AttractorMutation(
                parent_id=parent_id,
                child_id=attractor_id,
                transformation=f"spawn_{attractor_type.value}",
                timestamp=float(len(self.mutation_history)),
                preserved_invariants=["basin_topology", "stability_class"]
            )
            self.mutation_history.append(mutation)

        return attractor_id

    def establish_causal_flow(
        self,
        source_id: str,
        target_id: Optional[str],
        mode: CausalMode = CausalMode.FORWARD
    ) -> bool:
        """
        Establish causal flow between attractors.

        Causal flows define information processing pathways.

        Args:
            source_id: Source attractor ID
            target_id: Target attractor ID (None for unbounded flow)
            mode: Causal flow mode

        Returns:
            Success status
        """
        if source_id not in self.attractors:
            return False

        if target_id and target_id not in self.attractors:
            return False

        source = self.attractors[source_id]

        if target_id:
            target = self.attractors[target_id]
            flow_vector = target.basin_of_attraction - source.basin_of_attraction
        else:
            # Unbounded flow in direction of maximum generative potential
            flow_vector = source.basin_of_attraction / np.linalg.norm(source.basin_of_attraction + 1e-10)

        strength = source.generative_potential

        flow = CausalFlow(
            source_attractor=source_id,
            target_attractor=target_id,
            flow_vector=flow_vector,
            strength=strength,
            mode=mode
        )

        self.causal_flows.append(flow)

        return True

    def _compute_attractor_entropy(self, basin: np.ndarray) -> float:
        """
        Compute intrinsic entropy of attractor.

        Uses basin volume and dimensionality to estimate
        information content.

        Args:
            basin: Basin of attraction vector

        Returns:
            Entropy (nats)
        """
        # Volume of basin (rough estimate)
        volume = np.linalg.norm(basin)

        # Entropy scales with log(volume)
        entropy = np.log(volume + 1.0)

        return float(entropy)

    def compute_lyapunov_spectrum(self, attractor_id: str) -> np.ndarray:
        """
        Compute Lyapunov spectrum for attractor.

        The Lyapunov spectrum characterizes stability and chaos.

        Args:
            attractor_id: Attractor ID

        Returns:
            Sorted Lyapunov exponents
        """
        if attractor_id not in self.attractors:
            return np.array([])

        attractor = self.attractors[attractor_id]

        # Lyapunov exponents are the stability eigenvalues
        spectrum = np.sort(attractor.stability_eigenvalues)[::-1]

        return spectrum

    def measure_dimensional_reduction(
        self,
        source_id: str,
        target_id: str
    ) -> float:
        """
        Measure dimensional reduction in attractor transformation.

        Args:
            source_id: Source attractor
            target_id: Target attractor

        Returns:
            Dimension reduction factor
        """
        if source_id not in self.attractors or target_id not in self.attractors:
            return 0.0

        source = self.attractors[source_id]
        target = self.attractors[target_id]

        reduction = source.dimension - target.dimension

        return float(reduction)

    def evolve_attractor(
        self,
        attractor_id: str,
        time: float,
        dynamics: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Evolve attractor under intrinsic dynamics.

        Args:
            attractor_id: Attractor to evolve
            time: Evolution time
            dynamics: Custom dynamics function (default: exponential relaxation)

        Returns:
            Evolved state
        """
        if attractor_id not in self.attractors:
            return np.array([])

        attractor = self.attractors[attractor_id]

        if dynamics is None:
            # Default: exponential relaxation to attractor
            eigenvalues = attractor.stability_eigenvalues
            evolved = attractor.basin_of_attraction * np.exp(eigenvalues * time)
        else:
            # Custom dynamics
            evolved = dynamics(attractor.basin_of_attraction, time)

        return evolved

    def query_possibility_space(
        self,
        query_point: np.ndarray,
        k_nearest: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Query possibility space for nearest attractors.

        Args:
            query_point: Point in possibility space
            k_nearest: Number of nearest attractors

        Returns:
            List of (attractor_id, distance) tuples
        """
        distances = []

        for attractor_id, attractor in self.attractors.items():
            dist = np.linalg.norm(attractor.basin_of_attraction - query_point)
            distances.append((attractor_id, float(dist)))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        return distances[:k_nearest]

    def get_statistics(self) -> Dict[str, float]:
        """Get module statistics."""
        total_attractors = len(self.attractors)
        total_flows = len(self.causal_flows)

        # Average dimension
        avg_dimension = np.mean([a.dimension for a in self.attractors.values()])

        # Average entropy
        avg_entropy = np.mean([a.entropy for a in self.attractors.values()])

        # Total generative potential
        total_potential = sum([a.generative_potential for a in self.attractors.values()])

        # Count by type
        type_counts = {}
        for attractor in self.attractors.values():
            type_name = attractor.attractor_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_attractors": total_attractors,
            "total_causal_flows": total_flows,
            "average_dimension": float(avg_dimension),
            "average_entropy": float(avg_entropy),
            "total_generative_potential": float(total_potential),
            "mutation_events": len(self.mutation_history),
            "possibility_space_dimension": self.possibility_space_dimension,
            **{f"type_{k}": v for k, v in type_counts.items()}
        }
