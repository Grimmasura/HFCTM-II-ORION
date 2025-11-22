"""
Manifold Expansion Engine

Implements fractal manifold expansion with self-similar growth patterns
according to HFCTM-II principle: DH ≈ e ≈ 2.718.

Reference: Section 5.2 of MIH-IIE specification

Key Concepts:
- Fractal self-similarity: A₀(λr) = λ^(-DH) A₀(r)
- Hausdorff dimension DH ≈ e (natural expansion rate)
- Scale-invariant knowledge representation
- Recursive manifold growth
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ExpansionStrategy(Enum):
    """Manifold expansion strategies."""
    FRACTAL = "fractal"  # Pure fractal expansion (DH ≈ e)
    ADAPTIVE = "adaptive"  # Adaptive based on information density
    UNIFORM = "uniform"  # Uniform expansion (for comparison)
    HIERARCHICAL = "hierarchical"  # Hierarchical multi-scale


@dataclass
class ManifoldNode:
    """Node in the expanding manifold."""
    id: str
    position: np.ndarray
    scale: float
    data: Optional[np.ndarray] = None
    children: List[str] = None
    parent: Optional[str] = None
    generation: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class ExpansionMetrics:
    """Metrics for manifold expansion."""
    generation: int
    total_nodes: int
    measured_hausdorff_dimension: float
    target_hausdorff_dimension: float
    dimension_error: float
    branching_factor: float
    coverage_density: float


class ManifoldExpansionEngine:
    """
    Fractal manifold expansion engine with DH ≈ e growth pattern.

    Implements recursive knowledge space expansion where new nodes
    are added following fractal self-similarity principle.
    """

    def __init__(
        self,
        target_dimension: float = np.e,
        dimension_tolerance: float = 0.05,
        strategy: ExpansionStrategy = ExpansionStrategy.FRACTAL,
        max_generations: int = 10
    ):
        """
        Initialize manifold expansion engine.

        Args:
            target_dimension: Target Hausdorff dimension (default: e ≈ 2.718)
            dimension_tolerance: Acceptable deviation from target DH
            strategy: Expansion strategy
            max_generations: Maximum recursion depth
        """
        self.target_dimension = target_dimension
        self.dimension_tolerance = dimension_tolerance
        self.strategy = strategy
        self.max_generations = max_generations

        # Manifold state
        self.nodes: Dict[str, ManifoldNode] = {}
        self.generation = 0
        self.root_id: Optional[str] = None

        # Expansion history
        self.expansion_history: List[ExpansionMetrics] = []

    def initialize_seed(
        self,
        seed_position: Optional[np.ndarray] = None,
        seed_data: Optional[np.ndarray] = None,
        dimension: int = 3
    ) -> str:
        """
        Initialize manifold with seed node (0D attractor).

        Args:
            seed_position: Initial position (default: origin)
            seed_data: Initial data payload
            dimension: Embedding space dimension

        Returns:
            Root node ID
        """
        if seed_position is None:
            seed_position = np.zeros(dimension)

        root_node = ManifoldNode(
            id="root",
            position=seed_position,
            scale=1.0,
            data=seed_data,
            generation=0
        )

        self.nodes["root"] = root_node
        self.root_id = "root"
        self.generation = 0

        return "root"

    def compute_branching_factor(self, scale_ratio: float) -> int:
        """
        Compute branching factor for fractal expansion.

        From fractal self-similarity:
        A₀(λr) = λ^(-DH) A₀(r)

        For scale reduction λ, number of self-similar copies:
        N(λ) = λ^(DH)

        Args:
            scale_ratio: Ratio of child scale to parent scale

        Returns:
            Number of children to create
        """
        if self.strategy == ExpansionStrategy.FRACTAL:
            # Fractal branching: N = λ^(-DH)
            branching = int(np.round((1.0 / scale_ratio) ** self.target_dimension))
        elif self.strategy == ExpansionStrategy.ADAPTIVE:
            # Adaptive: base fractal + adaptation
            base_branching = int(np.round((1.0 / scale_ratio) ** self.target_dimension))
            # Add variance for exploration
            branching = max(2, base_branching + np.random.randint(-1, 2))
        elif self.strategy == ExpansionStrategy.UNIFORM:
            # Uniform: fixed branching
            branching = int(np.ceil(self.target_dimension))
        else:  # HIERARCHICAL
            # Hierarchical: generation-dependent
            branching = max(2, int(self.target_dimension ** (1 + 0.1 * self.generation)))

        return max(1, min(branching, 20))  # Clamp to reasonable range

    def expand_node(
        self,
        node_id: str,
        scale_ratio: float = 0.5,
        expansion_function: Optional[Callable] = None
    ) -> List[str]:
        """
        Expand a node by creating children according to fractal pattern.

        Args:
            node_id: ID of node to expand
            scale_ratio: Scale reduction for children
            expansion_function: Custom function to generate child data

        Returns:
            List of child node IDs
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        parent = self.nodes[node_id]

        # Check if we've reached max depth
        if parent.generation >= self.max_generations:
            return []

        # Compute branching factor
        n_children = self.compute_branching_factor(scale_ratio)

        # Generate children
        child_ids = []
        child_scale = parent.scale * scale_ratio

        for i in range(n_children):
            child_id = f"{node_id}_c{i}"

            # Position children around parent (fractal distribution)
            angle = 2 * np.pi * i / n_children
            if len(parent.position) == 2:
                # 2D case
                offset = child_scale * np.array([np.cos(angle), np.sin(angle)])
            elif len(parent.position) == 3:
                # 3D case (spiral on sphere)
                phi = angle
                theta = np.pi * (i + 0.5) / n_children
                offset = child_scale * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
            else:
                # N-D case (random direction)
                offset = child_scale * np.random.randn(len(parent.position))
                offset = offset / np.linalg.norm(offset)

            child_position = parent.position + offset

            # Generate child data
            if expansion_function is not None:
                child_data = expansion_function(parent.data, i, n_children)
            elif parent.data is not None:
                # Default: scale and perturb parent data
                child_data = parent.data * scale_ratio + np.random.randn(*parent.data.shape) * 0.1 * child_scale
            else:
                child_data = None

            # Create child node
            child = ManifoldNode(
                id=child_id,
                position=child_position,
                scale=child_scale,
                data=child_data,
                parent=node_id,
                generation=parent.generation + 1
            )

            self.nodes[child_id] = child
            parent.children.append(child_id)
            child_ids.append(child_id)

        return child_ids

    def expand_generation(
        self,
        expansion_function: Optional[Callable] = None,
        scale_ratio: float = 0.5
    ) -> int:
        """
        Expand all nodes in current generation.

        Args:
            expansion_function: Custom expansion function
            scale_ratio: Scale reduction for children

        Returns:
            Number of new nodes created
        """
        # Find all nodes in current generation
        current_gen_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.generation == self.generation
        ]

        if not current_gen_nodes:
            return 0

        # Expand each node
        new_nodes = []
        for node_id in current_gen_nodes:
            children = self.expand_node(node_id, scale_ratio, expansion_function)
            new_nodes.extend(children)

        # Update generation
        self.generation += 1

        # Compute and store metrics
        metrics = self.compute_expansion_metrics()
        self.expansion_history.append(metrics)

        return len(new_nodes)

    def measure_hausdorff_dimension(self, max_scale: Optional[float] = None) -> float:
        """
        Measure Hausdorff dimension of current manifold using box-counting.

        DH = lim(r→0) log(N(r)) / log(1/r)

        Args:
            max_scale: Maximum scale to consider

        Returns:
            Measured Hausdorff dimension
        """
        if len(self.nodes) < 2:
            return 0.0

        # Extract positions
        positions = np.array([node.position for node in self.nodes.values()])

        # Determine scales
        if max_scale is None:
            max_scale = np.max(np.linalg.norm(positions, axis=1)) or 1.0

        scales = [max_scale / (2 ** i) for i in range(1, 10)]

        # Box counting
        counts = []
        for scale in scales:
            # Discretize space into boxes of size 'scale'
            if scale > 0:
                boxes = (positions / scale).astype(int)
                # Count unique boxes
                unique_boxes = len(np.unique(boxes, axis=0))
                counts.append(unique_boxes)
            else:
                counts.append(len(positions))

        # Linear regression: log(N) vs log(1/r)
        log_scales = np.log([1/s for s in scales])
        log_counts = np.log(counts)

        # Fit line
        coeffs = np.polyfit(log_scales, log_counts, 1)
        measured_dh = coeffs[0]

        return float(measured_dh)

    def compute_coverage_density(self) -> float:
        """
        Compute manifold coverage density.

        Measures how uniformly the manifold covers the embedding space.

        Returns:
            Coverage density (0 = sparse, 1 = dense)
        """
        if len(self.nodes) < 2:
            return 0.0

        positions = np.array([node.position for node in self.nodes.values()])

        # Compute pairwise distances
        n = len(positions)
        distances = []
        for i in range(min(n, 100)):  # Sample for efficiency
            for j in range(i + 1, min(n, 100)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)

        if not distances:
            return 0.0

        # Density is inversely related to average distance
        avg_distance = np.mean(distances)
        max_distance = np.max(np.linalg.norm(positions, axis=1)) or 1.0

        density = 1.0 - (avg_distance / max_distance)

        return float(np.clip(density, 0.0, 1.0))

    def compute_expansion_metrics(self) -> ExpansionMetrics:
        """Compute metrics for current manifold state."""
        measured_dh = self.measure_hausdorff_dimension()
        dimension_error = abs(measured_dh - self.target_dimension)

        # Compute average branching factor
        branching_factors = [
            len(node.children) for node in self.nodes.values()
            if node.children
        ]
        avg_branching = np.mean(branching_factors) if branching_factors else 0.0

        coverage = self.compute_coverage_density()

        return ExpansionMetrics(
            generation=self.generation,
            total_nodes=len(self.nodes),
            measured_hausdorff_dimension=measured_dh,
            target_hausdorff_dimension=self.target_dimension,
            dimension_error=dimension_error,
            branching_factor=avg_branching,
            coverage_density=coverage
        )

    def get_statistics(self) -> Dict[str, float]:
        """Get manifold expansion statistics."""
        if not self.expansion_history:
            metrics = self.compute_expansion_metrics()
        else:
            metrics = self.expansion_history[-1]

        return {
            "total_nodes": len(self.nodes),
            "generation": self.generation,
            "measured_hausdorff_dimension": metrics.measured_hausdorff_dimension,
            "target_hausdorff_dimension": self.target_dimension,
            "dimension_error": metrics.dimension_error,
            "dimension_within_tolerance": metrics.dimension_error < self.dimension_tolerance,
            "average_branching_factor": metrics.branching_factor,
            "coverage_density": metrics.coverage_density
        }

    def query_nearest(
        self,
        query_position: np.ndarray,
        k: int = 1,
        max_distance: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest nodes to query position.

        Args:
            query_position: Query point
            k: Number of neighbors
            max_distance: Maximum search distance

        Returns:
            List of (node_id, distance) tuples
        """
        if not self.nodes:
            return []

        # Compute distances
        distances = []
        for node_id, node in self.nodes.items():
            dist = np.linalg.norm(node.position - query_position)
            if max_distance is None or dist <= max_distance:
                distances.append((node_id, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        return distances[:k]
