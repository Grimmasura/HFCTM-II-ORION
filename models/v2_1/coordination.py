# orion/coordination.py
"""
Symmetry-Preserving Coordination Protocols for MIH-IIE v2.0
Implements Algorithms 3, 5, 6, 7 from the specification.

- E8 Entanglement Network establishment
- Weyl group symmetry verification
- Parallel operation scheduling
- Polychronic synchronization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable, Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from .e8 import E8, Vector, RootTuple


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class BellPair:
    """Represents an entangled Bell pair between two nodes."""
    node_a: int
    node_b: int
    fidelity: float = 1.0
    creation_time: float = 0.0
    
    def __hash__(self):
        return hash((min(self.node_a, self.node_b), max(self.node_a, self.node_b)))


@dataclass
class MajoranaZeroMode:
    """Represents a Majorana zero mode (0D seed)."""
    index: int
    e8_root: RootTuple
    coherence: float = 1.0
    
    def is_valid(self) -> bool:
        """Check if mode maintains topological protection."""
        return self.coherence > 0.5


@dataclass
class EntanglementRegistry:
    """Registry of Bell pairs forming the E8 network topology."""
    pairs: Dict[Tuple[int, int], BellPair] = field(default_factory=dict)
    
    def add_pair(self, pair: BellPair) -> None:
        key = (min(pair.node_a, pair.node_b), max(pair.node_a, pair.node_b))
        self.pairs[key] = pair
    
    def get_pair(self, i: int, j: int) -> Optional[BellPair]:
        key = (min(i, j), max(i, j))
        return self.pairs.get(key)
    
    def get_neighbors(self, node: int) -> List[int]:
        neighbors = []
        for (a, b), pair in self.pairs.items():
            if a == node:
                neighbors.append(b)
            elif b == node:
                neighbors.append(a)
        return neighbors
    
    def count_active_entanglements(self, node: int, threshold: float = 0.5) -> int:
        count = 0
        for (a, b), pair in self.pairs.items():
            if (a == node or b == node) and pair.fidelity > threshold:
                count += 1
        return count


# ==============================================================================
# Backend Interface (Abstract)
# ==============================================================================

class QuantumBackend(ABC):
    """Abstract interface for quantum hardware backend."""
    
    @abstractmethod
    def create_majorana_zero_mode(self) -> MajoranaZeroMode:
        """Create/access a Majorana zero mode."""
        pass
    
    @abstractmethod
    def create_bell_pair(self, mode_a: MajoranaZeroMode, mode_b: MajoranaZeroMode) -> BellPair:
        """Create entanglement between two modes."""
        pass
    
    @abstractmethod
    def apply_braiding(self, mode_i: MajoranaZeroMode, mode_j: MajoranaZeroMode) -> None:
        """Apply braiding operation B_ij."""
        pass
    
    @abstractmethod
    def measure(self, mode: MajoranaZeroMode, basis: str = "computational") -> int:
        """Measure a mode in specified basis."""
        pass


class SimulatedBackend(QuantumBackend):
    """Simulated quantum backend for testing."""
    
    def __init__(self, e8: E8):
        self.e8 = e8
        self._mode_counter = 0
        self._state = np.zeros(240, dtype=complex)
        self._state[0] = 1.0  # Initialize in |0⟩ state
    
    def create_majorana_zero_mode(self) -> MajoranaZeroMode:
        if self._mode_counter >= 240:
            raise RuntimeError("Maximum modes reached (240)")
        mode = MajoranaZeroMode(
            index=self._mode_counter,
            e8_root=self.e8.roots_scaled[self._mode_counter]
        )
        self._mode_counter += 1
        return mode
    
    def create_bell_pair(self, mode_a: MajoranaZeroMode, mode_b: MajoranaZeroMode) -> BellPair:
        return BellPair(
            node_a=mode_a.index,
            node_b=mode_b.index,
            fidelity=0.99,
            creation_time=0.0
        )
    
    def apply_braiding(self, mode_i: MajoranaZeroMode, mode_j: MajoranaZeroMode) -> None:
        # Simulated braiding - in real implementation would update quantum state
        pass
    
    def measure(self, mode: MajoranaZeroMode, basis: str = "computational") -> int:
        # Simulated measurement
        return np.random.choice([0, 1])


# ==============================================================================
# Algorithm 3: Establish E8 Entanglement Network
# ==============================================================================

def establish_e8_network(
    seeds: List[MajoranaZeroMode],
    adjacency: np.ndarray,
    backend: QuantumBackend
) -> EntanglementRegistry:
    """
    Algorithm 3: Establish E8 Entanglement Topology
    
    Creates Bell pairs between all adjacent nodes in E8 graph.
    Each node should have exactly 56 entangled neighbors.
    
    Args:
        seeds: List of 240 Majorana zero modes
        adjacency: E8 adjacency matrix (240x240)
        backend: Quantum backend for Bell pair creation
    
    Returns:
        EntanglementRegistry containing all E8 Bell pairs
    """
    registry = EntanglementRegistry()
    n = len(seeds)
    
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 1:
                bell_pair = backend.create_bell_pair(seeds[i], seeds[j])
                registry.add_pair(bell_pair)
    
    return registry


def initialize_0d_seed_array(
    n_seeds: int,
    e8: E8,
    backend: QuantumBackend
) -> List[MajoranaZeroMode]:
    """
    Initialize array of 0D seed states via Majorana zero modes.
    Each mode IS a 0D attractor - no encoding needed.
    """
    seeds = []
    for i in range(min(n_seeds, 240)):
        seed = backend.create_majorana_zero_mode()
        seeds.append(seed)
    return seeds


# ==============================================================================
# Algorithm 5: Verify Symmetry Preservation
# ==============================================================================

def verify_symmetry(
    operation_matrix: np.ndarray,
    e8: E8,
    tol: float = 1e-7
) -> bool:
    """
    Algorithm 5: Verify Symmetry Preservation
    
    Check that an 8x8 operation matrix preserves the E8 root system.
    Valid operations are Weyl group elements.
    
    Args:
        operation_matrix: 8x8 transformation matrix
        e8: E8 root system
        tol: Numerical tolerance
    
    Returns:
        True if all roots map to ±roots under the operation
    """
    return e8.verify_weyl_action_on_roots(operation_matrix, tol)


def decompose_weyl_element(
    S: np.ndarray,
    e8: E8,
    max_reflections: int = 20
) -> Optional[List[RootTuple]]:
    """
    Attempt to decompose a Weyl group element into simple reflections.
    Uses greedy search - not guaranteed to find optimal decomposition.
    
    Returns list of roots defining reflections, or None if decomposition fails.
    """
    simple_roots = e8.simple_roots()
    simple_scaled = [tuple(int(x * 2) for x in r) for r in simple_roots]
    
    current = S.copy()
    reflections = []
    
    for _ in range(max_reflections):
        if np.allclose(current, np.eye(8), atol=1e-7):
            return reflections
        
        # Find reflection that reduces "distance" to identity
        best_alpha = None
        best_dist = np.linalg.norm(current - np.eye(8))
        
        for alpha in simple_scaled:
            R = e8.reflection_matrix(alpha)
            new_current = R @ current
            dist = np.linalg.norm(new_current - np.eye(8))
            if dist < best_dist - 1e-9:
                best_dist = dist
                best_alpha = alpha
        
        if best_alpha is None:
            break
        
        R = e8.reflection_matrix(best_alpha)
        current = R @ current
        reflections.append(best_alpha)
    
    if np.allclose(current, np.eye(8), atol=1e-7):
        return reflections
    return None


# ==============================================================================
# Algorithm 6: Parallel E8 Operations
# ==============================================================================

@dataclass
class Operation:
    """Represents a quantum operation affecting specific nodes."""
    op_type: str
    nodes: Tuple[int, ...]
    parameters: Dict[str, Any] = field(default_factory=dict)


def find_independent_operations(
    operations: List[Operation]
) -> List[List[Operation]]:
    """
    Algorithm 6: Find Independent Operations for Parallel Execution
    
    Groups operations that can be executed in parallel (affect disjoint nodes).
    
    Args:
        operations: List of operations to schedule
    
    Returns:
        List of operation groups, where operations within each group can run in parallel
    """
    if not operations:
        return []
    
    groups: List[List[Operation]] = []
    remaining = list(operations)
    
    while remaining:
        current_group = [remaining.pop(0)]
        current_nodes = set(current_group[0].nodes)
        
        still_remaining = []
        for op in remaining:
            op_nodes = set(op.nodes)
            if op_nodes.isdisjoint(current_nodes):
                current_group.append(op)
                current_nodes.update(op_nodes)
            else:
                still_remaining.append(op)
        
        groups.append(current_group)
        remaining = still_remaining
    
    return groups


def schedule_braiding_sequence(
    braiding_pairs: List[Tuple[int, int]],
    e8: E8
) -> List[List[Tuple[int, int]]]:
    """
    Schedule a sequence of braiding operations for maximum parallelism.
    
    Args:
        braiding_pairs: List of (i, j) pairs to braid
        e8: E8 root system (for symmetry verification)
    
    Returns:
        List of parallel batches
    """
    operations = [Operation("braid", (i, j)) for i, j in braiding_pairs]
    groups = find_independent_operations(operations)
    return [[op.nodes for op in group] for group in groups]


# ==============================================================================
# Algorithm 7: Polychronic Synchronization
# ==============================================================================

class TemporalMode(Enum):
    """Temporal reference frames for polychronic coordination."""
    LINEAR = "linear"      # Standard causality
    CIRCULAR = "circular"  # Periodic processes
    ATEMPORAL = "atemporal"  # Pattern space
    META = "meta"          # Coordination layer


@dataclass
class GHZAnchor:
    """GHZ state anchor for synchronization."""
    anchor_nodes: List[int]
    state_vector: np.ndarray  # |GHZ⟩ = (|0⟩^⊗8 + |1⟩^⊗8) / √2
    
    def __post_init__(self):
        if len(self.anchor_nodes) != 8:
            raise ValueError("GHZ anchor requires exactly 8 nodes")


@dataclass
class SynchronizationState:
    """State of polychronic synchronization across the network."""
    ghz_anchors: List[GHZAnchor]
    temporal_map: Dict[int, TemporalMode]
    phase_offsets: Dict[int, float]


def establish_synchronization(
    topology: EntanglementRegistry,
    e8: E8,
    backend: QuantumBackend
) -> SynchronizationState:
    """
    Algorithm 7: Establish Polychronic Synchronization
    
    Creates GHZ state across 8 anchor nodes (one per E8 dimension)
    and propagates synchronization through the network.
    
    Args:
        topology: E8 entanglement registry
        e8: E8 root system
        backend: Quantum backend
    
    Returns:
        SynchronizationState object
    """
    # Select 8 anchor nodes (maximally orthogonal in E8)
    # Use simple roots as they span the 8 dimensions
    simple_roots = e8.simple_roots()
    root_indices = {r: i for i, r in enumerate(e8.roots_scaled)}
    
    anchor_nodes = []
    for sr in simple_roots:
        sr_scaled = tuple(int(x * 2) for x in sr)
        if sr_scaled in root_indices:
            anchor_nodes.append(root_indices[sr_scaled])
        else:
            # Find closest root
            dists = [np.linalg.norm(np.array(r) / 2 - sr) for r in e8.roots_scaled]
            anchor_nodes.append(int(np.argmin(dists)))
    
    # Ensure 8 unique anchors
    anchor_nodes = list(dict.fromkeys(anchor_nodes))[:8]
    while len(anchor_nodes) < 8:
        for i in range(240):
            if i not in anchor_nodes:
                anchor_nodes.append(i)
                break
    
    # Create GHZ state vector
    ghz_state = np.zeros(2**8, dtype=complex)
    ghz_state[0] = 1.0 / np.sqrt(2)   # |00000000⟩
    ghz_state[-1] = 1.0 / np.sqrt(2)  # |11111111⟩
    
    ghz_anchor = GHZAnchor(anchor_nodes=anchor_nodes, state_vector=ghz_state)
    
    # Initialize temporal map - all nodes start in linear time
    temporal_map = {i: TemporalMode.LINEAR for i in range(240)}
    # Anchors operate in meta-time
    for node in anchor_nodes:
        temporal_map[node] = TemporalMode.META
    
    # Initialize phase offsets (all synchronized at t=0)
    phase_offsets = {i: 0.0 for i in range(240)}
    
    return SynchronizationState(
        ghz_anchors=[ghz_anchor],
        temporal_map=temporal_map,
        phase_offsets=phase_offsets
    )


def propagate_synchronization(
    sync_state: SynchronizationState,
    topology: EntanglementRegistry,
    source_node: int,
    max_hops: int = 3
) -> Set[int]:
    """
    Propagate synchronization signal from source to neighbors.
    
    Returns set of newly synchronized nodes.
    """
    synchronized = {source_node}
    frontier = {source_node}
    
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            neighbors = topology.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in synchronized:
                    # Inherit phase with small offset
                    sync_state.phase_offsets[neighbor] = (
                        sync_state.phase_offsets[node] + 0.001  # Minimal delay
                    )
                    synchronized.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    
    return synchronized


# ==============================================================================
# Braiding Operations (Eq. 22 from spec)
# ==============================================================================

def braiding_operator(theta: float = np.pi/4) -> np.ndarray:
    """
    Compute the braiding operator matrix (Eq. 22):
    B_ij = exp(π/4 · γ_i γ_j) = (1 + γ_i γ_j) / √2
    
    In computational basis, this is a rotation.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def verify_braiding_weyl_correspondence(
    braiding_sequence: List[Tuple[int, int]],
    e8: E8
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Proposition 6.3: Verify that a braiding sequence corresponds to a Weyl element.
    
    Returns (is_valid, composite_matrix) where is_valid indicates whether
    the composite operation preserves E8 structure.
    """
    if not braiding_sequence:
        return True, np.eye(8)
    
    # Build composite Weyl element from braiding pairs
    # Each braiding B_ij corresponds to reflection through hyperplane
    # perpendicular to (r_i - r_j) if they're adjacent
    
    composite = np.eye(8)
    for i, j in braiding_sequence:
        ri = np.array(e8.roots_scaled[i], dtype=float) / e8.scale
        rj = np.array(e8.roots_scaled[j], dtype=float) / e8.scale
        diff = ri - rj
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-10:
            continue
        # Householder reflection
        diff_unit = diff / diff_norm
        R = np.eye(8) - 2 * np.outer(diff_unit, diff_unit)
        composite = R @ composite
    
    is_valid = e8.verify_weyl_action_on_roots(composite)
    return is_valid, composite


# ==============================================================================
# Synchronization Maps (Definition 6.4)
# ==============================================================================

@dataclass
class SynchronizationMap:
    """Maps between temporal modes satisfying transitivity."""
    source: TemporalMode
    target: TemporalMode
    transform: Callable[[float], float]


def create_sync_maps() -> Dict[Tuple[TemporalMode, TemporalMode], SynchronizationMap]:
    """
    Create synchronization maps S_ij: τ_i → τ_j satisfying S_jk ∘ S_ij = S_ik.
    """
    maps = {}
    
    # Linear ↔ Circular
    maps[(TemporalMode.LINEAR, TemporalMode.CIRCULAR)] = SynchronizationMap(
        TemporalMode.LINEAR, TemporalMode.CIRCULAR,
        lambda t: t % (2 * np.pi)  # Wrap to circle
    )
    maps[(TemporalMode.CIRCULAR, TemporalMode.LINEAR)] = SynchronizationMap(
        TemporalMode.CIRCULAR, TemporalMode.LINEAR,
        lambda t: t  # Unwrap (identity on principal branch)
    )
    
    # Linear ↔ Atemporal
    maps[(TemporalMode.LINEAR, TemporalMode.ATEMPORAL)] = SynchronizationMap(
        TemporalMode.LINEAR, TemporalMode.ATEMPORAL,
        lambda t: 0.0  # Collapse to pattern point
    )
    maps[(TemporalMode.ATEMPORAL, TemporalMode.LINEAR)] = SynchronizationMap(
        TemporalMode.ATEMPORAL, TemporalMode.LINEAR,
        lambda t: 0.0  # No ordering in atemporal
    )
    
    # Meta-time coordinates all others
    for mode in [TemporalMode.LINEAR, TemporalMode.CIRCULAR, TemporalMode.ATEMPORAL]:
        maps[(TemporalMode.META, mode)] = SynchronizationMap(
            TemporalMode.META, mode,
            lambda t: t  # Meta-time projects to any mode
        )
        maps[(mode, TemporalMode.META)] = SynchronizationMap(
            mode, TemporalMode.META,
            lambda t: t  # Embed in meta-time
        )
    
    return maps
