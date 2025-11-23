# orion/error_correction.py
"""
Topological Error Correction for MIH-IIE v2.0
Implements Algorithms 8, 9, 10 from the specification.

E8-based stabilizer construction using graph cliques rather than geometric locality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum
import numpy as np

from models.v2_1.e8 import E8
from models.v2_1.coordination import (
    MajoranaZeroMode, EntanglementRegistry, QuantumBackend
)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class Stabilizer:
    """
    E8 Stabilizer Generator (Definition 7.1)
    
    S_ijkl = γ_i γ_j γ_k γ_l where (i,j,k,l) form a 4-clique in E8 adjacency graph.
    
    Properties:
    - S² = I (involutory)
    - [S_ijkl, S_mnop] = 0 for disjoint index sets
    - Eigenvalues ±1
    """
    indices: Tuple[int, int, int, int]
    
    def __post_init__(self):
        # Canonicalize order
        self.indices = tuple(sorted(self.indices))
    
    def __hash__(self):
        return hash(self.indices)
    
    def __eq__(self, other):
        return isinstance(other, Stabilizer) and self.indices == other.indices
    
    def affects(self, node: int) -> bool:
        return node in self.indices
    
    def overlaps(self, other: "Stabilizer") -> bool:
        return bool(set(self.indices) & set(other.indices))


@dataclass
class Syndrome:
    """Syndrome measurement result."""
    stabilizer: Stabilizer
    eigenvalue: int  # +1 or -1
    
    def is_violated(self) -> bool:
        return self.eigenvalue == -1


@dataclass
class ErrorLocation:
    """Identified error location from syndrome decoding."""
    nodes: Set[int]
    error_type: str  # "X", "Z", "Y", or "unknown"
    confidence: float


class CorrectionResult(Enum):
    NO_ERROR = "no_error"
    CORRECTED = "corrected"
    UNCORRECTABLE = "uncorrectable"


# ==============================================================================
# Algorithm 8: Construct E8 Stabilizers
# ==============================================================================

def construct_stabilizers(
    e8: E8,
    adjacency: Optional[np.ndarray] = None
) -> List[Stabilizer]:
    """
    Algorithm 8: Construct E8 Stabilizers
    
    Stabilizer generators are constructed from 4-cliques in the E8 adjacency graph.
    Each 4-clique (i,j,k,l) defines S_ijkl = γ_i γ_j γ_k γ_l.
    
    Args:
        e8: E8 root system
        adjacency: Optional pre-computed adjacency matrix
    
    Returns:
        List of Stabilizer objects (deduplicated)
    """
    if adjacency is None:
        adjacency = e8.adjacency_matrix()
    
    # Find all 4-cliques using the E8 engine
    cliques = e8.find_k_cliques(k=4, limit=None)
    
    # Convert to stabilizers (automatically deduplicated by set)
    stabilizers = set()
    for clique in cliques:
        stabilizers.add(Stabilizer(indices=clique))
    
    return list(stabilizers)


def construct_stabilizers_for_subgraph(
    e8: E8,
    selected_nodes: List[int]
) -> List[Stabilizer]:
    """
    Construct stabilizers for a subset of E8 nodes.
    Only includes stabilizers where all 4 nodes are in the selected set.
    """
    node_set = set(selected_nodes)
    full_stabilizers = construct_stabilizers(e8)
    
    return [s for s in full_stabilizers if all(i in node_set for i in s.indices)]


# ==============================================================================
# Algorithm 9: Error Correction Cycle
# ==============================================================================

@dataclass
class AncillaQubit:
    """Ancilla qubit for syndrome measurement."""
    index: int
    state: int = 0  # 0 or 1


class SyndromeMeasurer:
    """
    Measures stabilizer syndromes using ancilla qubits.
    """
    
    def __init__(self, backend: QuantumBackend):
        self.backend = backend
        self._ancilla_counter = 0
    
    def measure_stabilizer(
        self,
        stabilizer: Stabilizer,
        seeds: List[MajoranaZeroMode]
    ) -> Syndrome:
        """
        Measure a single stabilizer and return syndrome.
        
        In simulation, we model error processes probabilistically.
        In real hardware, this couples an ancilla to the 4-mode product.
        """
        # Create ancilla
        ancilla = AncillaQubit(index=self._ancilla_counter)
        self._ancilla_counter += 1
        
        # Couple ancilla to γ_i γ_j γ_k γ_l
        # In simulation, we just return a random eigenvalue with bias toward +1
        # Real implementation would perform actual measurement
        eigenvalue = 1 if np.random.random() > 0.01 else -1  # 1% error rate
        
        return Syndrome(stabilizer=stabilizer, eigenvalue=eigenvalue)
    
    def measure_all(
        self,
        stabilizers: List[Stabilizer],
        seeds: List[MajoranaZeroMode]
    ) -> List[Syndrome]:
        """Measure all stabilizers and return syndromes."""
        return [self.measure_stabilizer(s, seeds) for s in stabilizers]


class MinimumWeightDecoder:
    """
    Minimum weight decoder for E8 stabilizer code.
    
    Finds the minimum weight error consistent with observed syndrome.
    Uses the E8 graph structure for efficient decoding.
    """
    
    def __init__(self, e8: E8, stabilizers: List[Stabilizer]):
        self.e8 = e8
        self.stabilizers = stabilizers
        self._build_syndrome_table()
    
    def _build_syndrome_table(self) -> None:
        """Pre-compute syndrome patterns for single-node errors."""
        self.single_error_syndromes: Dict[int, Set[int]] = {}
        
        for node in range(len(self.e8.roots_scaled)):
            # Find all stabilizers affected by error at this node
            affected = set()
            for i, stab in enumerate(self.stabilizers):
                if stab.affects(node):
                    affected.add(i)
            self.single_error_syndromes[node] = affected
    
    def decode(
        self,
        violations: Set[int]
    ) -> ErrorLocation:
        """
        Decode syndrome violations to error location.
        
        Args:
            violations: Set of indices of violated stabilizers
        
        Returns:
            ErrorLocation indicating likely error site(s)
        """
        if not violations:
            return ErrorLocation(nodes=set(), error_type="none", confidence=1.0)
        
        # Try single-node errors first
        for node, affected in self.single_error_syndromes.items():
            if affected == violations:
                return ErrorLocation(
                    nodes={node},
                    error_type="X",
                    confidence=0.95
                )
        
        # Try two-node errors
        nodes = list(self.single_error_syndromes.keys())
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                combined = self.single_error_syndromes[n1] ^ self.single_error_syndromes[n2]
                if combined == violations:
                    return ErrorLocation(
                        nodes={n1, n2},
                        error_type="X",
                        confidence=0.8
                    )
        
        # Fall back to heuristic: nodes appearing most in violated stabilizers
        node_counts: Dict[int, int] = {}
        for v in violations:
            stab = self.stabilizers[v]
            for node in stab.indices:
                node_counts[node] = node_counts.get(node, 0) + 1
        
        if node_counts:
            max_count = max(node_counts.values())
            likely_nodes = {n for n, c in node_counts.items() if c == max_count}
            return ErrorLocation(
                nodes=likely_nodes,
                error_type="unknown",
                confidence=0.5
            )
        
        return ErrorLocation(
            nodes=set(),
            error_type="unknown",
            confidence=0.0
        )


def error_correction_cycle(
    stabilizers: List[Stabilizer],
    seeds: List[MajoranaZeroMode],
    backend: QuantumBackend,
    decoder: MinimumWeightDecoder
) -> CorrectionResult:
    """
    Algorithm 9: Error Correction Cycle
    
    1. Measure syndrome
    2. Decode syndrome
    3. Apply correction
    
    Args:
        stabilizers: List of E8 stabilizers
        seeds: Majorana zero mode array
        backend: Quantum backend
        decoder: Syndrome decoder
    
    Returns:
        CorrectionResult indicating outcome
    """
    # Measure syndrome
    measurer = SyndromeMeasurer(backend)
    syndromes = measurer.measure_all(stabilizers, seeds)
    
    # Find violations
    violations = {i for i, s in enumerate(syndromes) if s.is_violated()}
    
    if not violations:
        return CorrectionResult.NO_ERROR
    
    # Decode syndrome
    error_location = decoder.decode(violations)
    
    if error_location.confidence < 0.3:
        return CorrectionResult.UNCORRECTABLE
    
    # Apply correction
    for node in error_location.nodes:
        apply_majorana_correction(seeds[node], error_location.error_type, backend)
    
    return CorrectionResult.CORRECTED


def apply_majorana_correction(
    mode: MajoranaZeroMode,
    error_type: str,
    backend: QuantumBackend
) -> None:
    """Apply correction operation to a Majorana mode."""
    # In real implementation, this would apply physical correction
    # In simulation, we just mark it as corrected
    pass


# ==============================================================================
# Algorithm 10: E8 Symmetry Error Detection
# ==============================================================================

@dataclass
class SymmetryViolation:
    """Record of an E8 symmetry violation."""
    violation_type: str
    nodes: Tuple[int, ...]
    expected: any
    actual: any
    severity: float  # 0 to 1


def check_e8_symmetry(
    topology: EntanglementRegistry,
    e8: E8,
    adjacency: np.ndarray
) -> List[SymmetryViolation]:
    """
    Algorithm 10: E8 Symmetry Error Detection
    
    Detects violations of E8 structure:
    1. Coordination number violations (each node should have 56 neighbors)
    2. Inner product structure violations (entanglement where there shouldn't be)
    
    Args:
        topology: Current entanglement registry
        e8: E8 root system
        adjacency: Expected E8 adjacency matrix
    
    Returns:
        List of detected violations
    """
    violations = []
    n = len(e8.roots_scaled)
    
    # Check coordination numbers
    for i in range(n):
        active_neighbors = topology.count_active_entanglements(i)
        expected = int(adjacency[i].sum())
        if active_neighbors != expected:
            violations.append(SymmetryViolation(
                violation_type="coordination_number",
                nodes=(i,),
                expected=expected,
                actual=active_neighbors,
                severity=abs(active_neighbors - expected) / expected
            ))
    
    # Check inner product structure (sample-based for efficiency)
    sample_pairs = []
    for i in range(min(n, 50)):
        for j in range(i + 1, min(n, 50)):
            sample_pairs.append((i, j))
    
    for i, j in sample_pairs:
        expected_adjacent = adjacency[i, j] == 1
        pair = topology.get_pair(i, j)
        
        if expected_adjacent:
            if pair is None or pair.fidelity < 0.9:
                violations.append(SymmetryViolation(
                    violation_type="missing_entanglement",
                    nodes=(i, j),
                    expected="entangled",
                    actual="not entangled" if pair is None else f"low fidelity ({pair.fidelity})",
                    severity=1.0 if pair is None else 1.0 - pair.fidelity
                ))
        else:
            if pair is not None and pair.fidelity > 0.5:
                violations.append(SymmetryViolation(
                    violation_type="spurious_entanglement",
                    nodes=(i, j),
                    expected="not entangled",
                    actual=f"entangled ({pair.fidelity})",
                    severity=pair.fidelity
                ))
    
    return violations


def verify_weyl_preservation(
    operation_sequence: List[np.ndarray],
    e8: E8
) -> Tuple[bool, List[str]]:
    """
    Verify that a sequence of operations preserves E8 Weyl group structure.
    
    Returns (all_valid, list of error messages)
    """
    errors = []
    composite = np.eye(8)
    
    for i, op in enumerate(operation_sequence):
        composite = op @ composite
        if not e8.verify_weyl_action_on_roots(composite):
            errors.append(f"Operation {i} breaks Weyl structure")
    
    return len(errors) == 0, errors


# ==============================================================================
# Stabilizer Code Distance Analysis
# ==============================================================================

def compute_code_distance(
    stabilizers: List[Stabilizer],
    n_nodes: int,
    sample_size: int = 1000
) -> int:
    """
    Estimate the code distance (minimum weight of undetectable error).
    
    Uses sampling to estimate - exact computation is NP-hard.
    """
    # Build stabilizer node coverage
    covered_by: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    for si, stab in enumerate(stabilizers):
        for node in stab.indices:
            covered_by[node].add(si)
    
    # Test random error patterns
    min_undetected_weight = n_nodes
    
    for _ in range(sample_size):
        weight = np.random.randint(1, min(6, n_nodes + 1))
        error_nodes = set(np.random.choice(n_nodes, weight, replace=False))
        
        # Check if error is detected
        affected_stabilizers: Set[int] = set()
        for node in error_nodes:
            affected_stabilizers ^= covered_by[node]  # XOR for parity
        
        if not affected_stabilizers and weight < min_undetected_weight:
            min_undetected_weight = weight
    
    return min_undetected_weight


# ==============================================================================
# Real-time Error Monitoring
# ==============================================================================

@dataclass
class ErrorStatistics:
    """Running statistics on error rates."""
    total_cycles: int = 0
    no_error_count: int = 0
    corrected_count: int = 0
    uncorrectable_count: int = 0
    
    def record(self, result: CorrectionResult) -> None:
        self.total_cycles += 1
        if result == CorrectionResult.NO_ERROR:
            self.no_error_count += 1
        elif result == CorrectionResult.CORRECTED:
            self.corrected_count += 1
        else:
            self.uncorrectable_count += 1
    
    @property
    def error_rate(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return (self.corrected_count + self.uncorrectable_count) / self.total_cycles
    
    @property
    def correction_success_rate(self) -> float:
        errors = self.corrected_count + self.uncorrectable_count
        if errors == 0:
            return 1.0
        return self.corrected_count / errors


class ErrorMonitor:
    """
    Continuous monitoring of error correction performance.
    """
    
    def __init__(
        self,
        e8: E8,
        stabilizers: List[Stabilizer],
        backend: QuantumBackend
    ):
        self.e8 = e8
        self.stabilizers = stabilizers
        self.backend = backend
        self.decoder = MinimumWeightDecoder(e8, stabilizers)
        self.stats = ErrorStatistics()
    
    def run_cycle(self, seeds: List[MajoranaZeroMode]) -> CorrectionResult:
        """Run one error correction cycle and update statistics."""
        result = error_correction_cycle(
            self.stabilizers, seeds, self.backend, self.decoder
        )
        self.stats.record(result)
        return result
    
    def run_multiple(
        self,
        seeds: List[MajoranaZeroMode],
        n_cycles: int
    ) -> ErrorStatistics:
        """Run multiple cycles and return statistics."""
        for _ in range(n_cycles):
            self.run_cycle(seeds)
        return self.stats
    
    def check_threshold(self, max_error_rate: float = 0.01) -> bool:
        """Check if error rate is below threshold."""
        return self.stats.error_rate < max_error_rate
