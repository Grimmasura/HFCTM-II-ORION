# orion/holography.py
"""
Holographic State Readout for MIH-IIE v2.0
Implements Algorithms 11, 12, 13 from the specification.

- Boundary identification using E8 graph structure
- Tensor network reconstruction
- Inference vector computation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Sequence, Tuple, Optional, Set
import numpy as np
from abc import ABC, abstractmethod

from .e8 import E8, Vector


# ==============================================================================
# Data Structures
# ==============================================================================

Measurement = np.ndarray
Index = int


@dataclass
class BoundaryMeasurement:
    """Measurement result from a boundary node."""
    node: int
    z_basis: int  # Computational basis measurement (0 or 1)
    x_basis: int  # Hadamard basis measurement (0 or 1)
    coordinate: Vector  # E8 root coordinate
    confidence: float = 1.0


@dataclass
class TensorNode:
    """Node in the E8 tensor network."""
    index: int
    tensor: np.ndarray
    neighbor_indices: List[int]
    neighbor_axis_map: Dict[int, int] = field(default_factory=dict)  # neighbor id -> tensor axis
    physical_dim: int = 2  # For qubit/Majorana mode
    is_boundary: bool = False


@dataclass
class TensorNetwork:
    """
    Tensor network over E8 topology (Definition 8.2).
    
    Each node has a tensor with:
    - One leg per neighbor (bond dimension)
    - One physical index (for measurement/state)
    
    Edges correspond to contracted indices.
    """
    tensors: Dict[Index, TensorNode]
    edges: List[Tuple[Index, Index]]
    bond_dim: int = 2
    
    def get_tensor(self, node: int) -> np.ndarray:
        return self.tensors[node].tensor
    
    def set_tensor(self, node: int, tensor: np.ndarray) -> None:
        self.tensors[node].tensor = tensor
    
    def get_neighbors(self, node: int) -> List[int]:
        return self.tensors[node].neighbor_indices
    
    def _ensure_neighbor_axis(self, node: TensorNode, neighbor: int) -> int:
        """
        Ensure a tensor has an explicit axis for a given neighbor.

        If the neighbor was not part of the compressed axis set, append a new
        bond_dim axis directly before the physical dimension.
        """
        if neighbor in node.neighbor_axis_map:
            return node.neighbor_axis_map[neighbor]

        # Add a new bond axis before the physical dimension
        tensor = node.tensor
        tensor = np.repeat(tensor[..., None], self.bond_dim, axis=-1)
        tensor = np.moveaxis(tensor, -1, -2)  # place before physical dimension

        node.tensor = tensor
        new_axis = tensor.ndim - 2  # physical dim is last
        node.neighbor_axis_map[neighbor] = new_axis
        return new_axis

    def contract_edge(self, i: int, j: int) -> np.ndarray:
        """Contract tensors at nodes i and j along their shared edge."""
        node_i = self.tensors[i]
        node_j = self.tensors[j]

        ai = self._ensure_neighbor_axis(node_i, j)
        aj = self._ensure_neighbor_axis(node_j, i)

        return np.tensordot(node_i.tensor, node_j.tensor, axes=(ai, aj))


# ==============================================================================
# Boundary Identification (Definition 8.1)
# ==============================================================================

def identify_boundary(
    adjacency: np.ndarray,
    method: str = "degree",
    percentile: float = 25.0
) -> List[Index]:
    """
    Definition 8.1: E8 Network Boundary
    
    Boundary nodes are those with below-average connectivity 
    (information-theoretically peripheral).
    
    Methods:
        "degree": Select nodes with degree <= Q_{percentile}
        "spectral": Use Fiedler vector for graph partitioning
        "random": Random selection (for testing)
    
    Args:
        adjacency: E8 adjacency matrix
        method: Selection method
        percentile: Percentile threshold for degree method
    
    Returns:
        List of boundary node indices
    """
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1)
    
    if method == "degree":
        thresh = np.percentile(deg, percentile)
        boundary = np.where(deg <= thresh)[0].tolist()
    
    elif method == "spectral":
        # Compute Laplacian and Fiedler vector
        D = np.diag(deg)
        L = D - adjacency
        eigenvalues, eigenvectors = np.linalg.eigh(L.astype(float))
        # Fiedler vector is second eigenvector (first is constant)
        fiedler = eigenvectors[:, 1]
        # Boundary = nodes with extreme Fiedler values
        thresh_low = np.percentile(fiedler, percentile / 2)
        thresh_high = np.percentile(fiedler, 100 - percentile / 2)
        boundary = np.where((fiedler <= thresh_low) | (fiedler >= thresh_high))[0].tolist()
    
    elif method == "random":
        n_boundary = max(1, int(n * percentile / 100))
        boundary = list(np.random.choice(n, n_boundary, replace=False))
    
    else:
        raise ValueError(f"Unknown boundary selection method: {method}")
    
    return boundary


def boundary_by_degree(
    A: np.ndarray,
    percentile: float = 10.0
) -> List[Index]:
    """
    Select boundary nodes as lowest-degree percentile.
    Convenience wrapper for identify_boundary.
    """
    return identify_boundary(A, method="degree", percentile=percentile)


# ==============================================================================
# Algorithm 11: Holographic Boundary Measurement
# ==============================================================================

def measure_boundary(
    boundary_nodes: List[int],
    seeds: List,  # MajoranaZeroMode list
    e8: E8,
    backend  # QuantumBackend
) -> Dict[int, BoundaryMeasurement]:
    """
    Algorithm 11: Holographic Boundary Measurement
    
    Measures boundary nodes in both computational and Hadamard bases
    to extract holographic encoding of bulk state.
    
    Args:
        boundary_nodes: Indices of boundary nodes
        seeds: Majorana zero mode array
        e8: E8 root system
        backend: Quantum backend
    
    Returns:
        Dictionary mapping node index to measurement result
    """
    measurements = {}
    roots = e8.roots()
    
    for node in boundary_nodes:
        # Measure in computational basis (Z)
        z = backend.measure(seeds[node], basis="computational")
        
        # Measure in Hadamard basis (X)
        # Note: In real implementation, this requires state preparation
        x = backend.measure(seeds[node], basis="hadamard")
        
        measurements[node] = BoundaryMeasurement(
            node=node,
            z_basis=z,
            x_basis=x,
            coordinate=roots[node],
            confidence=1.0
        )
    
    return measurements


# ==============================================================================
# Tensor Network Construction
# ==============================================================================

def build_e8_tensor_network(
    e8: E8,
    adjacency: np.ndarray,
    bond_dim: int = 2,
    physical_dim: int = 2,
    initialization: str = "random",
    max_tensor_rank: int = 8
) -> TensorNetwork:
    """
    Build tensor network over E8 graph topology.
    
    Note: E8 nodes have 56 neighbors each. Creating tensors with 56 bond indices
    would require 2^56 elements which is intractable. Instead, we use a compressed
    representation where each tensor has at most max_tensor_rank virtual indices,
    representing a low-rank approximation of the full tensor.
    
    For production use, this should interface with opt_einsum or cotengra for
    efficient contraction ordering and PEPS/MERA decompositions.
    
    Args:
        e8: E8 root system
        adjacency: E8 adjacency matrix
        bond_dim: Bond dimension for entanglement
        physical_dim: Dimension of physical index
        initialization: "random", "identity", or "hadamard"
        max_tensor_rank: Maximum number of bond indices per tensor (compressed)
    
    Returns:
        TensorNetwork object
    """
    n = len(e8.roots_scaled)
    tensors: Dict[int, TensorNode] = {}
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        neighbors = list(np.where(adjacency[i] == 1)[0])
        
        # Use compressed representation: limit to max_tensor_rank indices
        # The tensor represents a low-rank approximation
        effective_rank = min(len(neighbors), max_tensor_rank)
        
        # Shape: (bond_dim^effective_rank, physical_dim)
        shape = tuple([bond_dim] * effective_rank + [physical_dim])
        
        if initialization == "random":
            tensor = np.random.randn(*shape) / np.sqrt(np.prod(shape))
        elif initialization == "identity":
            tensor = np.zeros(shape)
            # Set diagonal elements to 1
            for idx in np.ndindex(*([bond_dim] * effective_rank)):
                if all(x == idx[0] for x in idx):
                    tensor[idx + (0,)] = 1.0
        elif initialization == "hadamard":
            tensor = np.ones(shape) / np.sqrt(np.prod(shape))
        else:
            tensor = np.random.randn(*shape)

        # Map only the first effective_rank neighbors to axes; others are lazily added
        neighbor_axis_map = {neighbor: axis for axis, neighbor in enumerate(neighbors[:effective_rank])}

        tensors[i] = TensorNode(
            index=i,
            tensor=tensor,
            neighbor_indices=neighbors,
            neighbor_axis_map=neighbor_axis_map,
            physical_dim=physical_dim
        )
        
        # Add edges (avoid duplicates)
        for j in neighbors:
            if j > i:
                edges.append((i, j))
    
    return TensorNetwork(tensors=tensors, edges=edges, bond_dim=bond_dim)


# ==============================================================================
# Algorithm 12: Bulk State Reconstruction
# ==============================================================================

def _edge_score(network: TensorNetwork, edge: Tuple[int, int]) -> float:
    """Heuristic score for choosing contraction order (lower is better)."""
    i, j = edge
    ni = len(network.get_neighbors(i))
    nj = len(network.get_neighbors(j))
    ti = network.tensors[i].tensor
    tj = network.tensors[j].tensor
    return ni + nj + ti.ndim + tj.ndim


def optimize_contraction_order(
    network: TensorNetwork,
    boundary_nodes: Set[int],
    max_edges: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Optimize tensor network contraction order.
    
    Uses a simple heuristic: contract from boundary inward,
    preferring edges where one node has fewer remaining connections.
    
    For production, use opt_einsum or cotengra.
    """
    remaining_edges = list(network.edges)
    order = []
    contracted = set()

    # Start from boundary
    active = set(boundary_nodes)

    processed = 0
    while remaining_edges:
        # Find edge involving an active node
        candidate_edges = [e for e in remaining_edges if e[0] in active or e[1] in active]
        if not candidate_edges:
            candidate_edges = remaining_edges

        # Choose edge with lowest heuristic score
        best_edge = min(candidate_edges, key=lambda e: _edge_score(network, e))
        order.append(best_edge)
        remaining_edges.remove(best_edge)
        processed += 1

        # Update active set
        i, j = best_edge
        active.add(i)
        active.add(j)
        contracted.add((min(i, j), max(i, j)))
        if max_edges is not None and processed >= max_edges:
            break

    return order


def reconstruct_bulk(
    boundary_measurements: Dict[int, BoundaryMeasurement],
    network: TensorNetwork,
    boundary_nodes: List[int],
    edge_limit: Optional[int] = None,
) -> np.ndarray:
    """
    Algorithm 12: Bulk State Reconstruction
    
    Fix boundary tensors from measurements, then contract network inward.
    
    Args:
        boundary_measurements: Measurements from boundary nodes
        network: E8 tensor network
        boundary_nodes: List of boundary node indices
    
    Returns:
        Reconstructed bulk state tensor
    """
    # Fix boundary tensors from measurements
    for node in boundary_nodes:
        if node in boundary_measurements:
            meas = boundary_measurements[node]
            tensor = network.get_tensor(node)
            
            # Project physical index based on measurement
            # Z-basis measurement collapses to |0⟩ or |1⟩
            projected = np.zeros_like(tensor)
            if tensor.ndim > 0:
                # Take slice corresponding to measured value
                idx = tuple([slice(None)] * (tensor.ndim - 1) + [meas.z_basis])
                projected[idx] = tensor[idx]
            
            network.set_tensor(node, projected)
            network.tensors[node].is_boundary = True
    
    # Optimize contraction order
    order = optimize_contraction_order(network, set(boundary_nodes), max_edges=edge_limit)
    
    # Contract network
    # This is a simplified sequential contraction
    # Production code should use einsum optimization
    
    if not order:
        # Return first tensor if no edges
        return next(iter(network.tensors.values())).tensor
    
    # Start with first edge
    i, j = order[0]
    result = network.contract_edge(i, j)
    contracted_nodes = {i, j}
    
    for edge in order[1:]:
        i, j = edge
        if i in contracted_nodes and j not in contracted_nodes:
            tj = network.get_tensor(j)
            result = np.tensordot(result, tj, axes=0)
            contracted_nodes.add(j)
        elif j in contracted_nodes and i not in contracted_nodes:
            ti = network.get_tensor(i)
            result = np.tensordot(result, ti, axes=0)
            contracted_nodes.add(i)
        elif i not in contracted_nodes and j not in contracted_nodes:
            contracted = network.contract_edge(i, j)
            result = np.tensordot(result, contracted, axes=0)
            contracted_nodes.add(i)
            contracted_nodes.add(j)
    
    return result


# ==============================================================================
# Algorithm 13: Compute Inference Vector
# ==============================================================================

def compute_inference(
    query_encoding: Vector,
    e8: E8,
    seeds: List,  # MajoranaZeroMode
    backend,  # QuantumBackend
    evolution_time: float = 1.0
) -> Vector:
    """
    Algorithm 13: Compute Inference Vector
    
    1. Encode query into network by finding closest E8 nodes
    2. Apply Hadamard to create superposition
    3. Let system evolve
    4. Measure response nodes (E8 opposites)
    5. Compute inference vector from measurements
    
    Args:
        query_encoding: Query vector in R^8
        e8: E8 root system
        seeds: Majorana zero mode array
        backend: Quantum backend
        evolution_time: Time to let system evolve
    
    Returns:
        Inference vector in R^8
    """
    roots = e8.root_vectors()
    
    # Find closest nodes to query encoding
    query_nodes = e8.find_closest_nodes(query_encoding, n_closest=8)
    
    # Apply Hadamard to query nodes (create superposition)
    for node in query_nodes:
        # In simulation, this just marks the node as in superposition
        # Real implementation applies H gate
        pass
    
    # Let system evolve (simulated)
    # Real implementation would let quantum state evolve under Hamiltonian
    
    # Find response nodes (E8 antipodal points)
    response_nodes = e8.find_opposite_nodes(query_nodes)
    
    # Measure response nodes
    response_coords = []
    for node in response_nodes:
        measurement = backend.measure(seeds[node], basis="computational")
        sign = 1 if measurement == 0 else -1
        response_coords.append(sign * roots[node])
    
    # Compute mean inference vector
    if response_coords:
        inference_vector = np.mean(response_coords, axis=0)
    else:
        inference_vector = np.zeros(8)
    
    # Normalize
    norm = np.linalg.norm(inference_vector)
    if norm > 1e-10:
        inference_vector = inference_vector / norm
    
    return inference_vector


# ==============================================================================
# Holographic State Projector (Section 9.1)
# ==============================================================================

class HolographicStateProjector:
    """
    Holographic State Projector (HSP)
    
    Maps quantum states to tensor representations using AdS/CFT-like
    correspondence. Implements Eq. 26-27 from spec.
    """
    
    def __init__(self, e8: E8, max_rank: int = 8):
        self.e8 = e8
        self.max_rank = max_rank
    
    def project(
        self,
        quantum_state: np.ndarray,
        subsystem_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        HSP: H_Q → T_C
        
        Project quantum state to tensor representation.
        Rank scales with entanglement entropy.
        """
        if subsystem_indices is None:
            subsystem_indices = list(range(min(8, len(quantum_state.shape))))
        
        # Compute entanglement entropy to determine rank
        # For pure state |ψ⟩, compute S(ρ_A) for subsystem A
        n = len(quantum_state.shape)
        if n < 2:
            return quantum_state
        
        # Reshape to bipartite system
        mid = n // 2
        shape_a = int(np.prod(quantum_state.shape[:mid]))
        shape_b = int(np.prod(quantum_state.shape[mid:]))
        psi_matrix = quantum_state.reshape(shape_a, shape_b)
        
        # SVD to get Schmidt decomposition
        try:
            U, S, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
            # Entanglement entropy
            S_normalized = S / (np.sum(S) + 1e-10)
            entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-10))
        except:
            entropy = 0.0
        
        # Rank proportional to entropy (Eq. 27)
        rank = max(1, min(self.max_rank, int(entropy * 2) + 1))
        
        # Construct tensor representation
        tensor_shape = (2,) * rank
        tensor = np.zeros(tensor_shape, dtype=complex)
        
        # Fill with projected amplitudes
        flat_state = quantum_state.flatten()
        for i, amp in enumerate(flat_state[:np.prod(tensor_shape)]):
            idx = np.unravel_index(i, tensor_shape)
            tensor[idx] = amp
        
        return tensor
    
    def inverse_project(
        self,
        tensor: np.ndarray,
        target_dim: int = 256
    ) -> np.ndarray:
        """
        Inverse projection: T_C → H_Q
        
        Reconstruct quantum state from tensor representation.
        """
        flat = tensor.flatten()
        if len(flat) >= target_dim:
            return flat[:target_dim]
        else:
            padded = np.zeros(target_dim, dtype=complex)
            padded[:len(flat)] = flat
            return padded


# ==============================================================================
# Utility Functions
# ==============================================================================

def boundary_measurements_to_dict(
    measurements: Dict[int, BoundaryMeasurement]
) -> Dict[int, Measurement]:
    """Convert BoundaryMeasurement objects to simple measurement arrays."""
    return {
        node: np.array([m.z_basis, m.x_basis], dtype=float)
        for node, m in measurements.items()
    }


def reconstruct_bulk_from_boundary(
    tn: TensorNetwork,
    boundary_meas: Dict[int, Measurement],
    boundary_nodes: Sequence[int]
) -> np.ndarray:
    """
    Simplified bulk reconstruction for testing.
    
    Apply boundary constraints then contract.
    """
    # Scale boundary tensors by measurement
    for b in boundary_nodes:
        if b in tn.tensors and b in boundary_meas:
            scale = float(boundary_meas[b][0]) if len(boundary_meas[b]) > 0 else 1.0
            tn.tensors[b].tensor = tn.tensors[b].tensor * (scale + 0.5)
    
    # Simple contraction (prototype)
    if not tn.edges:
        tensors = list(tn.tensors.values())
        if tensors:
            return tensors[0].tensor
        return np.array([1.0])
    
    # Contract first few edges
    result = None
    for i, (a, b) in enumerate(tn.edges[:10]):  # Limit for tractability
        if a in tn.tensors and b in tn.tensors:
            contracted = tn.contract_edge(a, b)
            if result is None:
                result = contracted
            else:
                result = np.tensordot(result, contracted, axes=0)
    
    return result if result is not None else np.array([1.0])
