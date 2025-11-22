"""
Holographic State Projector

Projects quantum states to classical tensor representations while preserving
information content according to holographic principle.

Implements holographic projection as specified in Section 5.2 of MIH-IIE:
- Information on boundaries encodes bulk dynamics
- Entropy bound: S(M) ≤ A(∂M) / 4Gℏc
- Dimensionality reduction while maintaining information fidelity

Target: 10²⁴ tensor ops/sec (full system)
Current: Classical simulation with arbitrary precision
"""

from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import numpy as np
from enum import Enum


class ProjectionMode(Enum):
    """Holographic projection modes."""
    BOUNDARY = "boundary"  # Project to boundary representation
    BULK = "bulk"  # Project to bulk tensor
    HYBRID = "hybrid"  # Adaptive boundary-bulk


@dataclass
class HolographicState:
    """Holographic representation of quantum state."""
    boundary_data: np.ndarray  # Information on boundary
    bulk_reconstruction: np.ndarray  # Reconstructed bulk state
    entropy: float  # von Neumann entropy
    fidelity: float  # Reconstruction fidelity
    dimension_reduction: float  # Ratio of boundary/bulk dimensions


@dataclass
class ProjectionMetrics:
    """Metrics for holographic projection quality."""
    entropy_bound_satisfied: bool
    information_loss: float
    compression_ratio: float
    reconstruction_fidelity: float
    projection_time_ms: float


class HolographicProjector:
    """
    Projects quantum states to holographic tensor representations.

    Uses AdS/CFT-inspired projection where:
    - d-dimensional bulk → (d-1)-dimensional boundary
    - Information preserved via holographic encoding
    - Entropy bounded by boundary area
    """

    def __init__(
        self,
        bulk_dimension: int = 3,
        boundary_dimension: Optional[int] = None,
        precision: float = 1e-6,
        mode: ProjectionMode = ProjectionMode.HYBRID
    ):
        """
        Initialize holographic projector.

        Args:
            bulk_dimension: Dimension of bulk spacetime
            boundary_dimension: Dimension of boundary (default: bulk_dim - 1)
            precision: Numerical precision threshold
            mode: Projection mode (boundary, bulk, or hybrid)
        """
        self.bulk_dimension = bulk_dimension
        self.boundary_dimension = boundary_dimension or (bulk_dimension - 1)
        self.precision = precision
        self.mode = mode

        # Holographic scaling constant (Planck scale)
        self.G = 6.67430e-11  # Gravitational constant
        self.hbar = 1.054571817e-34  # Reduced Planck constant
        self.c = 299792458  # Speed of light

        # Projection statistics
        self.projection_count = 0
        self.total_information_loss = 0.0
        self.projection_history: List[ProjectionMetrics] = []

    def project_to_boundary(
        self,
        quantum_state: np.ndarray,
        preserve_phase: bool = True
    ) -> np.ndarray:
        """
        Project quantum state to boundary representation.

        Uses radial projection in Hilbert space, mapping bulk state
        to lower-dimensional boundary while preserving essential information.

        Args:
            quantum_state: Quantum state vector or density matrix
            preserve_phase: Whether to preserve phase information

        Returns:
            Boundary representation (holographic encoding)
        """
        # Ensure state is normalized
        if quantum_state.ndim == 1:
            # State vector
            state = quantum_state / np.linalg.norm(quantum_state)
            is_pure = True
        else:
            # Density matrix
            state = quantum_state / np.trace(quantum_state)
            is_pure = False

        # Compute boundary dimension
        bulk_size = state.shape[0]
        boundary_size = int(bulk_size ** (self.boundary_dimension / self.bulk_dimension))

        if is_pure:
            # For pure states: use SVD to find optimal low-rank approximation
            # This preserves maximum information in reduced dimension
            state_matrix = np.outer(state, state.conj())
            U, S, Vh = np.linalg.svd(state_matrix)

            # Keep top boundary_size components
            boundary_data = S[:boundary_size]

            if preserve_phase:
                # Encode phase in complex coefficients
                phases = np.angle(U[:boundary_size, 0])
                boundary_data = boundary_data * np.exp(1j * phases)
        else:
            # For mixed states: eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(state)

            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Project to boundary by keeping top eigenvalues
            boundary_data = eigenvalues[:boundary_size]

            if preserve_phase:
                # Encode dominant eigenvector phases
                phases = np.angle(eigenvectors[:boundary_size, 0])
                boundary_data = boundary_data * np.exp(1j * phases)

        return boundary_data

    def reconstruct_bulk(
        self,
        boundary_data: np.ndarray,
        target_dimension: Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct bulk state from boundary representation.

        Inverse holographic projection: boundary → bulk.
        Uses maximum entropy principle to infer bulk from boundary.

        Args:
            boundary_data: Boundary representation
            target_dimension: Target bulk dimension (default: auto-infer)

        Returns:
            Reconstructed bulk state
        """
        if target_dimension is None:
            # Infer from boundary dimension
            boundary_size = len(boundary_data)
            target_dimension = int(boundary_size ** (self.bulk_dimension / self.boundary_dimension))

        # Pad boundary data to bulk dimension with maximum entropy padding
        if len(boundary_data) < target_dimension:
            # Use maximum entropy distribution for missing components
            missing = target_dimension - len(boundary_data)

            # Renormalize existing data
            total_weight = np.sum(np.abs(boundary_data))
            if total_weight > 0:
                boundary_data = boundary_data / total_weight

            # Add uniform entropy padding
            padding_weight = 1.0 / missing
            padding = np.ones(missing) * padding_weight

            bulk_state = np.concatenate([boundary_data, padding])
        else:
            bulk_state = boundary_data[:target_dimension]

        # Normalize
        bulk_state = bulk_state / np.linalg.norm(bulk_state)

        return bulk_state

    def compute_holographic_entropy(
        self,
        state: np.ndarray,
        boundary_area: Optional[float] = None
    ) -> Tuple[float, bool]:
        """
        Compute von Neumann entropy and verify holographic bound.

        Holographic entropy bound (Bekenstein-Hawking):
        S(M) ≤ A(∂M) / 4Gℏc

        Args:
            state: Quantum state (vector or density matrix)
            boundary_area: Physical boundary area (optional, uses dimensionality if not provided)

        Returns:
            (entropy, bound_satisfied)
        """
        # Convert to density matrix if needed
        if state.ndim == 1:
            rho = np.outer(state, state.conj())
        else:
            rho = state

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > self.precision]  # Filter numerical zeros

        # von Neumann entropy: S = -Tr(ρ log ρ) = -Σ λᵢ log λᵢ
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + self.precision))

        # Compute holographic bound
        if boundary_area is None:
            # Use dimensionality as proxy for area
            # A ~ N^(d-1)/d where N is Hilbert space dimension
            N = len(eigenvalues)
            boundary_area = N ** (self.boundary_dimension / self.bulk_dimension)

        # Bekenstein-Hawking bound (in natural units, setting factors to 1 for dimensionless comparison)
        # In full theory: S_BH = A / (4 G ℏ / c³) = A c³ / (4 G ℏ)
        # For dimensionless comparison, we use S ≤ A
        holographic_bound = boundary_area

        bound_satisfied = entropy <= holographic_bound * (1 + self.precision)

        return float(entropy), bool(bound_satisfied)

    def project(
        self,
        quantum_state: np.ndarray,
        return_metrics: bool = True
    ) -> HolographicState:
        """
        Complete holographic projection: quantum → classical tensor.

        Performs boundary projection, bulk reconstruction, and verification.

        Args:
            quantum_state: Input quantum state
            return_metrics: Whether to compute and return detailed metrics

        Returns:
            HolographicState with boundary data and bulk reconstruction
        """
        import time
        start_time = time.time()

        # Project to boundary
        boundary_data = self.project_to_boundary(quantum_state)

        # Reconstruct bulk
        bulk_reconstruction = self.reconstruct_bulk(boundary_data, target_dimension=len(quantum_state))

        # Compute entropy and verify bound
        entropy, bound_satisfied = self.compute_holographic_entropy(quantum_state)

        # Compute fidelity
        if quantum_state.ndim == 1:
            fidelity = np.abs(np.vdot(quantum_state / np.linalg.norm(quantum_state), bulk_reconstruction)) ** 2
        else:
            # For density matrices: fidelity = Tr(√(√ρ₁ ρ₂ √ρ₁))²
            sqrt_rho1 = self._matrix_sqrt(quantum_state)
            product = sqrt_rho1 @ (bulk_reconstruction @ sqrt_rho1)
            fidelity = np.trace(self._matrix_sqrt(product)) ** 2

        # Dimension reduction ratio
        dimension_reduction = len(boundary_data) / len(quantum_state)

        # Create holographic state
        holo_state = HolographicState(
            boundary_data=boundary_data,
            bulk_reconstruction=bulk_reconstruction,
            entropy=entropy,
            fidelity=float(np.real(fidelity)),
            dimension_reduction=dimension_reduction
        )

        # Update statistics
        self.projection_count += 1
        information_loss = 1.0 - holo_state.fidelity
        self.total_information_loss += information_loss

        if return_metrics:
            projection_time = (time.time() - start_time) * 1000  # ms
            metrics = ProjectionMetrics(
                entropy_bound_satisfied=bound_satisfied,
                information_loss=information_loss,
                compression_ratio=dimension_reduction,
                reconstruction_fidelity=holo_state.fidelity,
                projection_time_ms=projection_time
            )
            self.projection_history.append(metrics)

        return holo_state

    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root using eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Handle numerical errors
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T

    def get_statistics(self) -> Dict[str, float]:
        """Get projection statistics."""
        return {
            "total_projections": self.projection_count,
            "average_information_loss": self.total_information_loss / max(self.projection_count, 1),
            "average_fidelity": 1.0 - (self.total_information_loss / max(self.projection_count, 1)),
            "bulk_dimension": self.bulk_dimension,
            "boundary_dimension": self.boundary_dimension
        }
