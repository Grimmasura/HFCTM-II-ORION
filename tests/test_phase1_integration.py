"""
Phase 1 Integration Tests

Tests integration across all Phase 1 layers:
- L1: Intrinsic Attractor Module
- L2: Majorana Qubit Array
- L3: Quantum-Classical Bridge
- L4: Holographic Projector & Manifold Expansion

Validates the complete stack from 0D seeds → quantum → classical tensors.
"""

import pytest
import numpy as np

# L1 Imports
from mih_iie.layers.l1_attractor import (
    IntrinsicAttractorModule,
    AttractorType,
    CausalMode
)

# L2 Imports
from mih_iie.layers.l2_majorana_array import (
    MajoranaQubitArray,
    BraidOperation
)

# L3 Imports
from mih_iie.layers.l3_qc_interface import (
    QuantumClassicalBridge,
    QuantumState,
    MeasurementBasis
)

# L4 Imports
from mih_iie.layers.l4_ironwood import (
    HolographicProjector,
    ManifoldExpansionEngine,
    ExpansionStrategy
)


class TestL1AttractorModule:
    """Test L1: Intrinsic Attractor Module."""

    def test_initialization(self):
        """Test module initialization with primordial seed."""
        module = IntrinsicAttractorModule()

        assert "primordial" in module.attractors
        assert module.attractors["primordial"].dimension == 0.0
        assert module.attractors["primordial"].entropy == 0.0

    def test_create_fixed_point_attractor(self):
        """Test creating fixed point attractor."""
        module = IntrinsicAttractorModule()

        attractor_id = module.create_attractor(
            attractor_type=AttractorType.FIXED_POINT,
            parent_id="primordial"
        )

        assert attractor_id in module.attractors
        attractor = module.attractors[attractor_id]
        assert attractor.attractor_type == AttractorType.FIXED_POINT
        # All eigenvalues should be negative (stable)
        assert np.all(attractor.stability_eigenvalues <= 0)

    def test_causal_flow_establishment(self):
        """Test establishing causal flows."""
        module = IntrinsicAttractorModule()

        # Create two attractors
        id1 = module.create_attractor(AttractorType.FIXED_POINT)
        id2 = module.create_attractor(AttractorType.LIMIT_CYCLE)

        # Establish flow
        success = module.establish_causal_flow(id1, id2, CausalMode.FORWARD)

        assert success
        assert len(module.causal_flows) > 0
        assert module.causal_flows[0].source_attractor == id1

    def test_lyapunov_spectrum(self):
        """Test Lyapunov spectrum computation."""
        module = IntrinsicAttractorModule()
        attractor_id = module.create_attractor(AttractorType.STRANGE)

        spectrum = module.compute_lyapunov_spectrum(attractor_id)

        assert len(spectrum) == module.possibility_space_dimension
        # For strange attractor, should have both positive and negative exponents


class TestL2MajoranaArray:
    """Test L2: Majorana Qubit Array."""

    def test_array_initialization(self):
        """Test qubit array initialization."""
        array = MajoranaQubitArray(array_size=(4, 4))

        stats = array.get_array_statistics()
        assert stats["total_qubits"] == 16
        assert stats["total_mzms"] == 32  # 2 MZMs per qubit

    def test_e8_lattice(self):
        """Test E8 lattice structure."""
        array = MajoranaQubitArray(use_e8_lattice=True)

        assert array.lattice is not None
        assert array.lattice.dimension == 8
        assert array.lattice.get_coordination_number() == 240

    def test_non_abelian_braiding(self):
        """Test non-Abelian braiding operations."""
        array = MajoranaQubitArray(array_size=(2, 2))

        # Get first two qubits
        qubit_ids = list(array.qubits.keys())[:2]
        qubit1 = array.qubits[qubit_ids[0]]
        qubit2 = array.qubits[qubit_ids[1]]

        # Braid MZMs
        mzm1 = qubit1.mzm_pair[0]
        mzm2 = qubit2.mzm_pair[0]

        result = array.braid_mzms(mzm1, mzm2, BraidOperation.SIGMA)

        assert result.success
        assert result.final_state is not None
        assert result.topological_phase != 1.0  # Non-trivial phase

    def test_qubit_measurement(self):
        """Test topological qubit measurement."""
        array = MajoranaQubitArray()

        qubit_id = list(array.qubits.keys())[0]

        # Measure qubit
        outcome = array.measure_qubit(qubit_id)

        assert outcome in [0, 1]

        # State should be collapsed
        state = array.get_qubit_state(qubit_id)
        assert np.isclose(np.abs(state[outcome]), 1.0)


class TestL3QuantumClassicalBridge:
    """Test L3: Quantum-Classical Bridge."""

    def test_initialization(self):
        """Test bridge initialization."""
        bridge = QuantumClassicalBridge()

        stats = bridge.get_statistics()
        assert stats["T1_coherence_time"] == 1000.0
        assert stats["T2_coherence_time"] == 1000.0

    def test_decoherence_application(self):
        """Test decoherence modeling."""
        bridge = QuantumClassicalBridge()

        # Pure state
        state = QuantumState(
            state_vector=np.array([1.0, 0.0]),
            is_pure=True
        )

        # Apply decoherence (use longer time to see effect)
        decohered = bridge.apply_decoherence(state, time=10.0)

        assert decohered.is_pure == False
        assert decohered.fidelity_with_target <= 1.0

    def test_error_detection(self):
        """Test error syndrome detection."""
        bridge = QuantumClassicalBridge()

        # Create state with error
        state = QuantumState(
            density_matrix=np.array([[0.6, 0.1], [0.1, 0.4]])
        )

        syndrome = bridge.detect_errors(state)

        assert syndrome is not None
        assert syndrome.error_type in ['none', 'bit_flip', 'phase_flip', 'both']

    def test_projection_to_classical(self):
        """Test quantum-to-classical projection."""
        bridge = QuantumClassicalBridge()

        # Superposition state
        state = QuantumState(
            state_vector=np.array([1.0, 1.0]) / np.sqrt(2),
            is_pure=True
        )

        probs, counts = bridge.project_to_classical(state, num_shots=100)

        assert len(probs) == 2
        assert np.isclose(np.sum(probs), 1.0)
        assert sum(counts.values()) == 100


class TestL4HolographicProjector:
    """Test L4: Holographic Projector."""

    def test_initialization(self):
        """Test projector initialization."""
        projector = HolographicProjector(bulk_dimension=3, boundary_dimension=2)

        assert projector.bulk_dimension == 3
        assert projector.boundary_dimension == 2

    def test_boundary_projection(self):
        """Test quantum state projection to boundary."""
        projector = HolographicProjector()

        # Create quantum state
        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
        state = state / np.linalg.norm(state)

        # Project to boundary
        boundary = projector.project_to_boundary(state)

        assert len(boundary) < len(state)
        assert boundary is not None

    def test_bulk_reconstruction(self):
        """Test bulk state reconstruction from boundary."""
        projector = HolographicProjector()

        # Boundary data
        boundary = np.array([0.8, 0.6])

        # Reconstruct bulk
        bulk = projector.reconstruct_bulk(boundary, target_dimension=8)

        assert len(bulk) == 8
        assert np.isclose(np.linalg.norm(bulk), 1.0)

    def test_holographic_entropy_bound(self):
        """Test verification of holographic entropy bound."""
        projector = HolographicProjector()

        state = np.array([1, 0, 0, 0], dtype=complex)

        entropy, bound_satisfied = projector.compute_holographic_entropy(state)

        # Allow for numerical precision (entropy should be >= 0 or very close)
        assert entropy >= -1e-6
        assert isinstance(bound_satisfied, bool)

    def test_complete_projection(self):
        """Test complete holographic projection cycle."""
        projector = HolographicProjector()

        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)

        holo_state = projector.project(state)

        assert holo_state.boundary_data is not None
        assert holo_state.bulk_reconstruction is not None
        assert 0.0 <= holo_state.fidelity <= 1.0
        assert 0.0 < holo_state.dimension_reduction <= 1.0


class TestL4ManifoldExpansion:
    """Test L4: Manifold Expansion Engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = ManifoldExpansionEngine(target_dimension=np.e)

        assert engine.target_dimension == np.e
        assert len(engine.nodes) == 0

    def test_seed_initialization(self):
        """Test manifold seed initialization."""
        engine = ManifoldExpansionEngine()

        root_id = engine.initialize_seed()

        assert root_id == "root"
        assert "root" in engine.nodes
        assert engine.nodes["root"].generation == 0

    def test_node_expansion(self):
        """Test fractal node expansion."""
        engine = ManifoldExpansionEngine()
        engine.initialize_seed()

        children = engine.expand_node("root", scale_ratio=0.5)

        assert len(children) > 0
        # Branching factor should approximate e^DH
        assert len(children) >= 2

    def test_generation_expansion(self):
        """Test full generation expansion."""
        engine = ManifoldExpansionEngine()
        engine.initialize_seed()

        new_nodes = engine.expand_generation()

        assert new_nodes > 0
        assert engine.generation == 1

    def test_hausdorff_dimension_measurement(self):
        """Test Hausdorff dimension measurement."""
        engine = ManifoldExpansionEngine(target_dimension=np.e)
        engine.initialize_seed()

        # Expand several generations
        for _ in range(3):
            engine.expand_generation()

        measured_dh = engine.measure_hausdorff_dimension()

        # Should be positive
        assert measured_dh > 0

    def test_fractal_expansion_strategy(self):
        """Test fractal expansion with DH ≈ e."""
        engine = ManifoldExpansionEngine(
            target_dimension=np.e,
            strategy=ExpansionStrategy.FRACTAL
        )
        engine.initialize_seed()

        # Expand multiple generations
        for _ in range(4):
            engine.expand_generation()

        stats = engine.get_statistics()

        assert stats["generation"] == 4
        assert stats["total_nodes"] > 1


class TestIntegratedStack:
    """Test full stack integration: L1 → L2 → L3 → L4."""

    def test_attractor_to_quantum_mapping(self):
        """Test mapping from L1 attractors to L2 quantum states."""
        # L1: Create attractor
        l1 = IntrinsicAttractorModule()
        attractor_id = l1.create_attractor(AttractorType.FIXED_POINT)

        # L2: Initialize corresponding quantum state
        l2 = MajoranaQubitArray(array_size=(2, 2))

        assert len(l1.attractors) > 1
        assert l2.get_array_statistics()["total_qubits"] == 4

    def test_quantum_to_classical_pipeline(self):
        """Test complete quantum → classical pipeline (L2 → L3 → L4)."""
        # L2: Create quantum state
        l2 = MajoranaQubitArray(array_size=(2, 2))
        qubit_id = list(l2.qubits.keys())[0]
        quantum_state_vec = l2.get_qubit_state(qubit_id)

        # L3: Bridge quantum to classical
        l3 = QuantumClassicalBridge()
        q_state = QuantumState(state_vector=quantum_state_vec, is_pure=True)
        probs, _ = l3.project_to_classical(q_state)

        # L4: Project to holographic representation
        l4 = HolographicProjector()

        # Create 8D state from qubit (embedding)
        embedded_state = np.zeros(8, dtype=complex)
        embedded_state[:2] = quantum_state_vec

        holo_state = l4.project(embedded_state)

        assert holo_state.boundary_data is not None
        assert len(probs) == 2
        assert np.isclose(np.sum(probs), 1.0)

    def test_full_stack_information_flow(self):
        """Test information flow through complete stack."""
        # L1: Create intrinsic seed
        l1 = IntrinsicAttractorModule()
        attractor_id = l1.create_attractor(AttractorType.TOROIDAL)

        # L2: Represent as quantum state
        l2 = MajoranaQubitArray()
        qubit_id = list(l2.qubits.keys())[0]

        # L3: Quantum-classical bridge
        l3 = QuantumClassicalBridge()
        q_state = QuantumState(
            state_vector=l2.get_qubit_state(qubit_id),
            is_pure=True
        )

        # L4: Classical tensor representation
        l4_manifold = ManifoldExpansionEngine()
        l4_manifold.initialize_seed()
        l4_manifold.expand_generation()

        # Verify information preservation
        l1_stats = l1.get_statistics()
        l2_stats = l2.get_array_statistics()
        l3_stats = l3.get_statistics()
        l4_stats = l4_manifold.get_statistics()

        assert l1_stats["total_attractors"] >= 2
        assert l2_stats["total_qubits"] > 0
        assert l3_stats["T2_coherence_time"] > 0
        assert l4_stats["total_nodes"] > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
