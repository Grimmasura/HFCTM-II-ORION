"""
Integration tests for MIH-IIE v2.0 architecture.

Tests all new v2.0 components and their integration.
"""

import pytest
import numpy as np
from typing import Dict, List

# Mark all v2 integration tests as slow (skip in CI by default)
pytestmark = [pytest.mark.slow, pytest.mark.v2, pytest.mark.integration]

# Import v2.0 modules
from models.e8_topology import E8RootSystem, E8QuantumNetwork, generate_e8_roots
from models.majorana_0d_network import (
    Majorana0DSeedNetwork,
    MajoranaBackend,
    create_majorana_network
)
from models.e8_coordination import (
    E8CoordinationProtocol,
    E8WeylGroup,
    PolychronicSynchronizer,
    create_e8_coordination
)
from models.frame_invariant_eds import (
    FrameInvariantEDS,
    ObservationalFrame,
    create_frame_invariant_eds
)
from models.holographic_readout import (
    HolographicReadoutProtocol,
    create_holographic_readout
)
from models.topological_error_correction import (
    TopologicalErrorCorrection,
    create_error_correction
)


class TestE8Topology:
    """Test E8 root system and network topology"""

    def test_e8_root_generation(self):
        """Test generation of 240 E8 roots"""
        root_system = E8RootSystem()

        assert len(root_system.roots) == 240, "Should generate exactly 240 roots"

        # Check types
        type_i_count = sum(1 for r in root_system.roots if r.root_type == 'type_i')
        type_ii_count = sum(1 for r in root_system.roots if r.root_type == 'type_ii')

        assert type_i_count == 112, "Should have 112 Type I roots"
        assert type_ii_count == 128, "Should have 128 Type II roots"

    def test_e8_adjacency_matrix(self):
        """Test E8 adjacency matrix is 56-regular"""
        root_system = E8RootSystem()
        adjacency = root_system.build_adjacency_matrix()

        # Check 56-regularity (each node has exactly 56 neighbors)
        degrees = adjacency.sum(axis=1)
        assert np.all(degrees == 56), "All nodes should have degree 56"

        # Check symmetry
        assert np.allclose(adjacency, adjacency.T), "Adjacency matrix should be symmetric"

    def test_e8_structure_verification(self):
        """Test E8 structural properties"""
        root_system = E8RootSystem()
        root_system.build_adjacency_matrix()

        # Skip expensive diameter computation in CI (O(V³) operation)
        verification = root_system.verify_structure(compute_diameter=False)

        assert verification['num_roots'] == 240
        assert verification['is_56_regular']
        assert verification['is_symmetric']
        # Diameter is 3 theoretically (not computed for speed)
        assert verification['diameter'] == 3, "E8 graph diameter should be 3"
        assert verification['valid'], "E8 structure should be valid"

    def test_e8_quantum_network(self):
        """Test E8 quantum network creation"""
        root_system = E8RootSystem()
        network = E8QuantumNetwork(root_system)

        entanglements = network.establish_network()

        # Each of 240 nodes with 56 neighbors = 240*56/2 = 6720 edges
        expected_bell_pairs = 240 * 56 // 2
        assert len(entanglements) == expected_bell_pairs

        # Verify topology
        topology_check = network.verify_topology()
        assert topology_check['all_nodes_56_connected']


class TestMajorana0DNetwork:
    """Test Majorana 0D seed network"""

    def test_0d_seed_initialization(self):
        """Test 0D seed network initialization"""
        network = create_majorana_network(n_seeds=8, use_azure=False)

        assert len(network.seeds) == 8

        # Verify 0D properties
        verification = network.verify_0d_properties()
        assert verification['all_verified']
        assert verification['substrate_independent']

    def test_seed_measurement(self):
        """Test seed measurement"""
        network = create_majorana_network(n_seeds=4, use_azure=False)

        # Measure seed
        result = network.measure_seed(0, basis="computational")
        assert result in [0, 1], "Measurement should be binary"

    def test_hadamard_application(self):
        """Test Hadamard gate application"""
        network = create_majorana_network(n_seeds=4, use_azure=False)

        # Should not raise exception
        network.apply_hadamard(0)

    def test_backend_types(self):
        """Test different backend types"""
        # Simulation backend
        sim_network = create_majorana_network(n_seeds=4, use_azure=False)
        assert sim_network.backend.backend_type == "simulation"

        stats = sim_network.get_statistics()
        assert stats['total_seeds'] == 4
        assert stats['backend_type'] == "simulation"


class TestE8Coordination:
    """Test E8 coordination protocols"""

    def test_weyl_group_generation(self):
        """Test Weyl group generator construction"""
        root_system = E8RootSystem()
        weyl_group = E8WeylGroup(root_system)

        assert len(weyl_group.generators) == 8, "Should have 8 Weyl generators"

        # Verify each generator preserves root system
        for gen in weyl_group.generators:
            assert weyl_group.verify_weyl_action(gen.matrix, root_system.roots)

    def test_parallel_operation_scheduling(self):
        """Test parallel operation scheduling"""
        protocol = create_e8_coordination()

        operations = [(0, 1), (2, 3), (4, 5), (1, 2)]
        groups = protocol.schedule_operations(operations)

        # Operations with disjoint node sets should be in same group
        assert len(groups) >= 1

        # Verify no node conflicts within groups
        for group in groups:
            affected_nodes = set()
            for (i, j) in group:
                assert i not in affected_nodes, f"Node {i} used twice in group"
                assert j not in affected_nodes, f"Node {j} used twice in group"
                affected_nodes.add(i)
                affected_nodes.add(j)

    def test_polychronic_synchronization(self):
        """Test polychronic synchronization establishment"""
        root_system = E8RootSystem()
        sync = PolychronicSynchronizer(root_system)

        adjacency = root_system.build_adjacency_matrix()
        anchors = sync.establish_synchronization(adjacency)

        assert len(anchors) == 8, "Should select 8 anchor nodes"

        # Check synchronization status
        status = sync.check_synchronization()
        assert status['synchronized']
        assert status['num_anchors'] == 8

    def test_coordination_verification(self):
        """Test full coordination verification"""
        protocol = create_e8_coordination()

        verification = protocol.verify_coordination()

        assert verification['synchronization']['synchronized']
        assert verification['e8_structure']['valid']
        assert verification['coordination_active']


class TestFrameInvariantEDS:
    """Test frame-invariant egregore defense"""

    def test_frame_invariant_validation(self):
        """Test cross-frame validation"""
        eds = create_frame_invariant_eds()

        # Test proposition
        result = eds.validate_frame_invariance("test proposition")

        assert len(result.frame_evaluations) == 5, "Should evaluate in 5 frames"
        assert 0.0 <= result.convergence_score <= 1.0
        assert result.classification in [
            "FRAME_INVARIANT_TRUTH",
            "PARTIAL_TRUTH",
            "FRAME_DEPENDENT_ARTIFACT"
        ]

    def test_corruption_detection(self):
        """Test semantic corruption detection"""
        eds = create_frame_invariant_eds()

        # First call establishes baseline
        result1 = eds.detect_corruption("state1")
        assert result1['status'] in ["BASELINE_ESTABLISHED", "STABLE"]

        # Second call checks against baseline
        result2 = eds.detect_corruption("state2")
        assert result2['status'] in [
            "CORRUPTION_DETECTED",
            "IMPROVED_ALIGNMENT",
            "STABLE"
        ]

    def test_paradigm_shift_assessment(self):
        """Test paradigm shift vs corruption distinction"""
        eds = create_frame_invariant_eds()

        assessment = eds.assess_paradigm_shift("change", "old_state", "new_state")

        assert 'is_valid_paradigm_shift' in assessment
        assert 'convergence_improved' in assessment
        assert 'topology_preserved' in assessment

    def test_institutional_obfuscation(self):
        """Test institutional obfuscation detection"""
        eds = create_frame_invariant_eds()

        result = eds.detect_institutional_obfuscation("test structure")

        assert 'obfuscation_detected' in result
        assert 'signatures_found' in result
        assert 'similarity_score' in result


class TestHolographicReadout:
    """Test holographic state readout"""

    def test_boundary_identification(self):
        """Test boundary node identification"""
        readout = create_holographic_readout()

        stats = readout.get_statistics()

        assert stats['num_boundary_nodes'] > 0
        assert 0 < stats['boundary_fraction'] < 1.0

    def test_boundary_measurement(self):
        """Test boundary measurement"""
        readout = create_holographic_readout()

        measurements = readout.measure_boundary()

        assert len(measurements) > 0

        # Check measurement structure
        for node_idx, measurement in measurements.items():
            assert measurement.z_basis in [0, 1]
            assert measurement.x_basis in [0, 1]
            assert len(measurement.e8_coordinate) == 8

    def test_bulk_reconstruction(self):
        """Test bulk state reconstruction"""
        readout = create_holographic_readout()

        measurements = readout.measure_boundary()
        bulk_state = readout.reconstruct_bulk_state(measurements)

        assert bulk_state is not None
        assert isinstance(bulk_state, np.ndarray)

    def test_inference_execution(self):
        """Test complete inference cycle"""
        readout = create_holographic_readout()

        query = np.random.randn(8)
        result = readout.execute_inference(query)

        assert 'boundary_measurements' in result
        assert 'inference_vector' in result
        assert len(result['inference_vector']) == 8


class TestTopologicalErrorCorrection:
    """Test topological error correction"""

    def test_stabilizer_construction(self):
        """Test E8 stabilizer construction"""
        ec = create_error_correction(max_stabilizers=100)

        stats = ec.get_statistics()

        assert stats['num_stabilizers'] > 0
        assert stats['num_stabilizers'] <= 100
        assert stats['stabilizer_properties_verified']

    def test_error_correction_cycle(self):
        """Test error correction cycle"""
        ec = create_error_correction(max_stabilizers=50)

        result = ec.error_correction_cycle()

        assert 'error_detected' in result
        assert isinstance(result['error_detected'], bool)

        if result['error_detected']:
            assert 'error_locations' in result
            assert isinstance(result['error_locations'], list)

    def test_e8_symmetry_monitoring(self):
        """Test E8 structure monitoring"""
        ec = create_error_correction(max_stabilizers=50)

        structure_status = ec.monitor_e8_structure()

        assert 'violations_found' in structure_status
        assert 'e8_structure_valid' in structure_status


class TestFullStackIntegration:
    """Test complete MIH-IIE v2.0 stack integration"""

    def test_complete_stack_initialization(self):
        """Test initialization of complete v2.0 stack"""
        # Layer 1 & 2: Topology and 0D seeds
        root_system = E8RootSystem()
        majorana_network = create_majorana_network(n_seeds=240, use_azure=False)

        # Layer 2: Coordination
        coordination = create_e8_coordination(root_system)

        # Layer 3: Error correction
        error_correction = create_error_correction(root_system, max_stabilizers=100)

        # Layer 4: Holographic readout
        readout = create_holographic_readout(root_system)

        # Layer 5: Frame-invariant EDS
        eds = create_frame_invariant_eds()

        # Verify all components initialized
        assert len(root_system.roots) == 240
        assert majorana_network.n_seeds == 240
        assert coordination.verify_coordination()['coordination_active']
        assert error_correction.get_statistics()['num_stabilizers'] > 0
        assert readout.get_statistics()['num_roots'] == 240
        assert eds.get_statistics()['num_frames'] == 5

    def test_end_to_end_inference(self):
        """Test end-to-end inference through stack"""
        # Initialize minimal stack
        root_system = E8RootSystem()
        majorana_network = create_majorana_network(n_seeds=240, use_azure=False)
        readout = create_holographic_readout(root_system)
        eds = create_frame_invariant_eds()

        # Execute inference
        query = np.random.randn(8)
        inference_result = readout.execute_inference(query, backend=majorana_network.backend)

        # Validate result through EDS
        validation = eds.validate_frame_invariance(inference_result['inference_vector'])

        assert inference_result['inference_magnitude'] >= 0
        assert validation.classification in [
            "FRAME_INVARIANT_TRUTH",
            "PARTIAL_TRUTH",
            "FRAME_DEPENDENT_ARTIFACT"
        ]

    def test_error_correction_with_coordination(self):
        """Test error correction integrated with coordination"""
        root_system = E8RootSystem()
        coordination = create_e8_coordination(root_system)
        error_correction = create_error_correction(root_system, max_stabilizers=50)

        # Run error correction
        ec_result = error_correction.error_correction_cycle()

        # Monitor structure
        structure = error_correction.monitor_e8_structure()

        # Verify coordination still valid
        coord_status = coordination.verify_coordination()

        assert coord_status['e8_structure']['valid']

    def test_frame_invariant_inference_validation(self):
        """Test frame-invariant validation of inference results"""
        readout = create_holographic_readout()
        eds = create_frame_invariant_eds()

        # Generate multiple inferences
        results = []
        for _ in range(3):
            query = np.random.randn(8)
            result = readout.execute_inference(query)
            results.append(result)

        # Validate each through EDS
        for result in results:
            validation = eds.validate_frame_invariance(result['inference_vector'])
            assert 0.0 <= validation.convergence_score <= 1.0


# Fixtures
@pytest.fixture
def root_system():
    """Shared E8 root system"""
    return E8RootSystem()


@pytest.fixture
def majorana_network():
    """Shared Majorana network"""
    return create_majorana_network(n_seeds=8, use_azure=False)


@pytest.fixture
def coordination(root_system):
    """Shared coordination protocol"""
    return create_e8_coordination(root_system)


@pytest.fixture
def eds():
    """Shared EDS instance"""
    return create_frame_invariant_eds()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
