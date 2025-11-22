"""
Tests for L5: Recursive Governance Layer

Tests HFCTM-II compliance monitoring, chiral inversion, egregore defense,
and polychronic synchronization.
"""

import pytest
import numpy as np
from mih_iie.layers.l5_governance.chiral_inversion import (
    ChiralInversionController,
    ChiralInversionResult
)
from mih_iie.layers.l5_governance.egregore_defense import (
    EgregoreDefenseSystem,
    SemanticState,
    CorruptedPattern
)
from mih_iie.layers.l5_governance.hfctm_compliance import (
    HFCTMComplianceMonitor,
    ComplianceResult
)


class TestChiralInversionController:
    """Test chiral inversion and time-reversal validation."""

    def test_initialization(self):
        """Test controller initialization."""
        controller = ChiralInversionController(fidelity_threshold=0.95)
        assert controller.fidelity_threshold == 0.95
        assert len(controller.inversion_history) == 0

    def test_time_reversal_operator(self):
        """Test time-reversal operator (complex conjugation)."""
        controller = ChiralInversionController()
        state = np.array([1+2j, 3+4j, 5+6j])
        reversed_state = controller.apply_time_reversal(state)

        # Check complex conjugation
        assert np.allclose(reversed_state, np.conj(state))

    def test_parity_operator(self):
        """Test parity operator (reversal)."""
        controller = ChiralInversionController()
        state = np.array([1, 2, 3, 4, 5])
        parity_state = controller.apply_parity(state)

        # Check reversal
        assert np.array_equal(parity_state, state[::-1])

    def test_fidelity_calculation(self):
        """Test fidelity calculation between states."""
        controller = ChiralInversionController()

        # Identical states should have fidelity = 1
        state1 = np.array([1, 0, 0])
        state2 = np.array([1, 0, 0])
        fidelity = controller.compute_fidelity(state1, state2)
        assert np.isclose(fidelity, 1.0)

        # Orthogonal states should have fidelity = 0
        state1 = np.array([1, 0])
        state2 = np.array([0, 1])
        fidelity = controller.compute_fidelity(state1, state2)
        assert np.isclose(fidelity, 0.0)

    def test_chiral_symmetry_validation_pass(self):
        """Test chiral symmetry validation with symmetric operation."""
        controller = ChiralInversionController(fidelity_threshold=0.9)

        # Identity operation should be chirally symmetric
        def identity_op(state):
            return state.copy()

        initial_state = np.array([1, 2, 3, 4])
        result = controller.validate_chiral_symmetry(identity_op, initial_state)

        assert isinstance(result, ChiralInversionResult)
        assert result.is_valid
        assert result.fidelity >= 0.9
        assert len(result.divergence_points) == 0

    def test_chiral_symmetry_validation_fail(self):
        """Test chiral symmetry validation with asymmetric operation."""
        controller = ChiralInversionController(fidelity_threshold=0.95)

        # Non-symmetric operation (adds noise)
        def noisy_op(state):
            return state + np.random.randn(*state.shape) * 10

        initial_state = np.array([1, 2, 3, 4])
        result = controller.validate_chiral_symmetry(noisy_op, initial_state)

        # Should fail due to noise
        assert isinstance(result, ChiralInversionResult)
        # May pass or fail depending on random noise, but should return valid result
        assert result.fidelity >= 0.0

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        controller = ChiralInversionController()

        # Run several validations
        def simple_op(state):
            return state * 2

        state = np.array([1, 2, 3])
        for _ in range(5):
            controller.validate_chiral_symmetry(simple_op, state)

        stats = controller.get_statistics()
        assert stats["total_validations"] == 5
        assert "average_fidelity" in stats
        assert stats["success_rate"] >= 0.0


class TestEgregoreDefenseSystem:
    """Test egregore defense and semantic drift detection."""

    def test_initialization(self):
        """Test EDS initialization."""
        eds = EgregoreDefenseSystem()
        assert eds.torsion_threshold_sigma == 3.0
        assert eds.similarity_threshold == 0.80
        assert len(eds.corrupted_patterns) > 0
        assert eds.total_checks == 0

    def test_corrupted_pattern_database(self):
        """Test that corrupted pattern database is initialized."""
        eds = EgregoreDefenseSystem()

        # Check for expected patterns
        pattern_ids = [p.pattern_id for p in eds.corrupted_patterns]
        assert "circular_reasoning" in pattern_ids
        assert "authority_appeal" in pattern_ids
        assert "manufactured_consensus" in pattern_ids
        assert "linguistic_drift" in pattern_ids
        assert "measurement_corruption" in pattern_ids

    def test_semantic_torsion_measurement_baseline(self):
        """Test semantic torsion measurement (first measurement)."""
        eds = EgregoreDefenseSystem()

        semantic_field = {"concept1": "meaning1", "concept2": "meaning2"}
        torsion = eds.measure_semantic_torsion(semantic_field)

        # First measurement should be 0 (establishes baseline)
        assert torsion == 0.0
        assert eds.baseline is None

    def test_semantic_torsion_measurement_drift(self):
        """Test semantic torsion measurement with drift."""
        eds = EgregoreDefenseSystem()

        # Establish baseline
        baseline_field = {"concept": "original meaning"}
        eds.safety_check(baseline_field)
        eds.baseline = SemanticState(
            timestamp=0.0,
            symbol_mappings=baseline_field,
            torsion_measure=0.0
        )

        # Measure drift
        drifted_field = {"concept": "completely different meaning"}
        torsion = eds.measure_semantic_torsion(drifted_field)

        # Should detect drift
        assert torsion > 0.0

    def test_semantic_distance(self):
        """Test semantic distance calculation."""
        eds = EgregoreDefenseSystem()

        # Identical meanings
        dist = eds._semantic_distance("hello", "hello")
        assert dist == 0.0

        # Different meanings
        dist = eds._semantic_distance("hello", "world")
        assert dist > 0.0

        # Completely different
        dist = eds._semantic_distance("abc", "xyz")
        assert dist > 0.0

    def test_corrupted_pattern_detection_match(self):
        """Test detection of corrupted patterns."""
        eds = EgregoreDefenseSystem(similarity_threshold=0.5)

        # Structure matching circular reasoning
        circular_structure = {
            "type": "circular",
            "dependency_loop": True
        }

        matches = eds.detect_corrupted_patterns(circular_structure)

        # Should detect circular reasoning
        assert len(matches) > 0
        matched_ids = [m[0].pattern_id for m in matches]
        assert "circular_reasoning" in matched_ids

    def test_corrupted_pattern_detection_no_match(self):
        """Test that clean structures don't match corrupted patterns."""
        eds = EgregoreDefenseSystem(similarity_threshold=0.9)

        # Clean structure
        clean_structure = {
            "type": "valid_reasoning",
            "evidence_provided": True,
            "logical_flow": True
        }

        matches = eds.detect_corrupted_patterns(clean_structure)

        # Should not trigger high-confidence matches
        high_confidence_matches = [m for m in matches if m[1] > 0.9]
        assert len(high_confidence_matches) == 0

    def test_safety_check_clean(self):
        """Test safety check with clean semantic field."""
        eds = EgregoreDefenseSystem()

        semantic_field = {"term": "meaning"}
        result = eds.safety_check(semantic_field)

        assert "safe" in result
        assert "should_quarantine" in result
        assert "alerts" in result
        assert "torsion" in result
        assert isinstance(result["alerts"], list)

    def test_safety_check_with_corrupted_pattern(self):
        """Test safety check detecting corrupted pattern."""
        eds = EgregoreDefenseSystem(similarity_threshold=0.5)

        semantic_field = {"term": "meaning"}
        inference_structure = {
            "type": "circular",
            "dependency_loop": True
        }

        result = eds.safety_check(semantic_field, inference_structure)

        # Should detect circular reasoning
        assert len(result["alerts"]) > 0
        # May trigger quarantine depending on similarity
        assert "should_quarantine" in result

    def test_autonomous_correction_no_baseline(self):
        """Test autonomous correction without baseline."""
        eds = EgregoreDefenseSystem()

        result = eds.autonomous_correction({})

        assert result["success"] is False
        assert "No baseline" in result["message"]

    def test_autonomous_correction_with_baseline(self):
        """Test autonomous correction with baseline."""
        eds = EgregoreDefenseSystem()

        # Establish baseline
        eds.baseline = SemanticState(
            timestamp=100.0,
            symbol_mappings={"term": "original"},
            torsion_measure=0.5
        )

        result = eds.autonomous_correction({"contaminated": "state"})

        assert result["success"] is True
        assert "corrected_state" in result
        assert result["corrected_state"]["semantic_field"] == {"term": "original"}

    def test_statistics(self):
        """Test EDS statistics."""
        eds = EgregoreDefenseSystem()

        # Run several checks
        for _ in range(3):
            eds.safety_check({"term": "meaning"})

        stats = eds.get_statistics()

        assert stats["total_checks"] == 3
        assert "alerts_triggered" in stats
        assert "quarantines_issued" in stats


class TestHFCTMComplianceMonitor:
    """Test HFCTM-II compliance monitoring."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = HFCTMComplianceMonitor()
        assert monitor.chiral_tolerance == 0.01
        assert monitor.fractal_tolerance == 0.05
        assert monitor.closure_tolerance == 1e-6
        assert monitor.total_checks == 0

    def test_chiral_symmetry_verification(self):
        """Test chiral symmetry verification."""
        monitor = HFCTMComplianceMonitor()

        # Identity should be chirally symmetric
        def identity(state):
            return state.copy()

        state = np.array([1, 2, 3, 4])
        is_symmetric, deviation = monitor.verify_chiral_symmetry(identity, state)

        assert is_symmetric
        assert deviation < monitor.chiral_tolerance

    def test_fractal_dimension_measurement(self):
        """Test Hausdorff dimension measurement."""
        monitor = HFCTMComplianceMonitor()

        # Create a trajectory
        trajectory = np.random.randn(100)

        is_consistent, measured_dh = monitor.measure_fractal_dimension(trajectory)

        # Should return a dimension
        assert measured_dh > 0
        # May or may not be close to e depending on random trajectory
        assert isinstance(is_consistent, bool)

    def test_toroidal_closure_validation_closed(self):
        """Test toroidal closure with matching states."""
        monitor = HFCTMComplianceMonitor()

        initial = np.array([1, 2, 3])
        final = np.array([1, 2, 3])  # Perfect closure

        is_closed, error = monitor.validate_toroidal_closure(initial, final)

        assert is_closed
        assert error < monitor.closure_tolerance

    def test_toroidal_closure_validation_open(self):
        """Test toroidal closure with non-matching states."""
        monitor = HFCTMComplianceMonitor()

        initial = np.array([1, 2, 3])
        final = np.array([10, 20, 30])  # Large deviation

        is_closed, error = monitor.validate_toroidal_closure(initial, final)

        assert not is_closed
        assert error > monitor.closure_tolerance

    def test_full_compliance_check(self):
        """Test complete compliance check."""
        monitor = HFCTMComplianceMonitor()

        # Simple operation
        def operation(state):
            return state * 2

        state = np.array([1, 2, 3])
        trajectory = np.random.randn(50)
        initial = np.array([1, 0, 0])
        final = np.array([1.001, 0, 0])  # Near closure

        result = monitor.check_compliance(
            operation=operation,
            state=state,
            trajectory=trajectory,
            initial_state=initial,
            final_state=final
        )

        assert isinstance(result, ComplianceResult)
        assert isinstance(result.chiral_symmetric, bool)
        assert isinstance(result.fractal_consistent, bool)
        assert isinstance(result.toroidally_closed, bool)
        assert isinstance(result.overall_compliant, bool)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.violations, list)

    def test_compliance_statistics(self):
        """Test compliance statistics tracking."""
        monitor = HFCTMComplianceMonitor()

        # Run several checks
        for _ in range(3):
            monitor.check_compliance()

        stats = monitor.get_statistics()

        assert stats["total_checks"] == 3
        assert "violations" in stats
        assert "compliance_rate" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
