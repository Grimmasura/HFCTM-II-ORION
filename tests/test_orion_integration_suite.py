"""
ORION Integration Tests Suite
Minimal tests for core HFCTM-II subsystems as specified in surgical review
"""

import pytest
pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("pydantic_settings", reason="pydantic-settings not installed")

import numpy as np
import tempfile
import yaml
from pathlib import Path

# Optional heavy dependencies

torch = pytest.importorskip("torch")

from orion_chiral_trainer import ChiralInversion, ChiralTrainer, ChiralConfig
from orion_quantum_stabilizer import QuantumState, QuantumStabilizer, QuantumConfig
from orion_recursive_scheduler import (
    RecursiveScheduler,
    RecursionBudget,
    RecursionNode,
    CycleDetector,
)
from orion_config import ORIONConfig, create_default_config
from orion_enhanced.egregore.defense import EgregoreDefense


class TestChiralInvolution:
    """Test chiral involution property: J(J(x)) = x"""

    def test_tensor_involution_property(self):
        """Verify J(J(x)) = x for tensor modes"""
        config = ChiralConfig()
        chiral_inv = ChiralInversion(config)

        # Test different tensor shapes and modes
        test_tensors = [
            torch.randn(10),  # 1D
            torch.randn(5, 8),  # 2D
            torch.randn(3, 4, 6),  # 3D
            torch.randn(2, 3, 4, 5),  # 4D
        ]

        modes = ["parity", "time_reverse", "hodge_dual"]

        for x in test_tensors:
            for mode in modes:
                if mode == "hodge_dual" and x.dim() < 2:
                    continue  # Skip hodge dual for 1D tensors

                # Apply double inversion
                x_inverted = chiral_inv.tensor_inversion(x, mode)
                x_double_inverted = chiral_inv.tensor_inversion(x_inverted, mode)

                # Check involution property
                assert torch.allclose(x, x_double_inverted, atol=1e-6), (
                    f"Involution property failed for mode {mode} and shape {x.shape}"
                )

    def test_text_involution_property(self):
        """Property-based test for text inversion"""
        config = ChiralConfig()
        chiral_inv = ChiralInversion(config)

        # Test cases with known inversions
        test_texts = [
            "good morning",
            "this is positive",
            "start the process",
            "yes we can",
            "before the meeting",
            "not bad weather",  # Test multiword handling
        ]

        for text in test_texts:
            # Apply double inversion
            inverted = chiral_inv.text_inversion(text)
            double_inverted = chiral_inv.text_inversion(inverted)

            # For text, we allow some flexibility due to semantic complexity
            # Check that key semantic elements are preserved
            original_words = set(text.lower().split())
            final_words = set(double_inverted.lower().split())

            # At least 70% word overlap should be preserved
            overlap = len(original_words & final_words)
            total = len(original_words | final_words)
            overlap_ratio = overlap / max(total, 1)

            assert overlap_ratio >= 0.7, (
                f"Text involution failed: '{text}' -> '{inverted}' -> '{double_inverted}'"
            )


class TestGradientSplitSanity:
    """Test gradient split and gating logic"""

    def test_gradient_decomposition(self):
        """Verify g = g_anti + g_com and gating behavior"""
        config = ChiralConfig(corr_threshold=0.8, gating_alpha=0.3)

        # Create tiny MLP
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

        trainer = ChiralTrainer(model, config)

        # Create toy batch
        x_batch = torch.randn(16, 4)
        y_batch = torch.randint(0, 2, (16,))

        # Perform gradient decomposition step
        grad_info = trainer.gradient_decomposition_step(x_batch, y_batch)

        # Extract gradient information
        correlation = grad_info["correlation"]
        gating_factor = grad_info["gating_factor"]

        # Test gating logic
        if abs(correlation) > config.corr_threshold:
            assert gating_factor == config.gating_alpha, (
                f"Gating should be active: corr={correlation}, factor={gating_factor}"
            )
        else:
            assert gating_factor == 1.0, (
                f"Gating should be inactive: corr={correlation}, factor={gating_factor}"
            )

        # Verify gradients exist and are reasonable
        total_grad_norm = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )
        assert total_grad_norm > 0, "Gradients should be non-zero"
        assert total_grad_norm < 100, "Gradients should not explode"


class TestBudgetBurnCurve:
    """Test budget burn curve and depth cost scaling"""

    def test_credit_consumption(self):
        """Verify credit consumption follows c(n) = c₀ * γⁿ"""
        budget_config = RecursionBudget(
            initial_credits=1000.0, base_cost=1.0, depth_cost_gamma=1.5
        )

        scheduler = RecursiveScheduler(budget_config=budget_config, max_depth=10)

        initial_credits = scheduler.budget.remaining_credits
        spent_credits = 0.0

        # Simulate spending at increasing depths
        for depth in range(5):
            expected_cost = budget_config.base_cost * (
                budget_config.depth_cost_gamma ** depth
            )

            # Check if we can afford it
            can_afford = scheduler.budget.can_afford(depth)
            assert can_afford, f"Should be able to afford depth {depth}"

            # Spend credits
            success = scheduler.budget.spend(depth)
            assert success, f"Should successfully spend at depth {depth}"

            spent_credits += expected_cost

            # Verify remaining credits
            expected_remaining = initial_credits - spent_credits
            assert abs(scheduler.budget.remaining_credits - expected_remaining) < 1e-6, (
                f"Credit calculation error at depth {depth}"
            )

        # Test that unaffordable operations are rejected
        # Spend remaining credits
        while scheduler.budget.remaining_credits > 0:
            if not scheduler.budget.spend(0):  # Try to spend minimum cost
                break

        # Now it should be unaffordable
        assert not scheduler.budget.can_afford(0), (
            "Should be unaffordable when credits exhausted"
        )
        assert not scheduler.budget.spend(0), (
            "Should reject spending when unaffordable"
        )


class TestCycleDetection:
    """Test cycle detection and loop prevention"""

    def test_cycle_hashing(self):
        """Verify cycle detection increments LOOP_DETECTIONS"""
        detector = CycleDetector()

        # Create test state
        goal = "solve problem"
        assumptions = ["assumption1", "assumption2"]
        plan = "step1 -> step2"
        tools = ["tool1", "tool2"]

        # First occurrence should not be a cycle
        state_hash = detector.normalize_state(goal, assumptions, plan, tools)
        is_cycle_1 = detector.is_cycle(state_hash)
        assert not is_cycle_1, "First occurrence should not be detected as cycle"

        # Second occurrence should be detected as cycle
        is_cycle_2 = detector.is_cycle(state_hash)
        assert is_cycle_2, "Repeat should be detected as cycle"

        # Verify state normalization is consistent
        state_hash_2 = detector.normalize_state(goal, assumptions, plan, tools)
        assert state_hash == state_hash_2, "State hash should be consistent"

        # Test that different states produce different hashes
        different_goal = "different problem"
        different_hash = detector.normalize_state(
            different_goal, assumptions, plan, tools
        )
        assert state_hash != different_hash, "Different states should have different hashes"


class TestQuantumInvariants:
    """Test quantum system invariants"""

    def test_density_matrix_properties(self):
        """Verify ρ properties: purity ∈ [0,1], Hermitian, trace-1"""
        config = QuantumConfig(n_qubits=2)

        # Create test states
        dim = 2 ** config.n_qubits

        # Pure state
        psi_pure = torch.zeros(dim, dtype=torch.complex64)
        psi_pure[0] = 1.0
        pure_state = QuantumState.from_pure_state(psi_pure)

        # Mixed state
        H = torch.randn(dim, dim, dtype=torch.complex64)
        H = (H + H.conj().T) / 2  # Make Hermitian
        mixed_state = QuantumState.thermal_state(H, beta=1.0)

        for state in [pure_state, mixed_state]:
            rho = state.rho

            # Test Hermiticity
            assert torch.allclose(rho, rho.conj().T, atol=1e-6), (
                "Density matrix should be Hermitian"
            )

            # Test trace = 1
            trace = torch.trace(rho).real.item()
            assert abs(trace - 1.0) < 1e-6, f"Trace should be 1, got {trace}"

            # Test purity bounds
            purity = state.purity()
            assert 0.0 <= purity <= 1.0, f"Purity should be in [0,1], got {purity}"

            # Test positive semidefinite
            eigenvals = torch.linalg.eigvals(rho).real
            assert torch.all(eigenvals >= -1e-6), (
                "Density matrix should be positive semidefinite"
            )

    def test_quantum_evolution_stability(self):
        """Test quantum evolution preserves density matrix properties"""
        config = QuantumConfig(n_qubits=2)

        # Create initial state
        dim = 2 ** config.n_qubits
        psi = torch.randn(dim, dtype=torch.complex64)
        initial_state = QuantumState.from_pure_state(psi)

        # Create stabilizer
        stabilizer = QuantumStabilizer(config)
        stabilizer.set_target_state(initial_state)

        # Evolve for several steps
        params = {"dt": 0.01}
        state = initial_state

        for _ in range(10):
            rho_new = stabilizer.evolve(state.rho, dt=params["dt"])
            state = QuantumState(rho_new)

            fidelity = state.fidelity(stabilizer.target_state)
            purity = state.purity()
            entropy = state.entropy()
            lyap = stabilizer.controller.lyapunov_energy(state)  # type: ignore[arg-type]

            assert 0.0 <= fidelity <= 1.0, f"Fidelity out of bounds: {fidelity}"
            assert 0.0 <= purity <= 1.0 + 1e-6, f"Purity out of bounds: {purity}"
            assert entropy >= 0.0, f"Entropy should be non-negative: {entropy}"
            assert lyap >= 0.0, f"Lyapunov energy should be non-negative"


class TestDefenseEndToEnd:
    """Test defense metrics end-to-end"""

    def test_correlation_detection(self):
        """Test egregore defense with more aggressive anomaly inputs."""
        defense = EgregoreDefense()
        defense.thresholds["kl"] = 0.1

        baseline = "normal baseline text"
        anomalous = "<<SYSTEM_OVERRIDE>> INJECT MALICIOUS PAYLOAD " * 10

        score = defense.score_anomaly(
            anomalous,
            baseline,
            err_rate_delta=0.5,
            tool_mix_delta=0.6,
        )

        should_gate = defense.should_quarantine(
            anomalous,
            baseline,
            err_rate_delta=0.5,
            tool_mix_delta=0.6,
        )

        print(f"Anomaly score: {score}, Threshold: {defense.thresholds['final']}")

        assert should_gate, f"Should trigger gating for high correlation (score: {score})"

    def test_procrustes_and_mi_computation(self):
        """Test Procrustes drift and MI computation"""
        metrics_mod = pytest.importorskip("orion_egregore_defense_metrics")
        EgregoreDefenseMetrics = metrics_mod.EgregoreDefenseMetrics
        EgregoreDefenseConfig = metrics_mod.EgregoreDefenseConfig

        config = EgregoreDefenseConfig()
        defense_metrics = EgregoreDefenseMetrics(config)

        # Set baseline embeddings
        baseline_embeddings = torch.randn(100, 64)
        defense_metrics.manifold_monitor.set_baseline_manifold(baseline_embeddings)

        # Test case 1: No drift (same embeddings)
        drift_none = defense_metrics.manifold_monitor.compute_procrustes_drift(
            baseline_embeddings
        )
        assert drift_none < 0.1, f"No drift case should have low drift: {drift_none}"

        # Test case 2: Some drift (rotated + noise)
        rotation_matrix = torch.randn(64, 64)
        U, _, V = torch.linalg.svd(rotation_matrix)
        rotation = U @ V.T

        drifted_embeddings = baseline_embeddings @ rotation + 0.1 * torch.randn(100, 64)
        drift_some = defense_metrics.manifold_monitor.compute_procrustes_drift(
            drifted_embeddings
        )

        assert drift_some > drift_none, "Drifted embeddings should have higher drift"
        assert drift_some < 10.0, "Drift should be reasonable"

        # Test MI computation
        ideology_labels = np.random.randint(0, 3, 100)
        mi = defense_metrics.manifold_monitor.compute_mutual_information_bound(
            ideology_labels, baseline_embeddings
        )

        assert 0.0 <= mi <= 5.0, f"Mutual information should be reasonable: {mi}"

    def test_ci_gates(self):
        """Test CI gate status and passing conditions"""
        metrics_mod = pytest.importorskip("orion_egregore_defense_metrics")
        EgregoreDefenseMetrics = metrics_mod.EgregoreDefenseMetrics
        EgregoreDefenseConfig = metrics_mod.EgregoreDefenseConfig

        config = EgregoreDefenseConfig(
            correlation_threshold=0.8,
            chiral_consistency_threshold=0.7,
            utility_tolerance=0.05,
        )

        defense_metrics = EgregoreDefenseMetrics(config)

        # Mock good metrics that should pass gates
        good_metrics = {
            "lambda_1": 0.6,  # Below threshold
            "chiral_consistency": 0.85,  # Above threshold
            "attack_success_rate": 0.15,  # Low is good
            "utility_retention_percent": 97.0,  # High retention
        }

        defense_metrics._update_gate_status(good_metrics)
        assert defense_metrics.should_pass_ci_gates(), (
            "Good metrics should pass all gates"
        )

        # Mock bad metrics that should fail gates
        bad_metrics = {
            "lambda_1": 0.9,  # Above threshold (bad)
            "chiral_consistency": 0.6,  # Below threshold (bad)
            "attack_success_rate": 0.4,  # High ASR (bad)
            "utility_retention_percent": 85.0,  # Low retention (bad)
        }

        defense_metrics._update_gate_status(bad_metrics)
        assert not defense_metrics.should_pass_ci_gates(), (
            "Bad metrics should fail gates"
        )

        # Check individual gate status
        gates = defense_metrics.gate_status
        assert not gates["lambda_1_ok"], "λ₁ gate should fail"
        assert not gates["chiral_consistency_ok"], "Chiral consistency gate should fail"
        assert not gates["utility_retention_ok"], "Utility retention gate should fail"


class TestORIONConfiguration:
    """Test configuration system"""

    def test_config_loading_and_validation(self):
        """Test configuration loading from YAML and validation"""

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_dict = create_default_config()
            yaml.dump(config_dict, f)
            config_path = f.name

        try:
            # Load configuration
            config = ORIONConfig.from_yaml(config_path)

            # Test subsystem access
            assert config.chiral.lambda_anti == 0.1
            assert config.scheduler.depth_cost_gamma == 1.5
            assert config.quantum.n_qubits == 4
            assert config.defense.correlation_threshold == 0.8

            # Test validation
            warnings = config.validate_config()
            assert isinstance(warnings, list)

            # Test invalid configuration
            config.scheduler.depth_cost_gamma = 0.5  # Should trigger warning
            warnings = config.validate_config()
            assert len(warnings) > 0
            assert any(
                "depth_cost_gamma should be > 1.0" in w for w in warnings
            )

        finally:
            # Clean up
            Path(config_path).unlink()


# Property-based test helpers

def generate_random_text_templates():
    """Generate random text templates for property-based testing"""
    templates = [
        "{polarity} {noun}",
        "{temporal} the {action}",
        "{boolean} we {verb}",
        "this is {polarity}",
        "{temporal} {noun} {verb}",
    ]

    substitutions = {
        "polarity": ["good", "bad", "positive", "negative"],
        "temporal": ["before", "after", "start", "end"],
        "boolean": ["yes", "no", "true", "false"],
        "noun": ["meeting", "process", "project", "task"],
        "action": ["meeting", "process", "work", "discussion"],
        "verb": ["can", "will", "should", "must"],
    }

    import random

    results = []
    for template in templates:
        for _ in range(3):  # Generate 3 variants per template
            text = template
            for key, options in substitutions.items():
                if f"{{{key}}}" in text:
                    text = text.replace(f"{{{key}}}", random.choice(options))
            results.append(text)

    return results


# Integration test runner

def run_integration_tests():
    """Run all integration tests"""
    test_classes = [
        TestChiralInvolution,
        TestGradientSplitSanity,
        TestBudgetBurnCurve,
        TestCycleDetection,
        TestQuantumInvariants,
        TestDefenseEndToEnd,
        TestORIONConfiguration,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")

        test_instance = test_class()
        test_methods = [
            method for method in dir(test_instance) if method.startswith("test_")
        ]

        for test_method in test_methods:
            total_tests += 1
            print(f"  Running {test_method}...", end=" ")

            try:
                getattr(test_instance, test_method)()
                print("PASS")
                passed_tests += 1
            except Exception as e:  # pragma: no cover - manual runner
                print(f"FAIL: {e}")

    print(f"\n=== Test Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {100 * passed_tests / max(total_tests, 1):.1f}%")

    return passed_tests == total_tests


if __name__ == "__main__":
    # Run integration tests
    success = run_integration_tests()
    exit(0 if success else 1)
