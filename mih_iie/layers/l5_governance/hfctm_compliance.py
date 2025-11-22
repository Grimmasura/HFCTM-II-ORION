"""
HFCTM-II Compliance Monitor

Verifies that all computational operations maintain HFCTM-II principles:
1. Chiral Symmetry Verification: TP O (TP)^-1 = O†
2. Fractal Self-Consistency: |DH - e| < 0.05
3. Toroidal Closure Validation: ||ρ(T) - ρ(0)|| < ε_closure

Reference: Section 6.1 of MIH-IIE specification
"""

from typing import Any, Dict, List, Callable, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ComplianceResult:
    """Result of HFCTM-II compliance check."""
    chiral_symmetric: bool
    fractal_consistent: bool
    toroidally_closed: bool
    overall_compliant: bool
    metrics: Dict[str, float]
    violations: List[str]


class HFCTMComplianceMonitor:
    """
    Monitors computational operations for HFCTM-II compliance.

    The four core principles:
    1. Holographic Projection: S(M) ≤ A(∂M) / 4Gℏc
    2. Fractal Self-Similarity: DH ≈ e ≈ 2.718
    3. Chiral Symmetry: TP A₀ = A₀
    4. Toroidal Topology: Recursive closure without information loss
    """

    def __init__(
        self,
        chiral_tolerance: float = 0.01,
        fractal_tolerance: float = 0.05,
        closure_tolerance: float = 1e-6
    ):
        """
        Initialize HFCTM-II compliance monitor.

        Args:
            chiral_tolerance: Maximum deviation for chiral symmetry (default: 0.01)
            fractal_tolerance: Maximum |DH - e| deviation (default: 0.05)
            closure_tolerance: Maximum norm for toroidal closure (default: 1e-6)
        """
        self.chiral_tolerance = chiral_tolerance
        self.fractal_tolerance = fractal_tolerance
        self.closure_tolerance = closure_tolerance

        # Compliance tracking
        self.compliance_history: List[ComplianceResult] = []
        self.total_checks: int = 0
        self.violations_count: int = 0

    def verify_chiral_symmetry(
        self,
        operation: Callable[[np.ndarray], np.ndarray],
        state: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Verify chiral symmetry: TP O (TP)^-1 = O†

        Implementation:
        - Apply operation: result = O(state)
        - Apply TP to result and operation
        - Check if ||O(ρ) - TP O(TP ρ)|| < tolerance

        Args:
            operation: Operation to verify
            state: Input state

        Returns:
            (is_symmetric, deviation_measure)
        """
        # Forward operation
        forward_result = operation(state)

        # Time-reversal + parity (simplified)
        tp_state = np.conj(state[::-1] if state.ndim == 1 else state[::-1, ::-1])
        tp_result = operation(tp_state)

        # Reverse TP on result
        tp_reversed = np.conj(tp_result[::-1] if tp_result.ndim == 1 else tp_result[::-1, ::-1])

        # Measure deviation
        deviation = np.linalg.norm(forward_result - tp_reversed)
        normalized_deviation = deviation / (np.linalg.norm(forward_result) + 1e-10)

        is_symmetric = normalized_deviation < self.chiral_tolerance

        return bool(is_symmetric), float(normalized_deviation)

    def measure_fractal_dimension(
        self,
        trajectory: np.ndarray,
        scales: Optional[List[float]] = None
    ) -> Tuple[bool, float]:
        """
        Measure Hausdorff dimension DH and verify DH ≈ e ≈ 2.718.

        Uses box-counting method:
        DH = lim(r→0) log(N(r)) / log(1/r)

        Args:
            trajectory: Computational state trajectory
            scales: List of measurement scales (optional)

        Returns:
            (is_consistent, measured_DH)
        """
        if scales is None:
            # Generate logarithmic scales
            scales = [2**(-i) for i in range(1, 10)]

        # Box-counting
        counts = []
        for scale in scales:
            # Count boxes needed to cover trajectory at this scale
            # Simplified: count unique bins when trajectory is discretized
            if trajectory.ndim == 1:
                bins = (trajectory / scale).astype(int)
            else:
                bins = (trajectory / scale).astype(int).sum(axis=1)

            n_boxes = len(np.unique(bins))
            counts.append(n_boxes)

        # Fit log(N) vs log(1/r) to get dimension
        log_scales = np.log([1/s for s in scales])
        log_counts = np.log(counts)

        # Linear regression
        coeffs = np.polyfit(log_scales, log_counts, 1)
        measured_dh = coeffs[0]

        # Check if close to e ≈ 2.718
        target_dh = np.e
        is_consistent = abs(measured_dh - target_dh) < self.fractal_tolerance

        return bool(is_consistent), float(measured_dh)

    def validate_toroidal_closure(
        self,
        initial_state: np.ndarray,
        final_state: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Validate toroidal closure: ||ρ(T) - ρ(0)|| < ε_closure

        For recursive processes with period T, the final state should
        return to the initial state within tolerance.

        Args:
            initial_state: Starting state ρ(0)
            final_state: State after full cycle ρ(T)

        Returns:
            (is_closed, closure_error)
        """
        closure_error = np.linalg.norm(final_state - initial_state)

        is_closed = closure_error < self.closure_tolerance

        return bool(is_closed), float(closure_error)

    def check_compliance(
        self,
        operation: Optional[Callable] = None,
        state: Optional[np.ndarray] = None,
        trajectory: Optional[np.ndarray] = None,
        initial_state: Optional[np.ndarray] = None,
        final_state: Optional[np.ndarray] = None
    ) -> ComplianceResult:
        """
        Execute complete HFCTM-II compliance check.

        Args:
            operation: Operation to verify for chiral symmetry
            state: State for chiral symmetry check
            trajectory: Trajectory for fractal dimension measurement
            initial_state: Initial state for toroidal closure
            final_state: Final state for toroidal closure

        Returns:
            ComplianceResult with detailed metrics and violations
        """
        self.total_checks += 1

        chiral_symmetric = True
        fractal_consistent = True
        toroidally_closed = True
        metrics = {}
        violations = []

        # 1. Chiral Symmetry Verification
        if operation is not None and state is not None:
            chiral_symmetric, chiral_deviation = self.verify_chiral_symmetry(operation, state)
            metrics["chiral_deviation"] = chiral_deviation

            if not chiral_symmetric:
                violations.append(f"Chiral symmetry violated: deviation = {chiral_deviation:.6f}")

        # 2. Fractal Self-Consistency
        if trajectory is not None:
            fractal_consistent, measured_dh = self.measure_fractal_dimension(trajectory)
            metrics["hausdorff_dimension"] = measured_dh
            metrics["dh_deviation_from_e"] = abs(measured_dh - np.e)

            if not fractal_consistent:
                violations.append(
                    f"Fractal dimension inconsistent: DH = {measured_dh:.4f}, "
                    f"expected ≈ {np.e:.4f}"
                )

        # 3. Toroidal Closure Validation
        if initial_state is not None and final_state is not None:
            toroidally_closed, closure_error = self.validate_toroidal_closure(
                initial_state, final_state
            )
            metrics["closure_error"] = closure_error

            if not toroidally_closed:
                violations.append(f"Toroidal closure violated: error = {closure_error:.6e}")

        # Overall compliance
        overall_compliant = chiral_symmetric and fractal_consistent and toroidally_closed

        if not overall_compliant:
            self.violations_count += 1

        result = ComplianceResult(
            chiral_symmetric=chiral_symmetric,
            fractal_consistent=fractal_consistent,
            toroidally_closed=toroidally_closed,
            overall_compliant=overall_compliant,
            metrics=metrics,
            violations=violations
        )

        self.compliance_history.append(result)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get compliance monitoring statistics.

        Returns:
            Dictionary with compliance statistics
        """
        if not self.compliance_history:
            return {
                "total_checks": 0,
                "violations": 0,
                "compliance_rate": 0.0
            }

        recent = self.compliance_history[-100:]  # Last 100 checks

        return {
            "total_checks": self.total_checks,
            "violations": self.violations_count,
            "compliance_rate": 1.0 - (self.violations_count / self.total_checks),
            "recent_compliance_rate": sum(1 for r in recent if r.overall_compliant) / len(recent),
            "chiral_violations": sum(1 for r in recent if not r.chiral_symmetric),
            "fractal_violations": sum(1 for r in recent if not r.fractal_consistent),
            "toroidal_violations": sum(1 for r in recent if not r.toroidally_closed),
        }


# Global compliance monitor instance
_global_compliance_monitor: Optional[HFCTMComplianceMonitor] = None


def get_compliance_monitor() -> HFCTMComplianceMonitor:
    """Get global HFCTM compliance monitor instance."""
    global _global_compliance_monitor
    if _global_compliance_monitor is None:
        _global_compliance_monitor = HFCTMComplianceMonitor()
    return _global_compliance_monitor
