"""
HFCTM-II Compliance Monitoring Demo

Demonstrates verification of HFCTM-II principles:
- Chiral Symmetry
- Fractal Self-Consistency (DH ≈ e)
- Toroidal Closure

Reference: Section 6.1 of MIH-IIE specification
"""

import numpy as np
from mih_iie.layers.l5_governance.hfctm_compliance import HFCTMComplianceMonitor


def main():
    print("=== HFCTM-II Compliance Monitor Demo ===\n")

    # Initialize monitor
    monitor = HFCTMComplianceMonitor(
        chiral_tolerance=0.01,
        fractal_tolerance=0.05,
        closure_tolerance=1e-6
    )
    print(f"Initialized with tolerances:")
    print(f"  Chiral: {monitor.chiral_tolerance}")
    print(f"  Fractal: {monitor.fractal_tolerance}")
    print(f"  Closure: {monitor.closure_tolerance}\n")

    # Example 1: Chiral symmetry verification
    print("Example 1: Verifying chiral symmetry")
    def symmetric_op(state):
        """Identity-like operation that preserves chiral symmetry"""
        return state.copy()

    state = np.array([1.0, 2.0, 3.0, 4.0])
    is_symmetric, deviation = monitor.verify_chiral_symmetry(symmetric_op, state)

    print(f"  Is symmetric: {is_symmetric}")
    print(f"  Deviation: {deviation:.8f}\n")

    # Example 2: Fractal dimension measurement
    print("Example 2: Measuring fractal dimension")
    # Create a synthetic trajectory with known fractal properties
    trajectory = np.cumsum(np.random.randn(1000)) * 0.1

    is_consistent, measured_dh = monitor.measure_fractal_dimension(trajectory)
    print(f"  Is consistent with DH ≈ e: {is_consistent}")
    print(f"  Measured DH: {measured_dh:.4f}")
    print(f"  Target DH (e): {np.e:.4f}")
    print(f"  Deviation: {abs(measured_dh - np.e):.4f}\n")

    # Example 3: Toroidal closure validation
    print("Example 3: Validating toroidal closure")

    # Perfect closure
    initial = np.array([1.0, 0.0, 0.0])
    final_closed = np.array([1.0, 0.0, 0.0])

    is_closed, error = monitor.validate_toroidal_closure(initial, final_closed)
    print(f"  Perfect closure:")
    print(f"    Is closed: {is_closed}")
    print(f"    Error: {error:.8f}")

    # Imperfect closure
    final_open = np.array([0.9, 0.1, 0.05])
    is_closed, error = monitor.validate_toroidal_closure(initial, final_open)
    print(f"  Imperfect closure:")
    print(f"    Is closed: {is_closed}")
    print(f"    Error: {error:.8f}\n")

    # Example 4: Full compliance check
    print("Example 4: Complete compliance check")

    def test_operation(state):
        return state * 1.5

    result = monitor.check_compliance(
        operation=test_operation,
        state=state,
        trajectory=trajectory,
        initial_state=initial,
        final_state=final_closed
    )

    print(f"  Overall compliant: {result.overall_compliant}")
    print(f"  Chiral symmetric: {result.chiral_symmetric}")
    print(f"  Fractal consistent: {result.fractal_consistent}")
    print(f"  Toroidally closed: {result.toroidally_closed}")
    print(f"  Metrics: {result.metrics}")
    if result.violations:
        print(f"  Violations:")
        for violation in result.violations:
            print(f"    - {violation}")
    print()

    # Show statistics
    print("Statistics:")
    stats = monitor.get_statistics()
    print(f"  Total checks: {stats['total_checks']}")
    print(f"  Violations: {stats['violations']}")
    print(f"  Compliance rate: {stats['compliance_rate']:.2%}")


if __name__ == "__main__":
    main()
