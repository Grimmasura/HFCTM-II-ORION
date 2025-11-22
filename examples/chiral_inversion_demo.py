"""
Chiral Inversion Demo

Demonstrates time-reversal validation of computational operations
using the ChiralInversionController.

Reference: Section 6.2 of MIH-IIE specification
"""

import numpy as np
from mih_iie.layers.l5_governance.chiral_inversion import ChiralInversionController


def main():
    print("=== Chiral Inversion Demo ===\n")

    # Initialize controller with fidelity threshold
    controller = ChiralInversionController(fidelity_threshold=0.95)
    print(f"Initialized with fidelity threshold: {controller.fidelity_threshold}\n")

    # Example 1: Identity operation (should pass)
    print("Example 1: Validating identity operation")
    def identity_op(state):
        return state.copy()

    initial_state = np.array([1.0, 2.0, 3.0, 4.0])
    result = controller.validate_chiral_symmetry(identity_op, initial_state)

    print(f"  Is valid: {result.is_valid}")
    print(f"  Fidelity: {result.fidelity:.6f}")
    print(f"  Divergence points: {result.divergence_points}\n")

    # Example 2: Linear transformation (should pass)
    print("Example 2: Validating linear transformation")
    def linear_op(state):
        return state * 2.0

    result = controller.validate_chiral_symmetry(linear_op, initial_state)
    print(f"  Is valid: {result.is_valid}")
    print(f"  Fidelity: {result.fidelity:.6f}\n")

    # Example 3: Non-reversible operation (should fail)
    print("Example 3: Validating non-reversible operation (with noise)")
    def noisy_op(state):
        return state + np.random.randn(*state.shape) * 10

    result = controller.validate_chiral_symmetry(noisy_op, initial_state)
    print(f"  Is valid: {result.is_valid}")
    print(f"  Fidelity: {result.fidelity:.6f}")
    if result.divergence_points:
        print(f"  Divergence points: {result.divergence_points}")
    print()

    # Show statistics
    print("Statistics:")
    stats = controller.get_statistics()
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  Average fidelity: {stats['average_fidelity']:.6f}")
    print(f"  Success rate: {stats['success_rate']:.2%}")


if __name__ == "__main__":
    main()
