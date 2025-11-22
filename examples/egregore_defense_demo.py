"""
Egregore Defense System Demo

Demonstrates semantic drift protection and corrupted pattern detection
using the EgregoreDefenseSystem.

Reference: Section 6.4 of MIH-IIE specification
"""

from mih_iie.layers.l5_governance.egregore_defense import EgregoreDefenseSystem


def main():
    print("=== Egregore Defense System Demo ===\n")

    # Initialize EDS
    eds = EgregoreDefenseSystem(
        torsion_threshold_sigma=3.0,
        similarity_threshold=0.80
    )
    print(f"Initialized EDS with {len(eds.corrupted_patterns)} known corrupted patterns\n")

    # Example 1: Clean semantic field
    print("Example 1: Checking clean semantic field")
    clean_field = {
        "truth": "correspondence to reality",
        "evidence": "empirical observation",
        "reasoning": "logical inference from premises"
    }

    result = eds.safety_check(clean_field)
    print(f"  Safe: {result['safe']}")
    print(f"  Should quarantine: {result['should_quarantine']}")
    print(f"  Torsion measure: {result['torsion']:.6f}")
    print(f"  Alerts: {result['alerts']}\n")

    # Example 2: Clean reasoning structure
    print("Example 2: Checking clean reasoning structure")
    clean_structure = {
        "type": "empirical_inference",
        "evidence_provided": True,
        "logical_flow": True,
        "falsifiable": True
    }

    result = eds.safety_check(clean_field, clean_structure)
    print(f"  Safe: {result['safe']}")
    print(f"  Should quarantine: {result['should_quarantine']}")
    print(f"  Alerts: {result['alerts']}\n")

    # Example 3: Circular reasoning pattern
    print("Example 3: Detecting circular reasoning")
    circular_structure = {
        "type": "circular",
        "dependency_loop": True,
        "premises_depend_on_conclusion": True
    }

    result = eds.safety_check(clean_field, circular_structure)
    print(f"  Safe: {result['safe']}")
    print(f"  Should quarantine: {result['should_quarantine']}")
    print(f"  Alerts:")
    for alert in result['alerts']:
        print(f"    - {alert}\n")

    # Example 4: Authority-based reasoning
    print("Example 4: Detecting authority appeal")
    authority_structure = {
        "type": "authority_based",
        "appeals_to_authority": True,
        "evidence_required": False,
        "institutional_validation": True
    }

    result = eds.safety_check(clean_field, authority_structure)
    print(f"  Safe: {result['safe']}")
    print(f"  Should quarantine: {result['should_quarantine']}")
    print(f"  Alerts:")
    for alert in result['alerts']:
        print(f"    - {alert}\n")

    # Show statistics
    print("Statistics:")
    stats = eds.get_statistics()
    print(f"  Total checks: {stats['total_checks']}")
    print(f"  Alerts triggered: {stats['alerts_triggered']}")
    print(f"  Quarantines issued: {stats['quarantines_issued']}")


if __name__ == "__main__":
    main()
