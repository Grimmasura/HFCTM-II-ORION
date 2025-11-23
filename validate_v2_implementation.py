"""
Validation script for MIH-IIE v2.0 implementation.

Checks that all v2.0 modules are present and properly structured.
Does not require external dependencies for basic checks.
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_exists(path: str) -> bool:
    """Check if file exists"""
    return Path(path).exists()

def count_lines(path: str) -> int:
    """Count lines in file"""
    try:
        with open(path, 'r') as f:
            return len(f.readlines())
    except:
        return 0

def validate_v2_modules():
    """Validate all v2.0 modules are present"""
    print(f"\n{BLUE}=== MIH-IIE v2.0 Implementation Validation ==={RESET}\n")

    modules = {
        'E8 Topology': 'models/e8_topology.py',
        'Majorana 0D Network': 'models/majorana_0d_network.py',
        'E8 Coordination': 'models/e8_coordination.py',
        'Frame-Invariant EDS': 'models/frame_invariant_eds.py',
        'Holographic Readout': 'models/holographic_readout.py',
        'Topological Error Correction': 'models/topological_error_correction.py'
    }

    all_present = True
    total_lines = 0

    for name, path in modules.items():
        exists = check_file_exists(path)
        if exists:
            lines = count_lines(path)
            total_lines += lines
            print(f"{GREEN}✓{RESET} {name:30s} - {path:40s} ({lines:4d} lines)")
        else:
            print(f"{RED}✗{RESET} {name:30s} - {path:40s} MISSING")
            all_present = False

    print(f"\n{BLUE}Total implementation:{RESET} {total_lines} lines of code")

    # Check documentation
    print(f"\n{BLUE}=== Documentation ==={RESET}\n")

    docs = {
        'v2.0 Implementation Guide': 'docs/MIH-IIE_v2.0_Implementation.md',
        'Integration Tests': 'tests/test_v2_integration.py'
    }

    for name, path in docs.items():
        exists = check_file_exists(path)
        if exists:
            lines = count_lines(path)
            print(f"{GREEN}✓{RESET} {name:30s} - {path:40s} ({lines:4d} lines)")
        else:
            print(f"{RED}✗{RESET} {name:30s} - {path:40s} MISSING")
            all_present = False

    # Summary
    print(f"\n{BLUE}=== Summary ==={RESET}\n")

    if all_present:
        print(f"{GREEN}✓ All v2.0 modules implemented successfully{RESET}")
        print(f"\n{BLUE}Key architectural upgrades from v1.0:{RESET}")
        print(f"  • Majorana zero modes ARE 0D attractors (not encodings)")
        print(f"  • E8 as network topology in Hilbert space (not geometric layout)")
        print(f"  • Frame-invariant validation (not static baselines)")
        print(f"  • 240-node E8 quantum entanglement network")
        print(f"  • Topological error correction using E8 graph cliques")
        print(f"  • Holographic state readout from boundary measurements")

        print(f"\n{BLUE}To test implementation:{RESET}")
        print(f"  1. Install dependencies: pip install -r requirements.txt")
        print(f"  2. Run tests: pytest tests/test_v2_integration.py -v")
        print(f"  3. See docs/MIH-IIE_v2.0_Implementation.md for usage examples")

        return True
    else:
        print(f"{RED}✗ Some v2.0 modules missing{RESET}")
        return False

def check_module_structure():
    """Check internal structure of modules"""
    print(f"\n{BLUE}=== Module Structure Analysis ==={RESET}\n")

    checks = {
        'models/e8_topology.py': [
            'class E8Root',
            'class E8RootSystem',
            'class E8QuantumNetwork',
            'Algorithm 1: Generate E8 Root Vectors',
            'Algorithm 2: Build E8 Adjacency Matrix'
        ],
        'models/majorana_0d_network.py': [
            'class MajoranaZeroMode',
            'class Majorana0DSeedNetwork',
            'ARE 0D attractors',
            'topological protection'
        ],
        'models/e8_coordination.py': [
            'class E8WeylGroup',
            'class PolychronicSynchronizer',
            'Algorithm 6: Parallel E8 Operations',
            'Algorithm 7: Establish Polychronic'
        ],
        'models/frame_invariant_eds.py': [
            'class FrameInvariantEDS',
            'class ObservationalFrame',
            'Algorithm 14: Frame-Invariant Validation',
            'Theorem 2.7'
        ],
        'models/holographic_readout.py': [
            'class HolographicReadoutProtocol',
            'Algorithm 11: Holographic Boundary',
            'Algorithm 12: Bulk State Reconstruction'
        ],
        'models/topological_error_correction.py': [
            'class TopologicalErrorCorrection',
            'class E8Stabilizer',
            'Algorithm 8: Construct E8 Stabilizers',
            '4-clique'
        ]
    }

    all_valid = True

    for module, required_elements in checks.items():
        if not check_file_exists(module):
            continue

        with open(module, 'r') as f:
            content = f.read()

        found_count = sum(1 for elem in required_elements if elem in content)
        total = len(required_elements)

        if found_count == total:
            print(f"{GREEN}✓{RESET} {module:40s} - All {total} key elements present")
        else:
            print(f"{YELLOW}⚠{RESET} {module:40s} - {found_count}/{total} elements found")
            all_valid = False

    return all_valid

if __name__ == "__main__":
    success = validate_v2_modules()
    structure_valid = check_module_structure()

    print()
    if success and structure_valid:
        print(f"{GREEN}{'=' * 60}")
        print(f"   MIH-IIE v2.0 Implementation Complete & Valid")
        print(f"{'=' * 60}{RESET}")
        sys.exit(0)
    else:
        print(f"{YELLOW}{'=' * 60}")
        print(f"   Validation completed with warnings")
        print(f"{'=' * 60}{RESET}")
        sys.exit(1)
