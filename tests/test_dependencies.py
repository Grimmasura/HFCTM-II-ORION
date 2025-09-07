"""
HFCTM-II Dependency Validation Tests
Ensures recursive stability and compatibility
"""

import pytest
from packaging import version


def test_core_imports():
    """Test that all core dependencies can be imported"""
    import fastapi  # noqa: F401
    import cirq  # noqa: F401
    import networkx  # noqa: F401
    import torch  # noqa: F401
    import numpy  # noqa: F401

    assert True


def test_networkx_version_compatibility():
    """Ensure NetworkX version is compatible with Cirq"""
    import networkx as nx

    assert version.parse(nx.__version__) == version.parse("3.1")


def test_cirq_networkx_integration():
    """Test that Cirq works with the installed NetworkX version"""
    import cirq

    # Test basic cirq functionality that uses networkx
    qubits = cirq.GridQubit.rect(2, 2)
    cirq.Circuit()

    # This would fail if networkx compatibility is broken
    assert len(qubits) == 4


def test_quantum_backend_initialization():
    """Test HFCTM quantum backend initialization"""
    try:
        from orion_api.quantum.backend_adapter import HFCTMQuantumInterface

        qi = HFCTMQuantumInterface()
        assert qi is not None
        assert qi.backend is not None
    except ImportError:
        pytest.skip("Quantum backend not available for testing")


def test_fallback_mechanisms():
    """Test that fallback mechanisms work properly"""
    try:
        from orion_api.quantum.backend_adapter import ClassicalFallbackBackend

        backend = ClassicalFallbackBackend()
        circuit = backend.create_circuit(4)
        assert circuit["qubits"] == 4
        assert circuit["backend"] == "classical_fallback"
    except ImportError:
        pytest.skip("Fallback backend not available for testing")


@pytest.mark.parametrize(
    "package,min_version",
    [
        ("fastapi", "0.115.0"),
        ("torch", "2.0.0"),
        ("numpy", "1.24.0"),
        ("networkx", "3.1"),
    ],
)
def test_minimum_versions(package, min_version):
    """Test that packages meet minimum version requirements"""
    module = __import__(package)
    installed_version = getattr(module, "__version__", "0.0.0")
    assert version.parse(installed_version) >= version.parse(min_version)
