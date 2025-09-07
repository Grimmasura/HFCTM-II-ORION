def test_core_imports():
    import fastapi
    import cirq
    import networkx
    assert fastapi.__version__


def test_version_compatibility():
    import networkx as nx
    assert nx.__version__.startswith('3.1')


def test_quantum_fallback():
    from orion_api.quantum import QuantumInterface
    qi = QuantumInterface()
    assert qi is not None
