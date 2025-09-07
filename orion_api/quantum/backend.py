"""Quantum backend abstraction with graceful fallback."""

try:
    import cirq
    QUANTUM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cirq = None  # type: ignore
    QUANTUM_AVAILABLE = False


class QuantumInterface:
    """Simple interface that abstracts quantum circuit creation.

    If ``cirq`` is not available the interface falls back to a
    deterministic classical representation so that higher level
    components can continue operating.
    """

    def __init__(self) -> None:
        self.use_classical_fallback = not QUANTUM_AVAILABLE

    def create_circuit(self, *args, **kwargs):
        if QUANTUM_AVAILABLE:
            return cirq.Circuit(*args, **kwargs)
        return self._classical_simulation(*args, **kwargs)

    def _classical_simulation(self, *args, **kwargs):
        """Basic classical fallback for environments without cirq."""
        return {"classical": True, "args": args, "kwargs": kwargs}
