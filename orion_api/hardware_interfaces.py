import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

try:  # Majorana1 quantum libraries
    from azure.quantum import Workspace
    from azure.quantum.cirq import AzureQuantumService
    import cirq
    MAJORANA1_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    MAJORANA1_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Majorana1 libraries not available - falling back to classical"
    )

try:  # Ironwood TPU libraries
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import torch_xla.core.xla_model as xm
    IRONWOOD_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    IRONWOOD_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Ironwood TPU libraries not available - using CPU/GPU fallback"
    )
    import numpy as jnp  # type: ignore

    def jit(fn):  # type: ignore
        return fn

    def vmap(fn):  # type: ignore
        return lambda *a, **k: fn(*a, **k)

    class xm:  # type: ignore
        @staticmethod
        def xla_device():
            return None


@dataclass
class HardwareConfig:
    """Configuration for hardware acceleration."""

    use_majorana1: bool = MAJORANA1_AVAILABLE
    use_ironwood: bool = IRONWOOD_AVAILABLE
    quantum_shots: int = 1000


class Majorana1Interface:
    """Minimal interface for the Majorana1 quantum processor."""

    def __init__(self, config: HardwareConfig):
        self.config = config
        self.quantum_service = None
        if MAJORANA1_AVAILABLE and config.use_majorana1:
            self._initialize()

    def _initialize(self) -> None:
        try:  # pragma: no cover - network call
            workspace = Workspace(
                subscription_id="",
                resource_group="",
                name="",
                location="",
            )
            self.quantum_service = AzureQuantumService(workspace)
        except Exception as exc:  # pragma: no cover - optional path
            logging.getLogger(__name__).error(
                "Failed to initialize Majorana1: %s", exc
            )
            self.config.use_majorana1 = False

    async def amplitude_estimation_mi(self, latent_slice: np.ndarray) -> float:
        if not self.config.use_majorana1:
            return self._classical_mi_fallback(latent_slice)
        try:  # pragma: no cover - network call
            n_qubits = min(16, int(np.ceil(np.log2(len(latent_slice)))))
            circuit = cirq.Circuit()
            qubits = cirq.LineQubit.range(n_qubits)
            for i, qubit in enumerate(qubits):
                if i < len(latent_slice):
                    angle = np.arcsin(np.sqrt(abs(latent_slice[i])))
                    circuit.append(cirq.ry(2 * angle)(qubit))
            circuit.append(cirq.measure(*qubits, key="result"))
            result = await self.quantum_service.run_async(
                circuit, repetitions=self.config.quantum_shots
            )
            measurements = result.measurements["result"]
            bit_counts = np.sum(measurements, axis=1)
            correlation = np.corrcoef(
                bit_counts, latent_slice[: len(bit_counts)]
            )[0, 1]
            return float(abs(correlation))
        except Exception as exc:  # pragma: no cover - network call
            logging.getLogger(__name__).warning(
                "Quantum MI estimation failed: %s", exc
            )
            return self._classical_mi_fallback(latent_slice)

    def _classical_mi_fallback(self, latent_slice: np.ndarray) -> float:
        from sklearn.feature_selection import mutual_info_regression

        x = latent_slice[:-1].reshape(-1, 1)
        y = latent_slice[1:]
        try:
            return float(mutual_info_regression(x, y)[0])
        except Exception:
            return 0.0


class IronwoodTPUInterface:
    """Minimal interface for Ironwood TPUs."""

    def __init__(self, config: HardwareConfig):
        self.config = config
        if IRONWOOD_AVAILABLE and config.use_ironwood:
            self.device = xm.xla_device()
        else:
            self.device = None

    @jit
    def wavelet_transform_batch(self, latent_batch: jnp.ndarray) -> jnp.ndarray:
        def _single(signal):
            return jnp.fft.fft(signal)

        return vmap(_single)(latent_batch)

    @jit
    def lyapunov_batch_compute(
        self, states: jnp.ndarray, perturbations: jnp.ndarray
    ) -> jnp.ndarray:
        def _single(state, perturbation):
            perturbed = state + perturbation * 1e-8
            divergence = jnp.linalg.norm(perturbed - state)
            return jnp.log(divergence / 1e-8)

        return vmap(_single)(states, perturbations)

    @jit
    def e8_projection_batch(self, control_vectors: jnp.ndarray) -> jnp.ndarray:
        def _single(vector):
            if vector.shape[0] > 8:
                vector = vector[:8]
            elif vector.shape[0] < 8:
                vector = jnp.pad(vector, (0, 8 - vector.shape[0]))
            return jnp.round(vector * 2) / 2

        return vmap(_single)(control_vectors)
