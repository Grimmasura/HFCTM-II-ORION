"""ORION Quantum Stabilization Module.

Implements Lindblad master equation control with Lyapunov feedback,
optional counterdiabatic driving and Floquet analysis. SciPy is optional
and a simple Euler fallback is used when unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from prometheus_client import Gauge

try:  # Optional SciPy dependency
    import scipy.linalg as la  # noqa: F401
    from scipy.integrate import odeint
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_SCIPY = False

    def odeint(func, y0, t, args=()):  # type: ignore
        y = y0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            y = y + dt * func(y, t[i - 1], *args)
        return np.stack([y0, y])

from orion_config import QuantumConfig

logger = logging.getLogger(__name__)


# Prometheus metrics
LYAPUNOV_ENERGY = Gauge("orion_lyapunov_energy", "Lyapunov energy V(ρ)")
FIDELITY_TARGET = Gauge("orion_fidelity_target", "Fidelity to target state")
COHERENCE_TIME = Gauge("orion_coherence_time", "Estimated coherence time")
FLOQUET_EIGENVAL_MAX = Gauge(
    "orion_floquet_eigenval_max", "Maximum Floquet eigenvalue magnitude"
)


class QuantumState:
    """Density matrix representation with basic utilities."""

    def __init__(self, rho: torch.Tensor):
        self.rho = rho
        self.n_qubits = int(np.log2(rho.shape[0]))

    @classmethod
    def from_pure_state(cls, psi: torch.Tensor) -> "QuantumState":
        psi = psi / torch.norm(psi)
        rho = torch.outer(psi, psi.conj())
        return cls(rho)

    @classmethod
    def thermal_state(cls, hamiltonian: torch.Tensor, beta: float) -> "QuantumState":
        h_eig = torch.linalg.eigvals(hamiltonian).real
        z = torch.sum(torch.exp(-beta * h_eig))
        rho = torch.matrix_exp(-beta * hamiltonian) / z
        return cls(rho)

    def fidelity(self, other: "QuantumState") -> float:
        return torch.trace(self.rho @ other.rho).real.item()

    def purity(self) -> float:
        return torch.trace(self.rho @ self.rho).real.item()

    def entropy(self) -> float:
        vals = torch.linalg.eigvals(self.rho).real
        vals = vals[vals > 1e-12]
        return -(vals * torch.log(vals)).sum().item()


class LindbladMasterEquation:
    """Minimal Lindblad master equation evolution with controls."""

    def __init__(self, config: QuantumConfig):
        self.config = config
        self.dim = 2 ** config.n_qubits
        self.H0 = self._generate_system_hamiltonian()
        self.H_controls = self._generate_control_hamiltonians()
        self.lindblad_ops = self._generate_lindblad_operators()

    def _generate_system_hamiltonian(self) -> torch.Tensor:
        H = torch.randn(self.dim, self.dim, dtype=torch.complex64)
        return (H + H.conj().T) / 2

    def _generate_control_hamiltonians(self) -> List[torch.Tensor]:
        pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        controls: List[torch.Tensor] = []
        for i in range(self.config.n_qubits):
            for pauli in (pauli_x, pauli_y, pauli_z):
                H_control = torch.eye(1, dtype=torch.complex64)
                for j in range(self.config.n_qubits):
                    H_control = torch.kron(
                        H_control,
                        pauli if j == i else torch.eye(2, dtype=torch.complex64),
                    )
                controls.append(H_control)
        return controls

    def _generate_lindblad_operators(self) -> List[torch.Tensor]:
        ops: List[torch.Tensor] = []
        for _ in range(self.config.n_qubits):
            L = torch.zeros(self.dim, self.dim, dtype=torch.complex64)
            ops.append(L)
        return ops

    def lindblad_rhs(self, rho_vec: np.ndarray, t: float, controls: np.ndarray) -> np.ndarray:
        rho = torch.tensor(rho_vec.reshape(self.dim, self.dim), dtype=torch.complex64)
        H_total = self.H0
        for k, H_k in enumerate(self.H_controls):
            if k < len(controls):
                H_total = H_total + controls[k] * H_k
        coherent = -1j * (H_total @ rho - rho @ H_total)
        dissipation = torch.zeros_like(rho)
        for L in self.lindblad_ops:
            dissipation += L @ rho @ L.conj().T - 0.5 * (
                L.conj().T @ L @ rho + rho @ L.conj().T @ L
            )
        drho_dt = coherent + dissipation
        return drho_dt.flatten().numpy()


class LyapunovController:
    """Lyapunov-based feedback controller."""

    def __init__(self, config: QuantumConfig, target_state: QuantumState):
        self.config = config
        self.target_state = target_state
        self.lindblad = LindbladMasterEquation(config)

    def lyapunov_energy(self, current_state: QuantumState) -> float:
        diff = current_state.rho - self.target_state.rho
        return 0.5 * torch.trace(diff.conj().T @ diff).real.item()

    def compute_controls(self, current_state: QuantumState) -> np.ndarray:
        rho = current_state.rho
        rho_star = self.target_state.rho
        rho_diff = rho - rho_star
        controls: List[float] = []
        for k, H_k in enumerate(self.lindblad.H_controls):
            if k < len(self.config.lyapunov_kappa):
                comm = 1j * (H_k @ rho - rho @ H_k)
                control = -self.config.lyapunov_kappa[k] * torch.trace(
                    comm @ rho_diff
                ).real.item()
                controls.append(control)
            else:
                controls.append(0.0)
        return np.array(controls)

    def evolve_step(self, current_state: QuantumState, dt: float) -> QuantumState:
        controls = self.compute_controls(current_state)
        rho_vec = current_state.rho.flatten().numpy()
        rho_new_vec = odeint(
            self.lindblad.lindblad_rhs, rho_vec, [0.0, dt], args=(controls,)
        )[-1]
        rho_new = torch.tensor(
            rho_new_vec.reshape(self.lindblad.dim, self.lindblad.dim),
            dtype=torch.complex64,
        )
        rho_new = (rho_new + rho_new.conj().T) / 2
        if self.config.use_psd_projection:
            vals, vecs = torch.linalg.eigh(rho_new)
            vals = torch.clamp(vals.real, min=0.0)
            rho_new = vecs @ torch.diag(vals) @ vecs.conj().T
        rho_new = rho_new / torch.trace(rho_new)
        return QuantumState(rho_new)


class CounterdiabaticDriver:
    """Counterdiabatic driving for adiabatic processes."""

    def __init__(self, config: QuantumConfig):
        self.eta = config.counterdiabatic_eta

    def compute_cd_hamiltonian(
        self, H_lambda: torch.Tensor, dH_dt: torch.Tensor
    ) -> torch.Tensor:
        comm = dH_dt @ H_lambda - H_lambda @ dH_dt
        return self.eta * 1j * comm

    def adiabatic_evolution(
        self,
        H_func: Callable[[float], torch.Tensor],
        initial_state: QuantumState,
        T: float,
        n_steps: int = 100,
    ) -> List[QuantumState]:
        dt = T / n_steps
        traj = [initial_state]
        state = initial_state
        for i in range(n_steps):
            t = i * dt
            H_t = H_func(t)
            H_tp = H_func(t + dt / 2)
            dH_dt = (H_tp - H_t) / (dt / 2)
            H_cd = self.compute_cd_hamiltonian(H_t, dH_dt)
            U = torch.matrix_exp(-1j * (H_t + H_cd) * dt)
            state = QuantumState(U @ state.rho @ U.conj().T)
            traj.append(state)
        return traj


class FloquetStabilizer:
    """Floquet analysis for periodic systems."""

    def __init__(self, config: QuantumConfig):
        self.period = config.floquet_period

    def compute_floquet_operator(self, H_func: Callable[[float], torch.Tensor]) -> torch.Tensor:
        n_steps = 100
        dt = self.period / n_steps
        U = torch.eye(H_func(0).shape[0], dtype=torch.complex64)
        for i in range(n_steps):
            U_step = torch.matrix_exp(-1j * H_func(i * dt) * dt)
            U = U_step @ U
        return U

    def analyze_stability(self, U_F: torch.Tensor, config: QuantumConfig) -> Dict[str, float]:
        evals = torch.linalg.eigvals(U_F)
        max_mag = torch.max(torch.abs(evals)).item()
        FLOQUET_EIGENVAL_MAX.set(max_mag)
        return {
            "max_eigenval_magnitude": max_mag,
            "is_stable": max_mag <= config.max_eigenval_thresh,
        }


class QuantumStabilizer:
    """Main quantum stabilization interface for ORION."""

    def __init__(self, config: QuantumConfig):
        self.config = config
        self.target_state: QuantumState | None = None
        self.controller: LyapunovController | None = None
        self.cd_driver = CounterdiabaticDriver(config)
        self.floquet_analyzer = FloquetStabilizer(config)
        self.stability_history: List[float] = []
        self.fidelity_history: List[float] = []

    def set_target_state(self, target_state: QuantumState) -> None:
        self.target_state = target_state
        self.controller = LyapunovController(self.config, target_state)

    def evolve(self, rho: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        if self.target_state is None:
            dim = 2 ** self.config.n_qubits
            psi = torch.zeros(dim, dtype=torch.complex64)
            psi[0] = 1.0
            self.set_target_state(QuantumState.from_pure_state(psi))
        assert self.controller is not None
        state = QuantumState(rho)
        new_state = self.controller.evolve_step(state, dt)
        lyap = self.controller.lyapunov_energy(new_state)
        fid = new_state.fidelity(self.target_state) if self.target_state else 0.0
        LYAPUNOV_ENERGY.set(lyap)
        FIDELITY_TARGET.set(fid)
        self.stability_history.append(lyap)
        self.fidelity_history.append(fid)
        return new_state.rho

    def analyze_periodic_stability(
        self, H_func: Callable[[float], torch.Tensor]
    ) -> Dict[str, float]:
        U_F = self.floquet_analyzer.compute_floquet_operator(H_func)
        return self.floquet_analyzer.analyze_stability(U_F, self.config)


__all__ = ["QuantumState", "QuantumStabilizer"]

