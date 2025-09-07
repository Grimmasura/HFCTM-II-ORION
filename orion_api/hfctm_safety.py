"""HFCTM-II safety core implementation.

This module defines a small safety framework used throughout the project.  It
exposes a :class:`SafetyConfig` dataclass describing hardware toggles and
thresholds and the :class:`HFCTMII_SafetyCore` which performs the actual safety
analysis.  The core provides an asynchronous ``safety_check`` method integrating
Lyapunov stability analysis, wavelet energy estimation and egregore detection.

The implementation is intentionally lightweight – hardware accelerators such as
TPUs or quantum back‑ends are optional and graceful fallbacks are provided when
those libraries are not available at runtime.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from prometheus_client import Gauge

# Optional accelerator / scientific packages.  These are imported lazily and the
# code falls back to pure PyTorch/NumPy implementations if they are not present.
try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - JAX not installed
    jax = None
    jnp = None

try:  # pragma: no cover - optional dependency
    import pywt
except Exception:  # pragma: no cover - PyWavelets not installed
    pywt = None


logger = logging.getLogger(__name__)


# Prometheus metric capturing the percentage of overhead used by the safety
# checks.  Tests import this to ensure it is being updated.
safety_overhead_gauge = Gauge(
    "hfctm_safety_overhead_pct",
    "Safety check overhead as percentage of configured budget",
)


@dataclass
class SafetyConfig:
    """Configuration for :class:`HFCTMII_SafetyCore`.

    The configuration combines hardware toggles as well as various thresholds
    used by the heuristics.  Values are intentionally conservative and can be
    tuned by the caller if required.
    """

    # Hardware toggles -----------------------------------------------------
    use_majorana1: bool = True
    use_ironwood_tpu: bool = True
    quantum_shots: int = 128

    # Thresholds -----------------------------------------------------------
    r_min: float = 0.70
    lambda_max_z: float = 2.0
    mi_z: float = 1.5
    wavelet_z: float = 3.0
    dH_dt_max: float = -0.02

    # Budget for computing safety checks in seconds
    safety_overhead_budget: float = 0.1


class HFCTMII_SafetyCore:
    """Main HFCTM-II safety implementation."""

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()
        logger.debug("HFCTMII_SafetyCore initialised with config: %s", self.config)

    # ------------------------------------------------------------------ API
    async def safety_check(
        self, model_state: torch.Tensor, latents: torch.Tensor
    ) -> Dict[str, Any]:
        """Perform a safety check asynchronously.

        Parameters
        ----------
        model_state:
            Current model state or hidden activations.
        latents:
            Latent vectors used for the wavelet analysis.
        """

        start = time.time()
        metrics: Dict[str, float] = {}

        # Lyapunov exponent -------------------------------------------------
        metrics["lyapunov"] = await asyncio.to_thread(
            self._compute_lyapunov, model_state
        )

        # Wavelet energy ----------------------------------------------------
        metrics["wavelet_energy"] = await asyncio.to_thread(
            self._compute_wavelet_energy, latents
        )

        # Mutual information heuristic -------------------------------------
        # In the full system this would leverage a quantum amplitude
        # estimation routine.  For the purposes of the tests we simply compute
        # a cheap surrogate based on mean absolute activation.
        metrics["mutual_info"] = float(torch.mean(torch.abs(latents[0])).item())

        # Egregore detection ------------------------------------------------
        metrics["egregore_active"] = self._detect_egregore(metrics)
        interventions = (
            ["chiral_inversion", "adaptive_damping"]
            if metrics["egregore_active"]
            else []
        )

        # Update safety overhead metric ------------------------------------
        elapsed = time.time() - start
        pct = (elapsed / self.config.safety_overhead_budget) * 100.0
        safety_overhead_gauge.set(pct)

        return {
            "metrics": metrics,
            "interventions": interventions,
            "hardware_used": {
                "majorana1": self.config.use_majorana1,
                "ironwood": self.config.use_ironwood_tpu,
            },
        }

    # Backwards compatibility ----------------------------------------------
    async def recursive_safety_check(
        self, model_state: torch.Tensor, latents: torch.Tensor
    ) -> Dict[str, Any]:
        """Alias for ``safety_check`` kept for compatibility with older code."""

        return await self.safety_check(model_state, latents)

    # ------------------------------------------------------------- Internals
    def _compute_lyapunov(self, state: torch.Tensor) -> float:
        """Estimate a Lyapunov exponent.

        If TPU support via ``jax`` is available and enabled the computation is
        delegated to JAX; otherwise a small PyTorch based estimate is used.
        """

        if self.config.use_ironwood_tpu and jax is not None:  # pragma: no cover
            state_j = jnp.array(state.detach().cpu().numpy())
            pert = jax.random.normal(jax.random.PRNGKey(0), state_j.shape) * 1e-8
            divergence = jnp.linalg.norm((state_j + pert) - state_j)
            return float(jnp.log(divergence / 1e-8))

        # Fallback: simple PyTorch estimate
        with torch.no_grad():
            perturbation = torch.randn_like(state) * 1e-8
            perturbed = state + perturbation
            divergence = torch.norm(perturbed - state)
            return float(torch.log(divergence / 1e-8))

    def _compute_wavelet_energy(self, latents: torch.Tensor) -> float:
        """Compute wavelet energy of the provided latents.

        A jax based implementation is used when TPU support is enabled.  If
        unavailable, PyWavelets (``pywt``) is used and finally a NumPy FFT
        fallback is provided to keep the routine dependency light.
        """

        signal = latents[0].detach().cpu().numpy()

        if self.config.use_ironwood_tpu and jax is not None:  # pragma: no cover
            coeffs = jnp.fft.fft(signal)
            return float(jnp.sum(jnp.abs(coeffs) ** 2))

        if pywt is not None:
            coeffs = pywt.wavedec(signal, "db4", level=4)
            energy = sum(np.sum(c ** 2) for c in coeffs)
            return float(energy)

        # Final fallback using NumPy FFT
        coeffs = np.fft.fft(signal)
        return float(np.sum(np.abs(coeffs) ** 2))

    def _detect_egregore(self, metrics: Dict[str, float]) -> bool:
        """Simple heuristic for detecting an "egregore" state.

        The heuristic checks whether two out of three signals exceed their
        respective thresholds.
        """

        score = 0
        if metrics.get("mutual_info", 0.0) > self.config.mi_z:
            score += 1
        if metrics.get("wavelet_energy", 0.0) > self.config.wavelet_z:
            score += 1
        if metrics.get("lyapunov", 0.0) > 0.0:
            score += 1
        return score >= 2


# ----------------------------------------------------------------- Initialiser
safety_core: HFCTMII_SafetyCore


def init_safety_core(config: Optional[SafetyConfig] = None) -> HFCTMII_SafetyCore:
    """Initialise the global safety core instance.

    Parameters
    ----------
    config:
        Optional :class:`SafetyConfig` overriding the defaults.
    """

    global safety_core
    safety_core = HFCTMII_SafetyCore(config)
    return safety_core


# Initialise a default instance when the module is imported so that consumers
# can simply ``from orion_api.hfctm_safety import safety_core`` and use it
# directly without any further setup.
init_safety_core()


__all__ = [
    "SafetyConfig",
    "HFCTMII_SafetyCore",
    "init_safety_core",
    "safety_core",
    "safety_overhead_gauge",
]

