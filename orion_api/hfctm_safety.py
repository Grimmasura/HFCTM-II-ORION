"""HFCTM-II safety core implementation."""

from __future__ import annotations

import asyncio
import time
import numpy as np
import torch
from cryptography.hazmat.primitives import hashes
from prometheus_client import Gauge

from .config import settings
from .hardware_interfaces import (
    HardwareConfig,
    IronwoodTPUInterface,
    Majorana1Interface,
)


safety_overhead_gauge = Gauge(
    "hfctm_safety_overhead_pct",
    "Safety check overhead as percentage of configured budget",
)


class HFCTMII_SafetyCore:
    """Main HFCTM-II safety implementation."""

    def __init__(self) -> None:
        config = HardwareConfig(
            use_majorana1=settings.enable_majorana1,
            use_ironwood=settings.enable_ironwood_tpu,
            quantum_shots=settings.quantum_shots,
        )
        self.config = config
        self.majorana1 = Majorana1Interface(config)
        self.ironwood = IronwoodTPUInterface(config)
        self.thresholds = {
            "r_min": 0.70,
            "lambda_max_z": 2.0,
            "mi_z": 1.5,
            "wavelet_z": 3.0,
            "dH_dt_max": -0.02,
        }

    async def recursive_safety_check(
        self, model_state: torch.Tensor, latents: torch.Tensor
    ) -> dict:
        """Perform a hardware-accelerated safety check."""

        start = time.time()
        metrics: dict[str, float] = {}

        if self.config.use_ironwood:
            import jax
            import jax.numpy as jnp

            latents_jax = jnp.array(latents.cpu().numpy())
            state_jax = jnp.array(model_state.cpu().numpy())
            perturbations = jax.random.normal(
                jax.random.PRNGKey(0), state_jax.shape
            )
            lyapunov = self.ironwood.lyapunov_batch_compute(
                state_jax, perturbations
            )
            metrics["lyapunov"] = float(jnp.mean(lyapunov))
            wavelet_coeffs = self.ironwood.wavelet_transform_batch(latents_jax)
            wavelet_energy = jnp.sum(jnp.abs(wavelet_coeffs) ** 2, axis=-1)
            metrics["wavelet_energy"] = float(jnp.mean(wavelet_energy))
            control_8d = state_jax[:, :8] if state_jax.shape[1] >= 8 else state_jax
            projected = self.ironwood.e8_projection_batch(control_8d)
            e8_residual = jnp.linalg.norm(control_8d - projected)
            metrics["e8_residual"] = float(e8_residual)
        else:
            metrics["lyapunov"] = self._classical_lyapunov(model_state)
            metrics["wavelet_energy"] = self._classical_wavelet(latents)

        latent_slice = latents[0, :64].cpu().numpy()
        metrics["mutual_info"] = await self.majorana1.amplitude_estimation_mi(
            latent_slice
        )
        egregore_active = self._detect_egregore(metrics)
        metrics["egregore_active"] = egregore_active
        interventions = [
            "chiral_inversion",
            "adaptive_damping",
        ] if egregore_active else []
        state_hash = self._compute_state_hash(metrics, interventions)

        elapsed = time.time() - start
        pct = (elapsed / settings.safety_overhead_budget) * 100.0
        safety_overhead_gauge.set(pct)

        return {
            "metrics": metrics,
            "interventions": interventions,
            "hash": state_hash,
            "hardware_used": {
                "majorana1": self.config.use_majorana1,
                "ironwood": self.config.use_ironwood,
            },
        }

    def _detect_egregore(self, metrics: dict) -> bool:
        score = 0
        if metrics.get("mutual_info", 0) > self.thresholds["mi_z"]:
            score += 1
        if metrics.get("wavelet_energy", 0) > self.thresholds["wavelet_z"]:
            score += 1
        if metrics.get("lyapunov", 0) > 0:
            score += 1
        return score >= 2

    def _classical_lyapunov(self, state: torch.Tensor) -> float:
        with torch.no_grad():
            perturbation = torch.randn_like(state) * 1e-8
            perturbed = state + perturbation
            divergence = torch.norm(perturbed - state)
            return float(torch.log(divergence / 1e-8))

    def _classical_wavelet(self, latents: torch.Tensor) -> float:
        import pywt

        signal = latents[0, :].cpu().numpy()
        coeffs = pywt.wavedec(signal, "db4", level=4)
        energy = sum(np.sum(c ** 2) for c in coeffs)
        return float(energy)

    def _compute_state_hash(self, metrics: dict, interventions: list[str]) -> str:
        digest = hashes.Hash(hashes.SHA256())
        digest.update(str(metrics).encode())
        digest.update(str(interventions).encode())
        return digest.finalize().hex()


safety_core = HFCTMII_SafetyCore()
