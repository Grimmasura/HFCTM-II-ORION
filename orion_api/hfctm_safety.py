"""Minimal HFCTM-II safety core used for unit testing.

The real project features complex safety mechanisms that analyse model
state tensors using advanced metrics.  Here we provide a lightweight
NumPy based implementation that captures the essence of the algorithms
while remaining fully deterministic and dependency free.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SafetyConfig:
    """Configuration for :class:`HFCTMII_SafetyCore`."""

    enable_quantum: bool = False
    enable_tpu: bool = False
    overhead_budget: float = 2.0
    lyapunov_threshold: float = 0.0
    wavelet_threshold: float = 3.0


class HFCTMII_SafetyCore:
    """Simplified safety core implementing a few basic metrics."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self.intervention_count = 0

    async def safety_check(self, model_state: np.ndarray) -> Dict[str, object]:
        """Evaluate safety metrics for ``model_state``.

        The function calculates a crude Lyapunov estimate and a wavelet energy
        proxy.  When both metrics exceed their thresholds the function signals
        that an ``egregore`` has been detected.
        """

        metrics: Dict[str, float] = {}
        interventions: List[str] = []

        lyapunov = self._compute_lyapunov(model_state)
        metrics["lyapunov"] = lyapunov

        wavelet_energy = self._compute_wavelet_energy(model_state)
        metrics["wavelet_energy"] = wavelet_energy

        egregore = self._detect_egregore(metrics)
        metrics["egregore_active"] = egregore

        if egregore or lyapunov > self.config.lyapunov_threshold:
            interventions.extend(["chiral_inversion", "adaptive_damping"])
            self.intervention_count += 1

        return {"metrics": metrics, "interventions": interventions, "safe": not egregore}

    # Internal helpers -----------------------------------------------------
    def _compute_lyapunov(self, state: np.ndarray) -> float:
        perturbation = np.random.randn(*state.shape) * 1e-8
        perturbed = state + perturbation
        divergence = np.linalg.norm(perturbed - state)
        return float(np.log(divergence / 1e-8))

    def _compute_wavelet_energy(self, state: np.ndarray) -> float:
        try:  # pragma: no cover - pywt optional
            import pywt

            signal = state.flatten()
            coeffs = pywt.wavedec(signal, "db4", level=2)
            return float(sum(float(np.sum(c ** 2)) for c in coeffs))
        except Exception:
            return float(np.var(state))

    def _detect_egregore(self, metrics: Dict[str, float]) -> bool:
        score = 0
        if metrics.get("lyapunov", 0.0) > self.config.lyapunov_threshold:
            score += 1
        if metrics.get("wavelet_energy", 0.0) > self.config.wavelet_threshold:
            score += 1
        return score >= 2


# Global safety core instance ----------------------------------------------

safety_core: Optional[HFCTMII_SafetyCore] = None


def init_safety_core(config: SafetyConfig) -> HFCTMII_SafetyCore:
    """Initialise the global :class:`HFCTMII_SafetyCore` instance."""

    global safety_core
    safety_core = HFCTMII_SafetyCore(config)
    return safety_core
