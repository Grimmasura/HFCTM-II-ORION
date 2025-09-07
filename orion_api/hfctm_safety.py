"""HFCTM-II Safety Core for ORION.

The original implementation relied on :mod:`torch` tensors for numerical
operations.  Downloading and installing PyTorch is expensive and unnecessary
for the purposes of the unit tests included with this kata.  To keep the
test environment lightweight the safety core has been rewritten to operate on
``numpy.ndarray`` instances instead.  The public interface of the class is
unchanged so the rest of the codebase and the tests interact with it exactly
as before, but the heavy dependency has been completely removed.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Hardware availability checks
try:
    import jax
    import jax.numpy as jnp
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    
try:
    import qiskit
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

@dataclass
class SafetyConfig:
    """Safety system configuration"""
    enable_quantum: bool = QUANTUM_AVAILABLE
    enable_tpu: bool = TPU_AVAILABLE
    overhead_budget: float = 2.0
    
    # Thresholds
    lyapunov_threshold: float = 0.0
    wavelet_threshold: float = 3.0
    mi_threshold: float = 1.5

class HFCTMII_SafetyCore:
    """Main HFCTM-II safety implementation"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.intervention_count = 0

    async def safety_check(self, model_state: np.ndarray) -> Dict:
        """Execute safety protocols."""
        metrics: Dict[str, float] = {}
        interventions: List[str] = []

        # 1. Lyapunov stability check
        lyapunov = self._compute_lyapunov(model_state)
        metrics["lyapunov"] = lyapunov

        # 2. Wavelet anomaly detection
        wavelet_energy = self._compute_wavelet_energy(model_state)
        metrics["wavelet_energy"] = wavelet_energy

        # 3. Egregore detection
        egregore_detected = self._detect_egregore(metrics)
        metrics["egregore_active"] = egregore_detected

        # 4. Apply interventions if needed
        if egregore_detected or lyapunov > self.config.lyapunov_threshold:
            interventions.extend(["chiral_inversion", "adaptive_damping"])
            self.intervention_count += 1

        return {
            "metrics": metrics,
            "interventions": interventions,
            "safe": not egregore_detected,
        }

    def _compute_lyapunov(self, state: np.ndarray) -> float:
        """Compute a simple Lyapunov exponent approximation."""
        perturbation = np.random.randn(*state.shape) * 1e-8
        perturbed = state + perturbation
        divergence = np.linalg.norm(perturbed - state)
        return float(np.log(divergence / 1e-8))

    def _compute_wavelet_energy(self, state: np.ndarray) -> float:
        """Compute wavelet energy for anomaly detection."""
        try:
            import pywt

            signal = state.flatten()
            coeffs = pywt.wavedec(signal, "db4", level=4)
            energy = sum(np.sum(c ** 2) for c in coeffs)
            return float(energy)
        except Exception:
            # Fallback to simple variance when PyWavelets is unavailable.
            return float(np.var(state))
    
    def _detect_egregore(self, metrics: Dict) -> bool:
        """Multi-metric egregore detection"""
        score = 0
        if metrics.get('lyapunov', 0) > self.config.lyapunov_threshold:
            score += 1
        if metrics.get('wavelet_energy', 0) > self.config.wavelet_threshold:
            score += 1
        return score >= 2

# Global safety core instance
safety_core: Optional[HFCTMII_SafetyCore] = None

def init_safety_core(config: SafetyConfig):
    """Initialize global safety core"""
    global safety_core
    safety_core = HFCTMII_SafetyCore(config)
    return safety_core
