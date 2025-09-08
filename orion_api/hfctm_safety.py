"""HFCTM-II Safety Core for ORION"""

import asyncio
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - import error handling
    TORCH_AVAILABLE = False

    class _TorchStub:
        class Tensor:  # pragma: no cover - type stub
            pass

        def __getattr__(self, name):  # pragma: no cover - defensive
            raise RuntimeError("PyTorch is not installed")

    torch = _TorchStub()  # type: ignore

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
        
    async def safety_check(self, model_state: torch.Tensor) -> Dict:
        """Execute safety protocols"""
        metrics = {}
        interventions: List[str] = []
        
        # 1. Lyapunov stability check
        lyapunov = self._compute_lyapunov(model_state)
        metrics['lyapunov'] = lyapunov
        
        # 2. Wavelet anomaly detection  
        wavelet_energy = self._compute_wavelet_energy(model_state)
        metrics['wavelet_energy'] = wavelet_energy
        
        # 3. Egregore detection
        egregore_detected = self._detect_egregore(metrics)
        metrics['egregore_active'] = egregore_detected
        
        # 4. Apply interventions if needed
        if egregore_detected or lyapunov > self.config.lyapunov_threshold:
            interventions.extend(['chiral_inversion', 'adaptive_damping'])
            self.intervention_count += 1
            
        return {
            'metrics': metrics,
            'interventions': interventions,
            'safe': not egregore_detected
        }
    
    def _compute_lyapunov(self, state: torch.Tensor) -> float:
        """Compute Lyapunov exponent approximation"""
        with torch.no_grad():
            perturbation = torch.randn_like(state) * 1e-8
            perturbed = state + perturbation
            divergence = torch.norm(perturbed - state)
            return float(torch.log(divergence / 1e-8))
    
    def _compute_wavelet_energy(self, state: torch.Tensor) -> float:
        """Compute wavelet energy for anomaly detection"""
        try:
            import pywt
            signal = state.flatten().cpu().numpy()
            coeffs = pywt.wavedec(signal, 'db4', level=4)
            energy = sum(np.sum(c**2) for c in coeffs)
            return float(energy)
        except ImportError:
            # Fallback to simple variance
            return float(torch.var(state))
    
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
