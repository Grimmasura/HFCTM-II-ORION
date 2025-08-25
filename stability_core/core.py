from __future__ import annotations

from .config import StabilityConfig
from .lambda_monitor import LambdaMonitor
from .damping import AdaptiveDamping
from .chiral import ChiralInversion
from .wavelet import WaveletScanner
from .e8_anchor import E8Anchor


class StabilityCore:
    """Coordinate stability heads in a per-step loop."""

    def __init__(self, config: StabilityConfig) -> None:
        self.config = config
        self.lambda_monitor = LambdaMonitor(config.lambda_monitor)
        self.damping = AdaptiveDamping(config.damping)
        self.chiral = ChiralInversion(config.chiral)
        self.wavelet = WaveletScanner(config.wavelet)
        self.e8_anchor = E8Anchor(config.e8_anchor)

    def step(self, state: float, latents: float) -> tuple[float, dict[str, float | bool]]:
        """Process one step and return ``(state, metrics)``."""

        lam = self.lambda_monitor.compute(state, latents)
        damped_state, factor = self.damping.apply(state, lam)
        chiral_state = self.chiral.apply(damped_state)
        anomaly = self.wavelet.scan(chiral_state)
        final_state = self.e8_anchor.project(chiral_state)
        metrics = {"lambda": lam, "damping_factor": factor, "anomaly": anomaly}
        return final_state, metrics
