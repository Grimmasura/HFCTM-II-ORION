from __future__ import annotations

from .config import WaveletConfig


class WaveletScanner:
    """Minimal wavelet anomaly detector.

    Real implementations would use a discrete wavelet transform to inspect the
    signal. For simplicity, we treat amplitudes above ``threshold`` as anomalies.
    """

    def __init__(self, config: WaveletConfig) -> None:
        self.config = config

    def scan(self, state: float) -> bool:
        """Return ``True`` if ``state`` exceeds the anomaly threshold."""

        return abs(state) > self.config.threshold
