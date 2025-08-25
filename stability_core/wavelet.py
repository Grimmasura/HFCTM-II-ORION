"""Wavelet-based anomaly scanning."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel


class WaveletConfig(BaseModel):
    """Configuration for :class:`WaveletScanner`."""

    width: int = 4
    threshold: float = 1.0


class WaveletScanner:
    """Detects simple anomalies using a moving average as a pseudo wavelet."""

    def __init__(self, config: WaveletConfig | None = None):
        self.config = config or WaveletConfig()

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        arr = np.asarray(state, dtype=float)
        w = self.config.width
        if arr.size < w:
            return arr, {"anomaly_indices": []}
        kernel = np.ones(w) / float(w)
        smooth = np.convolve(arr, kernel, mode="same")
        diff = np.abs(arr - smooth)
        indices = np.where(diff > self.config.threshold)[0].tolist()
        return arr, {"anomaly_indices": indices}
