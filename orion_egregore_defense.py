"""Egregore defense metrics with optional sklearn/scipy/psutil fallbacks."""

from __future__ import annotations

import numpy as np

try:  # Optional heavy dependencies
    from sklearn.decomposition import PCA
    from sklearn.metrics import mutual_info_score
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_SKLEARN = False

    def mutual_info_score(x, y):  # type: ignore
        hist_2d, _, _ = np.histogram2d(x, y, bins=20)
        pxy = hist_2d / np.sum(hist_2d)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        nz = pxy > 0
        return np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz]))

    class PCA:  # type: ignore
        def __init__(self, n_components: int = 2):
            self.n_components = n_components

        def fit_transform(self, x: np.ndarray) -> np.ndarray:
            u, s, vh = np.linalg.svd(x, full_matrices=False)
            return (u[:, : self.n_components] * s[: self.n_components])

try:
    import scipy.linalg as la
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    la = np.linalg

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    _HAVE_PSUTIL = False
    psutil = None


class EgregoreDefense:
    """Compute simple metrics for egregore defense analysis."""

    def __init__(self) -> None:
        self.mem_percent = psutil.virtual_memory().percent if _HAVE_PSUTIL else 0.0

    def analyze(self, data: np.ndarray) -> dict[str, float]:
        reduced = PCA(n_components=2).fit_transform(data)
        mi = mutual_info_score(reduced[:, 0], reduced[:, 1])
        norm = float(la.norm(reduced))
        return {"mutual_info": float(mi), "norm": norm, "mem_percent": self.mem_percent}


__all__ = ["EgregoreDefense"]
