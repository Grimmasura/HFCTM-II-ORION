import numpy as np


def phase_order(phases: np.ndarray) -> float:
    """Kuramoto order parameter r(t) for array of phase angles."""
    phases = np.asarray(phases)
    if phases.size == 0:
        return 0.0
    return np.abs(np.exp(1j * phases).mean()).item()


def spectral_capacity(signal: np.ndarray) -> float:
    """Largest eigenvalue (lambda_max) of covariance matrix."""
    signal = np.asarray(signal)
    if signal.ndim == 1:
        cov = np.var(signal)
        return float(cov)
    cov = np.cov(signal)
    vals = np.linalg.eigvals(cov)
    return float(np.max(np.real(vals)))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """Estimate mutual information between two signals using histograms."""
    x = np.asarray(x)
    y = np.asarray(y)
    c_xy, _, _ = np.histogram2d(x, y, bins)
    p_xy = c_xy / np.sum(c_xy)
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    nzs = p_xy > 0
    mi = np.sum(p_xy[nzs] * np.log(p_xy[nzs] / (p_x @ p_y)[nzs]))
    return float(mi)


def _morlet(length: int, width: float = 5.0) -> np.ndarray:
    t = np.linspace(-2 * np.pi, 2 * np.pi, length)
    wavelet = np.exp(1j * t) * np.exp(-(t ** 2) / (2 * width ** 2))
    return wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))


def wavelet_burst_energy(signal: np.ndarray, width: float = 5.0) -> float:
    """Approximate burst energy using a Morlet wavelet convolution."""
    signal = np.asarray(signal)
    wavelet = _morlet(min(len(signal), 8 * int(width)), width).real
    conv = np.convolve(signal, wavelet, mode="same")
    energy = np.sum(conv ** 2)
    return float(energy)


def entropy_slope(signal: np.ndarray, window: int = 32, bins: int = 32) -> float:
    """Slope of Shannon entropy over moving windows."""
    signal = np.asarray(signal)
    if len(signal) < window:
        return 0.0
    entropies = []
    for i in range(len(signal) - window + 1):
        segment = signal[i : i + window]
        hist, _ = np.histogram(segment, bins=bins, density=True)
        nz = hist > 0
        ent = -np.sum(hist[nz] * np.log(hist[nz]))
        entropies.append(ent)
    x = np.arange(len(entropies))
    y = np.array(entropies)
    if y.size == 0:
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)
