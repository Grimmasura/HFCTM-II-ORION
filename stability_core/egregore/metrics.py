import numpy as np


def phase_order(phases: np.ndarray) -> float:
    """Compute Kuramoto phase order parameter r(t).

    Parameters
    ----------
    phases: np.ndarray
        Array of oscillator phases in radians.
    Returns
    -------
    float
        Order parameter magnitude in [0,1].
    """
    if phases.size == 0:
        return 0.0
    order = np.abs(np.sum(np.exp(1j * phases)) / phases.size)
    return float(order)


def spectral_capacity(matrix: np.ndarray) -> float:
    """Largest eigenvalue (spectral capacity) of a matrix."""
    if matrix.size == 0:
        return 0.0
    vals = np.linalg.eigvals(matrix)
    return float(np.max(np.real(vals)))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """Estimate mutual information between two signals."""
    if x.size == 0 or y.size == 0:
        return 0.0
    c_xy = np.histogram2d(x, y, bins)[0]
    p_xy = c_xy / np.sum(c_xy)
    p_x = np.sum(p_xy, axis=1)[:, None]
    p_y = np.sum(p_xy, axis=0)[None, :]
    nz = p_xy > 0
    mi = np.sum(p_xy[nz] * np.log(p_xy[nz] / (p_x * p_y)[nz]))
    return float(mi)


def wavelet_burst_energy(signal: np.ndarray, width: int = 5) -> float:
    """Compute energy of a Morlet wavelet burst in the signal."""
    if signal.size == 0:
        return 0.0
    t = np.arange(-5 * width, 5 * width)
    morlet = np.exp(-t**2 / (2 * width**2)) * np.cos(5 * t)
    conv = np.convolve(signal, morlet, mode="same")
    return float(np.sum(conv ** 2))


def entropy_slope(signal: np.ndarray, window: int = 50) -> float:
    """Slope of Shannon entropy over sliding windows."""
    n = signal.size
    if n < window * 2:
        return 0.0
    entropies = []
    for i in range(0, n - window + 1):
        segment = signal[i : i + window]
        hist, _ = np.histogram(segment, bins="auto", density=True)
        p = hist[hist > 0]
        entropies.append(-np.sum(p * np.log(p)))
    x = np.arange(len(entropies))
    slope, _ = np.polyfit(x, entropies, 1)
    return float(slope)
