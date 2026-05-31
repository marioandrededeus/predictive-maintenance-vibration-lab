from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class WelchConfig:
    """
    Configuration parameters for Welch PSD estimation.
    """

    nperseg: int = 2048
    noverlap: int = 1024
    window: str = "hann"
    scaling: str = "density"


def fft_magnitude_single_sided(
    x: np.ndarray,
    fs_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided FFT magnitude spectrum.
    """

    x = np.asarray(x, dtype=float)
    n = x.size

    if n < 2:
        raise ValueError("Signal length must be >= 2.")

    if fs_hz <= 0:
        raise ValueError("Sampling frequency must be positive.")

    fft_values = np.fft.rfft(x)
    magnitude = np.abs(fft_values) / n

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / float(fs_hz)
    )

    return frequencies, magnitude


def welch_psd(
    x: np.ndarray,
    fs_hz: float,
    cfg: WelchConfig = WelchConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch power spectral density for a 1D vibration signal.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be 1D.")

    if fs_hz <= 0:
        raise ValueError("Sampling frequency must be positive.")

    if cfg.noverlap >= cfg.nperseg:
        raise ValueError("noverlap must be smaller than nperseg.")

    frequencies, psd = welch(
        x,
        fs=fs_hz,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        window=cfg.window,
        scaling=cfg.scaling,
    )

    return frequencies, psd


def to_db(
    power: np.ndarray,
    eps: float = 1e-20
) -> np.ndarray:
    """
    Convert linear power values to dB.
    """

    power = np.asarray(power, dtype=float)

    return 10.0 * np.log10(power + eps)


def assert_strictly_increasing(
    time: np.ndarray
) -> None:
    """
    Validate that the time axis is strictly increasing.
    """

    time = np.asarray(time, dtype=float)

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis is not strictly increasing.")


def infer_sampling_rate(
    time: np.ndarray
) -> float:
    """
    Infer sampling frequency from the median time step.
    """

    time = np.asarray(time, dtype=float)
    dt = np.diff(time)

    if len(dt) == 0:
        raise ValueError("Time array is too short to infer sampling rate.")

    dt_median = float(np.median(dt))

    if dt_median <= 0:
        raise ValueError("Invalid time step detected.")

    return 1.0 / dt_median


def assert_uniform_sampling(
    time: np.ndarray,
    tolerance: float = 1e-4
) -> None:
    """
    Validate approximate uniform sampling.
    """

    time = np.asarray(time, dtype=float)
    dt = np.diff(time)

    if len(dt) == 0:
        raise ValueError("Time array is too short to validate sampling.")

    dt_median = np.median(dt)

    if dt_median <= 0:
        raise ValueError("Invalid time step detected.")

    relative_variation = np.abs(dt - dt_median) / dt_median

    if np.max(relative_variation) > tolerance:
        raise ValueError(
            f"Sampling is not uniform within tolerance {tolerance}. "
            f"Max relative deviation: {np.max(relative_variation)}"
        )