from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def robust_threshold(
    values: np.ndarray,
    k: float = 6.0
) -> float:
    """
    Robust threshold based on median absolute deviation (MAD).
    """

    values = np.asarray(values, dtype=float)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-12

    return median + k * mad


def extract_peaks_in_band(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    fmin: float,
    fmax: float,
    k: float = 6.0,
    min_distance_hz: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract spectral peaks within a frequency band using
    a robust MAD-based threshold.
    """

    frequencies = np.asarray(frequencies, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)

    selection = (frequencies >= fmin) & (frequencies <= fmax)

    freq_band = frequencies[selection]
    mag_band = magnitude[selection]

    if freq_band.size < 3:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            float("nan"),
        )

    threshold = robust_threshold(
        mag_band,
        k=k
    )

    df = float(np.median(np.diff(freq_band)))

    distance_bins = max(
        1,
        int(round(min_distance_hz / df))
    )

    peak_idx, _ = find_peaks(
        mag_band,
        height=threshold,
        distance=distance_bins
    )

    return (
        freq_band[peak_idx],
        mag_band[peak_idx],
        float(threshold),
    )