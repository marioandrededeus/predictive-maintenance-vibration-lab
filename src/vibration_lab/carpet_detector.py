from __future__ import annotations

import numpy as np


def broadband_energy_ratio(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    fmin: float = 1000.0,
) -> float:
    """
    Ratio between high-frequency energy and total energy.
    """

    frequencies = np.asarray(frequencies, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)

    total_energy = np.sum(magnitude ** 2)

    if total_energy <= 0:
        return 0.0

    high_freq_energy = np.sum(
        (magnitude[frequencies >= fmin]) ** 2
    )

    return float(high_freq_energy / total_energy)


def carpet_score(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
) -> float:
    """
    Simple explainable carpet score based on
    broadband high-frequency energy.
    """

    ratio = broadband_energy_ratio(
        frequencies,
        magnitude,
    )

    return min(
        100.0,
        ratio * 100.0
    )