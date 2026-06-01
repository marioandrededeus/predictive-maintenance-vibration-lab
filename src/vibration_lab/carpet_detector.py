from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarpetIndicators:
    high_frequency_energy_ratio: float
    spectral_flatness: float
    peak_to_floor_ratio: float
    carpet_score: float
    spectrum_status: str


def broadband_energy_ratio(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    fmin: float = 1000.0,
) -> float:
    frequencies = np.asarray(frequencies, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)

    total_energy = np.sum(magnitude ** 2)

    if total_energy <= 0:
        return 0.0

    high_freq_energy = np.sum(
        magnitude[frequencies >= fmin] ** 2
    )

    return float(high_freq_energy / total_energy)


def spectral_flatness(
    magnitude: np.ndarray,
    eps: float = 1e-12
) -> float:
    magnitude = np.asarray(magnitude, dtype=float)
    power = magnitude ** 2 + eps

    geometric_mean = np.exp(
        np.mean(np.log(power))
    )

    arithmetic_mean = np.mean(power)

    if arithmetic_mean <= 0:
        return 0.0

    return float(geometric_mean / arithmetic_mean)


def peak_to_floor_ratio(
    magnitude: np.ndarray,
    eps: float = 1e-12
) -> float:
    magnitude = np.asarray(magnitude, dtype=float)

    floor = np.median(magnitude) + eps
    peak = np.max(magnitude)

    return float(peak / floor)


def classify_spectrum(
    score: float
) -> str:
    if score >= 70:
        return "Strong carpet-like high-frequency broadband pattern"

    if score >= 40:
        return "Moderate carpet-like spectral behavior"

    return "No strong carpet-like pattern detected"


def compute_carpet_indicators(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    fmin: float = 1000.0,
) -> CarpetIndicators:
    hf_ratio = broadband_energy_ratio(
        frequencies,
        magnitude,
        fmin=fmin,
    )

    flatness = spectral_flatness(
        magnitude
    )

    pfr = peak_to_floor_ratio(
        magnitude
    )

    score = min(
        100.0,
        100.0 * (
            0.70 * hf_ratio
            + 0.30 * flatness
        )
    )

    return CarpetIndicators(
        high_frequency_energy_ratio=hf_ratio,
        spectral_flatness=flatness,
        peak_to_floor_ratio=pfr,
        carpet_score=score,
        spectrum_status=classify_spectrum(score),
    )