from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoosenessIndicators:
    low_frequency_energy_ratio: float
    harmonic_content_ratio: float
    subharmonic_content_ratio: float
    peak_dominance_ratio: float
    looseness_score: float
    spectrum_status: str


def _band_energy(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    selection = (frequencies >= fmin) & (frequencies <= fmax)

    return float(
        np.sum(magnitude[selection] ** 2)
    )


def low_frequency_energy_ratio(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    max_frequency: float = 200.0,
) -> float:
    total_energy = np.sum(magnitude ** 2)

    if total_energy <= 0:
        return 0.0

    low_frequency_energy = _band_energy(
        frequencies,
        magnitude,
        0.0,
        max_frequency,
    )

    return float(low_frequency_energy / total_energy)


def harmonic_content_ratio(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    rotational_frequency_hz: float,
    harmonics: tuple[int, ...] = (1, 2, 3),
    bandwidth_hz: float = 2.0,
) -> float:
    total_energy = np.sum(magnitude ** 2)

    if total_energy <= 0:
        return 0.0

    harmonic_energy = 0.0

    for harmonic in harmonics:
        center = harmonic * rotational_frequency_hz

        harmonic_energy += _band_energy(
            frequencies,
            magnitude,
            center - bandwidth_hz,
            center + bandwidth_hz,
        )

    return float(harmonic_energy / total_energy)


def subharmonic_content_ratio(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    rotational_frequency_hz: float,
    subharmonics: tuple[float, ...] = (0.5, 1.5),
    bandwidth_hz: float = 2.0,
) -> float:
    total_energy = np.sum(magnitude ** 2)

    if total_energy <= 0:
        return 0.0

    subharmonic_energy = 0.0

    for subharmonic in subharmonics:
        center = subharmonic * rotational_frequency_hz

        subharmonic_energy += _band_energy(
            frequencies,
            magnitude,
            center - bandwidth_hz,
            center + bandwidth_hz,
        )

    return float(subharmonic_energy / total_energy)


def peak_dominance_ratio(
    magnitude: np.ndarray,
    eps: float = 1e-12,
) -> float:
    magnitude = np.asarray(magnitude, dtype=float)

    floor = np.median(magnitude) + eps
    peak = np.max(magnitude)

    return float(peak / floor)


def classify_looseness_spectrum(
    score: float,
) -> str:
    if score >= 70:
        return "Strong low-frequency harmonic pattern"

    if score >= 40:
        return "Moderate looseness-like spectral behavior"

    return "No strong looseness-like pattern detected"


def compute_looseness_indicators(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    rpm: float,
) -> LoosenessIndicators:
    frequencies = np.asarray(frequencies, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)

    rotational_frequency_hz = rpm / 60.0

    lf_ratio = low_frequency_energy_ratio(
        frequencies,
        magnitude,
    )

    harmonic_ratio = harmonic_content_ratio(
        frequencies,
        magnitude,
        rotational_frequency_hz,
    )

    subharmonic_ratio = subharmonic_content_ratio(
        frequencies,
        magnitude,
        rotational_frequency_hz,
    )

    dominance_ratio = peak_dominance_ratio(
        magnitude,
    )

    dominance_score = min(
        1.0,
        dominance_ratio / 100.0,
    )

    score = min(
        100.0,
        100.0 * (
            0.35 * lf_ratio
            + 0.35 * harmonic_ratio
            + 0.20 * subharmonic_ratio
            + 0.10 * dominance_score
        )
    )

    return LoosenessIndicators(
        low_frequency_energy_ratio=lf_ratio,
        harmonic_content_ratio=harmonic_ratio,
        subharmonic_content_ratio=subharmonic_ratio,
        peak_dominance_ratio=dominance_ratio,
        looseness_score=score,
        spectrum_status=classify_looseness_spectrum(score),
    )