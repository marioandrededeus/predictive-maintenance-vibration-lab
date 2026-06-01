from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoosenessIndicators:
    higher_order_harmonic_ratio: float
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


def higher_order_harmonic_ratio(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    rotational_frequency_hz: float,
    bandwidth_hz: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """
    Ratio between higher-order harmonic energy and 1x energy.

    Structural looseness often increases 2x and 3x components
    relative to the fundamental rotational frequency.
    """

    energy_1x = _band_energy(
        frequencies,
        magnitude,
        rotational_frequency_hz - bandwidth_hz,
        rotational_frequency_hz + bandwidth_hz,
    )

    energy_2x = _band_energy(
        frequencies,
        magnitude,
        2.0 * rotational_frequency_hz - bandwidth_hz,
        2.0 * rotational_frequency_hz + bandwidth_hz,
    )

    energy_3x = _band_energy(
        frequencies,
        magnitude,
        3.0 * rotational_frequency_hz - bandwidth_hz,
        3.0 * rotational_frequency_hz + bandwidth_hz,
    )

    return float(
        (energy_2x + energy_3x) / (energy_1x + eps)
    )


def subharmonic_content_ratio(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    rotational_frequency_hz: float,
    bandwidth_hz: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """
    Ratio between sub-harmonic energy and 1x energy.

    Sub-harmonic components such as 0.5x and 1.5x are commonly
    associated with nonlinear vibration behavior.
    """

    energy_1x = _band_energy(
        frequencies,
        magnitude,
        rotational_frequency_hz - bandwidth_hz,
        rotational_frequency_hz + bandwidth_hz,
    )

    energy_05x = _band_energy(
        frequencies,
        magnitude,
        0.5 * rotational_frequency_hz - bandwidth_hz,
        0.5 * rotational_frequency_hz + bandwidth_hz,
    )

    energy_15x = _band_energy(
        frequencies,
        magnitude,
        1.5 * rotational_frequency_hz - bandwidth_hz,
        1.5 * rotational_frequency_hz + bandwidth_hz,
    )

    return float(
        (energy_05x + energy_15x) / (energy_1x + eps)
    )


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
        return "Strong looseness-like harmonic and sub-harmonic pattern"

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

    higher_order_ratio = higher_order_harmonic_ratio(
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
        dominance_ratio / 1500.0,
    )

    normalized_higher_order = min(
        1.0,
        higher_order_ratio / 0.50,
    )

    normalized_subharmonic = min(
        1.0,
        subharmonic_ratio / 0.15,
    )

    score = min(
        100.0,
        100.0 * (
            0.50 * normalized_higher_order
            + 0.35 * normalized_subharmonic
            + 0.15 * dominance_score
        )
    )

    return LoosenessIndicators(
        higher_order_harmonic_ratio=higher_order_ratio,
        subharmonic_content_ratio=subharmonic_ratio,
        peak_dominance_ratio=dominance_ratio,
        looseness_score=score,
        spectrum_status=classify_looseness_spectrum(score),
    )