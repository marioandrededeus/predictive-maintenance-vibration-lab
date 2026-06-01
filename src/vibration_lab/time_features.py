from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis


@dataclass(frozen=True)
class TimeDomainFeatures:
    rms: float
    peak: float
    peak_to_peak: float
    crest_factor: float
    kurtosis: float


def compute_time_domain_features(
    signal: np.ndarray,
) -> TimeDomainFeatures:

    signal = np.asarray(
        signal,
        dtype=float,
    )

    rms = float(
        np.sqrt(
            np.mean(signal ** 2)
        )
    )

    peak = float(
        np.max(
            np.abs(signal)
        )
    )

    peak_to_peak = float(
        np.max(signal)
        - np.min(signal)
    )

    crest_factor = float(
        peak / rms
    ) if rms > 0 else 0.0

    kurtosis_value = float(
        kurtosis(
            signal,
            fisher=False,
            bias=False,
        )
    )

    return TimeDomainFeatures(
        rms=rms,
        peak=peak,
        peak_to_peak=peak_to_peak,
        crest_factor=crest_factor,
        kurtosis=kurtosis_value,
    )