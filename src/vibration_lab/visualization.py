from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def create_time_signal_figure(
    time: np.ndarray,
    signal: np.ndarray,
    title: str = "Time Signal"
) -> go.Figure:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=signal,
            mode="lines",
            name="Signal"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white"
    )

    return fig


def create_fft_figure(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    title: str = "FFT Spectrum"
) -> go.Figure:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=magnitude,
            mode="lines",
            name="FFT"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white"
    )

    return fig


def create_psd_figure(
    frequencies: np.ndarray,
    psd: np.ndarray,
    title: str = "Welch Power Spectral Density"
) -> go.Figure:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=psd,
            mode="lines",
            name="PSD"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        template="plotly_white"
    )

    return fig