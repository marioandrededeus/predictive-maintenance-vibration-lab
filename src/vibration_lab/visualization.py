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


def create_carpet_annotated_fft_figure(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    high_frequency_start_hz: float = 1000.0,
    title: str = "FFT Spectrum with High-Frequency Carpet Region"
) -> go.Figure:

    fig = create_fft_figure(
        frequencies,
        magnitude,
        title=title,
    )

    max_frequency = float(np.max(frequencies))
    max_magnitude = float(np.max(magnitude))

    fig.add_vrect(
        x0=high_frequency_start_hz,
        x1=max_frequency,
        opacity=0.12,
        line_width=0,
        annotation_text="High-frequency region",
        annotation_position="top left",
    )

    fig.add_vline(
        x=high_frequency_start_hz,
        line_dash="dash",
        annotation_text=f"{high_frequency_start_hz:.0f} Hz",
        annotation_position="top",
    )

    fig.update_yaxes(
        range=[
            0,
            max_magnitude * 1.10
        ]
    )

    return fig


def create_looseness_annotated_fft_figure(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    rpm: float,
    title: str = "FFT Spectrum with Harmonic and Sub-Harmonic Markers"
) -> go.Figure:

    fig = create_fft_figure(
        frequencies,
        magnitude,
        title=title,
    )

    rotational_frequency_hz = rpm / 60.0

    markers = [
        (0.5, "0.5x"),
        (1.0, "1x"),
        (2.0, "2x"),
        (3.0, "3x"),
    ]

    max_frequency = float(np.max(frequencies))

    for multiplier, label in markers:

        marker_frequency = multiplier * rotational_frequency_hz

        if marker_frequency <= max_frequency:
            fig.add_vline(
                x=marker_frequency,
                line_dash="dash",
                annotation_text=label,
                annotation_position="top",
            )

    return fig