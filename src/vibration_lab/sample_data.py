import numpy as np
import pandas as pd


def generate_normal_signal(
    fs=10000,
    duration=5,
    rpm=1800
):

    t = np.arange(0, duration, 1 / fs)

    f_rot = rpm / 60

    signal = (
        0.20 * np.sin(2 * np.pi * f_rot * t)
        + 0.03 * np.random.randn(len(t))
    )

    return pd.DataFrame({
        "timestamp": t,
        "acceleration": signal,
        "rpm": rpm,
        "label": "normal"
    })


def generate_carpet_signal(
    fs=10000,
    duration=5,
    rpm=1800
):

    t = np.arange(0, duration, 1 / fs)

    f_rot = rpm / 60

    signal = (
        0.15 * np.sin(2 * np.pi * f_rot * t)
        + 0.25 * np.random.randn(len(t))
        + 0.08 * np.sin(2 * np.pi * 2500 * t)
    )

    return pd.DataFrame({
        "timestamp": t,
        "acceleration": signal,
        "rpm": rpm,
        "label": "spectral_carpet"
    })


def generate_looseness_signal(
    fs=10000,
    duration=5,
    rpm=1800
):

    t = np.arange(0, duration, 1 / fs)

    f_rot = rpm / 60

    signal = (
        0.40 * np.sin(2 * np.pi * f_rot * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f_rot * t)
        + 0.18 * np.sin(2 * np.pi * 3 * f_rot * t)
        + 0.12 * np.sin(2 * np.pi * 0.5 * f_rot * t)
        + 0.03 * np.random.randn(len(t))
    )

    return pd.DataFrame({
        "timestamp": t,
        "acceleration": signal,
        "rpm": rpm,
        "label": "structural_looseness"
    })