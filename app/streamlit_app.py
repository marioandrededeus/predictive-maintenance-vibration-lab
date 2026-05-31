import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.vibration_lab.sample_data import (
    generate_normal_signal,
    generate_carpet_signal,
    generate_looseness_signal,
)

from src.vibration_lab.signal_processing import (
    infer_sampling_rate,
    fft_magnitude_single_sided,
    welch_psd,
)

from src.vibration_lab.visualization import (
    create_time_signal_figure,
    create_fft_figure,
    create_psd_figure,
)


st.set_page_config(
    page_title="Predictive Maintenance Vibration Lab",
    layout="wide",
)

st.title("Predictive Maintenance Vibration Lab")

st.markdown(
    """
An open project exploring vibration analysis,
signal processing and explainable machine learning
for predictive maintenance.
"""
)

dataset = st.sidebar.selectbox(
    "Choose a dataset",
    [
        "Normal Condition",
        "Spectral Carpet",
        "Structural Looseness",
    ],
)

if dataset == "Normal Condition":
    df = generate_normal_signal()

elif dataset == "Spectral Carpet":
    df = generate_carpet_signal()

else:
    df = generate_looseness_signal()

time = df["timestamp"].values
signal = df["acceleration"].values

fs = infer_sampling_rate(time)

frequencies_fft, magnitude_fft = fft_magnitude_single_sided(
    signal,
    fs,
)

frequencies_psd, psd = welch_psd(
    signal,
    fs,
)

tab1, tab2, tab3 = st.tabs(
    [
        "Time Signal",
        "FFT Spectrum",
        "Welch PSD",
    ]
)

with tab1:
    st.plotly_chart(
        create_time_signal_figure(
            time,
            signal,
        ),
        use_container_width=True,
    )

with tab2:
    st.plotly_chart(
        create_fft_figure(
            frequencies_fft,
            magnitude_fft,
        ),
        use_container_width=True,
    )

with tab3:
    st.plotly_chart(
        create_psd_figure(
            frequencies_psd,
            psd,
        ),
        use_container_width=True,
    )