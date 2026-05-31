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

from src.vibration_lab.data_loader import load_csv


st.set_page_config(
    page_title="Predictive Maintenance Vibration Lab",
    layout="wide",
)

st.title("Predictive Maintenance Vibration Lab")

st.info(
    """
This project explores how vibration analysis, signal processing and
machine learning can support predictive maintenance in rotating machinery.

Current examples include:

• Normal operating condition

• Spectral carpet patterns associated with lubrication starvation

• Structural looseness characterized by harmonic and sub-harmonic behavior

You can use the example datasets or upload your own vibration signal.
"""
)

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
        "Upload CSV",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### Example datasets

**Normal Condition**

Baseline vibration signal.

**Spectral Carpet**

Broadband high-frequency energy increase.

**Structural Looseness**

Low-frequency harmonic and sub-harmonic behavior.
"""
)

uploaded_file = None

if dataset == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload vibration data",
        type=["csv"]
    )

    st.sidebar.markdown(
        """
Required columns:

- timestamp
- acceleration

Optional columns:

- rpm
- velocity
- label
- axis
"""
    )

if dataset == "Normal Condition":

    df = generate_normal_signal()

elif dataset == "Spectral Carpet":

    df = generate_carpet_signal()

elif dataset == "Structural Looseness":

    df = generate_looseness_signal()

else:

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to start the analysis."
        )

        st.stop()

    try:
        df = load_csv(uploaded_file)

    except Exception as e:
        st.error(str(e))
        st.stop()

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