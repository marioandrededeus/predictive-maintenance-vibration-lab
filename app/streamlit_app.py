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
    create_carpet_annotated_fft_figure,
    create_looseness_annotated_fft_figure,
)

from src.vibration_lab.data_loader import load_csv

from src.vibration_lab.carpet_detector import compute_carpet_indicators

from src.vibration_lab.looseness_analyzer import compute_looseness_indicators

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

carpet_indicators = compute_carpet_indicators(
    frequencies_fft,
    magnitude_fft,
)

rpm = float(df["rpm"].iloc[0]) if "rpm" in df.columns else 1800.0

looseness_indicators = compute_looseness_indicators(
    frequencies_fft,
    magnitude_fft,
    rpm=rpm,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
    "Time Signal",
    "FFT Spectrum",
    "Welch PSD",
    "Spectral Carpet",
    "Structural Looseness",
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

with tab4:
    st.subheader("Spectral Carpet Detector")

    st.metric(
        "Carpet Score",
        f"{carpet_indicators.carpet_score:.1f} / 100"
    )

    st.markdown(
        f"""
**Spectrum Status:** {carpet_indicators.spectrum_status}
"""
    )

    st.plotly_chart(
        create_carpet_annotated_fft_figure(
    frequencies_fft,
    magnitude_fft,
    ),
        use_container_width=True,
    )

    if carpet_indicators.carpet_score >= 50:
        st.warning(
            """
High broadband energy content detected.

This may indicate a spectral carpet-like pattern,
commonly associated with lubrication starvation,
friction-related behavior or distributed high-frequency excitation.
"""
        )
    else:
        st.success(
            """
No strong spectral carpet-like pattern detected.

High-frequency broadband energy is not dominant in this signal.
"""
        )

    st.markdown(
        """
### Interpretation

The carpet score is based on the ratio between high-frequency spectral energy
and total spectral energy.

It is an explainable indicator designed for exploration,
not a replacement for professional vibration analysis.
"""
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High Frequency Energy Ratio",
        f"{carpet_indicators.high_frequency_energy_ratio:.3f}"
    )

    col2.metric(
        "Spectral Flatness",
        f"{carpet_indicators.spectral_flatness:.3f}"
    )

    col3.metric(
        "Peak-to-Floor Ratio",
        f"{carpet_indicators.peak_to_floor_ratio:.2f}"
    )

with tab5:
    st.subheader("Structural Looseness Analyzer")

    st.metric(
        "Looseness Score",
        f"{looseness_indicators.looseness_score:.1f} / 100"
    )

    st.markdown(
        f"""
**Spectrum Status:** {looseness_indicators.spectrum_status}
"""
    )

    st.plotly_chart(
        create_looseness_annotated_fft_figure(
        frequencies_fft,
        magnitude_fft,
        rpm=rpm,
        ),
        use_container_width=True,
    )

    if looseness_indicators.looseness_score >= 50:
        st.warning(
            """
Low-frequency harmonic behavior detected.

This may indicate a looseness-like pattern,
commonly associated with structural looseness,
mechanical clearance or nonlinear vibration behavior.
"""
        )
    else:
        st.success(
            """
No strong looseness-like pattern detected.

Low-frequency harmonic and sub-harmonic energy is not dominant in this signal.
"""
        )

    st.markdown(
        """
### Interpretation

The looseness score is based on low-frequency energy,
harmonic content, sub-harmonic content and peak dominance.

It is an explainable indicator designed for exploration,
not a replacement for professional vibration analysis.
"""
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Low Frequency Energy Ratio",
        f"{looseness_indicators.low_frequency_energy_ratio:.3f}"
    )

    col2.metric(
        "Harmonic Content Ratio",
        f"{looseness_indicators.harmonic_content_ratio:.3f}"
    )

    col3.metric(
        "Subharmonic Content Ratio",
        f"{looseness_indicators.subharmonic_content_ratio:.3f}"
    )

    col4.metric(
        "Peak Dominance Ratio",
        f"{looseness_indicators.peak_dominance_ratio:.2f}"
    )