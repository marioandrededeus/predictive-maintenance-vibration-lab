# Vibration Fault Pattern Explorer

## Overview

Vibration Fault Pattern Explorer is an open technical study that combines vibration analysis, signal processing, physics-informed feature engineering, and explainable machine learning to support predictive maintenance in rotating machinery.

The project investigates how different failure mechanisms generate distinct vibration patterns and how these patterns can be identified through spectral analysis and machine learning techniques.

Rather than focusing solely on anomaly detection, the goal is to understand and explain the physical behavior behind vibration signatures commonly observed in industrial environments.

---

## Motivation

Vibration signals contain valuable information about the mechanical condition of rotating assets.

Different failure mechanisms produce different signatures:

* Lubrication starvation often manifests as broadband high-frequency energy and spectral floor elevation.
* Structural looseness typically appears as low-frequency harmonic and sub-harmonic content.
* Bearing defects may generate impulsive events and characteristic frequencies.
* Unbalance and misalignment produce specific harmonic patterns.

The objective of this project is to bridge mechanical engineering knowledge and machine learning, creating tools that help maintenance professionals explore and interpret vibration patterns.

---

## Current Modules

### Spectral Carpet Detector

The first module focuses on vibration patterns associated with lubrication starvation.

Typical characteristics include:

* High-frequency broadband energy
* Spectral floor elevation
* Increased spectral flatness
* Random micro-impact behavior

The implementation is inspired by published research on lubrication starvation detection and combines signal processing with explainable scoring methods.

---

### Structural Looseness Analyzer

The second module focuses on structural looseness.

Typical characteristics include:

* Strong low-frequency components
* Harmonic families (1x, 2x, 3x)
* Sub-harmonics (0.5x, 1.5x)
* Increased vibration velocity levels

The objective is to identify patterns commonly associated with loose foundations, mechanical clearances, and nonlinear vibration behavior.

---

## Web Application

The project includes an interactive Streamlit application where users can:

* Visualize vibration signals
* Generate FFT and Welch spectra
* Explore fault-related features
* Evaluate spectral carpet patterns
* Analyze structural looseness indicators
* Upload their own vibration datasets

---

## Example Datasets

The repository provides sample datasets for educational purposes:

* Normal operating condition
* Starved lubrication spectral carpet pattern
* Structural looseness pattern

These datasets are synthetic but generated to reproduce vibration characteristics commonly observed in real industrial environments.

---

## Uploading Your Own Data

The application accepts CSV files.

Required columns:

```text
timestamp
acceleration
```

Optional columns:

```text
velocity
rpm
label
axis
```

Example:

timestamp,acceleration,rpm
0.000,0.012,1800
0.001,0.018,1800
0.002,-0.004,1800

Recommended units:

* timestamp: seconds
* acceleration: g or m/s²
* velocity: mm/s
* rpm: rotational speed

---

## Project Structure

```text
app/
src/
data/
notebooks/
reports/
assets/
```

## Disclaimer

This project is intended for educational and exploratory purposes.

It does not replace professional vibration analysis or maintenance engineering judgment.

```
```