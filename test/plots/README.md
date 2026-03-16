# Spectral Analysis Diagnostic Plots

This directory contains visual checks for the spectral analysis module. These plots help us ensure that our math is correct when switching between different ways of calculating "vibrations" (spectra) in our data.

## What are we looking at?

Each diagnostic image now contains a **ground truth field plot** in the first column, followed by spectral recovery results from different backends.

*   **1st Column**: The original input field (e.g., $u(x,y)$), either on a uniform grid or scattered points.
*   **Subsequent Columns**: The power spectrum magnitude $|c(k_x, k_y)|$ recovered by different backends (Direct Sum, FINUFFT, FFT).
*   **Red markers** (dotted lines in 1D, crosses in 2D) indicate the **target frequencies** injected into the synthetic data. We expect the brightest peaks to align perfectly with these markers.

### 1. `verify_spectral_parity_uniform_grid.png`
**Goal**: Check if all math methods (FFT, NUFFT, and Direct Sum) give exactly the same answer on a perfect grid.
*   **Backends**: Direct Sum vs. FINUFFT vs. FFTW.
*   **Success**: The results from all three methods are visually identical and match to 11+ decimal places. Highlights DC consistency and normalization parity.

### 2. `verify_spectral_parity_scattered_grid.png`
**Goal**: Test if our method (NUFFT) can find the "notes" in messy, scattered data even with a coastal land mask (coastal exclusion).
*   **Backends**: Direct Sum (Scattered) vs. FINUFFT (Scattered).
*   **Success**: The field plot shows the scattered samples with hole/mask artifacts. The spectral results clearly align with target peaks despite the non-uniform sampling and masking.

### 3. `verify_spectral_sampling_validation.png`
**Goal**: Confirm that looking at scattered points (NUFFT) gives the same result as the slow, reliable ground truth (Direct Sum) when sampling high-resolution analytic signals.
*   **Backends**: NUFFT (Scattered Samples) vs. Direct Sum (Ground Truth).
*   **Success**: The two patterns look nearly identical, proving our scattered-data math is accurate and robust to sampling density variations.

---
*Note: These images are automatically generated during testing to help developers spot issues like shifts or scaling errors at a glance.*
