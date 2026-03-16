# Spectral Analysis Diagnostic Plots

This directory contains visual checks for the spectral analysis module. These plots help us ensure that our math is correct when switching between different ways of calculating "vibrations" (spectra) in our data.

## What are we looking at?

Think of these plots like an equalizer on a stereo. Each "panel" in the image represents a different math backend (Direct Sum, FINUFFT, or FFTW).

**Red markers** (dotted lines in 1D, crosses in 2D) indicate the **target frequencies** we injected into the synthetic data. We expect the brightest spots (peaks) to align perfectly with these markers.

### 1. `verify_spectral_parity_uniform_grid.png`
**Goal**: Check if all math methods (FFT, NUFFT, and Direct Sum) give exactly the same answer on a perfect grid.
*   **Backends**: Direct Sum vs. FINUFFT vs. FFTW.
*   **Success**: The results from all three methods are visually identical and match to 11+ decimal places.

### 2. `verify_spectral_parity_scattered_grid.png`
**Goal**: Test if our advanced method (NUFFT) can find the "notes" in messy, scattered data (like weather stations randomly placed on a map), even with a land mask.
*   **Backends**: Direct Sum (Scattered) vs. FINUFFT (Scattered).
*   **Success**: The dominant "notes" are clearly visible, align with the red crosses, and match between methods.

### 3. `verify_spectral_validation_scattered_vs_truth.png`
**Goal**: Confirm that looking at a few scattered points (NUFFT) gives us the same result as the slow, reliable ground truth (Direct Sum).
*   **Backends**: NUFFT (Scattered Samples) vs. Direct Sum (Ground Truth).
*   **Success**: The two patterns look nearly identical, proving our scattered-data math is accurate.

---
*Note: These images are automatically generated during testing to help developers spot issues like shifts or scaling errors at a glance.*
