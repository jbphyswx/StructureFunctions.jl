# StructureFunctions.jl


| **DOI**                           | [![DOI][zenodo-img]][zenodo-latest-url]          |
|-----------------------------------|--------------------------------------------------|

[zenodo-img]: https://zenodo.org/badge/734119226.svg
[zenodo-latest-url]: https://doi.org/10.5281/zenodo.14945669


Given data, calculates varying order structure functions for turbulence analysis.

## Features

- **Structure Functions**: Longitudinal and transverse SFs (1st, 2nd, 3rd order) in 1D, 2D, and 3D.
- **Spectral Analysis**: Unified API for power spectrum density calculation using:
  - `DirectSumBackend`: Accurate $O(N \cdot M)$ reference sum.
  - `FINUFFTBackend`: High-performance Non-Uniform Fast Fourier Transform.
  - `FFTBackend`: Standard FFT for uniform grids.
- **Modern Architecture**: Extension-based dependencies and type-stable dispatch.

## Quick Start

```julia
using StructureFunctions: StructureFunctions, SpectralAnalysis as SFSA

# Calculate spectrum from scattered data
r = SFSA.calculate_spectrum(SFSA.FINUFFTBackend(), (x, y), (u, v), ms; domain_size=(Lx, Ly))

# Visual comparison (requires CairoMakie)
using CairoMakie: CairoMakie
SFSA.compare_spectra(["Result" => r]; peaks=target_peaks)
```

