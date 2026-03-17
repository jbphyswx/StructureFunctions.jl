# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-03-17

### Added
- **GPU Acceleration**: Optional, portable GPU kernels for structure function and spectral analysis via `KernelAbstractions.jl`.
- **Spectral GPU Support**: `gpu_calculate_spectrum` added for Direct Sum method on any KA-supported backend (CUDA, ROCm, etc.).
- **Real-World Data Extensions**: Dedicated support for loading data from `JLD2`, `NetCDF`, `Zarr`, `HDF5`, and `CSV` files.
- **Scaling Benchmark Suite**: Added `test/benchmark_scaling.jl` to evaluate CPU threading and GPU performance (strong/weak scaling).
- **ND Data Utilities**: Added `flatten_data` and `remove_nans` to `HelperFunctions.jl` for robust data preprocessing.

### Changed
- **Parametric Refactor**: `StructureFunctionTypes` now uses parametric types (`ProjectedStructureFunction`, `FullVectorStructureFunction`) for improved type stability and performance.
- **Unified API**: `calculate_structure_function` now supports file paths as primary inputs, automatically dispatching to the correct extension.
- **Modernized Build**: Switched to Julia Extensions for optional dependencies (CairoMakie, FFTW, FINUFFT, KA, etc.).

### Fixed
- Improved type stability in core calculation loops.
- Resolved issues with unit vector normalization in high dimensions.
- Corrected binning logic for edge cases in structure function calculations.

### Removed
- Deprecated legacy calculation routines in favor of unified, optimized implementations.
- Removed unused archive candidates from the `src/` directory.

---
## [0.2.0] - Previous Release
- Initial implementation of spectral analysis and 2D/3D structure functions.
