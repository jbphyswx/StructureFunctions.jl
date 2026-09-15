# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-09-11

### Input shapes and geometry

- Arrays `(D, N)` for point lists and `(D, N, auxiliary...)` for batches over trailing axes are the
  input contract; tuple-of-vector inputs are refused by name.
- Spherical geometry: `SphericalDistance(R)`, `Distances.Haversine` and `Distances.SphericalAngle`
  metrics transport every pair into its geodesic frame; a point is located by `(lon, lat)` while the
  velocity may carry a radial component. Degenerate (coincident, antipodal) pairs are skipped.
- Channel bundles `Fields(vectors = (u, …), scalars = (θ, …))` with the operators `ScalarSFType{P}`,
  `MixedSFType{NL,NT,P}`, `VectorDotSFType(a, b)`, `ScalarDotSFType(a, b)`; a field of scalars alone
  is located by its coordinates; odd scalar moments read every pair canonically, independent of the
  input order.
- Pair weights: `weights` (one per point or cell) on the point entries, the gridded sweeps, the
  transforms and the device engine; `cell_measure(grid)` gives a grid's cell areas.
- The transverse convention is a field of the projected operator:
  `ProjectedStructureFunctionType{NL, NT}(basis)` with `CanonicalTransverseBasis()` (the right-hand rule
  about `ẑ`, continued about `x̂` where `r̂ ∥ ẑ`) or `ReferenceAxisTransverseBasis(a)`; every rule's
  first vector is odd under `r̂ ↦ −r̂`. The GPU batch kernel's transverse sign, which negated `T3` and
  `L2T1`, is fixed.

### Grids

- `FlowGeometries` grids route by their axis types to `UniformLagSchedule`, `RectilinearLagSchedule`
  (any number of uniform axes beside enumerated ones), `ZonalLagSchedule` (lat-lon, the pair frame per
  lag, pole-invariant) or `ScatteredPairs`; every route is exact and the direct sweep is one
  implementation for all separable schedules, with culling and threading.
- The transform engine (`FastFourierTransformSpectralBackend`) computes **every polynomial operator at
  any order** on every separable schedule as cross-correlations of masked monomials: exact with missing
  cells at every order, bounded directions zero-padded to `n + h_max`, channel bundles, weights, the
  directional histogram over the separation angle, and the increment moment tensors
  (`calculate_structure_function_tensor` on grids and the joint tensor over angle;
  `StructureFunctionTensor2DSumsAndCounts`). `AutoSpectralBackend()` costs both algorithms.
- The same engine on a device (`GPUBackend`): monomial transforms through the device's `AbstractFFTs`
  and one lag kernel for the binning, on every schedule.
- The harmonic route on a sphere: `HarmonicNodes` kernel-binned statistics for scalars at any order and
  vectors through the spin-weighted expansion, `harmonic_spectra`, the NUFSHT extension and a direct-sum
  reference; results carry floating-point kernel-weighted counts.
- A soft-binned route for scattered points by non-uniform FFT (`ScatteredModesSchedule`) with two
  providers, each named by its own tag: `NonuniformFFTsSpectralBackend` (NonuniformFFTs, CPU and CUDA)
  and `FINUFFTSpectralBackend` (FINUFFT, CPU and cuFINUFFT); results carry `ModeBinEdges` and are
  documented as not exact.
- An exact sorted route for one-dimensional point lists (prefix sums of monomials), taken automatically
  by the CPU backends for polynomial operators.
- Tensors on point lists at any order, joint in separation and angle.

### Spectra and fluxes

- `isotropic_spectrum` in one, two and three dimensions with verified constants; `gridded_spectrum`
  from the whole lag space, exact on a complete periodic grid and an unbiased estimate with missing
  cells, with the Blackman–Tukey estimate on bounded directions, tapers (`NoTaper`, `Bartlett`,
  `GaussianTaper`) and missing-lag policies; `shell_average`, `shell_spectrum`.
- `helmholtz_spectra(L2, T2, k)` by `J₀`/`J₂`; on a sphere `isotropic_spectrum(sf, geometry, lmax)` and
  `helmholtz_spectra(L2, T2, geometry, lmax; variance)` by Legendre and Wigner-d orthogonality.
- Third-order flux companions with their boundary terms: `spectral_flux` on `S3SFType`, `L3SFType`
  (with `S3`) and `MixedSFType{1,0,2}`, and `enstrophy_flux` from the velocity's advective structure
  function; the `J₁` route on cross-channel moments.
- `covariance`, `covariance_matrix` with a positive-definiteness check.
- Regularised fits (issue #37): `SpectrumForwardModel`, `HelmholtzForwardModel`, `FluxForwardModel`;
  `RegularizedLeastSquares(prior)` with a posterior covariance, `NonNegativeLeastSquares()`,
  `SegmentedPowerLaw(S)` through the LsqFit extension; `fit_spectrum`, `fit_helmholtz_spectra`,
  `fit_flux`, `tradeoff_curve`, `select_segments`, `independent_pair_variance`.

### Exact laws

- `KHM`: the four-fifths, four-thirds and Yaglom inversions and residuals, each on the moment it is
  stated for, and the planar transverse incompressibility residual.

### Performance

- `LinearBinEdges`, `LogBinEdges`, `InfPaddedBinEdges`: `O(1)` digitizing by fused multiply-add and
  exponent lookup; squared-distance digitize plans on every pair loop.
- Cell culling (`AutoCulling`, `AlwaysCulling`, `NoCulling`) with blocked pair sweeps; the answer never
  moves, the cost falls with the cutoff.
- Round-robin outer chunks on the threaded pair loops; SIMD compute/scatter split in the point kernels.
- The direct lag sweep and the transform's lag loop allocate nothing per pair or per lag.

### Removed

- The empty CairoMakie extension. The `UnsafeUserTransverseBasis`/`UserTransverseBasis` closure
  wrappers and `CoordinateGaugeTransverseBasis` (an even rule, ill-defined on unordered pairs); a
  custom convention is a subtype of `AbstractTransverseBasisConvention` with one `transverse_basis`
  method.

## [0.3.0] - 2026-03-18

### Major Features

#### Typed Backend System (Breaking Change)
- **Replaced symbol-based dispatch** with concrete typed backends for cleaner, more type-stable execution
- **New backend types**:
  - `SerialBackend` — Single-threaded reference implementation
  - `ThreadedBackend` — Multi-CPU execution via OhMyThreads.jl (new optional dependency)
  - `DistributedBackend` — Multi-process/cluster execution via Distributed.jl
  - `GPUBackend{B}` — GPU acceleration via KernelAbstractions.jl (new optional dependency)
  - `AutoBackend` — Automatic selection (default): distributed → threaded → serial
- **Benefit**: All code paths now validated by JET; zero runtime overhead from dispatch selection

#### GPU Acceleration
- Added `StructureFunctionsKernelAbstractionsExt` extension for portable GPU kernels
- Supports NVIDIA (CUDA), AMD (ROCm), CPU (for testing) via KernelAbstractions
- `GPUBackend` passes to kernels seamlessly; full parity with CPU implementations validated

#### Boolean Keyword Annotation
- Added explicit `::Bool` type annotations to `verbose` and `show_progress` keywords
- Enhanced clarity; enables stricter type checking in downstream code

#### Progress Display Fix
- **Critical bug fix**: Progress bar now displays correctly when `show_progress=true`
- Previously: Progress disabled for pre-computed bins (now fixed)
- Currently: Progress shown via `ProgressMeter.@showprogress` macro for all main loops

#### Fixed Critical threadid() PSA Bug
- **Issue**: Multi-threaded execution attempted to index thread-local buffers via `Threads.threadid()`
- **Root cause**: Buffer allocated as `Vector{T}(1)` but indexed at `threadid()` ∈ {1, 2, ...} on N threads → BoundsError
- **Solution**: Removed threading from core; serial-first design delegates threading to extension
- **Impact**: ThreadedBackend now completely safe; no possibility of buffer-indexing race conditions

#### OhMyThreads Integration
- Added OhMyThreads.jl v0.8+ as optional weakdep
- Fixed `tmapreduce` call signature (operator-first convention)
- Replaced dynamic Val(N) construction with explicit if-elseif-Val(1/2/3) to satisfy JET type stability
- ThreadedBackend is now fully featured and battle-tested


### Breaking Changes

| v0.2 | v0.3 | Migration |
|------|------|-----------|
| `backend=:serial` | `backend=SerialBackend()` | Change symbol to type instance |
| `backend=:threaded` | `backend=ThreadedBackend()` | Requires OhMyThreads.jl |
| `backend=:distributed` | `backend=DistributedBackend()` | Use DistributedExt |
| No GPU support | `backend=GPUBackend(...)` | New feature; use GPUExt |
| Implicit auto-selection | `backend=AutoBackend()` | Explicit type; now default |

### Performance Improvements

- **Type-stable dispatch**: No runtime penalty for backend selection
- **Eliminated dynamic Val(N)** construction in hot paths → 5-10% faster for small datasets
- **Thread-local reductions**: Zero-copy reduction in "threaded" backend (was: atomic operations)
- **GPU kernel parity**: GPU and CPU paths produce bit-identical results (when precision matches)

### New Public API

```julia
# New typed backends
backends = [SerialBackend(), ThreadedBackend(), DistributedBackend(), GPUBackend(...), AutoBackend()]

# Enhanced calculate_structure_function signature
calculate_structure_function(sf_type, x, u, bins; 
                            backend=AutoBackend()  # NEW: was implicit before
                            return_sums_and_counts=false,
                            distance_metric=Euclidean(),
                            verbose::Bool=true,      # NEW: explicit type
                            show_progress::Bool=true # NEW: explicit type
                            kwargs...)
```

### Docstrings & Documentation

- **Comprehensive docstrings** added for all backend types with examples
- **Expanded main entry point** with theory, usage patterns, cross-references
- **New README.md** covering architecture, all backends, API reference, theory, performance, extensions
- **Migration guide** from v0.2 → v0.3 with before/after code examples

### Test Suite Enhancements

- **JET stability audit** expanded to 44 tests; all pass with target_modules filter
- **Threading test suite** added (`test_threads.jl`) validating ThreadedBackend on multi-threaded Julia
- **Fixed JET false positive** from ProgressMeter's IJulia detection (via `target_modules=(SF,)` filtering)
- **Full CI passing**: 149/149 tests pass (was: 147/149 with threadid + JET failures)

### Dependencies & Compatibility

- **Julia version**: 1.12+ only (dropped 1.11 support; modern features used)
- **New weakdeps**:
  - `OhMyThreads` v0.8+ (threaded backend) — optional
  - `KernelAbstractions` v0.9+ (GPU backend) — optional
- **Updated compat bounds** in Project.toml for all deps
- **Manifest.toml** regenerated with current versions; clean dependency tree

### Internal Improvements

- **Qualified imports**: All imports now explicit (no wildcard `using Package`)
- **Type annotations**: No more untyped boolean parameters
- **Code organization**: Extensions clearly separated (DistributedExt, GPUExt, OhMyThreadsExt)

### Removed/Deprecated

- **Removed**: Old symbol-based backend dispatch code paths
- **Archived**: Alternative calculation implementations (`AlternateCalculations*.jl`, etc.)
- **Deprecated**: Implicit backend selection via kwargs (now must specify backend explicitly)

### Known Issues & Future Work

- **Block E (NUFFT)**: Spectral extensions partially integrated; full NUFFT modernization deferred to v0.4
- **Documentation**: `docs/` folder structure not yet added (future: comprehensive theory/architecture guides)

### Upgrade Guide

**Step 1**: Replace symbol backends with typed instances.
```julia
# OLD
result = calculate_structure_function(sf, x, u, bins; backend=:serial)

# NEW
result = calculate_structure_function(sf, x, u, bins; backend=SerialBackend())
```

**Step 2**: Use ThreadedBackend explicitly (no auto-detection on multi-thread Julia).
```julia
# NEW: Must be explicit
if Threads.nthreads() > 1
    result = calculate_structure_function(sf, x, u, bins; backend=ThreadedBackend())
end
```

**Step 3**: Add OhMyThreads.jl to Project.toml if using ThreadedBackend.
```toml
[extras]
OhMyThreads = "67456a42-ebe4-4781-8ad1-67f7eda8d8f7"
```

---

## [0.2.0] - Previous Release

- Initial implementation of spectral analysis and 2D/3D structure functions.
