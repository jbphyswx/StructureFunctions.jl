# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Breaking changes

- The count type and the result representation are positional arguments of every public entry, after the bins:
  `calculate_structure_function(sf, x, u, bins[, CT][, OT]; …)`, and likewise for the joint, single-pass, tensor,
  multi-field, harmonic, scattered-modes and grid forms. The keywords `count_eltype` and `output_type` are gone. A
  call without `CT` counts in `UInt32`.
- `verbose`, `show_progress` and the ProgressMeter dependency are removed. Passing either keyword raises, as any
  unknown keyword does.
- The backend-keyword entries are the public surface; the `serial_*`, `threaded_*`, `gpu_*` and `auxiliary_*` drivers
  are internal.
- A result computed on a GPU backend stays on the device; `to_host(result)` copies it to the host. In-place calls on a
  GPU backend take device output buffers.
- A vector of edges is `BinEdges`; `LinearBinEdges(::AbstractVector)` and `LogBinEdges(::AbstractVector)` raise. A
  uniform grid of edges is given by its endpoints and count, or as a range.
- `NonuniformFFTsSpectralBackend{M}` carries the kernel half-support its tolerance implies as the type parameter `M`,
  and is built only as `NonuniformFFTsSpectralBackend(; tolerance)`.
- `StructureFunction2DSumsAndCounts` records the second axis it was binned over; adding two results binned over
  different axes raises.
- The grid entries default their spectral backend to `AutoSpectralBackend()`.
- `independent_pair_variance` raises for a histogram over the separation angle and for floating-point (weighted or
  split) counts, where the counts are not a number of independent pairs.
- Fitting a result with batch axes returns an array of the per-slice fits over those axes.
- `isotropic_spectrum` and `helmholtz_spectra` take the field's `variance` as a required keyword, in place of an
  `asymptote` that defaulted to the largest value of the structure function; `helmholtz_spectra(h, k; variance)`
  replaces `rotational_asymptote` and `divergent_asymptote`.

### New

- `TransformWorkspace` keeps a grid transform's buffers and plans, including non-uniform FFT plans set to a scattered
  schedule's points, from one call to the next.
- On grids: the six single-pass invariants, by the lag sweep and by the transform; the joint histogram over the
  operator value; device routes for the joint histograms, the batches and the tensor.
- Culling on every CPU and device pair route, batches, tensors and multi-fields included, with or without a
  workspace.
- On a device: tensors of any order, multi-fields, the sorted line, harmonic coefficients, weighted calls and any width
  on the native CUDA kernels.
- Spectra, fluxes, the covariance and the Helmholtz split of a device result are computed in the result's own array
  family.
- The transform shares one grid's forward transforms, inverse columns and lags over the backend's tasks, and a batch
  of fewer slices than tasks runs its slices one after another on every task. With FFTW loaded, the engine's plans
  are single-threaded: FFTW's global thread count, which NonuniformFFTs raises, made the threaded transform
  run each of its tasks' plans on that many threads again.
- CPU batches on a flat metric of width 2 or 3: over shared positions, each run of pairs takes its geometry, its
  in-range choice and its bins once for every slice, and each slice a vectorized value pass, in place of a
  per-pair scalar loop for the joint and single-pass 2D histograms; over positions varying per slice, each slice's
  pairs take the vectorized pair kernel in place of a scalar loop over pairs and slices.

### Fixed

- Every digitizer, on every backend, places a value in the bin `searchsortedfirst` on the edges gives it.
- The CPU single-pass joint histogram counted in the sums' element type, so `Float32` sums stopped counting at 2²⁴
  pairs per cell; it counts in the count type.
- Weighted single-pass calls on `DistributedBackend` and `MPIBackend` returned the unweighted result, and an explicit
  `culling` was replaced by the default there.
- `AlwaysCulling()` on a device batch over varying positions without a workspace swept every pair instead of culling.
- `isotropic_spectrum`, `helmholtz_spectra` and `helmholtz_decompose_2d` integrate from zero separation by the
  trapezoid rule. `isotropic_spectrum` weighted its first sample as a whole bin and left out the separations below
  it, which in one dimension added a constant to the spectrum at every wavenumber; `helmholtz_decompose_2d` began
  its integral at the first bin.
- The native CUDA plan of a call class is chosen by timing every candidate after all have compiled, and a candidate
  leaves the timing only from its second round. A candidate's first timed launch held host time, so the plan the
  class's first call had run could be kept over a faster one.
- The spectra of a structure function that overshoots its large-separation limit, such as the transverse function
  of a solenoidal field, rang and went negative: the transforms subtracted the function's largest value, so the
  integrand did not decay before the last separation. They subtract twice the variance.
- `helmholtz_spectra(h, k)` transformed the rotational and divergent functions separately, each with a limit of its
  own; it transforms the trace and `D_LL − D_TT`, as the form taking `L2` and `T2` does.

## [0.4.0] - 2026-09-11

### Names and namespace

- The vocabulary is the domain's: several quantities at one set of points are **fields**, not
  "channels". `Fields` now lives in the `MultiFields` submodule, with `FieldIncrement`,
  `n_vector_fields`, `n_scalar_fields` and `field_dimension`.
- The package exports only its own names — the bin-edge types, the tapers and `HarmonicNodes`, the
  result types, `KHM` and `midpoints`. Everything else is reached through the submodule that owns it:
  `Calculations` for the entries, `StructureFunctionTypes` for the operators, `HelperFunctions` for
  the geometry, `MultiFields` for `Fields`.

### Input shapes and geometry

- Arrays `(D, N)` for point lists and `(D, N, auxiliary...)` for batches over trailing axes are the
  input contract; tuple-of-vector inputs are refused by name.
- Spherical geometry: `SphericalDistance(R)`, `Distances.Haversine` and `Distances.SphericalAngle`
  metrics transport every pair into its geodesic frame; a point is located by `(lon, lat)` while the
  velocity may carry a radial component. Degenerate (coincident, antipodal) pairs are skipped.
- Multi-fields `Fields(vectors = (u, …), scalars = (θ, …))` with the operators `ScalarSFType{P}`,
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
  cells at every order, bounded directions zero-padded to `n + h_max`, multi-fields, weights, the
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
- A batch over a trailing slice axis for a field sampled repeatedly on one grid:
  `calculate_structure_function_batch!(sums, counts, sf, grid, u, bins[, axis_bins][, tag])` with
  `(component, cells..., T)` in and `(n_distance, T)` — or `(n_distance, n_angle, T)` — out, on the
  lag sweep, the transform and the device engine, with validity per slice and the cell weights shared.
  The lags are enumerated once for the whole batch. A schedule names through
  `batch_shares_lag_geometry` whether the transform holds every slice's columns of a slab pair
  together: a curved schedule's per-lag geodesic frame and transport matrices are the same for every
  slice, while a flat schedule's are a displacement and a bin, and it takes its slices one at a time.
  The same entry takes a `ScatteredModesSchedule` for a fixed set of stations sampled over time.

### Spectra and fluxes

- `isotropic_spectrum` in one, two and three dimensions with verified constants; `gridded_spectrum`
  from the whole lag space, exact on a complete periodic grid and an unbiased estimate with missing
  cells, with the Blackman–Tukey estimate on bounded directions, tapers (`NoTaper`, `Bartlett`,
  `GaussianTaper`) and missing-lag policies; `shell_average`, `shell_spectrum`.
- `helmholtz_spectra(L2, T2, k)` by `J₀`/`J₂`; on a sphere `isotropic_spectrum(sf, geometry, lmax)` and
  `helmholtz_spectra(L2, T2, geometry, lmax; variance)` by Legendre and Wigner-d orthogonality.
- Third-order flux companions with their boundary terms: `spectral_flux` on `S3SFType`, `L3SFType`
  (with `S3`) and `MixedSFType{1,0,2}`, and `enstrophy_flux` from the velocity's advective structure
  function; the `J₁` route on cross-field moments.
- `covariance`, `covariance_matrix` with a positive-definiteness check.
- Regularised fits (issue #18): `SpectrumForwardModel`, `HelmholtzForwardModel`, `FluxForwardModel`;
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
- The device transform builds each monomial for every slab in one broadcast and transforms the slabs
  in one batch, and writes the spectra in the order the binning stage reads them, so assembling its
  input is a reshape.
- The device binning kernel launches each slab pair over its own lag box, which a schedule reports
  through `uniform_lag_box`. On a sphere each row pair's box narrows with the largest bin edge, so
  the kernel's cost falls as the sweep narrows.

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
| `backend=:distributed` | `backend=DistributedBackend()` | `using Distributed` loads `StructureFunctionsDistributedExt` |
| No GPU support | `backend=GPUBackend(...)` | New feature; `using KernelAbstractions` loads `StructureFunctionsKernelAbstractionsExt` |
| Implicit auto-selection | `backend=AutoBackend()` | Explicit type; now default |

### Performance Improvements

- **Type-stable dispatch**: No runtime penalty for backend selection
- **Eliminated dynamic Val(N)** construction in hot paths
- **Thread-local reductions**: private partial reductions in the threaded backend

### New Public API

```julia
# New typed backends
backends = [SerialBackend(), ThreadedBackend(), DistributedBackend(), GPUBackend(...), AutoBackend()]

# calculate_structure_function signature
calculate_structure_function(sf_type, x, u, bins[, CT][, OT]; backend=AutoBackend())
```

### Docstrings & Documentation

- **Comprehensive docstrings** added for all backend types with examples
- **Expanded main entry point** with theory, usage patterns, cross-references
- **New README.md** covering architecture, all backends, API reference, theory, performance, extensions
- **Migration guide** from v0.2 → v0.3 with before/after code examples

### Test Suite Enhancements

- **JET stability audit** expanded; runs with a `target_modules` filter
- **Threading test suite** added (`test_threads.jl`) validating ThreadedBackend on multi-threaded Julia

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
- **Code organization**: Extensions separated (`StructureFunctionsDistributedExt`, `StructureFunctionsKernelAbstractionsExt`, `StructureFunctionsOhMyThreadsExt`)

### Removed/Deprecated

- **Removed**: Old symbol-based backend dispatch code paths
- **Archived**: Alternative calculation implementations (`AlternateCalculations*.jl`, etc.)
- **Deprecated**: Implicit backend selection via kwargs (now must specify backend explicitly)

### Known Issues & Future Work

- **Block E (NUFFT)**: Spectral extensions partially integrated; full NUFFT modernization deferred to v0.4

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
