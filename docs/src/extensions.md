# Extensions

The core package computes structure functions of arrays and multi-fields on the serial backend
and depends on nothing but `ComputationalBackends`, `Distances`, `LinearAlgebra`,
`PrecompileTools`, `SpectralBackends` and `StaticArrays`. Everything else —
parallel execution, transforms, grids, spectral providers, fits — is a package extension that loads
when its trigger packages are loaded. A
method that needs an extension which is not loaded throws an `ArgumentError` naming the package to
load: an explicit request never falls back silently. The `Auto` choices are the exception, and only
because choosing is what they are for — `AutoBackend()` runs serially when the OhMyThreads extension
is absent and `AutoSpectralBackend()` sweeps the lags when no transform is loaded, both without
complaint.

The algorithm tags — `AutoSpectralBackend()`, `DirectSumSpectralBackend()`,
`FastFourierTransformSpectralBackend()`, `NUFSHTSpectralBackend()` and the non-uniform FFT tags — come
from `SpectralBackends`, a dependency, so they are always available (also as
`StructureFunctions.SpectralBackends`); the two provider-specific non-uniform FFT tags,
`NonuniformFFTsSpectralBackend(; tolerance)` and `FINUFFTSpectralBackend(; tolerance)`, are this
package's own. A tag whose transform is not loaded refuses by name.

| Extension | Load | Adds |
|---|---|---|
| `StructureFunctionsOhMyThreadsExt` | `using OhMyThreads` | `ThreadedBackend()` on every CPU path: point lists, multi-fields, gridded sweeps and transforms, tensors, the sorted line route |
| `StructureFunctionsDistributedExt` | `using Distributed` | `DistributedBackend()` for every entry family across worker processes: point lists, multi-fields, tensors, single-pass invariants, the batch drivers and the gridded sweeps |
| `StructureFunctionsMPIExt` | `using MPI` | `MPIBackend()` for every entry family across ranks, each rank taking a share and the partials reduced with `Allreduce!` |
| `StructureFunctionsKernelAbstractionsExt` | `using KernelAbstractions` | `GPUBackend(device)` kernels: point lists, joint histograms, single-pass invariants, batches over auxiliary axes, tensors, multi-fields, the sorted line, the gridded direct lag sweep and the harmonic direct sum; `KernelAbstractions.CPU()` runs them on the host |
| `StructureFunctionsCUDAExt` | `using CUDA` with `KernelAbstractions` | CUDA-specific launch configuration and shared-memory routes for the device kernels |
| `StructureFunctionsAbstractFFTsExt` | `using FFTW` (any `AbstractFFTs` implementation) | the transform engine: every polynomial operator on every separable schedule, masks, weights, joint histograms over angle, moment tensors, the six single-pass invariants from one set of forward transforms, `gridded_spectrum`, and the `Auto` cost model |
| `StructureFunctionsAbstractFFTsKernelAbstractionsExt` | the one above with `KernelAbstractions` | the transform engine on a device: monomial transforms through the device's `AbstractFFTs` and one lag kernel for the binning |
| `StructureFunctionsNonuniformFFTsExt` | `using NonuniformFFTs` | the NonuniformFFTs provider of the soft-binned route for scattered points on a `ScatteredModesSchedule` (`NonuniformFFTsSpectralBackend`): type-1 non-uniform FFTs of the masked, weighted monomials |
| `StructureFunctionsNonuniformFFTsKernelAbstractionsExt` | the one above with `KernelAbstractions` | the same transforms on the device the points live on |
| `StructureFunctionsFINUFFTExt` | `using FINUFFT` | the FINUFFT provider of the same route (`FINUFFTSpectralBackend`), two real monomials per complex transform in one batched plan |
| `StructureFunctionsFINUFFTKernelAbstractionsExt` | the one above with `KernelAbstractions` (and `CUDA`) | cuFINUFFT for points on a CUDA device |
| `StructureFunctionsNUFSHTExt` | `using NUFSHT` | the fast spherical harmonic pseudo-coefficients behind the harmonic route on `HarmonicNodes` and `harmonic_spectra` |
| `StructureFunctionsFlowGeometriesExt` | `using FlowGeometries` | the grid entries: `calculate_structure_function(sf, grid, u, bins[, spectral_backend][, CT][, OT]; …)`, `calculate_structure_functions_single_pass(grid, u, bins[, spectral_backend][, CT][, OT]; …)`, `calculate_structure_function_tensor(order, grid, …)`, their slice-batch forms, `cell_measure(grid)`, routing a grid's axis types to the uniform, rectilinear, zonal or scattered schedule |
| `StructureFunctionsBesselsExt` | `using Bessels` | the Bessel functions `J₀`–`J₃`: the two-dimensional isotropic kernel, the flux relations, the Helmholtz spectra and forward models |
| `StructureFunctionsLsqFitExt` | `using LsqFit` | the bounded Levenberg–Marquardt fit behind `SegmentedPowerLaw` |

## Examples

Threads:

```@example extensions
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads
using Random: Random
Random.seed!(17)

x = Random.rand(2, 32)
u = Random.randn(2, 32)
bins = range(0.0, 0.5; length = 21)
res = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.ThreadedBackend())
res.values[1:4]
```

A device. This block needs a CUDA device, so the documentation build shows it without running it:

```julia
using KernelAbstractions: KernelAbstractions, CUDA: CUDA
res = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.GPUBackend(CUDA.CUDABackend()))
```

`CB.GPUBackend(KernelAbstractions.CPU())` runs the same kernels on the host, which is how the test
suite covers them without a device.

The transform on a grid:

```@example extensions
using FFTW: FFTW
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB

geo = FG.Geometry.CartesianGeometry()
grid = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.1, length = 8),
                               range(0.0, step = 0.1, length = 8))
ug = randn(2, 8, 8)
gbins = collect(range(0.0, 0.8; length = 9))
sf = SFC.calculate_structure_function(SFT.L3SFType(), grid, ug, gbins,
                                      SB.FastFourierTransformSpectralBackend(), UInt64)
sf.values
```

`SB.AutoSpectralBackend()`, the default when no tag is given, costs the transform and the direct sweep
and takes the cheaper.

## Not provided

The package reads arrays, multi-fields and `FlowGeometries` grids. File formats, preprocessing and
metadata belong to the caller.

## Related pages

- [Backends](backends.md)
- [GPU acceleration](gpu.md)
- [Architecture](architecture.md)
