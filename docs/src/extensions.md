# Extensions

The core package computes structure functions of arrays and multi-fields on the serial backend
and depends on nothing but `ComputationalBackends`, `Distances`, `LinearAlgebra`, `ProgressMeter`,
`SpectralBackends` and `StaticArrays`. Everything else — parallel execution, transforms, grids,
spectral providers, fits — is a package extension that loads when its trigger packages are loaded. A
method that needs an extension which is not loaded throws an `ArgumentError` naming the package to
load; nothing falls back silently.

The algorithm tags — `AutoSpectralBackend()`, `DirectSumSpectralBackend()`,
`FastFourierTransformSpectralBackend()`, `NUFSHTSpectralBackend()` and the non-uniform FFT tags — come
from `SpectralBackends`, a dependency, so they are always available (also as
`StructureFunctions.SpectralBackends`); the two provider-specific non-uniform FFT tags,
`NonuniformFFTsSpectralBackend(; tolerance)` and `FINUFFTSpectralBackend(; tolerance)`, are this
package's own. A tag whose transform is not loaded refuses by name.

| Extension | Load | Adds |
|---|---|---|
| `StructureFunctionsOhMyThreadsExt` | `using OhMyThreads` | `ThreadedBackend()` on every CPU path: point lists, multi-fields, gridded sweeps and transforms, tensors, the sorted line route |
| `StructureFunctionsDistributedExt` | `using Distributed` | `DistributedBackend()` for point lists, multi-fields and tensors across worker processes |
| `StructureFunctionsMPIExt` | `using MPI` | the MPI backend for point lists |
| `StructureFunctionsKernelAbstractionsExt` | `using KernelAbstractions` | `GPUBackend(device)` kernels: point lists, joint histograms, single-pass invariants, batches over auxiliary axes, tensors, multi-fields; `KernelAbstractions.CPU()` runs them on the host |
| `StructureFunctionsCUDAExt` | `using CUDA` with `KernelAbstractions` | CUDA-specific launch configuration and shared-memory routes for the device kernels |
| `StructureFunctionsAbstractFFTsExt` | `using FFTW` (any `AbstractFFTs` implementation) | the transform engine: every polynomial operator on every separable schedule, masks, weights, joint histograms over angle, moment tensors, `gridded_spectrum`, and the `Auto` cost model |
| `StructureFunctionsAbstractFFTsKernelAbstractionsExt` | the one above with `KernelAbstractions` | the transform engine on a device: monomial transforms through the device's `AbstractFFTs` and one lag kernel for the binning |
| `StructureFunctionsNonuniformFFTsExt` | `using NonuniformFFTs` | the NonuniformFFTs provider of the soft-binned route for scattered points on a `ScatteredModesSchedule` (`NonuniformFFTsSpectralBackend`): type-1 non-uniform FFTs of the masked, weighted monomials |
| `StructureFunctionsNonuniformFFTsKernelAbstractionsExt` | the one above with `KernelAbstractions` | the same transforms on the device the points live on |
| `StructureFunctionsFINUFFTExt` | `using FINUFFT` | the FINUFFT provider of the same route (`FINUFFTSpectralBackend`), two real monomials per complex transform in one batched plan |
| `StructureFunctionsFINUFFTKernelAbstractionsExt` | the one above with `KernelAbstractions` (and `CUDA`) | cuFINUFFT for points on a CUDA device |
| `StructureFunctionsNUFSHTExt` | `using NUFSHT` | the fast spherical harmonic pseudo-coefficients behind the harmonic route on `HarmonicNodes` and `harmonic_spectra` |
| `StructureFunctionsFlowGeometriesExt` | `using FlowGeometries` | the grid entries: `calculate_structure_function(sf, grid, u, bins[, spectral_backend]; …)`, `calculate_structure_function_tensor(order, grid, …)`, `cell_measure(grid)`, routing a grid's axis types to the uniform, rectilinear, zonal or scattered schedule |
| `StructureFunctionsBesselsExt` | `using Bessels` | the Bessel functions `J₀`–`J₃`: the two-dimensional isotropic kernel, the flux relations, the Helmholtz spectra and forward models |
| `StructureFunctionsLsqFitExt` | `using LsqFit` | the bounded Levenberg–Marquardt fit behind `SegmentedPowerLaw` |

## Examples

Threads:

```julia
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads

res = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.ThreadedBackend())
```

A device:

```julia
using KernelAbstractions, CUDA
res = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.GPUBackend(CUDA.CUDABackend()))
```

`CB.GPUBackend(KernelAbstractions.CPU())` runs the same kernels on the host, which is how the test
suite covers them without a device.

The transform on a grid:

```julia
using FFTW, FlowGeometries
using SpectralBackends: SpectralBackends as SB
sf = calculate_structure_function(SFT.L3SFType(), grid, u, bins, SB.FastFourierTransformSpectralBackend())
```

`SB.AutoSpectralBackend()` costs the transform and the direct sweep and takes the cheaper; the direct
sweep is the default when no tag is given.

## Not provided

The package reads arrays, multi-fields and `FlowGeometries` grids. File formats, preprocessing and
metadata belong to the caller.

## Related pages

- [Backends](backends.md)
- [GPU acceleration](gpu.md)
- [Architecture](architecture.md)
