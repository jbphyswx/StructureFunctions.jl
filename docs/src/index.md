```@meta
CurrentModule = StructureFunctions
```

# StructureFunctions.jl

StructureFunctions computes structure functions: averages over pairs of points of a polynomial in the increment
`δu = u(x + r) − u(x)`, binned in the separation `|r|`. It takes vector and scalar fields on point lists, batches of
snapshots, structured grids and the sphere, and derives spectra, fluxes, covariances, fits and the third-order exact
laws from the results.

![Second-order structure function of a synthetic 2-D field](assets/sf_kolmogorov.png)

## What it computes

| Capability | Where |
|---|---|
| Second- and third-order, longitudinal, transverse, scalar, mixed and cross-field moments | [Mathematical definitions](theory.md) |
| The six isotropic invariants and the Helmholtz split in one pass over the pairs | [Calculations](examples.md#Six-invariants-in-one-pass) |
| Joint histograms in separation and operator value, or separation and angle | [Calculations](examples.md#Joint-histograms) |
| Moment tensors of any order | [Calculations](examples.md#Moment-tensors) |
| Batches of snapshots over shared or varying positions | [Calculations](examples.md#Batches-of-snapshots) |
| Exact routes: cell culling, the sorted line, grid lag sweeps and grid transforms | [Calculations](examples.md#Exact-routes) |
| Scattered points through a non-uniform FFT | [Calculations](examples.md#Scattered-points-through-a-non-uniform-FFT) |
| Pair weights, masks, multi-fields and the sphere | [Data and geometry](data.md) |
| Spectra, Helmholtz spectra, covariances, spectral fluxes and fits | [Spectra, fluxes and fitting](spectra.md) |
| The four-fifths, four-thirds and Yaglom laws | [Exact laws](khm.md) |
| Serial, threaded, Distributed, MPI and GPU execution | [Execution backends](backends.md) |

## Installation

Julia 1.12 or later is required.

```julia
using Pkg
Pkg.add(url="https://github.com/jbphyswx/StructureFunctions.jl.git")
```

[Extensions](extensions.md) load with the packages they need: threading with `OhMyThreads`, GPUs with
`KernelAbstractions` and `CUDA`, grids with `FlowGeometries`, transforms with an `AbstractFFTs` provider such as
`FFTW`, and the spectral kernels with `Bessels`.

Start with [Getting started](getting_started.md).
