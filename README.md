# StructureFunctions.jl

[Documentation](https://jbphyswx.github.io/StructureFunctions.jl/dev/) · [DOI](https://doi.org/10.5281/zenodo.14945669)

StructureFunctions computes statistics of field increments between pairs of positions. It supports vector and scalar fields, point lists, structured grids, spherical geometry, and batches of snapshots. It also provides spectrum and flux transforms, fitting routines, and homogeneous-isotropic exact-law diagnostics.

Julia 1.12 or later is required.

## Installation

```julia
using Pkg: Pkg; Pkg.add(url="https://github.com/jbphyswx/StructureFunctions.jl.git")
```

## Calculate a structure function

Columns contain points; rows contain vector components. This example computes the mean squared longitudinal velocity increment in each separation bin.

```julia
using Random: Random
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionSumsAndCounts
using ComputationalBackends: SerialBackend

rng = Random.MersenneTwister(42)
x = Random.rand(rng, 2, 32)
u = Random.randn(rng, 2, 32)
bins = range(0.0, 1.5; length=9)
sf = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins;
    backend=SerialBackend())
sf.values
```

`sf.distance` contains bin edges. `sf.values` contains one mean per interval `(left, right]`; an empty bin has value `NaN`. Coordinate and velocity units are supplied by the caller.

## Accumulate snapshots

Raw sums and counts can be combined before taking a mean; pass the count type and the result type positionally after the bins. Mutating calculations add to their output buffers.

```julia
raw = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, UInt64, StructureFunctionSumsAndCounts;
    backend=SerialBackend())
sums = copy(raw.sums)
counts = copy(raw.counts)
SFC.calculate_structure_function!(sums, counts, SFT.L2SFType(), x, u, bins;
    backend=SerialBackend())
@assert counts == 2 .* raw.counts
```

For shared positions, a field of shape `(D, N, snapshots...)` computes a separate histogram for each snapshot. Reusable workspaces retain preparation and scratch storage across repeated calculations.

## Mathematical quantities

| Quantity | Pair value |
|---|---|
| `S2SFType()` | `‖δu‖²` |
| `L2SFType()` | `δu_L²` |
| `T2SFType()` | `‖δu_T‖²`, summed over transverse components |
| `S3SFType()` | `δu_L ‖δu‖²` |
| `L3SFType()` | `δu_L³` |
| Scalar and mixed operators | Scalar increments and velocity–scalar products |
| Tensor calculation | Products of specified increment components |

A single-pass calculation shares pair geometry across six second- and third-order invariants. Joint histograms resolve separation and operator value or separation angle. Spherical calculations use transported pair frames.

## Methods and execution

Serial, threaded, Distributed, MPI, and GPU execution are selected through `ComputationalBackends`. Optional packages enable the corresponding extensions. Numerical methods include direct pairs, exact culling, sorted-line polynomial moments, direct gridded reductions, FFT correlations, and explicitly selected nonuniform or spherical transforms. Applicability depends on the operator and geometry; see the [backend guide](https://jbphyswx.github.io/StructureFunctions.jl/dev/backends/).

Load `OhMyThreads` for threaded execution and `KernelAbstractions, CUDA` for CUDA execution. Grid adapters use `FlowGeometries`; FFT methods use an `AbstractFFTs` provider such as `FFTW`. Unstructured inputs must contain valid points. Grid calculations support masks and invalid cells.

Spectrum and flux routines specify their dimensionality, normalization, and sampling assumptions in the [scientific analysis guide](https://jbphyswx.github.io/StructureFunctions.jl/dev/spectra/). A masked pair average does not by itself guarantee an unbiased estimate of an unobserved field's spectrum.

## Examples and development

See [`examples/`](examples/) for small executable examples. Performance experiments, scientific validation, and figure generation have separate entry points; they do not run during the documentation build.

On clima, GPU work and substantial CPU calculations require Slurm. Package development instructions are in [`AGENTS.md`](AGENTS.md).
