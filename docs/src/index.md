```@meta
CurrentModule = StructureFunctions
```

# StructureFunctions.jl

**Structure functions of turbulent and spatially varying fields, exact wherever the data allow, on
point lists, channel bundles, grids and the sphere; with the spectra, fluxes and fits that follow.**

## Features

- **Operators** of any order: the longitudinal, transverse and full increments, signed transverse
  components with a stated convention, scalar and mixed velocity–scalar moments, cross-channel
  advective moments, the whole increment moment tensor.
- **Point lists** `(D, N)` on the serial, threaded, distributed, MPI and GPU backends, with cell
  culling, pair weights, joint (distance × value) and (distance × angle) histograms, the six isotropic
  invariants and the Helmholtz split in one pass, and batches over auxiliary axes.
- **Grids** through FlowGeometries: uniform, stretched and lat-lon grids, masks at any order, and an
  exact transform engine that computes every polynomial operator at every lag at once — on the CPU or
  a device — plus a direct sweep to check it against.
- **The sphere**: parallel-transported pair frames, `SphericalDistance(R)`, the zonal transform, and a
  harmonic route with Legendre and Wigner-d kernels.
- **One-dimensional lists** in `O(N log N)` and **scattered points by non-uniform FFT** (soft bins,
  documented as such).
- **Spectra** from structure functions in one, two and three dimensions and on the sphere, exact on
  grids through the lag space; the Helmholtz `E`/`B` split; **fluxes** from third-order functions with
  their boundary terms; **regularised fits** of spectra and fluxes with posterior covariances.
- **Exact laws**: four-fifths, four-thirds and Yaglom inversions.
- Every route checked against an independent oracle; every backend against every other.

## Quick start

```julia
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LogBinEdges
using ComputationalBackends: ComputationalBackends as CB

x = rand(2, 2048) .* 1.0e4          # (D, N) coordinates
u = randn(2, 2048)                  # (D, N) velocity components
bins = LogBinEdges(collect(exp10.(range(log10(50.0), log10(5.0e3); length = 41))))

sf = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.AutoBackend())
sf.distance, sf.values              # the bin edges and the averaged ⟨δu_L²⟩ per bin

res = SFC.calculate_structure_functions_single_pass(x, u, bins)
res.L2, res.T2, res.helmholtz       # all six invariants and the Helmholtz split in one pass
```

## Where to next

- [Theory](theory.md) — the definitions, conventions and relations the code implements.
- [Walkthrough](walkthrough.md) — one analysis end to end, on fields whose answers are known.
- [Architecture](architecture.md) — operator × schedule × backend, and how a call becomes a kernel.
- [Backends](backends.md) and [GPU acceleration](gpu.md) — where each calculation runs.
- [Extensions](extensions.md) — what each optional package adds.
- [Exact laws](khm.md), [Validation](validation.md), [API reference](api/operators.md).

## Installation

```julia
using Pkg
Pkg.add("StructureFunctions")
```

The package depends on `ComputationalBackends` for the backend types; bring the trigger packages of
the [extensions](extensions.md) you need (`OhMyThreads`, `KernelAbstractions` + `CUDA`, `FFTW`,
`FlowGeometries`, `Bessels`, `NUFSHT`, `NonuniformFFTs` or `FINUFFT`, `LsqFit`). The algorithm tags come
from `SpectralBackends`, a dependency.
