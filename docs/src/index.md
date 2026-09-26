```@meta
CurrentModule = StructureFunctions
```

# StructureFunctions.jl

StructureFunctions computes pair statistics of vector and scalar field increments. Inputs include point lists, structured grids, spherical coordinates, and batches of snapshots. The package also implements spectra, fluxes, regularized fits, and homogeneous-isotropic exact-law diagnostics.

Start with [Getting started](getting_started.md), then use [Data and geometry](data.md) to choose layouts, masks, weights, and bin edges. [Execution backends](backends.md) explains hardware selection and reusable storage. [Recipes](examples.md) covers common calculations.

For interpretation, read [Mathematical definitions](theory.md), [Spectra, fluxes, and fitting](spectra.md), and [Exact laws](khm.md). The [validation guide](validation.md) distinguishes numerical checks from statistical assumptions and performance measurements.

## Installation

Julia 1.12 or later is required.

```julia
using Pkg
Pkg.add(url="https://github.com/jbphyswx/StructureFunctions.jl.git")
```

[Optional extensions](extensions.md) provide threading, GPU execution, grids, and transform or fitting providers. Load only the providers needed for the calculation.
