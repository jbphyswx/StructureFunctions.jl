```@meta
CurrentModule = StructureFunctions
```

# StructureFunctions.jl

**High-performance structure function calculations for turbulence and spatial correlation analysis.**

StructureFunctions.jl computes structure functions (SFs) from scattered or gridded data,
characterizing the spatial correlations and scaling properties of turbulent / spatially-varying
fields. It is optimized for multi-dimensional data with a typed backend system supporting
**serial, threaded, distributed, and GPU** execution, and a fused **single-pass** path that
produces the six isotropic invariants (plus the Helmholtz rotational/divergent decomposition) in
one O(N²) pair pass.

## Features

- **Operators**: 1st/2nd/3rd-order; longitudinal & transverse projections in 1D/2D/3D.
- **Single pass**: six isotropic invariants `S2, L2, T2, S3, L3, L1T2` (+ `helmholtz`) in one pass.
- **2D joint binning**: distance × value histograms (`StructureFunction2DSumsAndCounts`).
- **Typed backends**: `SerialBackend`, `ThreadedBackend`, `DistributedBackend`, `GPUBackend`, `AutoBackend`.
- **Native batching**: vectorized over a trailing `(D, N, T)` axis (CPU batch-leading SoA / GPU batch kernels).
- **In-place API**: pre-allocated mutating drivers for zero-allocation loops.
- **Fast bins**: O(1) `LinearBinEdges` / `LogBinEdges` digitizers; `Float32` & `Float64` first-class.
- **Type-stable & validated**: JET / Aqua, serial≡threaded≡distributed≡GPU parity tests.

## Quick start

```julia
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LogBinEdges

x = rand(2, 2048) .* 1.0e4          # (D, N) coordinates
u = randn(2, 2048)                  # (D, N) velocity components
bins = LogBinEdges(collect(exp10.(range(log10(50.0), log10(5.0e3); length = 41))))

# Second-order longitudinal SF, averaged S₂(r):
sf = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.AutoBackend())
sf.distance, sf.values

# All six invariants (+ Helmholtz) in one pass:
res = SFC.calculate_structure_functions_single_pass(x, u, bins)
res.L2, res.T2, res.helmholtz
```

## Where to next

- **[Theory](theory.md)** — what structure functions are and why they matter.
- **[Architecture](architecture.md)** — operator × result-container design and dispatch.
- **[Backends](backends.md)** — serial / threaded / distributed / GPU, and when to use each.
- **[GPU Acceleration](gpu.md)** — GPU kernels, batching, and `GPUSFWorkspace`.
- **[Extensions](extensions.md)** — the optional weakdep-gated integrations.
- **[Examples](examples.md)** — runnable scripts and featured snippets.
- **[API Reference](api.md)** — the complete public surface.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/jbphyswx/StructureFunctions.jl.git")
```

Optional backends/visualization load via package extensions when you bring their trigger packages
(`OhMyThreads`, `Distributed`, `KernelAbstractions` + `CUDA`, `MPI`, `CairoMakie`);
see [Extensions](extensions.md).
