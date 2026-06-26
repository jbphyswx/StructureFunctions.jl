# StructureFunctions.jl v0.3.0

[![Docs (stable)][docs-stable-img]][docs-stable-url] [![Docs (dev)][docs-dev-img]][docs-dev-url] [![DOI][zenodo-img]][zenodo-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jbphyswx.github.io/StructureFunctions.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jbphyswx.github.io/StructureFunctions.jl/dev/
[zenodo-img]: https://zenodo.org/badge/734119226.svg
[zenodo-latest-url]: https://doi.org/10.5281/zenodo.14945669

**High-performance structure function calculations for turbulence and spatial correlation analysis.**

StructureFunctions.jl computes structure functions (SFs) from scattered data, characterizing spatial correlations and scaling properties of turbulent/spatially-varying fields. Optimized for multi-dimensional data with typed backends supporting serial, threaded, distributed, and GPU execution.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Backends](#backends)
- [API Reference](#api-reference)
- [Theory & References](#theory--references)
- [Performance](#performance)
- [Extensions](#extensions)
- [Migration from v0.2](#migration-from-v02)
- [Examples](#examples)

## Features

- **Structure Functions**: 1st, 2nd, 3rd order; longitudinal & transverse projections in 1D, 2D, 3D
- **In-place Mutating API**: Pre-allocated mutating functions (`calculate_structure_function!`) for zero-allocation loops (O(n_threads) multi-threaded chunked allocations)
- **2D Joint-Probability Binning**: Natively accumulates both exact sums and contribution counts across distance and structure function value increment bins (`StructureFunction2DSumsAndCounts`)
- **Typed Backend System**: Serial, Threaded, Distributed, GPU, Auto — choose your parallelization strategy
- **Type-Stable Dispatch**: No runtime overhead from symbolic dispatch; all paths validated with JET
- **Extensible Architecture**: Optional extensions for parallelization and GPU acceleration
- **Production Ready**: Comprehensive test coverage, numerical validation, performance benchmarking
- **Modern Julia**: Julia 1.12+ with qualified imports and explicit type annotations

## Quick Start

```julia
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT

# 2D data: 3 points
x = ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
u = ([1.0, 1.1, 1.2], [0.0, 0.05, 0.1])

# Distance bins (physical units)
bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

# Calculate 2nd-order longitudinal SF
sf_type = SFT.LongitudinalSecondOrderStructureFunctionType()
result = SFC.calculate_structure_function(sf_type, x, u, bins)

# result.values contains the SF values for each bin
println("SF values: ", result.values)

# Speed it up with threading (if available)
using Base.Threads
if nthreads() > 1
    result_threaded = SFC.calculate_structure_function(
        sf_type, x, u, bins;
        backend=SFC.ThreadedBackend()
    )
end
```

### Pre-allocated In-place Calculation

For high-performance loops (e.g. over timesteps), you can pre-allocate memory buffers and run mutating calculations with zero heap allocation:

```julia
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT

x = ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
u = ([1.0, 1.1, 1.2], [0.0, 0.05, 0.1])
bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
sf_type = SFT.L2SFType()

# Pre-allocate output arrays
n_bins = length(bins)
sums = zeros(Float64, n_bins)
counts = zeros(Float64, n_bins)

# Compute in-place (accumulates into provided buffers)
SFC.calculate_structure_function!(sums, counts, sf_type, x, u, bins; backend=SFC.ThreadedBackend())

# Obtain structure function values via division
sf_values = sums ./ counts
```


## Architecture

### Operator Types ✕ Result Container Pattern

The v0.3.0 API separates **operators** (structure function definitions) from **result containers** (computed outcomes):

```
AbstractStructureFunctionType (operators)
  ├── LongitudinalSecondOrderStructureFunctionType
  ├── TransverseSecondOrderStructureFunctionType
  ├── LongitudinalThirdOrderStructureFunctionType
  └── ... (3+ other variants)

StructureFunction (result container)
  ├── operator::AbstractStructureFunctionType
  ├── distance_bins::AbstractVector
  ├── values::AbstractVector
  └── order::Int
```

This split ensures:
- Clear semantics: operators are **inputs**, containers are **outputs**
- Type stability: dispatch happens at compilation time
- Extensibility: custom operators and containers are easy to add

### Backend Dispatch System

```
calculate_structure_function(sf_type, x, u, bins; backend=AutoBackend())
    ↓
_dispatch_execution_backend(backend, ...)
    ├── SerialBackend       → serial_calculate_structure_function
    ├── ThreadedBackend     → threaded_calculate_structure_function (from OhMyThreadsExt)
    ├── DistributedBackend  → parallel_calculate_structure_function (from DistributedExt)
    ├── GPUBackend(b)       → gpu_calculate_structure_function (from GPUExt)
    └── AutoBackend         → (tries distributed → threaded → serial)
```

All code paths produce **numerically identical results** (validated by intensive test suite).

## Backends

### SerialBackend (Default Reference)

Single-threaded CPU execution. Use when:
- Debugging or validating calculations
- Data is small
- Deterministic execution is required

```julia
result = SFC.calculate_structure_function(sf_type, x, u, bins)  # Defaults to Serial
result = SFC.calculate_structure_function(sf_type, x, u, bins; 
                                        backend=SFC.SerialBackend())
```

**Performance**: O(N²) pairwise distance/SF evaluations.  
**Memory**: O(N + B) where N = points, B = distance bins.

### ThreadedBackend (Multi-CPU)

Multi-threaded execution using [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl).

```julia
using Base.Threads

result = SFC.calculate_structure_function(sf_type, x, u, bins;
                                        backend=SFC.ThreadedBackend())
```

- **Prerequisites**: `Threads.nthreads() > 1`, OhMyThreads.jl loaded (extension auto-loads)
- **Thread-local reductions**: No locks or atomic operations; no `threadid()` buffer indexing
- **Outer-loop scheduling**: Round-robin partition of index `i` for O(N²) pair loops (balanced
  when inner work per `i` is `(N - i)`; contiguous chunks would skew load ~× number of threads)
- **Speedup**: Near-linear at low thread count for large N; memory bandwidth limits scaling beyond

### DistributedBackend (Multi-Process/Cluster)

Multi-worker execution using [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/).

```julia
using Distributed: addprocs

addprocs(4)  # Or specify SSH workers, etc.

result = SFC.calculate_structure_function(sf_type, x, u, bins;
                                        backend=SFC.DistributedBackend())
```

- **Prerequisites**: Workers launched via `addprocs()` or similar
- **Communication overhead**: One `@distributed` reduction loop
- **Ideal for**: Large datasets, compute clusters

### GPUBackend (GPU Acceleration)

GPU execution via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

```julia
using KernelAbstractions as KA

# NVIDIA GPU (after loading CUDA.jl)
using CUDA
result = SFC.calculate_structure_function(sf_type, x, u, bins;
                                        backend=SFC.GPUBackend(CUDA.CUDABackend()))

# AMD GPU (after loading AMDGPU.jl)
using AMDGPU
result = SFC.calculate_structure_function(sf_type, x, u, bins;
                                        backend=SFC.GPUBackend(AMDGPU.ROCBackend()))

# CPU backend for testing (no GPU required)
result = SFC.calculate_structure_function(sf_type, x, u, bins;
                                        backend=SFC.GPUBackend(KA.CPU()))
```

- **Ideal for**: Large datasets (few×10³+ points) where GPU memory is sufficient
- **Docs**: [`docs/src/gpu.md`](docs/src/gpu.md) — workspace, batch driver, testing tiers

### AutoBackend (Recommended Default)

Automatic selection based on environment:

```julia
result = SFC.calculate_structure_function(sf_type, x, u, bins;
                                        backend=SFC.AutoBackend())

# Selection order:
# 1. Distributed  (if nworkers() > 1)
# 2. Threaded     (if nthreads() > 1)
# 3. Serial       (fallback)
```

## API Reference

### Main Entry Points

**1. Standard Allocating API:**

```julia
calculate_structure_function(sf_type::AbstractStructureFunctionType,
                            x::Union{Tuple, Matrix},
                            u::Union{Tuple, Matrix},
                            distance_bins::AbstractVector{<:Tuple};
                            backend=SerialBackend(),
                            output_type=StructureFunction,   # or StructureFunctionSumsAndCounts for raw
                            distance_metric=Euclidean(),
                            verbose=true,
                            show_progress=true) → StructureFunction  # (or StructureFunctionSumsAndCounts)
```

**2. 2D Joint-Probability Allocating API:**

```julia
calculate_structure_function(sf_type::AbstractStructureFunctionType,
                            x::Union{Tuple, Matrix},
                            u::Union{Tuple, Matrix},
                            distance_bins::AbstractVector{<:Tuple},
                            value_bins::AbstractVector;
                            backend=SerialBackend(),
                            distance_metric=Euclidean(),
                            verbose=true,
                            show_progress=true) → StructureFunction2DSumsAndCounts
```

**3. In-place Mutating API (Zero-Allocation):**

```julia
calculate_structure_function!(sums::AbstractVector,
                             counts::AbstractVector,
                             sf_type::AbstractStructureFunctionType,
                             x::Union{Tuple, Matrix},
                             u::Union{Tuple, Matrix},
                             distance_bins::AbstractVector;
                             backend=SerialBackend(),
                             distance_metric=Euclidean(),
                             verbose=true,
                             show_progress=true) → Nothing
```

**4. 2D Joint-Probability Mutating API (Zero-Allocation):**

```julia
calculate_structure_function!(sums_2d::AbstractMatrix,
                             counts_2d::AbstractMatrix,
                             sf_type::AbstractStructureFunctionType,
                             x::Union{Tuple, Matrix},
                             u::Union{Tuple, Matrix},
                             distance_bins::AbstractVector,
                             value_bins::AbstractVector;
                             backend=SerialBackend(),
                             distance_metric=Euclidean(),
                             verbose=true,
                             show_progress=true) → Nothing
```

*Note: The mutating APIs accumulate (`+=` and `.+=`) directly into the provided output buffers. The caller is responsible for pre-zeroing the arrays.*

### Operator Types

All inherit from `AbstractStructureFunctionType`. Instantiate with `()` or use shorthands:

```julia
SFT.LongitudinalSecondOrderStructureFunctionType()    # 2nd order, longitudinal
SFT.TransverseSecondOrderStructureFunctionType()      # 2nd order, transverse
SFT.LongitudinalThirdOrderStructureFunctionType()     # 3rd order, longitudinal
# ... shorthands: L2SFType, T2SFType, L3SFType, T3SFType, S2SFType, S3SFType
```

Each operator is **callable** (functors):
```julia
sf_op = SFT.L2SFType()
sf_op(du, rhat)  # Computes L2SF increment value
```

### Result Containers

**1. 1D Structure Function Container (`StructureFunction`):**

```julia
struct StructureFunction{FT, OT, BT, VT} <: AbstractStructureFunction
    operator::OT                   # AbstractStructureFunctionType
    distance_bins::BT              # AbstractVector of (r_min, r_max)
    values::VT                     # AbstractVector{FT} — computed SF
    order::Int                     # 1, 2, 3, ...
end
```

**2. 2D Joint-Probability Container (`StructureFunction2DSumsAndCounts`):**

```julia
struct StructureFunction2DSumsAndCounts{FT, OT, BT, VT, MT, CT} <: AbstractStructureFunction
    operator::OT                   # AbstractStructureFunctionType
    distance_bins::BT              # AbstractVector of distance bin edges
    value_bins::VT                 # AbstractVector of value bin edges
    sums::MT                       # AbstractMatrix{FT} (distance x value)
    counts::CT                     # AbstractMatrix (distance x value)
end
```

**Access results**:
```julia
# 1D
result.values         # SF values, one per bin
result.distance_bins  # Original input bins

# 2D
result_2d.sums        # Sum of SF values in each 2D cell
result_2d.counts      # Count of point pairs in each 2D cell
```

### Fast Bin Edges (`AbstractBinEdges`)

For large datasets ($N \ge 2000$), looking up the correct distance bin for each of the $O(N^2)$ point pairs can become a major CPU bottleneck (occupying over 50% of total runtime with standard arrays/ranges due to binary search overhead and cache misses). 

StructureFunctions.jl provides optimized, zero-allocation custom collections subtyping `AbstractBinEdges{T}` to bypass binary search and achieve $O(1)$-like lookup speeds:

* **`LinearBinEdges(edges::AbstractRange)`**: Wraps uniformly-spaced ranges. Bypasses the Twice-Precision calculations in standard ranges, performing searches in **~3 ns** (a **15x+ speedup**) using Fused Multiply-Add (FMA) instructions and ULP corrections.
* **`LogBinEdges(edges::AbstractVector)`**: Wraps log-spaced (geometric) ranges. Bypasses hardware `log(x)` latency by extracting the binary float exponent to perform octal Lookup Table (LUT) queries in **~5-8 ns** (a **5x+ speedup**).
* **`InfPaddedBinEdges(edges::AbstractBinEdges)`**: Wraps any custom bin edges collection to implicitly prepend $-\infty$ (or `typemin(T)`) and append $+\infty$ (or `typemax(T)`).
* **`BinEdges(edges::AbstractVector)`**: General fallback wrapper that automatically routes to `LinearBinEdges` for ranges or wraps vectors.

#### Usage Example

To enable O(1) binning in single-pass calculations, wrap your raw bin vector before calling the calculation:

```julia
using StructureFunctions: Calculations as SFC
using StructureFunctions: LogBinEdges

# Generate log-spaced boundaries
log_bins_raw = collect(exp.(range(log(0.01), log(10.0), length=51)))

# Wrap them in LogBinEdges to activate O(1) Exponent LUT Hybrid Search
distance_bins = LogBinEdges(log_bins_raw)

# Run calculation (bypasses standard binary search bottleneck completely).
# Returns a NamedTuple keyed by invariant: results.S2, results.L2, results.T2,
# results.S3, results.L3, results.L1T2 (+ results.helmholtz for point-field input).
# Each entry is a StructureFunction (pass output_type=StructureFunctionSumsAndCounts for raw).
results = SFC.calculate_structure_functions_single_pass(x, u, distance_bins; backend=SFC.SerialBackend())
```

## Theory & References

Structure functions quantify spatial correlations of a field **u** at separation distance **r**:

$$S_p(r) = \langle |u(\mathbf{x} + \mathbf{r}) - u(\mathbf{x})|^p \rangle$$

where $\langle \cdot \rangle$ is ensemble/spatial average over all displacement vectors $\mathbf{r}$.

### Dimensional Variants

- **1D**: Single coordinate axis (e.g., time series)
- **2D**: Horizontal plane (e.g., satellite imagery)
- **3D**: Full spatial field (e.g., atmospheric snapshots)

### Order Variants

- **1st order** ($p=1$): Absolute increment
- **2nd order** ($p=2$): Energy-like; related to kinetic energy spectrum by Wiener-Khinchin
- **3rd order** ($p=3$): Skewness; tests Kolmogorov refined similarity hypotheses

### References

1. **Kolmogorov (1941)**: _The Local Structure of Turbulence in Incompressible Viscous Fluid for Very Large Reynolds Numbers_  
   - Foundational theory; predicts $S_2(r) \sim r^{2/3}$ in inertial range

2. **Balwada et al. (2016)**: _Scale-aware analysis of satellite sea surface temperature variability_  
   - Applied SF analysis to geophysical gridded data; demonstrates multi-scale recovery

3. **Wikipedia**: [Turbulence](https://en.wikipedia.org/wiki/Turbulence#Kolmogorov's_theory_of_1941)  
   - Accessible overview of Kolmogorov theory

**See also**: [`docs/src/theory.md`](docs/src/theory.md) for detailed mathematical formulations and dimensional projections.

## Example Figures

### 2nd-Order Structure Function — Kolmogorov Scaling

![Structure Function S2](docs/src/assets/sf_kolmogorov.png)

*2nd-order longitudinal structure function on a 2D turbulent field. Dashed line: K41 prediction S₂(r) ~ r^(2/3).*

### Longitudinal vs Transverse Structure Functions

![Longitudinal vs Transverse](docs/src/assets/sf_long_vs_trans.png)

*Comparison of longitudinal (L2SF) and transverse (T2SF) 2nd-order structure functions on the same field.*

### Single-Pass Invariants + Helmholtz Decomposition

![Single-pass invariants and Helmholtz](docs/src/assets/sf_single_pass.png)

*All six isotropic invariants (S2, L2, T2, S3, L3, L1T2) plus the rotational/divergent Helmholtz decomposition — computed in **one** O(N²) pair pass via `calculate_structure_functions_single_pass`.*

### 2D Joint-Probability Binning — all invariants, with vs without a cascade

![2D joint-probability binning](docs/src/assets/sf_2d_binning.png)

*Conditional PDFs `P(value | r)` for all six single-pass invariants (one `calculate_structure_functions_single_pass_2d` call per field), comparing a symmetric random field (top) with a shock/forward-cascade field (bottom). 2nd-order PDFs broaden with separation in both; the **signed 3rd-order** panels (zero line + orange conditional-mean `⟨value|r⟩`) are symmetric for the random field but **skew negative for the cascade field** — the 4/5-law / energy-flux sign. White = low, gray = empty bins.*

### Backend Parity Validation

![Backend Parity](docs/src/assets/sf_backend_parity.png)

*Serial vs Threaded backend results on identical data — differences are at floating-point rounding level.*

---

## Performance

### Scaling Characteristics

| Dimension | Metric | Value |
|-----------|--------|-------|
| N points  | Algorithm | O(N²) |
| B bins    | Space | O(N + B) |
| D dim's   | CPU ops | ~D² per pair |
| Threads   | Speedup | ~0.8–0.9× per thread (dims ≤ 3) |

### Benchmark Figures (v0.3.0, Julia 1.12)

> **Hardware:** 2× Intel Xeon Gold 6426Y (16 cores / 32 threads each, 64 logical CPUs total).
> Benchmarks were run on this machine. Results on other hardware will differ.
> To regenerate with your own hardware, see [`benchmark/benchmark_scaling.jl`](benchmark/benchmark_scaling.jl).
> Output figures land in `benchmark/benchmark_results/` (gitignored).

#### CPU strong scaling — fixed N, increasing threads

![CPU strong scaling](docs/src/assets/strong_scaling.png)

*Fixed problem size (N = 4000 points, 3D, longitudinal 2nd-order SF). Speedup approaches linear up to ~4 threads; NUMA effects reduce efficiency beyond 8 threads on a dual-socket system. Generated by [`benchmark/benchmark_scaling.jl`](benchmark/benchmark_scaling.jl).*

#### CPU weak scaling — N ∝ √p, constant work per thread

![CPU weak scaling](docs/src/assets/weak_scaling.png)

*Problem size grows as N = N_base × √p so each thread has constant O(N²/p) pair work. Ideal wall-clock time is flat; observed rise reflects inter-socket memory traffic.*

#### GPU problem-size scaling — 1 GPU vs CPU (vary N)

![GPU problem-size scaling](docs/src/assets/gpu_problem_size_scaling.png)

*Fixed hardware (**1 GPU + serial CPU**, always 1 worker), sweep N. **Not** HPC strong/weak scaling. CPU threading is in the strong/weak figures above. Regenerate: [`gpu/collect_benchmark_assets.jl`](gpu/collect_benchmark_assets.jl), then [`generate_gpu_figures.jl`](docs/generate_assets/generate_gpu_figures.jl).*

#### GPU batch scaling — fixed N, vary T (time slices)

![GPU batch scaling](docs/src/assets/gpu_slice_batch_scaling.png)

*Fixed `N_SLICE` (default 1000), sweep T. CPU per-slice loop vs GPU naive loop vs GPU batch driver (`*_batch!`) on **1 GPU**. Not HPC weak scaling.*

#### GPU kernel parity (KA.CPU vs serial)

![GPU Parity](docs/src/assets/sf_gpu_parity.png)

*Serial CPU reference vs `gpu_calculate_structure_function` on `KA.CPU()` — proves kernel logic in default CI; does not test CUDA.*

### Optimization Tips

1. **Use AutoBackend** for deployment (automatic tuning)
2. **Prefer larger datasets** for threading overhead to amortize
3. **Pre-sort bins** by distance to improve cache locality
4. **Use Float32** if precision allows (faster GPU transfers)
5. **Batch multiple SFs** by reusing distance calculations

## Extensions

Optional packages extend StructureFunctions with additional functionality:

### OhMyThreadsExt (ThreadedBackend)

Loaded automatically when `OhMyThreads.jl` is in `Project.toml`:

```toml
[extras]
OhMyThreads = "67456a42-ebe4-4781-8ad1-67f7eda8d8f7"

[extensions]
StructureFunctionsOhMyThreadsExt = "OhMyThreads"
```

### DistributedExt (DistributedBackend)

Requires `Distributed.jl` (stdlib) + `SharedArrays.jl` (stdlib):

```julia
using Distributed: addprocs
addprocs(4)
backend = StructureFunctions.DistributedBackend()
```

### GPUExt (GPUBackend)

Requires `KernelAbstractions.jl` + GPU package (CUDA.jl, AMDGPU.jl, etc.):

```toml
[extras]
KernelAbstractions = "63c18a36-062a-441e-b365-b594b6ce51b1"

[extensions]
StructureFunctionsGPUExt = "KernelAbstractions"
```

## Migration from v0.2

### Breaking Changes

| v0.2 | v0.3 |
|------|------|
| Symbol-based backend selection | Typed backend objects |
| `backend=:serial` | `backend=SerialBackend()` |
| `backend=:threaded` | `backend=ThreadedBackend()` |
| `backend=:distributed` | `backend=DistributedBackend()` |
| No GPU support | `backend=GPUBackend(...)` |

### Recommended Updates

```julia
# OLD (v0.2)
result = calculate_structure_function(sf, x, u, bins; backend=:threaded)

# NEW (v0.3)
result = calculate_structure_function(sf, x, u, bins; backend=ThreadedBackend())

# Or use AutoBackend for automatic selection:
result = calculate_structure_function(sf, x, u, bins)  # Defaults to AutoBackend()
```

### Compatibility

- v0.3 is **not** backward-compatible with v0.2 scripts
- Update scripts by replacing symbol backends with typed backends
- See `CHANGELOG.md` for full change log

## Examples

Detailed worked examples are in `examples/` directory:

- `simple_2d.jl`: Basic 2D structure function calculation
- `threaded_calculation.jl`: Multi-threaded execution
- `gpu_acceleration.jl`: GPU acceleration with KernelAbstractions  
- `distributed_parallel.jl`: Multi-process execution
- `custom_operator.jl`: Defining custom SF operators

Clone and run:

```bash
cd examples/
julia simple_2d.jl
julia threaded_calculation.jl
```

## Contributing

Contributions welcome! Please:

1. Fork and create a feature branch
2. Add tests for new functionality
3. Ensure full test suite passes: `julia test/runtests.jl`
4. Document changes in docstrings and `CHANGELOG.md`

## License

See `LICENSE` file.

## Citation

If you use StructureFunctions.jl in research, please cite:

```bibtex
@software{structurefunctions_jl_2024,
  author = {Benjamin, Jordan and Contributors},
  title = {StructureFunctions.jl: High-performance structure function calculations},
  year = {2024},
  doi = {10.5281/zenodo.14945669},
  url = {https://zenodo.org/records/14945669}
}
```

---

**Last Updated**: March 2026 | **Version**: 0.3.0 | **Julia**: 1.12+
