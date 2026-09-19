# Examples

Runnable scripts live in [`examples/`](https://github.com/jbphyswx/StructureFunctions.jl/tree/main/examples)
and have their own environment:

```bash
julia --project=examples examples/simple_2d.jl
```

| Script | Shows |
|---|---|
| [`simple_2d.jl`](https://github.com/jbphyswx/StructureFunctions.jl/blob/main/examples/simple_2d.jl) | 2D field, 2nd-order SF, K41 scaling, plotting |
| [`threaded_calculation.jl`](https://github.com/jbphyswx/StructureFunctions.jl/blob/main/examples/threaded_calculation.jl) | `ThreadedBackend` (OhMyThreads), serial-vs-threaded speedup |
| [`distributed_parallel.jl`](https://github.com/jbphyswx/StructureFunctions.jl/blob/main/examples/distributed_parallel.jl) | `DistributedBackend` across worker processes |
| [`gpu_acceleration.jl`](https://github.com/jbphyswx/StructureFunctions.jl/blob/main/examples/gpu_acceleration.jl) | `GPUBackend` + reusable `GPUSFWorkspace` |
| [`gpu_time_slices.jl`](https://github.com/jbphyswx/StructureFunctions.jl/blob/main/examples/gpu_time_slices.jl) | native batch over the trailing `(D, N, T)` axis |
| [`single_pass.jl`](https://github.com/jbphyswx/StructureFunctions.jl/blob/main/examples/single_pass.jl) | six invariants + Helmholtz in one pair pass |

## Featured snippets

### Single pass: six invariants + Helmholtz

![Single-pass invariants and Helmholtz decomposition](assets/sf_single_pass.png)

One O(N²) pair pass returns all six isotropic invariants (and, for point-field input, the
rotational/divergent Helmholtz decomposition) as a `NamedTuple` keyed by invariant:

```@example examples
using StructureFunctions: Calculations as SFC, LogBinEdges, StructureFunctionObjects as SFO
using ComputationalBackends: ComputationalBackends as CB

x = rand(2, 4096) .* 1.0e4          # (D, N) coordinates
u = randn(2, 4096)                  # (D, N) velocities
bins = LogBinEdges(collect(exp10.(range(log10(50.0), log10(5.0e3); length = 41))))

res = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = CB.AutoBackend())
propertynames(res)   # one entry per invariant, and the Helmholtz split
```

Each entry carries raw sums and counts, which is what adds across slices and processes:

```@example examples
propertynames(res.L2)
```

`res.helmholtz` is a `HelmholtzDecomposition2D` carrying the rotational and divergent parts, for
point-field input only. `output_type` asks for the bin averages instead:

```@example examples
avg = SFC.calculate_structure_functions_single_pass(
    x, u, bins; output_type = SFO.StructureFunction,
)
propertynames(avg.L2)
```

### 2D joint (distance × value) histogram

![2D joint-probability binning across all invariants, with vs without a cascade](assets/sf_2d_binning.png)

```@example examples
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges

dist = LogBinEdges(collect(exp10.(range(log10(50.0), log10(5.0e3); length = 41))))
vbins = LinearBinEdges(collect(range(-5.0, 5.0; length = 51)))
sf2d = SFC.calculate_structure_function(SFT.L2SFType(), x, u, dist, vbins; backend = CB.AutoBackend())
size(sf2d.sums), size(sf2d.counts)   # the (n_dist, n_val) joint histogram
```

### Native batch over time slices

Pass a `(D, N, T)` array (shared positions may be a single `(D, N)` matrix). Geometry is computed
once per pair and the batch axis is vectorized — far faster than looping `t`:

```@example examples
n_slices = 4
u_batch = randn(2, size(x, 2), n_slices)      # (D, N, T); x stays a single (D, N) matrix
sums = zeros(SFC.SINGLE_PASS_N, length(dist) - 1, length(vbins) - 1, n_slices)
counts = zeros(Int, size(sums))
SFC.calculate_structure_functions_single_pass_2d_batch!(sums, counts, x, u_batch, dist, vbins;
                                                        backend = CB.AutoBackend())
size(sums)
```

See [Backends](backends.md) for choosing serial / threaded / distributed / GPU, and
[GPU Acceleration](gpu.md) for the GPU batch path and `GPUSFWorkspace`.
