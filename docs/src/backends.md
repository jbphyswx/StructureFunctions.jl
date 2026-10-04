# Execution backends and workspaces

Execution backends select hardware and parallel execution. Spectral backends select a numerical method. Both choices must support the requested operator, geometry, and data layout.

## Execution choices

| Backend | Required package | Execution |
|---|---|---|
| `SerialBackend()` | Core dependencies | One CPU thread |
| `ThreadedBackend()` | `OhMyThreads` | Tasks with private partial reductions |
| `DistributedBackend()` | `Distributed` | Julia worker processes |
| `MPIBackend()` | `MPI` | MPI ranks |
| `GPUBackend(device)` | `KernelAbstractions`, device provider | Device kernels |
| `AutoBackend()` | Available extensions | An available execution method |

Import these types from `ComputationalBackends`. Load `CUDA` and use `GPUBackend(CUDA.CUDABackend())` for CUDA. KernelAbstractions' CPU backend exercises portable kernels on a CPU; it does not establish GPU correctness or performance.

## Explicit CPU calculation

```@example backends
using Random: Random
using ComputationalBackends: SerialBackend
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
rng = Random.MersenneTwister(7)
x = Random.rand(rng, 2, 32)
u = Random.randn(rng, 2, 32)
bins = range(0.0, 1.5; length=9)
s = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins;
    backend=SerialBackend())
s.values
```

For threading, start Julia with the allocated thread count, load `OhMyThreads`, and select `ThreadedBackend()`. Distributed and MPI workers must use a compatible project and package resolution. Configure the local backend and worker threads within the allocated CPU budget.

## Result location

GPU individual, joint-value, and single-pass point calculations retain device
arrays for snapshots and batches. `to_host(result)` explicitly copies numerical
buffers to host memory. Mutating forms require outputs on the selected backend
and add to existing values. Allocating results own their buffers across workspace
reuse.
## Repeated calculations

`CPUSFWorkspace` and `GPUSFWorkspace` retain buffers for compatible calculations. Construct a workspace for the calculation's layout, bins, precision, and backend; pass it with `workspace=...`. A workspace serves one call at a time. On a grid, a `TransformWorkspace()` keeps the transform's spectra, plans and scratch from one call to the next, as a time series on one grid makes; it takes no arguments and rebuilds what it keeps when a call's sizes change.

Mutating entries add to output buffers; zero them when starting an independent result. A workspace carries no result from one call to the next: allocating entries return fresh buffers. Integer counts must represent both existing counts and the new contributions. Weighted normalization uses floating-point pair mass.

Workspace compatibility and supported families are specified in the [calculation reference](api/calculations.md). Memory and allocation claims require measurements of the specific prepared route.

## Numerical methods

Direct pairs support general pairwise operators. Culling restricts enumeration using an exact geometric cutoff. Sorted-line polynomial moments and gridded FFT correlations exploit additional structure. Explicit nonuniform and harmonic methods have their own resolution and tolerance parameters.

Use `SpectralBackends` method tags where the interface accepts them. An FFT requires a compatible provider and a polynomial moment representation. Approximate spectral methods must be chosen with their resolution assumptions understood.

