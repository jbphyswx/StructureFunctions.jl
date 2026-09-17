# Backends

Where a calculation runs is a type passed as `backend`. The result never depends on it: every backend
gives the same counts exactly and the same sums to round-off, and the tests hold each pair of
backends to that (see [Validation](validation.md) for the tolerance policy).

| backend | needs | runs |
|---|---|---|
| `SerialBackend()` | nothing | every route, on one thread; the reference the others are checked against |
| `ThreadedBackend()` | `using OhMyThreads`, `julia -t N` | point lists, multi-fields, tensors, the gridded sweeps and transforms, the sorted line route |
| `DistributedBackend()` | `using Distributed`, `addprocs` | point lists, multi-fields and tensors, each worker taking a share of the outer index; `DistributedBackend(ThreadedBackend())` threads inside each worker |
| `MPIBackend()` | `using MPI` | point lists across ranks |
| `GPUBackend(device)` | `using KernelAbstractions` and a device package | point lists, joint histograms, single-pass invariants, batches over auxiliary axes, tensors, multi-fields, and the gridded transform engine; `GPUBackend(KernelAbstractions.CPU())` runs the same kernels on the host |
| `AutoBackend()` | — | the default: the distributed backend when workers are present, the threaded one when Julia has more than one thread and its extension is loaded, the serial one otherwise |

The backend types come from `ComputationalBackends`:

```julia
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT

x = rand(2, 50_000) .* 100.0          # (D, N) coordinates
u = randn(2, 50_000)                  # (D, N) velocities
bins = range(0.0, 20.0; length = 41)  # a range is wrapped as LinearBinEdges, O(1) digitizing

SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.SerialBackend())
SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.ThreadedBackend())
SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins)   # AutoBackend()
```

## In place

Every entry has a mutating form that accumulates into caller-owned buffers, for loops over time
steps or for adding partial results across processes:

```julia
sums = zeros(Float64, length(bins) - 1)
counts = zeros(UInt32, length(bins) - 1)
SFC.calculate_structure_function!(sums, counts, SFT.L2SFType(), x, u, bins; backend = CB.ThreadedBackend())
```

The mutating entries add into the buffers they are given; zero them first. The count element type
must hold the worst-case pair count `N(N−1)/2` (`UInt32` up to `N = 92 682`), and weighted results
take a floating-point count type.

## What each backend does

**Serial.** Flat two- and three-dimensional point lists take a SIMD compute/scatter kernel over
blocked pair tiles: the pair distances and operator values of a block are computed in a vectorised
loop and scattered into the histogram in a scalar one. When the last bin edge bounds the separations
of interest, cells beyond it are never enumerated (`AutoCulling()`, the default; `AlwaysCulling()`
insists and errors where culling is not implemented; `NoCulling()` sweeps every pair). Curved geometry
takes the scalar per-point kernel through the pair frame; one-dimensional lists take the sorted line
route.

**Threaded.** The outer index is split round-robin over tasks — work for index `i` is `N − i`, so a
contiguous split would skew the load by the thread count — each task accumulates into private
histograms, and the partials are added. Throughput saturates near one socket's memory bandwidth on
the batched paths; `JULIA_EXCLUSIVE=1` pins threads.

The gridded routes split **slab pairs** across tasks, and the direct lag sweep additionally splits a
single slab's lags, so a one-slab schedule still threads. A transform splits slab pairs only: its
unit of work is a pair, because the pair's inverse transform must finish before any of that pair's
lags can be read. On a [`UniformLagSchedule`](@ref StructureFunctions.Calculations.UniformLagSchedule) or a
[`ScatteredModesSchedule`](@ref StructureFunctions.Calculations.ScatteredModesSchedule), which have exactly one slab
pair, a transform therefore sweeps its lags on one task whatever backend is asked for.

Most of a one-slab transform is the transforms themselves rather than the lag sweep, and those are
FFTW's to parallelise. FFTW keeps its own thread count, defaulting to one, and the package never sets
it, because it is global to the session rather than a property of a call. `FFTW.set_num_threads(n)`
is the caller's to set, and the two thread pools do not compose: raising it while the package is also
threading over many slab pairs oversubscribes the cores and loses badly. Raise one or the other —
FFTW's for a one-slab schedule, the package's for a many-slab one — and measure on your own grid,
since which is larger depends on the grid, the operator and the FFTW build.

**Distributed.** Each worker receives a balanced share of the outer index (`w:k:N`) and computes its
partial sums and counts with the serial kernel — or the threaded one under
`DistributedBackend(ThreadedBackend())`, one process per NUMA node being the way past a single
socket's bandwidth. The partials are reduced on the caller.

**GPU.** See [GPU acceleration](gpu.md). One thread per point walks its partners with block-local
histograms; the gridded transform engine takes its forward transforms through the device's
`AbstractFFTs` implementation and bins every lag of every slab pair in one kernel.

## Choosing

- Under a few thousand points, or for a check, the serial backend.
- Otherwise the threaded backend on one node, which is what `AutoBackend()` picks with
  `julia -t N` and `using OhMyThreads`.
- A GPU from a few thousand points up when the data fit in device memory; the gridded transform on a
  device from about a million cells.
- Several nodes: the distributed backend, threaded inside each worker.
- On a grid, prefer the transform (`FastFourierTransformSpectralBackend()`) to any pair loop: it is
  exact and its cost does not grow with the number of pairs. `AutoSpectralBackend()` costs the two
  gridded algorithms and takes the cheaper.

## Bin edges on the hot path

Digitizing each of the `O(N²)` pairs is the inner loop. Pass a `range` (wrapped as `LinearBinEdges`)
or `LogBinEdges(edges)` for `O(1)` digitizing by a fused multiply-add or an exponent lookup; a plain
`Vector` of edges falls back to binary search. See [Binning Internals](uniform_bin_digitize.md).

## Related pages

- [Architecture](architecture.md) — how a call becomes a kernel.
- [Extensions](extensions.md) — which package each backend needs.
