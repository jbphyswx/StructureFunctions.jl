# Calculations Execution Backends

abstract type AbstractExecutionBackend end

"""
    SerialBackend <: AbstractExecutionBackend

Serial (CPU, single-threaded) execution backend for structure function calculations.

Use this backend when:
- Running on a single thread (default for Julia)
- Parallelization is not available or not desired
- Debugging or validating calculations

This is the reference implementation that all other backends are validated against.

# Examples
```julia
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT

sf_type = SFT.LongitudinalSecondOrderStructureFunctionType()
x = ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
u = ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
bins = [0.0, 1.0, 2.0]

result = SFC.calculate_structure_function(sf_type, x, u, bins; backend=SFC.SerialBackend())
```
"""
struct SerialBackend <: AbstractExecutionBackend end

"""
    ThreadedBackend <: AbstractExecutionBackend

Multi-threaded (CPU) execution backend using OhMyThreads for structure function calculations.

Requires the `OhMyThreads.jl` package to be loaded. Use this backend when:
- Multiple CPU threads are available (`Threads.nthreads() > 1`)
- Speed is important and shared-memory parallelism is suitable
- Dataset fits in memory

Partitions the outer loop index `i` across CPU tasks (via OhMyThreads when loaded).
For O(N²) pair loops, the threaded extension uses round-robin outer-index chunks
(`OMT.RoundRobin`) so each task gets ~equal pair work; contiguous equal-size chunks
would severely load-imbalance this kernel. Thread-local reductions ensure thread safety
without locks or `threadid()` indexing.

# Examples
```julia
using Base.Threads: nthreads

if nthreads() > 1
    result = SFC.calculate_structure_function(sf_type, x, u, bins; 
                                            backend=SFC.ThreadedBackend())
end
```
"""
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    DistributedBackend{Inner} <: AbstractExecutionBackend
    DistributedBackend(inner = SerialBackend())

Distributed (multi-process/multi-node) execution backend using Distributed.jl.

Parametric on the per-worker `inner` backend (like [`GPUBackend`](@ref) is parametric on its
device backend): distribution across processes and local execution within a process are
orthogonal axes. `inner` selects how each worker computes its share of the pairs:

- `DistributedBackend()` / `DistributedBackend(SerialBackend())` — each worker runs serially.
- `DistributedBackend(ThreadedBackend())` — hybrid: each worker threads over its share
  (requires `OhMyThreads` and worker threads, e.g. `addprocs(n; exeflags="-t k")`). On a
  multi-socket node this enables one-process-per-socket × threaded-within-socket, which scales
  past the single-socket memory-bandwidth ceiling of pure threading.

Requires workers started via `addprocs()`.

# Examples
```julia
using Distributed: addprocs
addprocs(4)
result = SFC.calculate_structure_function(sf_type, x, u, bins; backend=SFC.DistributedBackend())

# hybrid distributed + threaded (workers each with 8 threads):
# addprocs(2; exeflags="-t 8"); @everywhere using OhMyThreads
# backend = SFC.DistributedBackend(SFC.ThreadedBackend())
```
"""
struct DistributedBackend{Inner <: AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    GPUBackend{B} <: AbstractExecutionBackend

GPU-accelerated execution backend using KernelAbstractions.jl.

Parameterized by the target GPU backend (e.g., `KernelAbstractions.CPU()`, `CUDA.CUDABackend()`).
Requires the `KernelAbstractions.jl` package. Use this backend when:
- NVIDIA GPU (CUDA) or other supported GPU hardware is available
- Computation time is critical
- Kernel compatibility is high (most SF calculations are embarrassingly parallel)

# Examples
```julia
using KernelAbstractions: CPU

# CPU backend (for testing parity)
result = SFC.calculate_structure_function(sf_type, x, u, bins; 
                                         backend=SFC.GPUBackend(CPU()))

# NVIDIA GPU backend (requires CUDA.jl)
using CUDA
result = SFC.calculate_structure_function(sf_type, x, u, bins; 
                                         backend=SFC.GPUBackend(CUDABackend()))
```
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"""
    AutoBackend <: AbstractExecutionBackend

Automatic backend selection based on availability and runtime state.

The selection order is:
  1. Distributed when workers are available (`nworkers() > 1`)
  2. Threaded when multiple CPU threads available (`nthreads() > 1`)
  3. Serial

Use this backend when:
- You want the package to choose the best available backend automatically
- You are writing portable code that should adapt to the execution environment

# Examples
```julia
# Package automatically chooses the best backend
result = SFC.calculate_structure_function(sf_type, x, u, bins)  # Defaults to AutoBackend

# Or explicitly:
result = SFC.calculate_structure_function(sf_type, x, u, bins; 
                                         backend=SFC.AutoBackend())
```
"""
struct AutoBackend <: AbstractExecutionBackend end

# Backward-compatible aliases for in-flight API migration.
const AbstractThreadingBackend = AbstractExecutionBackend
const AutoThreadingBackend = AutoBackend

function threaded_calculate_structure_function(args...; kwargs...)
    throw(
        ArgumentError(
            "Threaded backend is unavailable. Load the OhMyThreads extension or use backend=SerialBackend().",
        ),
    )
end

function threaded_calculate_structure_function!(args...; kwargs...)
    throw(
        ArgumentError(
            "Threaded backend is unavailable. Load the OhMyThreads extension or use backend=SerialBackend().",
        ),
    )
end

function _threaded_backend_available(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins,
    vrsac,
)
    return hasmethod(
        threaded_calculate_structure_function,
        Tuple{
            typeof(structure_function_type),
            typeof(x),
            typeof(u),
            typeof(distance_bins),
            typeof(vrsac),
        },
    )
end

function _threaded_backend_available!(
    sums,
    counts,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins,
)
    return hasmethod(
        threaded_calculate_structure_function!,
        Tuple{
            typeof(sums),
            typeof(counts),
            typeof(structure_function_type),
            typeof(x),
            typeof(u),
            typeof(distance_bins),
        },
    )
end

function _threaded_backend_available!(
    sums,
    counts,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector,
)
    return hasmethod(
        threaded_calculate_structure_function!,
        Tuple{
            typeof(sums),
            typeof(counts),
            typeof(structure_function_type),
            typeof(x),
            typeof(u),
            typeof(distance_bins),
            typeof(value_bins),
        },
    )
end

function _dispatch_execution_backend(
    ::DistributedBackend, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, vrsac; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed/SharedArrays extension or use backend=SerialBackend()."))
end

function _dispatch_execution_backend!(
    ::DistributedBackend, sums::AbstractArray, counts::AbstractArray, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed/SharedArrays extension or use backend=SerialBackend()."))
end

function _dispatch_execution_backend!(
    ::DistributedBackend, sums_2d::AbstractArray, counts_2d::AbstractArray, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed/SharedArrays extension or use backend=SerialBackend()."))
end

distributed_workers_available(::Val) = false
