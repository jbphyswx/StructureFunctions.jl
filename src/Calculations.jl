"""
The workhorse of this package
"""
module Calculations
using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using ..HelperFunctions: HelperFunctions as SFH
using ..StructureFunctionTypes: StructureFunctionTypes as SFT
using ..StructureFunctionObjects: StructureFunctionObjects as SFO

using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using Base.Threads: Threads
using LoopVectorization: LoopVectorization as LV # TODO: Move to extension or replace with Polyester/SIMD as part of modernization

export calculate_structure_function, parallel_calculate_structure_function,
    gpu_calculate_structure_function, calculate_structure_function_from_file,
    AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend,
    GPUBackend, AutoBackend, AbstractThreadingBackend, AutoThreadingBackend,
    serial_calculate_structure_function, threaded_calculate_structure_function,
    calculate_structure_functions_single_pass,
    serial_calculate_structure_function!, threaded_calculate_structure_function!,
    calculate_structure_function!

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
bins = [(0.0, 1.0), (1.0, 2.0)]

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

Controls the parallelization of the outer loop over point pairs. Thread-local reductions
ensure thread safety without locks.

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
    DistributedBackend <: AbstractExecutionBackend

Distributed (multi-process/multi-node) execution backend using Distributed.jl.

Requires workers to be started via `addprocs()` or similar. Use this backend when:
- Computing across multiple processes or machines
- Dataset is large but computation must remain in-core on each worker
- You have a compute cluster available

# Examples
```julia
using Distributed: addprocs

addprocs(4)  # Start 4 worker processes
result = SFC.calculate_structure_function(sf_type, x, u, bins; 
                                         backend=SFC.DistributedBackend())
```
"""
struct DistributedBackend <: AbstractExecutionBackend end

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
  3. Serial (fallback)

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

function parallel_calculate_structure_function!(args...; kwargs...)
    throw(
        ArgumentError(
            "Distributed backend is unavailable. Load the Distributed extension or use backend=SerialBackend().",
        ),
    )
end

distributed_workers_available(::Val) = false

function _threaded_backend_available(
    structure_function_type::SFT.AbstractStructureFunctionType,
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
    structure_function_type::SFT.AbstractStructureFunctionType,
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
    structure_function_type::SFT.AbstractStructureFunctionType,
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

function _distributed_backend_available!(
    sums,
    counts,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
)
    return hasmethod(
        parallel_calculate_structure_function!,
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

function _distributed_backend_available!(
    sums,
    counts,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector,
)
    return hasmethod(
        parallel_calculate_structure_function!,
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

# --- Public Mutating API ---

function calculate_structure_function!(
    sums, counts, sf_type, x, u, distance_bins;
    backend=SerialBackend(), kwargs...
)
    _dispatch_execution_backend!(backend, sums, counts, sf_type, x, u, distance_bins; kwargs...)
    return nothing
end

function calculate_structure_function!(
    sums_2d, counts_2d, sf_type, x, u, distance_bins, value_bins;
    backend=SerialBackend(), kwargs...
)
    _dispatch_execution_backend!(backend, sums_2d, counts_2d, sf_type, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

# Backend Dispatch Layers for !

function _dispatch_execution_backend!(
    ::SerialBackend,
    sums,
    counts,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins;
    kwargs...,
)
    serial_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x,
        u,
        distance_bins;
        kwargs...,
    )
    return nothing
end

function _dispatch_execution_backend!(
    ::ThreadedBackend,
    sums,
    counts,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins;
    kwargs...,
)
    threaded_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x,
        u,
        distance_bins;
        kwargs...,
    )
    return nothing
end

function _dispatch_execution_backend!(
    backend::GPUBackend,
    sums,
    counts,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins;
    kwargs...,
)
    throw(ArgumentError("In-place calculate_structure_function! is not supported on GPU backend."))
end

function _dispatch_execution_backend!(
    ::AutoBackend,
    sums,
    counts,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins;
    kwargs...,
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_backend_available!(sums, counts, structure_function_type, x, u, distance_bins)
        return parallel_calculate_structure_function!(
            sums,
            counts,
            structure_function_type,
            x,
            u,
            distance_bins;
            kwargs...,
        )
    end

    if Threads.nthreads() > 1 &&
       _threaded_backend_available!(sums, counts, structure_function_type, x, u, distance_bins)
        return threaded_calculate_structure_function!(
            sums,
            counts,
            structure_function_type,
            x,
            u,
            distance_bins;
            kwargs...,
        )
    end

    return serial_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x,
        u,
        distance_bins;
        kwargs...,
    )
end

# 2D Backend Dispatch Layers for !

function _dispatch_execution_backend!(
    ::SerialBackend,
    sums_2d,
    counts_2d,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    serial_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
    return nothing
end

function _dispatch_execution_backend!(
    ::ThreadedBackend,
    sums_2d,
    counts_2d,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    threaded_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
    return nothing
end

function _dispatch_execution_backend!(
    backend::GPUBackend,
    sums_2d,
    counts_2d,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    throw(ArgumentError("In-place calculate_structure_function! is not supported on GPU backend."))
end

function _dispatch_execution_backend!(
    ::AutoBackend,
    sums_2d,
    counts_2d,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_backend_available!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins)
        return parallel_calculate_structure_function!(
            sums_2d,
            counts_2d,
            structure_function_type,
            x,
            u,
            distance_bins,
            value_bins;
            kwargs...,
        )
    end

    if Threads.nthreads() > 1 &&
       _threaded_backend_available!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins)
        return threaded_calculate_structure_function!(
            sums_2d,
            counts_2d,
            structure_function_type,
            x,
            u,
            distance_bins,
            value_bins;
            kwargs...,
        )
    end

    return serial_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend!(
    backend::AbstractExecutionBackend,
    sums,
    counts,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins;
    kwargs...,
)
    return _dispatch_execution_backend!(
        SerialBackend(),
        sums,
        counts,
        structure_function_type,
        x,
        u,
        distance_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend!(
    backend::AbstractExecutionBackend,
    sums_2d,
    counts_2d,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    return _dispatch_execution_backend!(
        SerialBackend(),
        sums_2d,
        counts_2d,
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    ::SerialBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    vrsac;
    kwargs...,
)
    return serial_calculate_structure_function(
        structure_function_type,
        x,
        u,
        distance_bins,
        vrsac;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    ::ThreadedBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    vrsac;
    kwargs...,
)
    return threaded_calculate_structure_function(
        structure_function_type,
        x,
        u,
        distance_bins,
        vrsac;
        kwargs...,
    )
end

function _distributed_backend_available(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    vrsac,
)
    return hasmethod(
        parallel_calculate_structure_function,
        Tuple{
            typeof(structure_function_type),
            typeof(x),
            typeof(u),
            typeof(distance_bins),
            typeof(vrsac),
        },
    )
end

function _dispatch_execution_backend(
    ::DistributedBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    vrsac;
    kwargs...,
)
    if !_distributed_backend_available(structure_function_type, x, u, distance_bins, vrsac)
        throw(
            ArgumentError(
                "Distributed backend is unavailable. Load Distributed/SharedArrays extension or use backend=SerialBackend().",
            ),
        )
    end
    return parallel_calculate_structure_function(
        structure_function_type,
        x,
        u,
        distance_bins,
        vrsac;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    backend::GPUBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    vrsac;
    kwargs...,
)
    return gpu_calculate_structure_function(
        structure_function_type,
        backend.backend,
        x,
        u,
        distance_bins,
        vrsac;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    ::AutoBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    vrsac;
    kwargs...,
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_backend_available(structure_function_type, x, u, distance_bins, vrsac)
        return _dispatch_execution_backend(
            DistributedBackend(),
            structure_function_type,
            x,
            u,
            distance_bins,
            vrsac;
            kwargs...,
        )
    end

    if Threads.nthreads() > 1 &&
       _threaded_backend_available(structure_function_type, x, u, distance_bins, vrsac)
        return _dispatch_execution_backend(
            ThreadedBackend(),
            structure_function_type,
            x,
            u,
            distance_bins,
            vrsac;
            kwargs...,
        )
    end
    return _dispatch_execution_backend(
        SerialBackend(),
        structure_function_type,
        x,
        u,
        distance_bins,
        vrsac;
        kwargs...,
    )
end

"""
    calculate_structure_function(sf_type::SFT.AbstractStructureFunctionType, path::String, bins; kwargs...)

Entry point for calculating structure functions from a file (e.g. NetCDF, JLD2, CSV).
The file extension is used to dispatch to an appropriate extension method.
"""
function calculate_structure_function(
    sf_type::SFT.AbstractStructureFunctionType,
    path::String,
    bins;
    kwargs...,
)
    ext_str = lowercase(split(path, '.')[end])
    return calculate_structure_function_from_file(
        Val(Symbol(ext_str)),
        path,
        bins,
        sf_type;
        kwargs...,
    )
end

"""
    calculate_structure_function(sf_sym::Symbol, args...; kwargs...)

Shorthand that looks up the operator by name and calls the core calculation.
"""
function calculate_structure_function(
    sf_sym::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
)
    return calculate_structure_function(Val(sf_sym), x, u, bins; kwargs...)
end

function calculate_structure_function(
    sf_sym::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    return calculate_structure_function(Val(sf_sym), x, u, bins, Val(RSAC); kwargs...)
end

function calculate_structure_function(
    ::Val{sf_sym},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
) where {sf_sym}
    return calculate_structure_function(
        SFT.get_structure_function_type(Val(sf_sym)),
        x,
        u,
        bins;
        kwargs...,
    )
end

function calculate_structure_function(
    ::Val{sf_sym},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {sf_sym, RSAC}
    return calculate_structure_function(
        SFT.get_structure_function_type(Val(sf_sym)),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

"""
    calculate_structure_function(order::Int, mode::Symbol, args...; kwargs...)

Shorthand that looks up the operator by order and mode and calls the core calculation.
"""
function calculate_structure_function(
    order::Int,
    mode::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
)
    return calculate_structure_function(Val(order), Val(mode), x, u, bins; kwargs...)
end

function calculate_structure_function(
    order::Int,
    mode::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    return calculate_structure_function(
        Val(order),
        Val(mode),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

function calculate_structure_function(
    ::Val{order},
    ::Val{mode},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
) where {order, mode}
    return calculate_structure_function(
        SFT.get_structure_function_type(Val(order), Val(mode)),
        x,
        u,
        bins;
        kwargs...,
    )
end

function calculate_structure_function(
    ::Val{order},
    ::Val{mode},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {order, mode, RSAC}
    return calculate_structure_function(
        SFT.get_structure_function_type(Val(order), Val(mode)),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

# Shorthand for parallel version too
function parallel_calculate_structure_function(
    sf_sym::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
)
    return parallel_calculate_structure_function(Val(sf_sym), x, u, bins; kwargs...)
end

function parallel_calculate_structure_function(
    sf_sym::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    return parallel_calculate_structure_function(
        Val(sf_sym),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

function parallel_calculate_structure_function(
    ::Val{sf_sym},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
) where {sf_sym}
    return parallel_calculate_structure_function(
        SFT.get_structure_function_type(Val(sf_sym)),
        x,
        u,
        bins;
        kwargs...,
    )
end

function parallel_calculate_structure_function(
    ::Val{sf_sym},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {sf_sym, RSAC}
    return parallel_calculate_structure_function(
        SFT.get_structure_function_type(Val(sf_sym)),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

function parallel_calculate_structure_function(
    order::Int,
    mode::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
)
    return parallel_calculate_structure_function(
        Val(order),
        Val(mode),
        x,
        u,
        bins;
        kwargs...,
    )
end

function parallel_calculate_structure_function(
    order::Int,
    mode::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    return parallel_calculate_structure_function(
        Val(order),
        Val(mode),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

function parallel_calculate_structure_function(
    ::Val{order},
    ::Val{mode},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
) where {order, mode}
    return parallel_calculate_structure_function(
        SFT.get_structure_function_type(Val(order), Val(mode)),
        x,
        u,
        bins;
        kwargs...,
    )
end

function parallel_calculate_structure_function(
    ::Val{order},
    ::Val{mode},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {order, mode, RSAC}
    return parallel_calculate_structure_function(
        SFT.get_structure_function_type(Val(order), Val(mode)),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

"""
    StructureFunction(order::Int, mode::Symbol, x, u, bins; kwargs...)

Factory constructor that calculates a structure function and returns a result object.
This redirects to `calculate_structure_function`.
"""
function SFO.StructureFunction(
    order::Int,
    mode::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
)
    return calculate_structure_function(order, mode, x, u, bins; kwargs...)
end

function SFO.StructureFunction(
    order::Int,
    mode::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    return calculate_structure_function(order, mode, x, u, bins, Val(RSAC); kwargs...)
end

function SFO.StructureFunction(
    ::Val{order},
    ::Val{mode},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
) where {order, mode}
    return calculate_structure_function(Val(order), Val(mode), x, u, bins; kwargs...)
end

function SFO.StructureFunction(
    ::Val{order},
    ::Val{mode},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {order, mode, RSAC}
    return calculate_structure_function(
        Val(order),
        Val(mode),
        x,
        u,
        bins,
        Val(RSAC);
        kwargs...,
    )
end

function SFO.StructureFunction(
    sf_sym::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
)
    return calculate_structure_function(Val(sf_sym), x, u, bins; kwargs...)
end

function SFO.StructureFunction(
    sf_sym::Symbol,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    return calculate_structure_function(Val(sf_sym), x, u, bins, Val(RSAC); kwargs...)
end

function SFO.StructureFunction(
    ::Val{sf_sym},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector;
    kwargs...,
) where {sf_sym}
    return calculate_structure_function(Val(sf_sym), x, u, bins; kwargs...)
end

function SFO.StructureFunction(
    ::Val{sf_sym},
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {sf_sym, RSAC}
    return calculate_structure_function(Val(sf_sym), x, u, bins, Val(RSAC); kwargs...)
end

"""
    gpu_calculate_structure_function(...)

GPU-accelerated structure function calculation. Requires loading `KernelAbstractions.jl`
to activate the `GPUExt` extension. The backend can be `KernelAbstractions.CPU()`
(for testing parity) or any GPU backend like `CUDABackend()` from `CUDA.jl`.

This stub exists so the extension can legally extend this function.
"""
function gpu_calculate_structure_function end

# Stub for file extensions
function calculate_structure_function_from_file end

# ---------------------------------------------------------------------------
# Functor Support: Operator(x, u, bins; kwargs...) -> calculate_structure_function
# ---------------------------------------------------------------------------

function (sf::SFT.AbstractStructureFunctionType)(x, u, bins; kwargs...)
    return calculate_structure_function(sf, x, u, bins; kwargs...)
end

function (sf::SFT.AbstractStructureFunctionType)(
    x,
    u,
    bins,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    return calculate_structure_function(sf, x, u, bins, Val(RSAC); kwargs...)
end

function (::Type{T})(x, u, bins; kwargs...) where {T <: SFT.AbstractStructureFunctionType}
    return calculate_structure_function(T(), x, u, bins; kwargs...)
end

function (::Type{T})(
    x,
    u,
    bins,
    ::Val{RSAC};
    kwargs...,
) where {T <: SFT.AbstractStructureFunctionType, RSAC}
    return calculate_structure_function(T(), x, u, bins, Val(RSAC); kwargs...)
end

"""
    calculate_structure_function(sf_type, x, u, bin_edges::AbstractVector{<:Number}, ...)

Convenience method that converts a vector of bin edges `[e1, e2, e3]` into 
adjacent bins `[(e1, e2), (e2, e3)]` and calls the core calculation.
"""
function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x::Union{Tuple, AbstractArray},
    u::Union{Tuple, AbstractArray},
    bin_edges::AbstractVector{<:Number},
    args...;
    kwargs...,
)
    bin_tuples = [(bin_edges[i], bin_edges[i + 1]) for i in 1:(length(bin_edges) - 1)]
    return calculate_structure_function(
        structure_function_type,
        x,
        u,
        bin_tuples,
        args...;
        kwargs...,
    )
end

###########
# Core Calculation Methods
########

"""
    calculate_structure_function(structure_function_type::AbstractStructureFunctionType, 
                                  x, u, distance_bins; backend=SerialBackend(), 
                                  return_sums_and_counts=false, kwargs...)

Core entry point for calculating structure functions from scattered data.

Structure functions characterize spatial correlations and scaling properties of turbulent fields.
This function computes pairwise distances between points and evaluates the structure function
operator for each pair, binning results by distance.

# Arguments

- `structure_function_type::AbstractStructureFunctionType`: The SF operator (e.g., 
  `LongitudinalSecondOrderStructureFunctionType()`)
- `x::Union{Tuple, AbstractArray}`: Position data, either:
  - `Tuple{T1, Vararg{T1}}`: Tuple of 1D position vectors, e.g., `(x_coords, y_coords)` for 2D
  - `AbstractArray{FT}`: Matrix where columns are points, e.g., `[x1 x2 x3; y1 y2 y3]` (2×3 for 3 2D points)
- `u::Union{Tuple, AbstractArray}`: Velocity/field data with same structure as `x`
- `distance_bins::AbstractVector{<:Tuple}`: Distance bin edges as tuples `[(r_min₁, r_max₁), ...]`

# Keyword Arguments

- `backend::AbstractExecutionBackend=SerialBackend()`: Execution backend:
  - `SerialBackend()`: Single-threaded reference implementation
  - `ThreadedBackend()`: Multi-threaded using OhMyThreads (requires `nthreads() > 1`)
  - `DistributedBackend()`: Multi-process/node using Distributed.jl
  - `GPUBackend(backend)`: GPU acceleration via KernelAbstractions.jl
  - `AutoBackend()`: Automatic selection based on availability
- `return_sums_and_counts::Bool=false`: If true, returns raw sums and counts before averaging
- `distance_metric::PreMetric=Euclidean()`: Distance metric (default: Euclidean)
- `verbose::Bool=true`: Print informational messages
- `show_progress::Bool=true`: Display progress bar during computation

# Returns

- `StructureFunction`: Result container with fields: `order`, `operator`, `distance_bins`, `values`
  - If `return_sums_and_counts=true`, returns `StructureFunctionSumsAndCounts` instead

# Examples

```julia
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT

# 2D velocity field with 3 points
x = ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
u = ([1.0, 1.1, 1.2], [0.0, 0.05, 0.1])

# Define distance bins in physical units
bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

# Calculate 2nd-order longitudinal structure function
sf_type = SFT.LongitudinalSecondOrderStructureFunctionType()
result = SFC.calculate_structure_function(sf_type, x, u, bins)

# With N points in D dimensions: O(N²) pairwise distance calculations

# Use threading for speed (if `Threads.nthreads() > 1`):
result = SFC.calculate_structure_function(sf_type, x, u, bins; 
                                         backend=SFC.ThreadedBackend())
```

See also: `serial_calculate_structure_function`, `parallel_calculate_structure_function`, 
`gpu_calculate_structure_function`, `StructureFunction`
"""
function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}};
    return_sums_and_counts::Bool = false,
    kwargs...,
) where {T1, T2, FT3}
    return calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        Val(return_sums_and_counts);
        kwargs...,
    )
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    ::Val{RSAC};
    backend::AbstractExecutionBackend = SerialBackend(),
    kwargs...,
) where {T1, T2, FT3, RSAC}
    return _dispatch_execution_backend(
        backend,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        Val(RSAC);
        kwargs...,
    )
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    backend::AbstractExecutionBackend = SerialBackend(),
    kwargs...,
) where {T1, T2, FT3}
    return _dispatch_execution_backend(
        backend,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    ::SerialBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    return serial_calculate_structure_function(
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    ::ThreadedBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    return threaded_calculate_structure_function(
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    ::DistributedBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    return parallel_calculate_structure_function(
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    ::AutoBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_backend_available(structure_function_type, x, u, distance_bins, value_bins)
        return _dispatch_execution_backend(
            DistributedBackend(),
            structure_function_type,
            x,
            u,
            distance_bins,
            value_bins;
            kwargs...,
        )
    end

    if Threads.nthreads() > 1 &&
       _threaded_backend_available(structure_function_type, x, u, distance_bins, value_bins)
        return _dispatch_execution_backend(
            ThreadedBackend(),
            structure_function_type,
            x,
            u,
            distance_bins,
            value_bins;
            kwargs...,
        )
    end

    return _dispatch_execution_backend(
        SerialBackend(),
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    backend::GPUBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    return _dispatch_execution_backend(
        SerialBackend(),
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function _dispatch_execution_backend(
    backend::AbstractExecutionBackend,
    structure_function_type::SFT.AbstractStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins::AbstractVector;
    kwargs...,
)
    return _dispatch_execution_backend(
        SerialBackend(),
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function serial_calculate_structure_function!(
    output::AbstractVector{OT},
    counts::AbstractVector{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, T1, T2, FT3}
    N = length(x_vecs)
    N3 = length(distance_bins)

    # Create a stable Vector for bins edges
    distance_bins_vec = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        distance_bins_vec[k] = distance_bins[k][1]
    end
    distance_bins_vec[end] = distance_bins[end][2]

    if verbose
        @info("calculating structure function (serial reduction)")
    end

    iter_inds = eachindex(x_vecs[1])
    # Pass Val(length(x_vecs)) to make it a type parameter in the work function
    vN = Val(length(x_vecs))
    PM.@showprogress enabled = show_progress for i in iter_inds
        calculate_structure_function_i!(
            output,
            counts,
            vN,
            structure_function_type,
            i,
            x_vecs,
            u_vecs,
            distance_bins_vec;
            distance_metric = distance_metric,
        )
    end
    return nothing
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    ::Val{RSAC};
    kwargs...,
) where {T1, T2, FT3, RSAC}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    OT = promote_type(float(FT1), float(FT2))
    N3 = length(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

    serial_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        kwargs...,
    )

    if RSAC # just return the sums and the counts, don't take the mean in each bin...
        return SFO.StructureFunctionSumsAndCounts(
            structure_function_type,
            distance_bins,
            output,
            counts,
        )
    else # do the mean in each bin.
        # Use explicit loop instead of broadcast to satisfy JET and avoid Makie dispatch
        output_div = similar(output)
        for k in eachindex(output)
            c = counts[k]
            output_div[k] = iszero(c) ? OT(NaN) : output[k] / c
        end
        return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
    end
end



@inline function flat_bin_edges(bins::AbstractVector{<:Tuple})
    N = length(bins)
    FT = eltype(bins[1])
    vec = Vector{FT}(undef, N + 1)
    for k in 1:N
        vec[k] = bins[k][1]
    end
    vec[end] = bins[end][2]
    return vec
end

@inline flat_bin_edges(bins::AbstractVector{<:Number}) = bins

function serial_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, T1, T2, FT3}
    distance_bins_vec = flat_bin_edges(distance_bins)
    value_bins_vec = flat_bin_edges(value_bins)

    if verbose
        @info("calculating 2D joint structure function (serial reduction)")
    end

    iter_inds = eachindex(x_vecs[1])
    vN = Val(length(x_vecs))
    
    PM.@showprogress enabled = show_progress for i in iter_inds
        calculate_structure_function_2d_i!(
            sums_2d,
            counts_2d,
            vN,
            structure_function_type,
            i,
            x_vecs,
            u_vecs,
            distance_bins_vec,
            value_bins_vec;
            distance_metric = distance_metric,
        )
    end
    return nothing
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    kwargs...,
) where {T1, T2, FT3}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    OT = promote_type(float(FT1), float(FT2))
    N3 = length(distance_bins)
    value_bins_vec = flat_bin_edges(value_bins)
    N4 = length(value_bins_vec) - 1

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(OT, N3, N4)

    serial_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        kwargs...,
    )

    return SFO.StructureFunction2D(
        structure_function_type,
        distance_bins,
        value_bins,
        sums_2d,
        counts_2d,
    )
end

function serial_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return serial_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return serial_calculate_structure_function(
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function calculate_structure_function_2d_i!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{OT},
    ::Val{N},
    structure_function_type::SFT.AbstractStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins_vec::AbstractVector{FT3},
    value_bins_vec::AbstractVector{FT4};
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, N, T1, T2, FT3, FT4}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins_vec)
    N4 = length(value_bins_vec)

    X1 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][i], Val(N)))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][i], Val(N)))

    iter_inds = eachindex(x_vecs[1])
    for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][j], Val(N)))
        U2 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][j], Val(N)))

        distance = distance_metric(X1, X2)
        dist_bin = SFH.digitize(distance, distance_bins_vec)
        if 1 <= dist_bin < N3
            val = structure_function_type(U2 - U1, SFH.r̂(X1, X2))
            val_bin = SFH.digitize(val, value_bins_vec)
            
            if 1 <= val_bin < N4
                @inbounds sums_2d[dist_bin, val_bin] += val
                @inbounds counts_2d[dist_bin, val_bin] += 1
            end
        end
    end
    return nothing
end

function calculate_structure_function_2d_i(
    structure_function_type::SFT.AbstractStructureFunctionType,
    i::Int,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
)
    N = length(x_vecs)
    FT1 = eltype(x_vecs[1])
    FT2 = eltype(u_vecs[1])
    N3 = length(distance_bins)
    
    distance_bins_vec = flat_bin_edges(distance_bins)
    value_bins_vec = flat_bin_edges(value_bins)
    N4 = length(value_bins_vec) - 1

    OT = promote_type(float(FT1), float(FT2))
    local_sums = zeros(OT, N3, N4)
    local_counts = zeros(OT, N3, N4)

    vN = Val(N)
    calculate_structure_function_2d_i!(
        local_sums,
        local_counts,
        vN,
        structure_function_type,
        i,
        x_vecs,
        u_vecs,
        distance_bins_vec,
        value_bins_vec;
        distance_metric = distance_metric,
    )

    return SFO.StructureFunction2D(
        structure_function_type,
        distance_bins,
        value_bins,
        local_sums,
        local_counts,
    )
end



function calculate_structure_function_i!(
    output::AbstractVector{OT},
    counts::AbstractVector{OT},
    ::Val{N},
    structure_function_type::SFT.AbstractStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins_vec::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, N, T1, T2, FT3}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins_vec)

    X1 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][i], Val(N)))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][i], Val(N)))

    iter_inds = eachindex(x_vecs[1])
    for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][j], Val(N)))
        U2 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][j], Val(N)))

        distance = distance_metric(X1, X2)
        bin = SFH.digitize(distance, distance_bins_vec)
        if 1 <= bin < N3
            @inbounds output[bin] += structure_function_type(U2 - U1, SFH.r̂(X1, X2))
            @inbounds counts[bin] += 1
        end
    end
    return nothing
end

"""
    calculate_structure_function_i(...)

Result-object wrapper for `calculate_structure_function_i!`.
Returns a `StructureFunctionSumsAndCounts` for a single point `i`'s contributions.
Used primary by the `@distributed` backend.
"""
function calculate_structure_function_i(
    structure_function_type::SFT.AbstractStructureFunctionType,
    i::Int,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
)
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    N3 = length(distance_bins)
    local_output = zeros(OT, N3)
    local_counts = zeros(OT, N3)

    # Kernel expects edges
    FT3 = eltype(eltype(distance_bins))
    bin_edges = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        bin_edges[k] = distance_bins[k][1]
    end
    bin_edges[end] = distance_bins[end][2]

    calculate_structure_function_i!(
        local_output, local_counts,
        Val(length(x_vecs)),
        structure_function_type, i, x_vecs, u_vecs, bin_edges;
        distance_metric = distance_metric,
    )
    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        local_output,
        local_counts,
    )
end




function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing = :logarithmic,
    verbose::Bool = true,
    show_progress::Bool = true,
    return_sums_and_counts::Bool = false,
) where {T1, T2}
    N = length(x_vecs)
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.

    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _min_distance, _max_distance = minmax_i(i, x_vecs, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
    end

    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin (note this is needed so things matching min_distance don't get assigned bin '0', alternative is to check for bin 0 every time... which sounds slower
    if bin_spacing == :linear
        distance_bins = range(min_distance, max_distance, length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
    elseif bin_spacing ∈ (:logarithmic, :log)
        distance_bins =
            10 .^
            range(log10(min_distance), log10(max_distance), length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
        distance_bins[1] = min_distance # combat floating point errors (may have rounded up or down during operations)
        distance_bins[end] = max_distance # combat floating point errors (may have rounded up or down during operations)
    else
        error("bin_spacing must be :linear or :logarithmic/:log")
    end
    FT3 = eltype(distance_bins)
    distance_bins = SA.SVector{n_distance_bins, Tuple{FT3, FT3}}(
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins]...,
    ) # convert to tuples of the bin edges
    return calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        return_sums_and_counts = return_sums_and_counts,
    )
end


function minmax_i(
    i::Int,
    x_vecs::Tuple{T, Vararg{T}},
    distance_metric = DI.Euclidean(),
) where {T}
    N = length(x_vecs)
    FT = eltype(T)
    """ calculate, bin, and mean the pairwise distances """


    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = SA.SVector{N, FT}(ntuple(k -> x_vecs[k][i], Val{N}()))
    for j in eachindex(x_vecs[1])
        if i != j
            @inbounds X2 = SA.SVector{N, FT}(ntuple(k -> x_vecs[k][j], Val{N}()))
            # update the min and max distances
            distance = distance_metric(X1, X2)
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end


### ====================================================== ###
### ====================================================== ###
### ====================================================== ###
"""
Array version (is slower lol)
"""
# array version seems to be slower lol, idk why (and we're even using views!)

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}};
    return_sums_and_counts::Bool = false,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    return calculate_structure_function(
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins,
        Val(return_sums_and_counts);
        kwargs...,
    )
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    ::Val{RSAC};
    backend::AbstractExecutionBackend = SerialBackend(),
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, RSAC}
    return _dispatch_execution_backend(
        backend,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins,
        Val(RSAC);
        kwargs...,
    )
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    backend::AbstractExecutionBackend = SerialBackend(),
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    return _dispatch_execution_backend(
        backend,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins,
        value_bins;
        kwargs...,
    )
end


function serial_calculate_structure_function!(
    output::AbstractVector{OT},
    counts::AbstractVector{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N3 = length(distance_bins)

    # Create a stable Vector for bins edges
    distance_bins_vec = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        distance_bins_vec[k] = distance_bins[k][1]
    end
    distance_bins_vec[end] = distance_bins[end][2]

    if verbose
        @info("calculating structure function (serial reduction)")
    end

    iter_inds = axes(x_arr, 2)
    N = size(x_arr, 1)
    if N == 1
        PM.@showprogress enabled = show_progress for i in iter_inds
            calculate_structure_function_i!(
                output,
                counts,
                Val(1),
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
        end
    elseif N == 2
        PM.@showprogress enabled = show_progress for i in iter_inds
            calculate_structure_function_i!(
                output,
                counts,
                Val(2),
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
        end
    elseif N == 3
        PM.@showprogress enabled = show_progress for i in iter_inds
            calculate_structure_function_i!(
                output,
                counts,
                Val(3),
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
        end
    else
        throw(ArgumentError("Array backend supports only 1D, 2D, or 3D inputs."))
    end
    return nothing
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    ::Val{RSAC};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, RSAC}
    OT = promote_type(float(FT1), float(FT2))
    N3 = length(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

    serial_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins;
        kwargs...,
    )

    if RSAC # just return the sums and the counts, don't take the mean in each bin...
        return SFO.StructureFunctionSumsAndCounts(
            structure_function_type,
            distance_bins,
            output,
            counts,
        )
    else # do the mean in each bin.
        # Use explicit loop instead of broadcast to satisfy JET and avoid Makie dispatch
        output_div = similar(output)
        for k in eachindex(output)
            c = counts[k]
            output_div[k] = iszero(c) ? OT(NaN) : output[k] / c
        end
        return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
    end
end

function calculate_structure_function_i!(
    output::AbstractVector{FT},
    counts::AbstractVector{FT},
    ::Val{N},
    structure_function_type::SFT.AbstractStructureFunctionType,
    i::Int,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins_vec::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {FT, N, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N3 = length(distance_bins_vec)
    iter_inds = axes(x_arr, 2)

    X1 = SA.SVector{N, FT1}(ntuple(k -> x_arr[k, i], Val(N)))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_arr[k, i], Val(N)))

    # Restore symmetry (j > i) for $O(N^2/2)$ performance
    for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{N, FT1}(ntuple(k -> x_arr[k, j], Val(N)))
        U2 = SA.SVector{N, FT2}(ntuple(k -> u_arr[k, j], Val(N)))

        distance = distance_metric(X1, X2)
        bin = SFH.digitize(distance, distance_bins_vec)
        if 1 <= bin < N3
            @inbounds output[bin] += structure_function_type(U2 - U1, SFH.r̂(X1, X2))
            @inbounds counts[bin] += 1
        end
    end
    return nothing
end



function calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing = :logarithmic,
    verbose::Bool = true,
    show_progress::Bool = true,
    return_sums_and_counts::Bool = false,
) where {FT1 <: Number, FT2 <: Number}
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.

    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    iter_inds = axes(x_arr, 2) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # iter_inds = axes(x_arr,1) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _min_distance, _max_distance = minmax_i(i, x_arr, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
    end

    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin (note this is needed so things matching min_distance don't get assigned bin '0', alternative is to check for bin 0 every time... which sounds slower
    if bin_spacing == :linear
        distance_bins = range(min_distance, max_distance, length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
    elseif bin_spacing ∈ (:logarithmic, :log)
        distance_bins =
            10 .^
            range(log10(min_distance), log10(max_distance), length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
        distance_bins[1] = min_distance # combat floating point errors (may have rounded up or down during operations)
        distance_bins[end] = max_distance # combat floating point errors (may have rounded up or down during operations)
    else
        error("bin_spacing must be :linear or :logarithmic/:log")
    end
    FT3 = eltype(distance_bins)
    distance_bins = SA.SVector{n_distance_bins, Tuple{FT3, FT3}}(
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins]...,
    ) # convert to tuples of the bin edges
    return calculate_structure_function(
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        return_sums_and_counts = return_sums_and_counts,
    )
end

"""
    minmax_i(i, x_vecs, distance_metric)

Calculate the min and max distances from point `i` to all other points `j != i`.
Used for automatic binning.
"""
function minmax_i(
    i::Int,
    x_vecs::Tuple,
    distance_metric = DI.Euclidean(),
)
    N = length(x_vecs)
    FT = eltype(x_vecs[1])
    X1 = SA.SVector{N, FT}(ntuple(k -> x_vecs[k][i], Val(N)))

    min_distance, max_distance = FT(Inf), FT(0.0)
    iter_inds = eachindex(x_vecs[1])
    for j in iter_inds
        if i != j
            X2 = SA.SVector{N, FT}(ntuple(k -> x_vecs[k][j], Val(N)))
            distance = distance_metric(X1, X2)
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end

function minmax_i(
    i::Int,
    x_arr::AbstractArray{FT},
    distance_metric = DI.Euclidean(),
) where {FT <: Real}
    """ calculate, bin, and mean the pairwise distances """


    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = @view(x_arr[:, i])
    # @inbounds X1 = @view(x_arr[i, :])

    for j in axes(x_arr, 2)
        # for j in axes(x_arr,1)
        if i != j
            @inbounds X2 = @view(x_arr[:, j])
            # @inbounds X2 = @view(x_arr[j, :]) 

            # update the min and max distances
            distance = distance_metric(X1, X2) # this is the slow part
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end






































function parallel_calculate_structure_function end # placeholder for parallel extension

function parallel_calculate_structure_function end # placeholder for parallel extension

"""
    serial_calculate_structure_functions_single_pass(x, u, distance_bins, sums, counts)

Calculates the core 8 structure function sums and counts on a single CPU thread.
To achieve absolute maximum performance and eliminate heap allocations:
1. Spatial coordinate differences are computed via `SFH.δr` and stack-allocated.
2. Distance binning is determined via the package's optimized `SFH.digitize`.
3. To avoid redundant projection dot product FLOPs (which would occur if we called the 8 inlined SFT functors independently),
   we precompute the longitudinal `du_L` and transverse `du_T` increments once per coordinate pair:
   - `du_L = LA.dot(du, rh)` where `rh = SFH.r̂(x_i, x_j)`
   - `du_T = LA.dot(du, nh)` where `nh = SFH.n̂(rh)`
   These projections are mathematically identical to calling `mδu_l` and `mδu_t` in the `ProjectedStructureFunctionType` functors,
   preserving perfect physical alignment with the package's type hierarchy.
"""
function serial_calculate_structure_functions_single_pass(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{Int64}
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT}
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    
    fill!(sums, zero(OT))
    fill!(counts, 0)
    
    for i in 1:n_points
        x_i = SA.SVector{2, FT1}(x[1, i], x[2, i])
        u_i = SA.SVector{2, FT2}(u[1, i], u[2, i])
        
        for j in (i+1):n_points
            x_j = SA.SVector{2, FT1}(x[1, j], x[2, j])
            
            dx = SFH.δr(x_i, x_j)
            r = LA.norm(dx)
            
            bin_idx = SFH.digitize(r, distance_bins)
            
            if 1 <= bin_idx <= n_bins
                u_j = SA.SVector{2, FT2}(u[1, j], u[2, j])
                du = u_j - u_i
                
                rh = SFH.r̂(x_i, x_j)
                nh = SFH.n̂(rh)
                
                du_L = LA.dot(du, rh)
                du_T = LA.dot(du, nh)
                
                du_L2 = du_L * du_L
                du_T2 = du_T * du_T
                
                # Natively accumulate the 8 core physical structure functions:
                @inbounds sums[1, bin_idx] += du_L2 + du_T2            # S2SF (Full Vector 2nd-order)
                @inbounds sums[2, bin_idx] += du_L2                   # L2SF (Longitudinal 2nd-order)
                @inbounds sums[3, bin_idx] += du_T2                   # T2SF (Transverse 2nd-order)
                @inbounds sums[4, bin_idx] += du_L * (du_L2 + du_T2)  # S3SF (Full Vector 3rd-order)
                @inbounds sums[5, bin_idx] += du_L * du_L2             # L3SF (Longitudinal 3rd-order)
                @inbounds sums[6, bin_idx] += du_L2 * du_T             # L2T1SF (Diagonal Inconsistent)
                @inbounds sums[7, bin_idx] += du_L * du_T2             # L1T2SF (Off-Diagonal Inconsistent)
                @inbounds sums[8, bin_idx] += du_T * du_T2             # T3SF (Transverse 3rd-order)
                
                for t in 1:8
                    @inbounds counts[t, bin_idx] += 1
                end
            end
        end
    end
    
    return sums, counts
end

# ---------------------------------------------------------------------------
# Modular Execution Backend Dispatch System (Single-Pass)
# ---------------------------------------------------------------------------

function _dispatch_single_pass end

"""
    _dispatch_single_pass(::SerialBackend, x, u, distance_bins; thread_sums=nothing, thread_counts=nothing)

Calculates the single-pass structure functions on a single thread using SerialBackend.
"""
function _dispatch_single_pass(
    ::SerialBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    thread_sums = nothing,
    thread_counts = nothing,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    
    ts = isnothing(thread_sums) ? zeros(OT, 8, n_bins) : thread_sums
    tc = isnothing(thread_counts) ? zeros(Int64, 8, n_bins) : thread_counts
    
    sums, counts = serial_calculate_structure_functions_single_pass(x, u, distance_bins, ts, tc)
    return postprocess_single_pass_results(sums, counts, distance_bins)
end

"""
    _dispatch_single_pass(::ThreadedBackend, x, u, distance_bins; kwargs...)

Stub/placeholder for OhMyThreads-based CPU multi-threaded execution.
Overridden by `StructureFunctionsOhMyThreadsExt.jl` when the OhMyThreads package is loaded.
"""
function _dispatch_single_pass(
    ::ThreadedBackend,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector;
    kwargs...
)
    throw(
        ArgumentError(
            "Threaded single-pass backend is unavailable. Load the OhMyThreads extension or use backend=SerialBackend().",
        ),
    )
end

"""
    _dispatch_single_pass(::DistributedBackend, x, u, distance_bins; kwargs...)

Stub/placeholder for Distributed.jl-based multi-process/cluster execution.
Overridden by `StructureFunctionsDistributedExt.jl` when the Distributed package is loaded.
"""
function _dispatch_single_pass(
    ::DistributedBackend,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector;
    kwargs...
)
    throw(
        ArgumentError(
            "Distributed single-pass backend is unavailable. Load the Distributed/SharedArrays extension or use backend=SerialBackend().",
        ),
    )
end

"""
    _dispatch_single_pass(::GPUBackend, x, u, distance_bins; kwargs...)

Stub/placeholder for KernelAbstractions.jl-based GPU execution.
Overridden by `StructureFunctionsGPUExt.jl` when the KernelAbstractions package is loaded.
"""
function _dispatch_single_pass(
    ::GPUBackend,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector;
    kwargs...
)
    throw(
        ArgumentError(
            "GPU single-pass backend is unavailable. Load the GPUExt extension or use backend=SerialBackend().",
        ),
    )
end

# Checkers to detect dynamically if extensions loaded and overrode dispatch methods
_threaded_single_pass_available(x, u, distance_bins) = hasmethod(
    _dispatch_single_pass,
    Tuple{ThreadedBackend, typeof(x), typeof(u), typeof(distance_bins)}
)

_distributed_single_pass_available(x, u, distance_bins) = hasmethod(
    _dispatch_single_pass,
    Tuple{DistributedBackend, typeof(x), typeof(u), typeof(distance_bins)}
)

"""
    _dispatch_single_pass(::AutoBackend, x, u, distance_bins; kwargs...)

Resolves the execution backend automatically depending on active resources and workers.
"""
function _dispatch_single_pass(
    ::AutoBackend,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector;
    kwargs...
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass(DistributedBackend(), x, u, distance_bins; kwargs...)
    end
    
    if Threads.nthreads() > 1 &&
       _threaded_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass(ThreadedBackend(), x, u, distance_bins; kwargs...)
    end
    
    return _dispatch_single_pass(SerialBackend(), x, u, distance_bins; kwargs...)
end

# ---------------------------------------------------------------------------
# Public Entrypoint
# ---------------------------------------------------------------------------

"""
    calculate_structure_functions_single_pass(x, u, distance_bins; backend=AutoBackend(), kwargs...)

Primary high-performance pipeline entrypoint for SMODE.
Calculates all 10 structure functions (the 8 core 2nd/3rd order ones plus the rotational/divergent Helmholtz integrations)
in a single pass over position differences, dispatching dynamically to the specified `backend`.

# Arguments
- `x::AbstractMatrix`: Spatial positions of shape `(2, N)`.
- `u::AbstractMatrix`: Horizontal velocities of shape `(2, N)`.
- `distance_bins::AbstractVector`: Monotonically increasing bin edges (typically log-spaced).
- `backend::AbstractExecutionBackend`: Execution target (`SerialBackend()`, `ThreadedBackend()`, `DistributedBackend()`, `GPUBackend()`, or `AutoBackend()`).

# Reusable Buffers (Keyword Arguments)
- `thread_sums`: Optional pre-allocated buffer of shape `(8, n_bins, n_threads)` for ThreadedBackend, or `(8, n_bins)` for SerialBackend.
- `thread_counts`: Optional pre-allocated buffer of shape `(8, n_bins, n_threads)` for ThreadedBackend, or `(8, n_bins)` for SerialBackend.
"""
function calculate_structure_functions_single_pass(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    backend::AbstractExecutionBackend = AutoBackend(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    return _dispatch_single_pass(backend, x, u, distance_bins; kwargs...)
end

# ---------------------------------------------------------------------------
# Isotropic Helmholtz Decomposition Postprocessing
# ---------------------------------------------------------------------------

"""
    postprocess_single_pass_results(sums, counts, distance_bins)

Runs the 2D isotropic Helmholtz decomposition natively using the trapezoidal rule over the binned structure functions.
This implements the cumulative integral equations described by Erik Lindborg (JFM 2015) and Bühler, Callies, and Ferrari (JFM 2014)
to separate longitudinal (\$D_{LL}\$) and transverse (\$D_{TT}\$) structure functions into rotational (\$D_{\\text{rot}}\$) and divergent (\$D_{\\text{div}}\$) modes.

# Mathematical Formulation
Given the isotropic horizontal kinetic energy structure functions:
- \$D_{LL}(r) = \\langle (\\delta u_L)^2 \\rangle\$
- \$D_{TT}(r) = \\langle (\\delta u_T)^2 \\rangle\$

The rotational and divergent structure functions satisfy:
\$\$D_{\\text{rot}}(r) = D_{TT}(r) + r \\int_0^r \\frac{D_{TT}(s) - D_{LL}(s)}{s} \\, ds\$\$
\$\$D_{\\text{div}}(r) = D_{LL}(r) - r \\int_0^r \\frac{D_{TT}(s) - D_{LL}(s)}{s} \\, ds\$\$

We evaluate the cumulative integral using a trapezoidal rule on log-spaced bin midpoints \$r_k\$:
\$\$I(r_k) = \\sum_{j=2}^{k} \\frac{1}{2} \\left( \\frac{D_{TT}(r_j) - D_{LL}(r_j)}{r_j} + \\frac{D_{TT}(r_{j-1}) - D_{LL}(r_{j-1})}{r_{j-1}} \\right) (r_j - r_{j-1})\$\$

The derived structure functions are returned as the 9th (rotational) and 10th (divergent) indices of the final matrices.
"""
function postprocess_single_pass_results(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{Int64},
    distance_bins::AbstractVector{FT3}
) where {OT, FT3}
    n_bins = size(sums, 2)
    final_sums = zeros(OT, 10, n_bins)
    final_counts = zeros(Int64, 10, n_bins)
    
    # Core 8 structure functions are copied unchanged
    final_sums[1:8, :] .= sums
    final_counts[1:8, :] .= counts
    
    # Calculate bin midpoints from log-spaced edges
    min_log_dist = log(distance_bins[1])
    max_log_dist = log(distance_bins[end])
    log_step = (max_log_dist - min_log_dist) / n_bins
    
    bin_mids = zeros(FT3, n_bins)
    for k in 1:n_bins
        bin_mids[k] = exp(min_log_dist + (k - 0.5f0) * log_step)
    end
    
    # Compute normalized second-order longitudinal (2) and transverse (3) functions
    D_LL = sums[2, :] ./ max.(counts[2, :], 1)
    D_TT = sums[3, :] ./ max.(counts[3, :], 1)
    
    # Evaluate cumulative trapezoidal integral
    I = zeros(OT, n_bins)
    for k in 2:n_bins
        F_prev = (D_TT[k-1] - D_LL[k-1]) / bin_mids[k-1]
        F_curr = (D_TT[k] - D_LL[k]) / bin_mids[k]
        ds = bin_mids[k] - bin_mids[k-1]
        I[k] = I[k-1] + 0.5f0 * (F_prev + F_curr) * ds
    end
    
    # Rotational and divergent functions copy the count bounds of the longitudinal function
    final_counts[9, :] .= final_counts[2, :]
    final_counts[10, :] .= final_counts[2, :]
    
    for k in 1:n_bins
        D_rot = D_TT[k] + bin_mids[k] * I[k]
        D_div = D_LL[k] - bin_mids[k] * I[k]
        
        final_sums[9, k] = D_rot * final_counts[9, k]
        final_sums[10, k] = D_div * final_counts[10, k]
    end
    
    return final_sums, final_counts
end

end
