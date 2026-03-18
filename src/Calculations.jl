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
    serial_calculate_structure_function, threaded_calculate_structure_function

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

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {T1, T2, FT3, RSAC}
    N = length(x_vecs)
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop

    # preallocate output as vector of length of distance_bins
    # Use promote_type and float to ensure output can hold float results/NaN
    OT = promote_type(float(FT1), float(FT2))
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

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

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, RSAC}
    N3 = length(distance_bins)
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop

    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    OT = promote_type(float(FT1), float(FT2))
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

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

end
