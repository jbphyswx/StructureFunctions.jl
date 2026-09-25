# Backend tags come from ComputationalBackends; methods dispatch on the abstract types so
# downstream concrete backends reach these kernels.

using ComputationalBackends: ComputationalBackends as CB

"""
    resolve_auto_backend(shape, threaded_available; nthreads=Threads.nthreads())

What `AutoBackend` resolves to. `threaded_available` is a zero-argument predicate, since the
OhMyThreads extension is optional; [`distributed_adds_hardware`](@ref) is the whole test for the
Distributed backend, which the extension covers for every entry family.

Every candidate is tested before it is named, so `AutoBackend` only ever resolves to a backend that
can run the request.

`nthreads` defaults to `Threads.nthreads()`, which is fixed at process start.
"""
function resolve_auto_backend(
    shape, threaded_available::F; nthreads::Int = Threads.nthreads(),
) where {F}
    distributed_adds_hardware(Val(:distributed)) && return CB.DistributedBackend()
    (nthreads > 1 && threaded_available()) && return CB.ThreadedBackend()
    return CB.SerialBackend()
end

function _dispatch_execution_backend(
    ::CB.AbstractMPIBackend, args...; kwargs...,
)
    throw(ArgumentError("MPI backend is unavailable. Load MPI (`using MPI`) to enable StructureFunctionsMPIExt, or use a different backend."))
end

"""
    threaded_calculate_structure_function(sf, x, u, distance_bins[, value_bins], CT; kwargs...)

The point-list structure function on the threaded CPU backend, returning the raw sums and counts.
Takes the arguments of [`calculate_structure_function`](@ref) without `backend`; supplied by the
OhMyThreads extension, so `using OhMyThreads` is required.
"""
function threaded_calculate_structure_function(args...; kwargs...)
    throw(
        ArgumentError(
            "Threaded backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend().",
        ),
    )
end

"""
    threaded_calculate_structure_function!(sums, counts, sf, x, u, distance_bins[, value_bins]; kwargs...)

The in-place form of [`threaded_calculate_structure_function`](@ref), accumulating into `sums` and
`counts`; supplied by the OhMyThreads extension.
"""
function threaded_calculate_structure_function!(args...; kwargs...)
    throw(
        ArgumentError(
            "Threaded backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend().",
        ),
    )
end

# Set to `true` by the OhMyThreads extension's `__init__`. This is what `AutoBackend` tests: the
# throwing stub above makes `hasmethod` true whether or not the extension is loaded. A `Ref` set at
# load time, because overwriting a method during the extension's precompilation is illegal.
const _OHMYTHREADS_LOADED = Ref(false)
_ohmythreads_loaded() = _OHMYTHREADS_LOADED[]

"""
    _auto_local_backend()

The CPU backend `AutoBackend` resolves to: `ThreadedBackend()` when the process has more than one
thread and the OhMyThreads extension supplies the threaded driver, `SerialBackend()` otherwise.
"""
_auto_local_backend() =
    (Threads.nthreads() > 1 && _ohmythreads_loaded()) ? CB.ThreadedBackend() : CB.SerialBackend()

function _threaded_backend_available(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins,
)
    return _ohmythreads_loaded()
end

function _threaded_backend_available!(
    sums,
    counts,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins,
)
    return _ohmythreads_loaded()
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
    return _ohmythreads_loaded()
end

function _dispatch_execution_backend(
    ::CB.AbstractDistributedBackend, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_execution_backend(
    ::CB.AbstractDistributedBackend, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector, ::Type; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend, sums::AbstractArray, counts::AbstractArray, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend, sums_2d::AbstractArray, counts_2d::AbstractArray, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

"""
    distributed_adds_hardware(::Val{:distributed}) -> Bool

Whether the worker pool reaches cores this process cannot reach on its own — the question
`AutoBackend` asks before it names [`DistributedBackend`](@ref).

Workers on the driver's own node redistribute the cores the threaded backend already uses and pay
serialisation on top, so `Auto` prefers the local backend for them; workers placed by a cluster
manager bring hardware threads cannot. Supplied by the Distributed extension, which reads the
worker's `ClusterManager`; `false` without it.

An explicit `DistributedBackend()` is unaffected — this decides only what `Auto` chooses.
"""
distributed_adds_hardware(::Val) = false
