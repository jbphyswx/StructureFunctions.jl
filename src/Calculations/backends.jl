# Backend tags come from ComputationalBackends; methods dispatch on the abstract types so
# downstream concrete backends reach these kernels.

using ComputationalBackends: ComputationalBackends as CB

"""
    resolve_auto_backend(shape, threaded_available, distributed_available; nthreads=Threads.nthreads())

What `AutoBackend` resolves to. `threaded_available`/`distributed_available` are zero-argument
predicates because each entry family probes a different dispatch.

`nthreads` defaults to `Threads.nthreads()`, which is fixed at process start.
"""
function resolve_auto_backend(
    shape, threaded_available::F, distributed_available::G;
    nthreads::Int = Threads.nthreads(),
) where {F, G}
    (distributed_workers_available(Val(:distributed)) && distributed_available()) &&
        return CB.DistributedBackend()
    has_auxiliary_axes(shape) &&
        return nthreads > 1 ? CB.ThreadedBackend() : CB.SerialBackend()
    (nthreads > 1 && threaded_available()) && return CB.ThreadedBackend()
    return CB.SerialBackend()
end

function _dispatch_execution_backend(
    ::CB.AbstractMPIBackend, args...; kwargs...,
)
    throw(ArgumentError("MPI backend is unavailable. Load MPI (`using MPI`) to enable StructureFunctionsMPIExt, or use a different backend."))
end

function threaded_calculate_structure_function(args...; kwargs...)
    throw(
        ArgumentError(
            "Threaded backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend().",
        ),
    )
end

function threaded_calculate_structure_function!(args...; kwargs...)
    throw(
        ArgumentError(
            "Threaded backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend().",
        ),
    )
end

# Set to `true` by the OhMyThreads extension's `__init__` when it loads. AutoBackend gates its
# threaded choice on this rather than `hasmethod`, because the throwing stub above makes
# `hasmethod` always true — which previously fooled AutoBackend into the threaded path (then it
# threw) when OhMyThreads was not loaded. With this flag, AutoBackend falls back to serial.
# A `Ref` (set at load via `__init__`) is used instead of a method override, which would be an
# illegal method-overwrite during the extension's precompilation.
const _OHMYTHREADS_LOADED = Ref(false)
_ohmythreads_loaded() = _OHMYTHREADS_LOADED[]

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
    ::CB.AbstractDistributedBackend, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    throw(ArgumentError("Distributed backend is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_execution_backend(
    ::CB.AbstractDistributedBackend, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
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

distributed_workers_available(::Val) = false
