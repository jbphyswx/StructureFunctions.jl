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

"""
    threaded_calculate_structure_function(sf, x, u, distance_bins[, value_bins], CT; kwargs...)

The point-list structure function on the threaded CPU backend, returning the raw sums and counts.
Takes the arguments of [`calculate_structure_function`](@ref) without `backend`; supplied by the
OhMyThreads extension.
"""
function threaded_calculate_structure_function end

"""
    threaded_calculate_structure_function!(sums, counts, sf, x, u, distance_bins[, value_bins]; kwargs...)

The in-place form of [`threaded_calculate_structure_function`](@ref), accumulating into `sums` and
`counts`; supplied by the OhMyThreads extension.
"""
function threaded_calculate_structure_function! end

"""Whether an extension is loaded; each is set by that extension's `__init__`."""
const _OHMYTHREADS_LOADED = Ref(false)
const _KERNELABSTRACTIONS_LOADED = Ref(false)
const _DISTRIBUTED_LOADED = Ref(false)
const _MPI_LOADED = Ref(false)
const _ABSTRACTFFTS_LOADED = Ref(false)
_ohmythreads_loaded() = _OHMYTHREADS_LOADED[]

"""
    _require_backend(backend)

Throw an `ArgumentError` naming the package to load when an extension `backend` needs is not
loaded: OhMyThreads for a threaded backend, KernelAbstractions for a GPU backend, Distributed or MPI
for those wrappers and then whatever their local backend needs. Serial and `Auto` need none.
"""
_require_backend(::CB.AbstractExecutionBackend) = nothing
_require_backend(b::CB.AbstractThreadedBackend) = _require_package(b, _OHMYTHREADS_LOADED, "OhMyThreads")
_require_backend(b::CB.AbstractGPUBackend) = _require_package(b, _KERNELABSTRACTIONS_LOADED, "KernelAbstractions")
function _require_backend(b::CB.AbstractDistributedBackend)
    _require_package(b, _DISTRIBUTED_LOADED, "Distributed")
    return _require_backend(CB.local_backend(b))
end
function _require_backend(b::CB.AbstractMPIBackend)
    _require_package(b, _MPI_LOADED, "MPI")
    return _require_backend(CB.local_backend(b))
end

_require_package(backend, loaded::Base.RefValue{Bool}, package::String) = loaded[] ? nothing :
    throw(ArgumentError(
        "backend = $(nameof(typeof(backend))) needs `using $package`, which supplies it; or pass " *
        "backend = SerialBackend(), or AutoBackend() to take whichever backend is available.",
    ))

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
