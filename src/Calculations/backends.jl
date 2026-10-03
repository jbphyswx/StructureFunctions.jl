# Backend tags come from ComputationalBackends; methods dispatch on the abstract types so
# downstream concrete backends reach these kernels.

using ComputationalBackends: ComputationalBackends as CB

"""
    resolve_auto_backend(; nthreads = Threads.nthreads())

What `AutoBackend` resolves to for every entry: `DistributedBackend()` when Distributed workers reach
cores this process does not use ([`distributed_adds_hardware`](@ref)), else `ThreadedBackend()` when
there is more than one thread and the OhMyThreads extension is loaded, else `SerialBackend()`. Every
entry family has a method for each of the three.
"""
function resolve_auto_backend(; nthreads::Int = Threads.nthreads())
    distributed_adds_hardware(Val(:distributed)) && return CB.DistributedBackend()
    (nthreads > 1 && _ohmythreads_loaded()) && return CB.ThreadedBackend()
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
    distributed_adds_hardware(::Val{:distributed}) -> Bool

Whether Distributed workers reach cores this process does not use: any worker a cluster manager
placed elsewhere, or workers on this node while this process runs one thread. Supplied by the
Distributed extension; `false` without it. It decides only what `AutoBackend` resolves to.
"""
distributed_adds_hardware(::Val) = false
