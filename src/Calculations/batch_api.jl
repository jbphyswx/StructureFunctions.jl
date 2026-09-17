# Batch Drivers and Entry Points (batch over the trailing slice/time axis of `(D, N, T)` inputs)

"""
    calculate_structure_function_batch!(sums, counts, sf_type, x, u, distance_bins; backend=..., workspace=nothing, ...)
    calculate_structure_function_batch!(sums, counts, sf_type, grid, u, distance_bins[, axis_bins][, spectral_backend]; ...)

Structure functions of one fixed sampling observed over a trailing slice axis, accumulated into
`sums` and `counts` of shape `(NB, T)` with `NB = length(distance_bins) - 1`.

With coordinates `x` the sampling is a point list, `(N_dims, N_points, T)`, and the pair loop runs
once for the whole batch: `GPUBackend` keeps the batch on the device and synchronises once when
`KernelAbstractions` is loaded.

With a grid or a lag schedule in place of `x` the sampling is a grid, the field is
`(component, cells..., T)`, and the lag enumeration runs once for the whole batch; `axis_bins` makes
the histogram joint in separation and angle, `sums` and `counts` then being `(NB, n_angle, T)`. The
grid form is supplied by the FlowGeometries extension and the schedule forms by
[`gridded_sweep_batch!`](@ref).
"""
function calculate_structure_function_batch!(
    sums, counts, sf_type, x, u, distance_bins;
    backend = CB.SerialBackend(), kwargs...
)
    _dispatch_batch!(backend, sums, counts, sf_type, x, u, distance_bins; kwargs...)
    return nothing
end

"""
    _require_threading(what)

Refuse an explicit `ThreadedBackend()` when the OhMyThreads extension is not loaded. The threaded
batch drivers are defined in core as the serial ones so that results stay correct, and a caller who
asked for threading is told rather than quietly given one task; `AutoBackend()` reaches the serial
driver directly and is unaffected.
"""
_require_threading(what::AbstractString) = _ohmythreads_loaded() ? nothing : throw(ArgumentError(
    "backend = ThreadedBackend() needs the OhMyThreads extension for $what; `using OhMyThreads` " *
    "enables it, or pass backend = SerialBackend() for the serial driver, or AutoBackend() to take " *
    "whichever is available.",
))

function _dispatch_batch!(
    ::CB.AbstractSerialBackend, sums, counts, sf_type, x, u, distance_bins; kwargs...
)
    auxiliary_structure_function!(sums, counts, sf_type, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_batch!(
    ::CB.AbstractThreadedBackend, sums, counts, sf_type, x, u, distance_bins; kwargs...
)
    _require_threading("the slice batch driver")
    auxiliary_structure_function_threaded!(sums, counts, sf_type, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_batch!(
    ::CB.AbstractDistributedBackend, sums, counts, sf_type, x, u, distance_bins; kwargs...
)
    throw(
        ArgumentError(
            "Distributed slice batch driver is not implemented yet. Loop over time slices or use backend=CB.GPUBackend(...).",
        ),
    )
end

function _dispatch_batch!(
    backend::CB.AbstractGPUBackend, sums, counts, sf_type, x, u, distance_bins; kwargs...
)
    gpu_calculate_structure_function_batch!(
        sums, counts, sf_type, backend.backend, x, u, distance_bins; kwargs...
    )
    return nothing
end

"""
    calculate_structure_function_2d_batch!(sums, counts, sf_type, x, u, distance_bins, value_bins; backend=..., ...)

Batch 2D joint histograms over `(N_dims, N_points, T)`; outputs `(n_dist, n_val, T)`.
"""
function calculate_structure_function_2d_batch!(
    sums, counts, sf_type, x, u, distance_bins, value_bins;
    backend = CB.SerialBackend(), kwargs...
)
    _dispatch_2d_batch!(backend, sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_2d_batch!(
    ::CB.AbstractSerialBackend, sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...
)
    auxiliary_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_2d_batch!(
    ::CB.AbstractThreadedBackend, sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...
)
    _require_threading("the 2D joint slice batch driver")
    auxiliary_joint2d_threaded!(sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_2d_batch!(
    ::CB.AbstractDistributedBackend, args...; kwargs...
)
    throw(ArgumentError("Distributed 2D joint slice batch not implemented yet; use GPUBackend or loop over t."))
end

function _dispatch_2d_batch!(
    backend::CB.AbstractGPUBackend, sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...
)
    gpu_calculate_structure_function_2d_batch!(
        sums, counts, sf_type, backend.backend, x, u, distance_bins, value_bins; kwargs...
    )
    return nothing
end

# --- Single-pass 2D value-axis types (defined before slice drivers that annotate them) ---

"""Value-axis specification for [`calculate_structure_functions_single_pass_2d`](@ref)."""
const SINGLE_PASS_N = 6
const SINGLE_PASS_WITH_HELMHOLTZ_N = 8

"""
    single_pass_invariants(δu_L, δu_norm2) -> NTuple{SINGLE_PASS_N}

The six single-pass invariants of one pair — `(S2, L2, T2, S3, L3, L1T2)`, the stacked-row order — from the
longitudinal increment and the squared norm, which is what every geometry returns
([`HelperFunctions.pair_invariants`](@ref)).

`S2` is the norm as measured and `S3` is `δu_L` times it, so every backend sums one expression.
"""
@inline function single_pass_invariants(du_L, du_norm2)
    du_L2 = du_L * du_L
    du_T2 = du_norm2 - du_L2
    return (du_norm2, du_L2, du_T2, du_L * du_norm2, du_L * du_L2, du_L * du_T2)
end

const SinglePass2DValueBins = Union{AbstractVector, Tuple{Vararg{AbstractVector, SINGLE_PASS_N}}}

@inline _sp2d_value_bin_at(value_bins, t::Int) =
    value_bins isa Tuple ? value_bins[t] : value_bins

"""
    @sp2d_each_invariant value_bins t vb body

Run `body` once per `SINGLE_PASS_N` invariant, with `t` bound to a literal index and `vb` to that
invariant's concretely-typed value bins. Each repetition gets its own `let` scope, so `vb` is a
distinct binding per invariant.

`value_bins` may be a heterogeneous `NTuple{6}` (log bins for the non-negative invariants, linear
for the signed ones is the natural choice). Indexing it with a *runtime* `t` makes `vb` a `Union`,
which turns the `digitize` call into a dynamic dispatch on every pair × invariant in the hot loop.
A macro, so `body` is emitted inline at each of the six literal indices.
"""
macro sp2d_each_invariant(value_bins, t, vb, body)
    blocks = map(1:SINGLE_PASS_N) do i
        Expr(
            :let,
            Expr(:block,
                Expr(:(=), esc(t), i),
                Expr(:(=), esc(vb), :($(_sp2d_value_bin_at)($(esc(value_bins)), $i)))),
            esc(body),
        )
    end
    return Expr(:block, blocks...)
end

function _validate_value_bins!(value_bins, n_val::Int)
    if value_bins isa Tuple
        length(value_bins) == SINGLE_PASS_N ||
            throw(DimensionMismatch("single-pass 2D value_bins tuple must have $SINGLE_PASS_N entries; got $(length(value_bins))"))
        for t in 1:SINGLE_PASS_N
            n_edges = length(value_bins[t])
            n_edges >= n_val + 1 ||
                throw(DimensionMismatch(
                    "value_bins[$t] needs at least $(n_val + 1) edges for n_val=$n_val (got $n_edges)",
                ))
        end
    else
        length(value_bins) >= n_val + 1 ||
            throw(DimensionMismatch(
                "value_bins needs at least $(n_val + 1) edges for n_val=$n_val (got $(length(value_bins)))",
            ))
    end
    return nothing
end

"""
    calculate_structure_functions_single_pass_batch!(sums, counts, x, u, distance_bins; backend=..., ...)

Batch six invariant 1D distance histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, T)`.
"""
function calculate_structure_functions_single_pass_batch!(
    sums, counts, x, u, distance_bins;
    backend = CB.SerialBackend(), kwargs...
)
    _dispatch_single_pass_batch!(backend, sums, counts, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_single_pass_batch!(
    ::CB.AbstractSerialBackend, sums, counts, x, u, distance_bins; kwargs...
)
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_single_pass_batch!(
    ::CB.AbstractThreadedBackend, sums, counts, x, u, distance_bins; kwargs...
)
    _require_threading("the single-pass slice batch driver")
    threaded_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_single_pass_batch!(
    ::CB.AbstractDistributedBackend, args...; kwargs...
)
    throw(ArgumentError("Distributed single-pass slice batch not implemented yet; use GPUBackend or loop over t."))
end

function _dispatch_single_pass_batch!(
    backend::CB.AbstractGPUBackend, sums, counts, x, u, distance_bins; kwargs...
)
    gpu_calculate_structure_functions_single_pass_batch!(
        sums, counts, backend.backend, x, u, distance_bins; kwargs...
    )
    return nothing
end

"""
    calculate_structure_functions_single_pass_2d_batch!(sums, counts, x, u, distance_bins, value_bins; backend=..., ...)

Batch six invariant distance × value joint histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, n_val, T)`. Pass shared bin types or `NTuple{6,...}`; use `Tuple(v...)`
if you have a length-6 vector of bin objects.
"""
function calculate_structure_functions_single_pass_2d_batch!(
    sums, counts, x, u, distance_bins, value_bins::SinglePass2DValueBins;
    backend = CB.SerialBackend(), kwargs...
)
    _dispatch_single_pass_2d_batch!(
        backend, sums, counts, x, u, distance_bins, value_bins; kwargs...
    )
    return nothing
end

function _dispatch_single_pass_2d_batch!(
    ::CB.AbstractSerialBackend, sums, counts, x, u, distance_bins, value_bins::SinglePass2DValueBins; kwargs...
)
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_single_pass_2d_batch!(
    ::CB.AbstractThreadedBackend, sums, counts, x, u, distance_bins, value_bins::SinglePass2DValueBins; kwargs...
)
    _require_threading("the single-pass 2D slice batch driver")
    threaded_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_single_pass_2d_batch!(
    ::CB.AbstractDistributedBackend, args...; kwargs...
)
    throw(ArgumentError("Distributed single-pass 2D slice batch not implemented yet; use GPUBackend or loop over t."))
end

function _dispatch_single_pass_2d_batch!(
    backend::CB.AbstractGPUBackend, sums, counts, x, u, distance_bins, value_bins::SinglePass2DValueBins; kwargs...
)
    gpu_calculate_structure_functions_single_pass_2d_batch!(
        sums, counts, backend.backend, x, u, distance_bins, value_bins; kwargs...
    )
    return nothing
end

# --- Functor Support ---
function (sf::SFT.AbstractPairwiseStructureFunctionType)(x, u, bins; kwargs...)
    return calculate_structure_function(sf, x, u, bins; kwargs...)
end
