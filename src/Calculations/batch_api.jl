# Batch Drivers and Entry Points (batch over the trailing slice/time axis of `(D, N, T)` inputs)

"""
    calculate_structure_function_batch!(sums, counts, sf_type, x, u, distance_bins; backend, distance_metric, weights, culling, workspace)
    calculate_structure_function_batch!(sums, counts, sf_type, grid, u, distance_bins[, axis_bins][, spectral_backend]; ...)

Structure functions of one fixed sampling observed over a trailing slice axis, accumulated into
`sums` and `counts` of shape `(NB, T)` with `NB = length(distance_bins) - 1`.

With coordinates `x` the sampling is a point list, `(N_dims, N_points, T)`, and the pair loop runs
once for the whole batch; on a `GPUBackend` the batch stays on the device.

With a grid or a lag schedule in place of `x` the sampling is a grid, the field is
`(component, cells..., T)`, and the lag enumeration runs once for the whole batch; `axis_bins` makes
the histogram joint in separation and angle, `sums` and `counts` then being `(NB, n_angle, T)`. The
grid form is supplied by the FlowGeometries extension and the schedule forms by
[`gridded_sweep_batch!`](@ref).
"""
function calculate_structure_function_batch!(
    sums, counts, sf_type::SFT.AbstractPairwiseStructureFunctionType, x::BatchInput, u::BatchInput,
    distance_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing, culling::CullingPolicy = AutoCulling(), workspace = nothing,
)
    _require_backend(backend)
    validate_fields(sf_type, Val(1), Val(0))
    w = _batch_boundary(counts, x, u, distance_metric, weights, eltype(sums))
    _with_batch_geometry(u, distance_metric) do geometry
        _dispatch_batch!(backend, sums, counts, sf_type, x, u, distance_bins; geometry, weights = w, culling,
                         _workspace_kw(workspace)...)
    end
    return nothing
end

"""
    _batch_boundary(counts, x, u, distance_metric, weights, OT) -> weights

The one validation of a batch entry: the shapes, a trailing axis on `u`, the weights normalised to
`OT`, and room in `counts` for every pair.
"""
function _batch_boundary(counts, x, u, distance_metric, weights, ::Type{OT}) where {OT}
    xc, uc = _contract_layout(x), _contract_layout(u)
    _validate_array_shape(xc, uc, distance_metric)
    ndims(uc) >= 3 || throw(ArgumentError("batch fields require at least one trailing axis"))
    w = _pair_weights(weights, size(xc, 2), OT)
    _assert_counts_can_accumulate(counts, size(xc, 2), w)
    return w
end

"""`f(geometry)` with the pair geometry of `distance_metric` at the velocity width of the batch field `u`
([`_shaped`](@ref))."""
@inline _with_batch_geometry(f, u, distance_metric) =
    _shaped((_, geometry) -> f(geometry), SharedPositionField, size(_contract_layout(u), 1), distance_metric)

function _dispatch_batch!(
    ::CB.AbstractSerialBackend, sums, counts, sf_type, x, u, distance_bins; kwargs...
)
    auxiliary_structure_function!(sums, counts, sf_type, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_batch!(
    ::CB.AbstractThreadedBackend, sums, counts, sf_type, x, u, distance_bins; kwargs...
)
    auxiliary_structure_function_threaded!(sums, counts, sf_type, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_batch!(
    ::CB.AbstractAutoBackend, sums, counts, sf_type, x, u, distance_bins; kwargs...
)
    return _dispatch_batch!(resolve_auto_backend(), sums, counts, sf_type, x, u, distance_bins; kwargs...)
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
    calculate_structure_function_2d_batch!(sums, counts, sf_type, x, u, distance_bins, value_bins; backend, distance_metric, weights, second_axis, culling, workspace)

Batch 2D joint histograms over `(N_dims, N_points, T)`; outputs `(n_dist, n_val, T)`.
"""
function calculate_structure_function_2d_batch!(
    sums, counts, sf_type::SFT.AbstractPairwiseStructureFunctionType, x::BatchInput, u::BatchInput,
    distance_bins::AbstractVector, value_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing, second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
    culling::CullingPolicy = AutoCulling(), workspace = nothing,
)
    _require_backend(backend)
    validate_fields(sf_type, Val(1), Val(0))
    w = _batch_boundary(counts, x, u, distance_metric, weights, eltype(sums))
    _with_batch_geometry(u, distance_metric) do geometry
        _dispatch_2d_batch!(backend, sums, counts, sf_type, x, u, distance_bins, value_bins; geometry, weights = w,
                            second_axis, culling, _workspace_kw(workspace)...)
    end
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
    auxiliary_joint2d_threaded!(sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_2d_batch!(
    ::CB.AbstractAutoBackend, sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...
)
    return _dispatch_2d_batch!(resolve_auto_backend(), sums, counts, sf_type, x, u, distance_bins,
                               value_bins; kwargs...)
end

function _dispatch_2d_batch!(
    backend::CB.AbstractGPUBackend, sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...
)
    gpu_calculate_structure_function_2d_batch!(
        sums, counts, sf_type, backend.backend, x, u, distance_bins, value_bins; kwargs...
    )
    return nothing
end

# --- Single-pass invariants and 2D value bins ---

"""Number of native single-pass invariants."""
const SINGLE_PASS_N = 6
const SINGLE_PASS_WITH_HELMHOLTZ_N = 8

"""
    single_pass_invariants(δu_L, δu_norm2) -> NTuple{SINGLE_PASS_N}

The six single-pass invariants of one pair — `(S2, L2, T2, S3, L3, L1T2)`, the stacked-row order — from the
longitudinal increment and the squared norm of [`HelperFunctions.increment_invariants`](@ref).
"""
@inline function single_pass_invariants(du_L, du_norm2)
    du_L2 = du_L * du_L
    du_T2 = SFH.transverse_energy(du_L, du_norm2)
    return (du_norm2, du_L2, du_T2, du_L * du_norm2, du_L * du_L2, du_L * du_T2)
end

const SinglePass2DValueBins = Union{AbstractVector, Tuple{Vararg{AbstractVector, SINGLE_PASS_N}}}

@inline _sp2d_value_bin_at(value_bins, t::Int) =
    value_bins isa Tuple ? value_bins[t] : value_bins

"""
    @sp2d_each_invariant value_bins t vb body

Run `body` once per `SINGLE_PASS_N` invariant, with `t` bound to a literal index and `vb` to that
invariant's concretely-typed value bins. Each repetition gets its own `let` scope.

`value_bins` may be a heterogeneous `NTuple{6}`; `body` is emitted inline at each of the six literal
indices, so each `vb` has a concrete type.
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
    calculate_structure_functions_single_pass_batch!(sums, counts, x, u, distance_bins; backend, distance_metric, weights, culling, workspace)

Batch six invariant 1D distance histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, T)`.
"""
function calculate_structure_functions_single_pass_batch!(
    sums, counts, x::BatchInput, u::BatchInput, distance_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing, culling::CullingPolicy = AutoCulling(), workspace = nothing,
)
    _require_backend(backend)
    w = _batch_boundary(counts, x, u, distance_metric, weights, eltype(sums))
    _with_batch_geometry(u, distance_metric) do geometry
        _dispatch_single_pass_batch!(backend, sums, counts, x, u, distance_bins; geometry, weights = w, culling,
                                     _workspace_kw(workspace)...)
    end
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
    threaded_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_single_pass_batch!(
    ::CB.AbstractAutoBackend, sums, counts, x, u, distance_bins; kwargs...
)
    return _dispatch_single_pass_batch!(resolve_auto_backend(), sums, counts, x, u, distance_bins; kwargs...)
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
    calculate_structure_functions_single_pass_2d_batch!(sums, counts, x, u, distance_bins, value_bins; backend, distance_metric, weights, culling, workspace)

Batch six invariant distance × value joint histograms over `(N_dims, N_points, T)`;
outputs `(6, NB, n_val, T)`. Pass shared bin types or `NTuple{6,...}`; use `Tuple(v...)`
if you have a length-6 vector of bin objects.
"""
function calculate_structure_functions_single_pass_2d_batch!(
    sums, counts, x::BatchInput, u::BatchInput, distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing, culling::CullingPolicy = AutoCulling(), workspace = nothing,
)
    _require_backend(backend)
    w = _batch_boundary(counts, x, u, distance_metric, weights, eltype(sums))
    _with_batch_geometry(u, distance_metric) do geometry
        _dispatch_single_pass_2d_batch!(backend, sums, counts, x, u, distance_bins, value_bins; geometry, weights = w,
                                        culling, _workspace_kw(workspace)...)
    end
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
    threaded_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_single_pass_2d_batch!(
    ::CB.AbstractAutoBackend, sums, counts, x, u, distance_bins, value_bins::SinglePass2DValueBins; kwargs...
)
    return _dispatch_single_pass_2d_batch!(resolve_auto_backend(), sums, counts, x, u, distance_bins,
                                           value_bins; kwargs...)
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

function calculate_structure_functions_single_pass!(sums::AbstractArray, counts::AbstractArray,
        x::BatchInput, u::BatchInput, bins::AbstractVector; backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
        distance_metric::DI.PreMetric = DI.Euclidean(), weights = nothing, culling::CullingPolicy = AutoCulling(),
        workspace = nothing)
    _require_backend(backend)
    w = _batch_boundary(counts, x, u, distance_metric, weights, eltype(sums))
    _with_batch_geometry(u, distance_metric) do geometry
        _dispatch_single_pass_batch!(backend, sums, counts, x, u, bins; geometry, weights = w, culling,
                                     _workspace_kw(workspace)...)
    end
    return sums, counts
end

function calculate_structure_functions_single_pass_2d!(sums::AbstractArray, counts::AbstractArray,
        x::BatchInput, u::BatchInput, bins::AbstractVector, value_bins::SinglePass2DValueBins;
        backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
        weights = nothing, culling::CullingPolicy = AutoCulling(), workspace = nothing)
    _require_backend(backend)
    w = _batch_boundary(counts, x, u, distance_metric, weights, eltype(sums))
    _with_batch_geometry(u, distance_metric) do geometry
        _dispatch_single_pass_2d_batch!(backend, sums, counts, x, u, bins, value_bins; geometry, weights = w,
                                        culling, _workspace_kw(workspace)...)
    end
    return sums, counts
end
