# Frozen value-axis digitize plans for GPU joint 2D and single-pass 2D kernels.

"""Tag for shared O(1) linear value bins (all six invariant SF types use the same grid)."""
struct GPUValueLinearShared{T}
    first::T
    last::T
    inv_step::T
    step::T
end

"""Six independent O(1) linear value column parameter sets."""
struct GPUValueLinearCols{T}
    first::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inv_step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
end

"""Shared InfPadded(Linear) value bins with catch-all under/overflow."""
struct GPUValueInfLinearShared{T}
    first::T
    last::T
    inv_step::T
    step::T
    n_inner_edges::Int
    inner_last::T
end

"""Six InfPadded(Linear) value columns."""
struct GPUValueInfLinearCols{T}
    first::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inv_step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inner_last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    n_inner_edges::Int
end

"""Shared log-spaced value bins: FMA params on log grid; apply `log(val)` at digitize."""
struct GPUValueLogLinearShared{T}
    first::T
    last::T
    inv_step::T
    step::T
end

"""Six log-spaced value columns (FMA on log grid per column)."""
struct GPUValueLogLinearCols{T}
    first::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inv_step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
end

"""Plain `Vector` value columns: binary search on `(n_edges, 6)` edge matrix.
`edges_dev` is typed (param `E`) so the struct is isbits-after-adapt and can be
passed to a GPU kernel; the `adapt_structure` rule below converts the device
edge matrix to its in-kernel form when KA adapts kernel arguments."""
struct GPUValueVectorCols{T, E}
    edges_dev::E
end

# Keep the existing 1-type-param constructor calls working (E inferred).
GPUValueVectorCols{T}(edges_dev::E) where {T, E} = GPUValueVectorCols{T, E}(edges_dev)

# Make KA's argument adaptation recurse into the device edge matrix.
KA.Adapt.adapt_structure(to, p::GPUValueVectorCols{T}) where {T} =
    GPUValueVectorCols{T}(KA.Adapt.adapt(to, p.edges_dev))

const GPUValueDigitizePlan = Union{
    GPUValueLinearShared,
    GPUValueLinearCols,
    GPUValueInfLinearShared,
    GPUValueInfLinearCols,
    GPUValueLogLinearShared,
    GPUValueLogLinearCols,
    GPUValueVectorCols,
}

"""Histogram edge count for a single-pass 2D value-axis specification."""
function _sp2d_n_val_edges(value_bins::LinearBinEdges)
    return length(value_bins.edges)
end
function _sp2d_n_val_edges(value_bins::LogBinEdges)
    return length(value_bins)
end
function _sp2d_n_val_edges(value_bins::InfPaddedBinEdges)
    return length(value_bins)
end
function _sp2d_n_val_edges(value_bins::AbstractVector)
    return length(value_bins)
end
function _sp2d_n_val_edges(value_bins::Tuple)
    return _sp2d_n_val_edges(value_bins[1])
end

function _linear_plan_fields(lbe::LinearBinEdges)
    return (
        lbe.first_edge,
        lbe.last_edge,
        lbe.inv_step,
        lbe.step_val,
    )
end

function _inflinear_inner(vb::InfPaddedBinEdges)
    inner = vb.edges
    inner isa LinearBinEdges ||
        throw(ArgumentError("GPU InfPadded value bins require inner LinearBinEdges (got $(typeof(inner)))"))
    return inner
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::LinearBinEdges,
)
    f, l, inv, st = _linear_plan_fields(vb)
    return GPUValueLinearShared(f, l, inv, st)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::NTuple{SF_GPU_SINGLE_PASS_N, LinearBinEdges},
)
    T = eltype(vb[1].edges)
    f = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[1] for t in 1:SF_GPU_SINGLE_PASS_N)
    l = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[2] for t in 1:SF_GPU_SINGLE_PASS_N)
    inv = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[3] for t in 1:SF_GPU_SINGLE_PASS_N)
    st = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[4] for t in 1:SF_GPU_SINGLE_PASS_N)
    return GPUValueLinearCols(f, l, inv, st)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::InfPaddedBinEdges,
)
    inner = _inflinear_inner(vb)
    f, l, inv, st = _linear_plan_fields(inner)
    return GPUValueInfLinearShared(f, l, inv, st, length(inner.edges), inner.last_edge)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::NTuple{SF_GPU_SINGLE_PASS_N, InfPaddedBinEdges},
)
    inners = ntuple(t -> _inflinear_inner(vb[t]), SF_GPU_SINGLE_PASS_N)
    T = eltype(inners[1].edges)
    n_inner = length(inners[1].edges)
    for t in 2:SF_GPU_SINGLE_PASS_N
        length(inners[t].edges) == n_inner ||
            throw(DimensionMismatch("all InfPadded value columns must share inner edge length"))
    end
    f = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].first_edge for t in 1:SF_GPU_SINGLE_PASS_N)
    l = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].last_edge for t in 1:SF_GPU_SINGLE_PASS_N)
    inv = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].inv_step for t in 1:SF_GPU_SINGLE_PASS_N)
    st = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].step_val for t in 1:SF_GPU_SINGLE_PASS_N)
    il = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].last_edge for t in 1:SF_GPU_SINGLE_PASS_N)
    return GPUValueInfLinearCols(f, l, inv, st, il, n_inner)
end

function _log_linear_plan_fields(lbe::LogBinEdges)
    return _linear_plan_fields(lbe.log_linear)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::LogBinEdges,
)
    f, l, inv, st = _log_linear_plan_fields(vb)
    return GPUValueLogLinearShared(f, l, inv, st)
end

function _gpu_build_value_digitize_plan(
    backend::KA.Backend,
    vb::NTuple{SF_GPU_SINGLE_PASS_N, LogBinEdges},
)
    T = eltype(vb[1].log_edges)
    n_edges = length(vb[1].log_edges)
    for t in 2:SF_GPU_SINGLE_PASS_N
        length(vb[t].log_edges) == n_edges ||
            throw(DimensionMismatch("all log value columns must share edge length"))
    end
    f = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_log_linear_plan_fields(vb[t])[1] for t in 1:SF_GPU_SINGLE_PASS_N)
    l = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_log_linear_plan_fields(vb[t])[2] for t in 1:SF_GPU_SINGLE_PASS_N)
    inv = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_log_linear_plan_fields(vb[t])[3] for t in 1:SF_GPU_SINGLE_PASS_N)
    st = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_log_linear_plan_fields(vb[t])[4] for t in 1:SF_GPU_SINGLE_PASS_N)
    return GPUValueLogLinearCols{T}(f, l, inv, st)
end

function _gpu_build_value_vector_cols_plan(
    backend::KA.Backend,
    vb::Tuple{Vararg{AbstractVector, SF_GPU_SINGLE_PASS_N}},
)
    n_edges = length(vb[1])
    for t in 2:SF_GPU_SINGLE_PASS_N
        length(vb[t]) == n_edges ||
            throw(DimensionMismatch("all value columns must share edge length"))
    end
    FT = promote_type((eltype(vb[t]) for t in 1:SF_GPU_SINGLE_PASS_N)...)
    mat = Matrix{FT}(undef, n_edges, SF_GPU_SINGLE_PASS_N)
    for t in 1:SF_GPU_SINGLE_PASS_N
        @inbounds for e in 1:n_edges
            mat[e, t] = vb[t][e]
        end
    end
    edges_dev = KA.allocate(backend, FT, n_edges, SF_GPU_SINGLE_PASS_N)
    copyto!(edges_dev, mat)
    return GPUValueVectorCols{FT}(edges_dev)
end

function _gpu_build_value_vector_cols_plan(
    backend::KA.Backend,
    vb::Tuple,
)
    length(vb) == SF_GPU_SINGLE_PASS_N ||
        throw(ArgumentError("single-pass 2D value-bin tuples must have $SF_GPU_SINGLE_PASS_N entries"))
    n_edges = _sp2d_n_val_edges(vb[1])
    edge_cols = ntuple(t -> _gpu_host_edge_vector(vb[t]), SF_GPU_SINGLE_PASS_N)
    for t in 2:SF_GPU_SINGLE_PASS_N
        length(edge_cols[t]) == n_edges ||
            throw(DimensionMismatch("all value columns must share edge length"))
    end
    FT = promote_type((eltype(edge_cols[t]) for t in 1:SF_GPU_SINGLE_PASS_N)...)
    mat = Matrix{FT}(undef, n_edges, SF_GPU_SINGLE_PASS_N)
    for t in 1:SF_GPU_SINGLE_PASS_N
        @inbounds for e in 1:n_edges
            mat[e, t] = edge_cols[t][e]
        end
    end
    edges_dev = KA.allocate(backend, FT, n_edges, SF_GPU_SINGLE_PASS_N)
    copyto!(edges_dev, mat)
    return GPUValueVectorCols{FT}(edges_dev)
end

function _gpu_build_value_vector_cols_plan(
    backend::KA.Backend,
    vb::AbstractVector,
)
    n_edges = length(vb)
    FT = eltype(vb)
    mat = Matrix{FT}(undef, n_edges, SF_GPU_SINGLE_PASS_N)
    for t in 1:SF_GPU_SINGLE_PASS_N
        @inbounds for e in 1:n_edges
            mat[e, t] = vb[e]
        end
    end
    edges_dev = KA.allocate(backend, FT, n_edges, SF_GPU_SINGLE_PASS_N)
    copyto!(edges_dev, mat)
    return GPUValueVectorCols{FT}(edges_dev)
end

function _gpu_build_value_vector_cols_plan(
    backend::KA.Backend,
    vb::Union{LinearBinEdges, LogBinEdges, InfPaddedBinEdges},
)
    edges = _gpu_host_edge_vector(vb)
    return _gpu_build_value_vector_cols_plan(backend, edges)
end

function _gpu_build_value_digitize_plan(
    backend::KA.Backend,
    vb::Tuple{Vararg{AbstractVector, SF_GPU_SINGLE_PASS_N}},
)
    return _gpu_build_value_vector_cols_plan(backend, vb)
end

function _gpu_build_value_digitize_plan(
    backend::KA.Backend,
    vb::AbstractVector,
)
    return _gpu_build_value_vector_cols_plan(backend, vb)
end

function _validate_gpu_value_bins!(value_bins, n_val::Int)
    if value_bins isa Tuple
        for t in eachindex(value_bins)
            n_edges = _sp2d_n_val_edges(value_bins[t])
            n_edges >= n_val + 1 ||
                throw(DimensionMismatch(
                    "value_bins[$t] needs at least $(n_val + 1) edges for n_val=$n_val (got $n_edges)",
                ))
        end
    else
        _sp2d_n_val_edges(value_bins) >= n_val + 1 ||
            throw(DimensionMismatch(
                "value_bins needs at least $(n_val + 1) edges for n_val=$n_val",
            ))
    end
    return nothing
end

@inline function _gpu_digitize_value_plan(
    x,
    plan::GPUValueLinearShared,
    col::Int,
    n_edges::Int,
)
    return _gpu_digitize_linear(
        x, plan.first, plan.last, plan.inv_step, plan.step, n_edges,
    )
end

@inline function _gpu_digitize_value_plan(
    x,
    plan::GPUValueLinearCols,
    col::Int,
    n_edges::Int,
)
    return _gpu_digitize_linear(
        x,
        plan.first[col],
        plan.last[col],
        plan.inv_step[col],
        plan.step[col],
        n_edges,
    )
end

@inline function _gpu_digitize_value_plan(
    x,
    plan::GPUValueInfLinearShared,
    col::Int,
    n_edges::Int,
)
    return _gpu_digitize_inf_padded_linear(
        x,
        plan.first,
        plan.last,
        plan.inv_step,
        plan.step,
        plan.n_inner_edges,
        plan.inner_last,
    )
end

@inline function _gpu_digitize_value_plan(
    x,
    plan::GPUValueInfLinearCols,
    col::Int,
    n_edges::Int,
)
    return _gpu_digitize_inf_padded_linear(
        x,
        plan.first[col],
        plan.last[col],
        plan.inv_step[col],
        plan.step[col],
        plan.n_inner_edges,
        plan.inner_last[col],
    )
end

@inline function _gpu_digitize_value_plan(
    x,
    plan::GPUValueLogLinearShared,
    col::Int,
    n_edges::Int,
)
    return _gpu_digitize_log_spaced(
        x, plan.first, plan.last, plan.inv_step, plan.step, n_edges,
    )
end

@inline function _gpu_digitize_value_plan(
    x,
    plan::GPUValueLogLinearCols,
    col::Int,
    n_edges::Int,
)
    return _gpu_digitize_log_spaced_col(
        x,
        plan.first,
        plan.last,
        plan.inv_step,
        plan.step,
        col,
        n_edges,
    )
end

@inline function _gpu_digitize_value_plan(
    x,
    plan::GPUValueVectorCols,
    col::Int,
    n_edges::Int,
)
    return _gpu_digitize_general_col(x, plan.edges_dev, col, n_edges)
end

