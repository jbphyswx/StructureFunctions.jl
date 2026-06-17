# Frozen value-axis digitize plans for GPU single-pass 2D (built once at workspace creation).

"""Tag for shared O(1) linear value bins (all eight SF types use the same grid)."""
struct GPUValueLinearShared{T}
    first::T
    last::T
    inv_step::T
    offset::T
    step::T
end

"""Eight independent O(1) linear value column parameter sets."""
struct GPUValueLinearCols{T}
    first::SA.SVector{8, T}
    last::SA.SVector{8, T}
    inv_step::SA.SVector{8, T}
    offset::SA.SVector{8, T}
    step::SA.SVector{8, T}
end

"""Shared InfPadded(Linear) value bins with catch-all under/overflow."""
struct GPUValueInfLinearShared{T}
    first::T
    last::T
    inv_step::T
    offset::T
    step::T
    n_inner_edges::Int
    inner_last::T
end

"""Eight InfPadded(Linear) value columns."""
struct GPUValueInfLinearCols{T}
    first::SA.SVector{8, T}
    last::SA.SVector{8, T}
    inv_step::SA.SVector{8, T}
    offset::SA.SVector{8, T}
    step::SA.SVector{8, T}
    inner_last::SA.SVector{8, T}
    n_inner_edges::Int
end

"""Shared log-spaced value bins: FMA params on log grid; apply `log(val)` at digitize."""
struct GPUValueLogLinearShared{T}
    first::T
    last::T
    inv_step::T
    offset::T
    step::T
end

"""Eight log-spaced value columns (FMA on log grid per column)."""
struct GPUValueLogLinearCols{T}
    first::SA.SVector{8, T}
    last::SA.SVector{8, T}
    inv_step::SA.SVector{8, T}
    offset::SA.SVector{8, T}
    step::SA.SVector{8, T}
end

"""Plain `Vector` value columns — binary search on `(n_edges, 8)` edge matrix."""
struct GPUValueVectorCols{T}
    edges_dev
end

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
function _sp2d_n_val_edges(value_bins::NTuple{8, T}) where {T}
    return _sp2d_n_val_edges(value_bins[1])
end
function _sp2d_n_val_edges(value_bins::NTuple{8, Vector{FT}}) where {FT}
    return length(value_bins[1])
end

function _linear_plan_fields(lbe::LinearBinEdges)
    return (
        lbe.first_edge,
        lbe.last_edge,
        lbe.inv_step,
        lbe.offset,
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
    f, l, inv, off, st = _linear_plan_fields(vb)
    return GPUValueLinearShared(f, l, inv, off, st)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::NTuple{8, LinearBinEdges},
)
    T = eltype(vb[1].edges)
    f = SA.SVector{8, T}(_linear_plan_fields(vb[t])[1] for t in 1:8)
    l = SA.SVector{8, T}(_linear_plan_fields(vb[t])[2] for t in 1:8)
    inv = SA.SVector{8, T}(_linear_plan_fields(vb[t])[3] for t in 1:8)
    off = SA.SVector{8, T}(_linear_plan_fields(vb[t])[4] for t in 1:8)
    st = SA.SVector{8, T}(_linear_plan_fields(vb[t])[5] for t in 1:8)
    return GPUValueLinearCols(f, l, inv, off, st)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::InfPaddedBinEdges,
)
    inner = _inflinear_inner(vb)
    f, l, inv, off, st = _linear_plan_fields(inner)
    return GPUValueInfLinearShared(f, l, inv, off, st, length(inner.edges), inner.last_edge)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::NTuple{8, InfPaddedBinEdges},
)
    inners = ntuple(t -> _inflinear_inner(vb[t]), 8)
    T = eltype(inners[1].edges)
    n_inner = length(inners[1].edges)
    for t in 2:8
        length(inners[t].edges) == n_inner ||
            throw(DimensionMismatch("all InfPadded value columns must share inner edge length"))
    end
    f = SA.SVector{8, T}(inners[t].first_edge for t in 1:8)
    l = SA.SVector{8, T}(inners[t].last_edge for t in 1:8)
    inv = SA.SVector{8, T}(inners[t].inv_step for t in 1:8)
    off = SA.SVector{8, T}(inners[t].offset for t in 1:8)
    st = SA.SVector{8, T}(inners[t].step_val for t in 1:8)
    il = SA.SVector{8, T}(inners[t].last_edge for t in 1:8)
    return GPUValueInfLinearCols(f, l, inv, off, st, il, n_inner)
end

function _log_linear_plan_fields(lbe::LogBinEdges)
    return _linear_plan_fields(lbe.log_linear)
end

function _gpu_build_value_digitize_plan(
    ::KA.Backend,
    vb::LogBinEdges,
)
    f, l, inv, off, st = _log_linear_plan_fields(vb)
    return GPUValueLogLinearShared(f, l, inv, off, st)
end

function _gpu_build_value_digitize_plan(
    backend::KA.Backend,
    vb::NTuple{8, LogBinEdges},
)
    T = eltype(vb[1].log_edges)
    n_edges = length(vb[1].log_edges)
    for t in 2:8
        length(vb[t].log_edges) == n_edges ||
            throw(DimensionMismatch("all log value columns must share edge length"))
    end
    f = SA.SVector{8, T}(_log_linear_plan_fields(vb[t])[1] for t in 1:8)
    l = SA.SVector{8, T}(_log_linear_plan_fields(vb[t])[2] for t in 1:8)
    inv = SA.SVector{8, T}(_log_linear_plan_fields(vb[t])[3] for t in 1:8)
    off = SA.SVector{8, T}(_log_linear_plan_fields(vb[t])[4] for t in 1:8)
    st = SA.SVector{8, T}(_log_linear_plan_fields(vb[t])[5] for t in 1:8)
    return GPUValueLogLinearCols{T}(f, l, inv, off, st)
end

function _gpu_build_value_digitize_plan(
    backend::KA.Backend,
    vb::NTuple{8, Vector{FT}},
) where {FT}
    n_edges = length(vb[1])
    for t in 2:8
        length(vb[t]) == n_edges ||
            throw(DimensionMismatch("all vector value columns must share edge length"))
    end
    mat = Matrix{FT}(undef, n_edges, 8)
    for t in 1:8
        mat[:, t] .= vb[t]
    end
    edges_dev = KA.allocate(backend, FT, n_edges, 8)
    copyto!(edges_dev, mat)
    return GPUValueVectorCols{FT}(edges_dev)
end

function _validate_gpu_value_bins!(value_bins, n_val::Int)
    if value_bins isa Tuple
        for t in 1:8
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
