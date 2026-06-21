# Host-side privatization sizing for HTP-EJ six-invariant-type single-pass 2D.
#
# Two GPU algorithms (three `accum_mode` symbols):
#   On-chip (:shared, :typeplane) — shared histogram + joint-style flush to out_* (no merge).
#   Direct (:direct) — priv partition + merge kernel when even one n_dist×n_val plane does not fit in 48 KiB.
#
# Full design notes: gpu/SP2D_HTP_EJ.md

"""CUDA default per-block shared memory (bytes) when kernel does not opt in."""
const SF_GPU_SMEM_DEFAULT = 48 * 1024

"""Safety margin for driver / compiler static shared outside histogram."""
const SF_GPU_SMEM_COMPILER_RESERVE = 2048

"""`@localmem` width per coord buffer in tiled priv kernels (x/y per point)."""
const SP2D_PRIV_TILE_LOCALMEM = 256

# Frozen value-axis digitize plans for GPU single-pass 2D (built once at workspace creation).

"""Tag for shared O(1) linear value bins (all six invariant SF types use the same grid)."""
struct GPUValueLinearShared{T}
    first::T
    last::T
    inv_step::T
    offset::T
    step::T
end

"""Six independent O(1) linear value column parameter sets."""
struct GPUValueLinearCols{T}
    first::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inv_step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    offset::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
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

"""Six InfPadded(Linear) value columns."""
struct GPUValueInfLinearCols{T}
    first::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inv_step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    offset::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inner_last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
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

"""Six log-spaced value columns (FMA on log grid per column)."""
struct GPUValueLogLinearCols{T}
    first::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    last::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    inv_step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    offset::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
    step::SA.SVector{SF_GPU_SINGLE_PASS_N, T}
end

"""Plain `Vector` value columns — binary search on `(n_edges, 6)` edge matrix."""
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

"""
    SP2DPrivConfig

Frozen HTP-EJ histogram policy for one `(n_dist, n_val, FT)` workspace.
`accum_mode` is `:shared` when the full `6 × n_dist × n_val` histogram fits in
48 KiB shared memory; `:typeplane` when one or more type planes fit per pass
(`types_per_pass × n_dist × n_val` cells, `n_type_passes` pair traversals); otherwise
`:direct` uses block-partitioned global atomics (single pair pass).
`needs_priv_merge` is `true` only for `:direct` (priv partition + merge kernel).
"""
struct SP2DPrivConfig
    n_joint_cells::Int
    accum_mode::Symbol
    smem_per_block::Int
    max_shared_cells::Int
    plane_cells::Int
    types_per_pass::Int
    n_type_passes::Int
    needs_priv_merge::Bool
end

"""Total joint histogram cells `6 × n_dist × n_val`."""
@inline _sp2d_joint_cells(n_dist::Int, n_val::Int) = SF_GPU_SINGLE_PASS_N * n_dist * n_val

"""`shared_block_id`, `shared_tile[4]` (KA `@synchronize` metadata)."""
@inline _sp2d_priv_meta_smem_bytes() = 5 * sizeof(Int)

"""Four tiled coordinate buffers (`@localmem FT (256,)` × 4)."""
@inline function _sp2d_tile_smem_overhead(::Type{FT}) where {FT}
    return 4 * SP2D_PRIV_TILE_LOCALMEM * sizeof(FT)
end

@inline function _sp2d_hist_budget_bytes(smem_per_block::Int, ::Type{FT}) where {FT}
    return max(
        0,
        smem_per_block -
        _sp2d_tile_smem_overhead(FT) -
        _sp2d_priv_meta_smem_bytes() -
        SF_GPU_SMEM_COMPILER_RESERVE,
    )
end

"""Largest full histogram cell count whose static `@localmem` fits in `smem_limit`."""
function _sp2d_max_shared_cells(smem_limit::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return _sp2d_hist_budget_bytes(smem_limit, FT) ÷ cell_bytes
end

"""Static `@localmem` bytes for shared-histogram kernel at `max_cells` (compile-time width)."""
@inline function _sp2d_sharedhist_static_smem_bytes(max_cells::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return _sp2d_tile_smem_overhead(FT) +
           _sp2d_priv_meta_smem_bytes() +
           max_cells * cell_bytes +
           SF_GPU_SMEM_COMPILER_RESERVE
end

"""How many SF-type planes fit in one shared-histogram pass (`1…6`)."""
@inline function _sp2d_types_per_pass(plane::Int, max_shared::Int)
    plane <= 0 && return 1
    return min(SF_GPU_SINGLE_PASS_N, max(1, max_shared ÷ plane))
end

@inline function _sp2d_n_type_passes(types_per_pass::Int)
    return (SF_GPU_SINGLE_PASS_N + types_per_pass - 1) ÷ types_per_pass
end

"""
    _sp2d_priv_config(n_dist, n_val, FT) -> SP2DPrivConfig

Select `:shared`, `:typeplane`, or `:direct` from the 48 KiB shared-memory budget.
Typeplane packs `types_per_pass` SF planes per pair traversal (`n_type_passes` total).
Sets `needs_priv_merge = (mode == :direct)` for host launch/workspace routing.
"""
function _sp2d_priv_config(n_dist::Int, n_val::Int, ::Type{FT}) where {FT}
    C = _sp2d_joint_cells(n_dist, n_val)
    plane = n_dist * n_val
    smem = SF_GPU_SMEM_DEFAULT
    max_shared = _sp2d_max_shared_cells(smem, FT)
    mode = if C <= max_shared
        :shared
    elseif plane <= max_shared
        :typeplane
    else
        :direct
    end
    tpp = mode == :typeplane ? _sp2d_types_per_pass(plane, max_shared) : SF_GPU_SINGLE_PASS_N
    ntp = mode == :typeplane ? _sp2d_n_type_passes(tpp) : 1
    return SP2DPrivConfig(C, mode, smem, max_shared, plane, tpp, ntp, mode == :direct)
end

"""Block-private partition bytes for `n_tile_blocks` upper-triangle tile blocks."""
@inline function _sp2d_priv_partition_bytes(config::SP2DPrivConfig, n_tile_blocks::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return n_tile_blocks * config.n_joint_cells * cell_bytes
end

"""Compile-time `@localmem` histogram width for shared-histogram kernels."""
@inline function _sp2d_sharedhist_compile_cells(config::SP2DPrivConfig)
    return config.max_shared_cells
end

function _sp2d_dist_variant(::LinearBinEdges)
    return :linear
end
function _sp2d_dist_variant(::LogBinEdges)
    return :log_linear
end
# Joint 2D tiled-kernel shared-memory compile width (user-facing helpers + resolver).
# Included from StructureFunctionsGPUExt.jl after GPUValueDigitizePlan.jl.

"""
    joint2d_smem_max()

Compile-time `@localmem` width `SF_GPU_MAX_2D_HIST` (4096). Reuses one GPU kernel for
any joint grid with `n_dist × n_val ≤ 4096` — useful when many bin shapes are tried in
one Julia session.
"""
function SFC.joint2d_smem_max()
    return SF_GPU_MAX_2D_HIST
end

"""
    joint2d_smem_exact(n_dist, n_val)

Exact histogram cell count `n_dist × n_val` (same as omitting `joint2d_compile_cells`
on [`GPUSFWorkspace`](@ref)).
"""
function SFC.joint2d_smem_exact(n_dist::Int, n_val::Int)
    return n_dist * n_val
end

"""
    joint2d_smem_align256(n_dist, n_val)

Round `n_dist × n_val` up to a multiple of 256 (capped at [`joint2d_smem_max`](@ref)),
reusing one of at most 16 kernel sizes for bin-grid sweeps.
"""
function SFC.joint2d_smem_align256(n_dist::Int, n_val::Int)
    nb2 = n_dist * n_val
    return min(SF_GPU_MAX_2D_HIST, cld(nb2, 256) * 256)
end

"""Internal: distance-bin route symbol for joint tiled kernel dispatch."""
function _joint2d_dist_route(dist_bins)
    dist_bins isa LinearBinEdges && return :linear
    dist_bins isa LogBinEdges && return :log
    return :general
end

"""Internal: value-bin route symbol for joint tiled kernel dispatch."""
function _joint2d_val_route(::Nothing)
    return :general
end
function _joint2d_val_route(::GPUValueLinearShared)
    return :linear
end
function _joint2d_val_route(::GPUValueInfLinearShared)
    return :inflinear
end
function _joint2d_val_route(::GPUValueLogLinearShared)
    return :log_linear
end
function _joint2d_val_route(::GPUValueVectorCols)
    return :general
end
function _joint2d_val_route(val_plan)
    throw(ArgumentError(
        "joint2d value bins: expected LinearBinEdges, LogBinEdges, InfPaddedBinEdges, or Vector (got $(typeof(val_plan)))",
    ))
end

"""
Build a frozen value digitize plan for joint 2D (single shared column).
Returns `nothing` for plain `Vector` edges → `:general` kernel route.
"""
function _joint2d_build_val_plan(backend::KA.Backend, value_bins)
    value_bins isa LinearBinEdges && return _gpu_build_value_digitize_plan(backend, value_bins)
    value_bins isa InfPaddedBinEdges && return _gpu_build_value_digitize_plan(backend, value_bins)
    value_bins isa LogBinEdges && return _gpu_build_value_digitize_plan(backend, value_bins)
    value_bins isa BinEdges && return _joint2d_build_val_plan(backend, value_bins.edges)
    value_bins isa AbstractVector && return nothing
    throw(ArgumentError("joint2d unsupported value_bins type $(typeof(value_bins))"))
end

"""
Resolve compile-time histogram width from optional user override.
Default (`compile_cells === nothing`) is exact `NB2`.
"""
function _joint2d_resolve_compile_cells(NB2::Int, compile_cells::Union{Nothing, Int})
    cells = compile_cells === nothing ? NB2 : compile_cells
    cells >= NB2 ||
        throw(ArgumentError("joint2d_compile_cells=$cells is smaller than NB2=$NB2"))
    cells <= SF_GPU_MAX_2D_HIST ||
        throw(ArgumentError(
            "joint2d_compile_cells=$cells exceeds SF_GPU_MAX_2D_HIST=$(SF_GPU_MAX_2D_HIST)",
        ))
    return cells
end


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
    vb::NTuple{SF_GPU_SINGLE_PASS_N, LinearBinEdges},
)
    T = eltype(vb[1].edges)
    f = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[1] for t in 1:SF_GPU_SINGLE_PASS_N)
    l = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[2] for t in 1:SF_GPU_SINGLE_PASS_N)
    inv = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[3] for t in 1:SF_GPU_SINGLE_PASS_N)
    off = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[4] for t in 1:SF_GPU_SINGLE_PASS_N)
    st = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_linear_plan_fields(vb[t])[5] for t in 1:SF_GPU_SINGLE_PASS_N)
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
    off = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].offset for t in 1:SF_GPU_SINGLE_PASS_N)
    st = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].step_val for t in 1:SF_GPU_SINGLE_PASS_N)
    il = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(inners[t].last_edge for t in 1:SF_GPU_SINGLE_PASS_N)
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
    off = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_log_linear_plan_fields(vb[t])[4] for t in 1:SF_GPU_SINGLE_PASS_N)
    st = SA.SVector{SF_GPU_SINGLE_PASS_N, T}(_log_linear_plan_fields(vb[t])[5] for t in 1:SF_GPU_SINGLE_PASS_N)
    return GPUValueLogLinearCols{T}(f, l, inv, off, st)
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

function _gpu_build_value_digitize_plan(
    backend::KA.Backend,
    vb::Tuple{Vararg{AbstractVector, SF_GPU_SINGLE_PASS_N}},
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
