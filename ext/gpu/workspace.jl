# GPUSFWorkspace — device-resident histogram buffers and digitizers for GPU SF paths.

"""The workspace's device digitizer for the distance bins, or one built for this call's pass `kind`."""
_dist_digitizer(::Nothing, backend, bins, kind::Val) = _gpu_digitizer(backend, bins, kind)
_dist_digitizer(ws::GPUSFWorkspace, backend, bins, ::Val) = ws.dist_digitizer

"""The workspace's device value plan, or one built for this call."""
_value_digitizer(::Nothing, backend, value_bins) = _gpu_digitizer(backend, value_bins, Val(:value))
_value_digitizer(ws::GPUSFWorkspace, backend, value_bins) = ws.val_plan

"""The distance-bin count of `n_edges` edges, which must be at least two."""
function _workspace_check_nb!(n_edges::Int)
    NB = n_edges - 1
    NB > 0 || throw(ArgumentError("distance_bins must contain at least two edges"))
    return NB
end

"""
    GPUSFWorkspace(backend, distance_bins; kind=:sf1d)

Workspace for 1D tiled structure functions or, with `kind=:single_pass`, the
six-invariant-type single-pass distance histograms.
"""
function SFC.GPUSFWorkspace(
    backend::KA.Backend,
    distance_bins::AbstractVector{FT};
    kind::Symbol = :sf1d,
) where {FT}
    kind in (:sf1d, :single_pass) ||
        throw(ArgumentError("GPUSFWorkspace(...; kind=:sf1d|:single_pass); got kind=$kind"))
    NB = _workspace_check_nb!(length(distance_bins))

    if kind == :sf1d
        out_sums_dev = KA.zeros(backend, FT, NB)
        out_cnts_dev = KA.zeros(backend, UInt32, NB)
    else
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB)
        out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB)
    end
    dig = _gpu_digitizer(backend, distance_bins, Val(kind))

    return GPUSFWorkspace{kind, FT, typeof(backend), typeof(distance_bins), Nothing,
        typeof(out_sums_dev), typeof(out_cnts_dev), typeof(dig), Nothing, GPUSFLazyBuffers}(
        backend, distance_bins, nothing, out_sums_dev, out_cnts_dev, dig, nothing,
        NB, 0, 0, GPUSFLazyBuffers(),
    )
end

"""
    GPUSFWorkspace(backend, distance_bins, value_bins; kind=:joint2d, ...)

Workspace for 2D histograms. Routes on `kind`:

- `kind=:joint2d` — single distance × value joint histogram (see [`joint2d_smem_max`](@ref))
- `kind=:single_pass_2d` — six-invariant-type single-pass 2D

Typed `AbstractBinEdges` distance bins (`LogBinEdges`, etc.) subtype `AbstractVector`;
routing on `kind` avoids constructor ambiguity between joint and SP2D paths.
"""
function SFC.GPUSFWorkspace(
    backend::KA.Backend,
    distance_bins,
    value_bins;
    kind::Symbol = :joint2d,
    kwargs...,
)
    if kind == :joint2d && value_bins isa Tuple
        length(value_bins) == SF_GPU_SINGLE_PASS_N ||
            throw(ArgumentError(
                "tuple value_bins are reserved for single-pass 2D and must have " *
                "$SF_GPU_SINGLE_PASS_N entries; got $(length(value_bins))",
            ))
        kind = :single_pass_2d
    end
    if kind == :joint2d
        return _gpusf_workspace_joint2d!(backend, distance_bins, value_bins; kwargs...)
    elseif kind == :single_pass_2d
        return _gpusf_workspace_sp2d!(backend, distance_bins, value_bins; kwargs...)
    end
    throw(ArgumentError(
        "three-argument GPUSFWorkspace: kind must be :joint2d or :single_pass_2d (got $kind)",
    ))
end

"""
Build a `:joint2d` workspace (distance × SF value histogram).

`joint2d_compile_cells` sets the shared histogram width the tiled kernel is compiled with, at least
`n_dist × n_val` (the default); see [`joint2d_smem_max`](@ref) and [`joint2d_smem_align256`](@ref). A
call whose kernel at that width does not fit its device's shared memory takes the global-atomic
joint kernel.
"""
function _gpusf_workspace_joint2d!(
    backend::KA.Backend,
    distance_bins::AbstractVector{FT1},
    value_bins::AbstractVector{FT2};
    joint2d_compile_cells::Union{Nothing, Int} = nothing,
) where {FT1, FT2}
    FT = promote_type(FT1, FT2)
    NB = _workspace_check_nb!(length(distance_bins))
    n_val = length(value_bins) - 1
    n_val > 0 || throw(ArgumentError("value_bins must contain at least two edges"))
    compile_cells = _joint2d_resolve_compile_cells(NB * n_val, joint2d_compile_cells)
    dig = _gpu_digitizer(backend, distance_bins, Val(:joint2d))
    val_plan = _gpu_digitizer(backend, value_bins, Val(:value))

    out_sums_dev = KA.zeros(backend, FT, NB, n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, NB, n_val)

    return GPUSFWorkspace{:joint2d, FT, typeof(backend), typeof(distance_bins), typeof(value_bins),
        typeof(out_sums_dev), typeof(out_cnts_dev), typeof(dig), typeof(val_plan), GPUSFLazyBuffers}(
        backend, distance_bins, value_bins, out_sums_dev, out_cnts_dev, dig, val_plan,
        NB, n_val, compile_cells, GPUSFLazyBuffers(),
    )
end

"""Sum precision of a single-pass 2D histogram: the distance precision promoted by every value column's."""
_sp2d_value_eltype(value_bins::AbstractVector, FT3) = promote_type(FT3, eltype(value_bins))
_sp2d_value_eltype(value_bins::Tuple, FT3) = promote_type(FT3, map(eltype, value_bins)...)

"""
Build a `:single_pass_2d` workspace (six invariant distance × value joint histograms).
Pass one shared edge object or `NTuple{6,...}` when columns may differ.
"""
function _gpusf_workspace_sp2d!(
    backend::KA.Backend,
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    n_val::Union{Nothing, Int} = nothing,
) where {FT3}
    NB = _workspace_check_nb!(length(distance_bins))
    hist_n_val = n_val === nothing ? _n_value_edges(value_bins) - 1 : n_val
    SFC._validate_value_bins!(value_bins, hist_n_val)
    FT = _sp2d_value_eltype(value_bins, FT3)
    dig = _gpu_digitizer(backend, distance_bins, Val(:single_pass_2d))
    val_plan = _gpu_digitizer(backend, value_bins, Val(:value))

    out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB, hist_n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB, hist_n_val)

    return GPUSFWorkspace{:single_pass_2d, FT, typeof(backend), typeof(distance_bins),
        typeof(value_bins), typeof(out_sums_dev), typeof(out_cnts_dev), typeof(dig),
        typeof(val_plan), GPUSFLazyBuffers}(
        backend, distance_bins, value_bins, out_sums_dev, out_cnts_dev, dig, val_plan,
        NB, hist_n_val, 0, GPUSFLazyBuffers(),
    )
end

"""
Zeroed block-private SP2D partitions for `n_tile_blocks` tile blocks with counts of `CST`, kept on the
workspace and reallocated when a call needs more blocks or another count type.
"""
function _ensure_sp2d_partition_bufs!(
    ws::GPUSFWorkspace{:single_pass_2d, FT},
    n_tile_blocks::Int,
    ::Type{CST},
) where {FT, CST}
    lazy = ws.lazy
    cnts = lazy.partition_counts_dev
    if cnts === nothing || size(cnts, 4) < n_tile_blocks || eltype(cnts) !== CST
        lazy.partition_sums_dev, lazy.partition_counts_dev =
            _alloc_sp2d_partition_bufs(ws.backend, FT, CST, ws.NB, ws.n_val, n_tile_blocks)
    else
        fill!(view(lazy.partition_sums_dev, :, :, :, 1:n_tile_blocks), zero(FT))
        fill!(view(cnts, :, :, :, 1:n_tile_blocks), zero(CST))
    end
    return lazy.partition_sums_dev, lazy.partition_counts_dev
end

"""Zeroed block-private SP2D partitions: sums of `FT`, counts of `CST`."""
function _alloc_sp2d_partition_bufs(
    backend::KA.Backend,
    ::Type{FT},
    ::Type{CST},
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int,
) where {FT, CST}
    partition_sums = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_dist, n_val, n_tile_blocks)
    partition_counts = KA.zeros(backend, CST, SF_GPU_SINGLE_PASS_N, n_dist, n_val, n_tile_blocks)
    return partition_sums, partition_counts
end

function _reset_batch_histogram!(::Nothing)
    return nothing
end
function _reset_batch_histogram!(buffers::SFC.GPUBatchBuffers)
    fill!(buffers.sums, zero(eltype(buffers.sums)))
    fill!(buffers.counts, zero(eltype(buffers.counts)))
    return nothing
end

"""Zero snapshot, batch, and allocated partition histograms owned by `ws`."""
function SFC.reset_histogram!(ws::GPUSFWorkspace{<:Any, FT}) where {FT}
    fill!(ws.out_sums_dev, zero(FT))
    fill!(ws.out_cnts_dev, zero(eltype(ws.out_cnts_dev)))
    ws.lazy.snapshot_counts_dev === nothing ||
        fill!(ws.lazy.snapshot_counts_dev, zero(eltype(ws.lazy.snapshot_counts_dev)))
    _reset_batch_histogram!(ws.lazy.batch)
    return ws
end

function SFC.reset_histogram!(ws::GPUSFWorkspace{:single_pass_2d, FT}) where {FT}
    fill!(ws.out_sums_dev, zero(FT))
    fill!(ws.out_cnts_dev, zero(eltype(ws.out_cnts_dev)))
    _reset_batch_histogram!(ws.lazy.batch)
    if ws.lazy.partition_sums_dev !== nothing
        fill!(ws.lazy.partition_sums_dev, zero(FT))
        fill!(ws.lazy.partition_counts_dev, zero(eltype(ws.lazy.partition_counts_dev)))
    end
    return ws
end

"""Drop the lazily allocated device buffers; the immutable histogram buffers outlive this."""
function SFC.release!(ws::GPUSFWorkspace)
    lazy = ws.lazy
    lazy.partition_sums_dev = nothing
    lazy.partition_counts_dev = nothing
    lazy.x_dev_cache = nothing
    lazy.u_dev_cache = nothing
    lazy.snapshot_counts_dev = nothing
    lazy.batch = nothing
    lazy.active = nothing
    lazy.cull = nothing
    return nothing
end

"""Invalidate prepared inputs and culling decisions while retaining histogram storage."""
function SFC.refresh!(ws::GPUSFWorkspace)
    lazy = ws.lazy
    lazy.x_dev_cache = nothing
    lazy.u_dev_cache = nothing
    lazy.active = nothing
    lazy.cull = nothing
    return ws
end

function _workspace_snapshot_counts!(ws::GPUSFWorkspace, backend, ::Type{CT}, shape) where {CT}
    buf = ws.lazy.snapshot_counts_dev
    if buf === nothing || eltype(buf) !== CT || size(buf) != shape
        buf = KA.zeros(backend, CT, shape...)
        ws.lazy.snapshot_counts_dev = buf
    end
    return buf
end

function _validate_gpu_workspace!(
    ws::GPUSFWorkspace{kind},
    backend::KA.Backend,
    requested_kind::Symbol,
    NB::Int;
    n_val::Union{Nothing, Int} = nothing,
    distance_bins = nothing,
    value_bins = nothing,
    sum_type = nothing,
) where {kind}
    ws.backend == backend ||
        throw(ArgumentError("GPUSFWorkspace belongs to a different backend"))
    kind == requested_kind ||
        throw(ArgumentError("GPUSFWorkspace kind $kind incompatible with requested $requested_kind"))
    ws.NB == NB ||
        throw(ArgumentError("GPUSFWorkspace NB=$(ws.NB) incompatible with requested NB=$NB"))
    if n_val !== nothing && ws.n_val != n_val
        throw(ArgumentError("GPUSFWorkspace n_val=$(ws.n_val) incompatible with requested n_val=$n_val"))
    end
    sum_type === nothing || sum_type === eltype(ws.out_sums_dev) ||
        throw(ArgumentError("GPUSFWorkspace sum precision differs from the requested precision"))
    distance_bins === nothing || _workspace_bins_equal(ws.dist_bins, distance_bins) ||
        throw(ArgumentError("GPUSFWorkspace distance edges differ from the requested edges"))
    value_bins === nothing || _workspace_bins_equal(ws.val_bins, value_bins) ||
        throw(ArgumentError("GPUSFWorkspace value edges differ from the requested edges"))
    return ws
end

_workspace_bins_equal(a, b) = a == b
_workspace_bins_equal(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(_workspace_bins_equal, a, b))
_workspace_bins_equal(a::Tuple, b) = all(x -> _workspace_bins_equal(x, b), a)
_workspace_bins_equal(a, b::Tuple) = all(x -> _workspace_bins_equal(a, x), b)

function _validate_batch_workspace!(workspace, backend, kind, bins, ::Type{FT}; value_bins=nothing) where {FT}
    workspace === nothing && return nothing
    n_val = value_bins === nothing ? nothing : _n_value_edges(value_bins) - 1
    _validate_gpu_workspace!(workspace, backend, kind, length(bins)-1;
        n_val, distance_bins=bins, value_bins, sum_type=FT)
    return nothing
end
