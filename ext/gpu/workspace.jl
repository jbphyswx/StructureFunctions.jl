# GPUSFWorkspace — device-resident histogram buffers and digitizers for GPU SF paths.

"""The workspace's device digitizer for the distance bins, or one built for this call's pass `kind`."""
_dist_digitizer(::Nothing, backend, bins, kind::Val) = _gpu_digitizer(backend, bins, kind)
_dist_digitizer(ws::GPUSFWorkspace, backend, bins, ::Val) = ws.dist_digitizer

"""The workspace's device value plan, or one built for this call."""
_value_digitizer(::Nothing, backend, value_bins) = _gpu_digitizer(backend, value_bins, Val(:value))
_value_digitizer(ws::GPUSFWorkspace, backend, value_bins) = ws.val_plan

function _workspace_check_nb!(n_bins::Int)
    NB = n_bins - 1
    NB > 0 || throw(ArgumentError("distance_bins must contain at least two edges"))
    return NB, n_bins
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
    NB, n_bins = _workspace_check_nb!(length(distance_bins))

    if kind == :sf1d
        out_sums_dev = KA.zeros(backend, FT, NB)
        out_cnts_dev = KA.zeros(backend, UInt32, NB)
    else
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB)
        out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB)
    end
    dig = _gpu_digitizer(backend, distance_bins, Val(kind))

    return GPUSFWorkspace{kind, FT, typeof(backend), typeof(distance_bins), Nothing,
        typeof(out_sums_dev), typeof(out_cnts_dev), typeof(dig), Nothing, Nothing,
        GPUSFLazyBuffers}(
        backend, distance_bins, nothing,
        out_sums_dev, out_cnts_dev,
        dig, nothing,
        NB, n_bins, NB, 0, 0,
        Vector{FT}(undef, NB), Vector{UInt32}(undef, NB),
        nothing,
        0, 0, GPUSFLazyBuffers(),
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

Pass `joint2d_compile_cells` to override compile-time shared-histogram width (default
exact `n_dist × n_val`). See [`joint2d_smem_max`](@ref), [`joint2d_smem_align256`](@ref).
"""
function _gpusf_workspace_joint2d!(
    backend::KA.Backend,
    distance_bins::AbstractVector{FT1},
    value_bins::AbstractVector{FT2};
    joint2d_compile_cells::Union{Nothing, Int} = nothing,
) where {FT1, FT2}
    FT = promote_type(FT1, FT2)
    n_dist_edges = length(distance_bins)
    n_val_edges = length(value_bins)
    NB, n_bins = _workspace_check_nb!(n_dist_edges)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    n_dist > 0 && n_val > 0 ||
        throw(ArgumentError("distance_bins and value_bins must each have at least two edges"))
    nb2 = n_dist * n_val
    compile_cells = _joint2d_resolve_compile_cells(nb2, joint2d_compile_cells)
    dig = _gpu_digitizer(backend, distance_bins, Val(:joint2d))
    val_plan = _gpu_digitizer(backend, value_bins, Val(:value))

    out_sums_dev = KA.zeros(backend, FT, n_dist, n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, n_dist, n_val)

    return GPUSFWorkspace{:joint2d, FT, typeof(backend), typeof(distance_bins), typeof(value_bins),
        typeof(out_sums_dev), typeof(out_cnts_dev), typeof(dig), typeof(val_plan), Nothing,
        GPUSFLazyBuffers}(
        backend, distance_bins, value_bins,
        out_sums_dev, out_cnts_dev,
        dig, val_plan,
        NB, n_bins, n_dist, n_val, n_val_edges,
        Vector{FT}(undef, n_dist * n_val), Vector{UInt32}(undef, n_dist * n_val),
        nothing,
        nb2, compile_cells, GPUSFLazyBuffers(),
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
    n_dist_edges = length(distance_bins)
    NB, n_bins = _workspace_check_nb!(n_dist_edges)
    n_val_edges = _n_value_edges(value_bins)
    hist_n_val = n_val === nothing ? n_val_edges - 1 : n_val
    SFC._validate_value_bins!(value_bins, hist_n_val)
    FT = _sp2d_value_eltype(value_bins, FT3)
    dig = _gpu_digitizer(backend, distance_bins, Val(:single_pass_2d))
    val_plan = _gpu_digitizer(backend, value_bins, Val(:value))

    out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB, hist_n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB, hist_n_val)
    strategy = _sp2d_accumulation_strategy(NB, hist_n_val, FT, SFC.gpu_device_caps(backend))

    return GPUSFWorkspace{:single_pass_2d, FT, typeof(backend), typeof(distance_bins),
        typeof(value_bins), typeof(out_sums_dev), typeof(out_cnts_dev), typeof(dig),
        typeof(val_plan), typeof(strategy), GPUSFLazyBuffers}(
        backend, distance_bins, value_bins,
        out_sums_dev, out_cnts_dev,
        dig, val_plan,
        NB, n_bins, NB, hist_n_val, n_val_edges,
        Vector{FT}(undef, SF_GPU_SINGLE_PASS_N * NB * hist_n_val),
        Vector{UInt32}(undef, SF_GPU_SINGLE_PASS_N * NB * hist_n_val),
        strategy,
        0, 0, GPUSFLazyBuffers(),
    )
end

"""
Ensure block-private HTP-EJ partitions are allocated for `n_tile_blocks` CUDA tile blocks.
Reallocates when `N_points` (hence tile-block count) grows.
"""
function _ensure_sp2d_partition_bufs!(
    ws::GPUSFWorkspace{:single_pass_2d, FT},
    n_tile_blocks::Int,
) where {FT}
    cfg = ws.sp2d_accumulation_strategy
    cfg.needs_partition_merge ||
        throw(ArgumentError("_ensure_sp2d_partition_bufs! requires needs_partition_merge (direct mode)"))
    lazy = ws.lazy
    if _partition_n_tile_blocks(lazy) < n_tile_blocks
        lazy.partition_sums_dev = KA.zeros(ws.backend, FT, SF_GPU_SINGLE_PASS_N, ws.n_dist, ws.n_val, n_tile_blocks)
        lazy.partition_counts_dev = KA.zeros(ws.backend, UInt32, SF_GPU_SINGLE_PASS_N, ws.n_dist, ws.n_val, n_tile_blocks)
    end
    return lazy.partition_sums_dev, lazy.partition_counts_dev
end

"""Allocate ephemeral privatization partitions when no workspace is provided."""
function _alloc_sp2d_partition_bufs(
    backend::KA.Backend,
    FT::Type,
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int,
)
  partition_sums = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_dist, n_val, n_tile_blocks)
  partition_counts = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, n_dist, n_val, n_tile_blocks)
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
    if ws.sp2d_accumulation_strategy.needs_partition_merge && ws.lazy.partition_sums_dev !== nothing
        fill!(ws.lazy.partition_sums_dev, zero(FT))
        fill!(ws.lazy.partition_counts_dev, zero(UInt32))
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
# Reusable GPU buffers for batched structure-function launches (production).

"""
    GPUBatchWorkspace{FT}

Device buffers reused across batched SF calls at fixed `(N, B, NB)`.

`sums_dev` / `counts_dev` are `(NB, B)` or higher-rank batch histograms.
`partial_dev` is lazy block-private `(2·NB, strip_w, n_tile_blocks)` partition.
`u_dev` uses batch-major layout `(B, N, N_dims)` for coalesced inner-batch loads.
"""
mutable struct GPUBatchWorkspace{FT, S, C, P}
    N::Int
    B::Int
    NB::Int
    n_tile_blocks::Int
    fixed_x::Bool
    sums_dev::S
    counts_dev::C
    partial_dev::Union{Nothing, P}
    x_dev::Union{AbstractArray{FT, 2}, Nothing}
    u_dev::Union{AbstractArray{FT, 3}, Nothing}
end

function GPUBatchWorkspace(
    backend::KA.Backend,
    ::Type{FT},
    N::Int,
    B::Int,
    NB::Int;
    fixed_x::Bool = true,
) where {FT}
    n_tiles = cld(N, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    sums_dev = KA.zeros(backend, FT, NB, B)
    counts_dev = KA.zeros(backend, UInt32, NB, B)
    partial_placeholder = KA.zeros(backend, FT, 0, 0, 0)
    return GPUBatchWorkspace{FT, typeof(sums_dev), typeof(counts_dev), typeof(partial_placeholder)}(
        N, B, NB, n_tile_blocks, fixed_x,
        sums_dev, counts_dev, nothing, nothing, nothing,
    )
end

"""Device bytes for block-private partial `(NB, B_chunk, n_tile_blocks)` sums."""
function _batch_fixed_x_chunk_partial_bytes(N_points::Int, B_chunk::Int, NB::Int, ::Type{FT}) where {FT}
    _, n_tile_blocks, _, _ = _batch_tiled_launch_params(N_points)
    return n_tile_blocks * NB * B_chunk * sizeof(FT)
end

"""Split `1:B` into chunks whose `(NB, B_chunk, n_tile_blocks)` partial fits `max_partial_bytes`."""
function batch_fixed_x_chunk_ranges(
    B::Int,
    max_partial_bytes::Int,
    N_points::Int,
    NB::Int,
    ::Type{FT},
) where {FT}
    if max_partial_bytes <= 0 || B <= 0
        return [1:B]
    end
    per_b = _batch_fixed_x_chunk_partial_bytes(N_points, 1, NB, FT)
    per_b <= 0 && return [1:B]
    chunk = max(1, max_partial_bytes ÷ per_b)
    ranges = UnitRange{Int}[]
    b0 = 1
    while b0 <= B
        b1 = min(B, b0 + chunk - 1)
        push!(ranges, b0:b1)
        b0 = b1 + 1
    end
    return ranges
end

"""VRAM bytes for block-private partial `(2·NB, B, n_tile_blocks)` sums + counts."""
function estimate_batch_priv_bytes(N_points::Int, B::Int, NB::Int, ::Type{FT}) where {FT}
    n_tiles = cld(N_points, SF_GPU_TILE)
    n_priv = n_tiles * (n_tiles + 1) ÷ 2
    partition = 2 * NB * B * sizeof(FT)
    return (partial_bytes = n_priv * partition, n_priv = n_priv, n_tiles = n_tiles)
end

"""
Split linear batch axis `1:B` into sub-ranges so each partition's partial buffer fits
`max_partial_bytes` (0 = no splitting → single range `1:B`).
"""
function batch_partition_ranges(B::Int, max_partial_bytes::Int, N_points::Int, NB::Int, ::Type{FT}) where {FT}
    if max_partial_bytes <= 0 || B <= 0
        return [1:B]
    end
    est = estimate_batch_priv_bytes(N_points, 1, NB, FT)
    per_b_partial = est.n_priv * 2 * NB * sizeof(FT)
    per_b_partial <= 0 && return [1:B]
    chunk = max(1, max_partial_bytes ÷ per_b_partial)
    ranges = UnitRange{Int}[]
    b0 = 1
    while b0 <= B
        b1 = min(B, b0 + chunk - 1)
        push!(ranges, b0:b1)
        b0 = b1 + 1
    end
    return ranges
end

"""Upload host `x`, `u` once before timed kernel loops."""
function upload_batch!(ws::GPUBatchWorkspace{FT}, backend::KA.Backend, x, u) where {FT}
    x_dev, u_dev = _stage_batch_device(backend, x, u; fixed_x = ws.fixed_x)
    ws.x_dev = x_dev
    ws.u_dev = u_dev
    return ws
end

function reset_batch_output!(ws::GPUBatchWorkspace{FT}) where {FT}
    fill!(ws.sums_dev, zero(FT))
    fill!(ws.counts_dev, zero(UInt32))
    return ws
end

"""Allocate block-private partial buffer on first use."""
function ensure_batch_partial_dev!(ws::GPUBatchWorkspace{FT}, backend::KA.Backend, strip_w::Int) where {FT}
    if ws.partial_dev === nothing
        ws.partial_dev = KA.zeros(backend, FT, 2 * ws.NB, strip_w, ws.n_tile_blocks)
    end
    return ws.partial_dev
end

function download_batch!(sums, counts, ws::GPUBatchWorkspace{FT}) where {FT}
    copy!(sums, reshape(Array(ws.sums_dev), size(sums)))
    copy!(counts, reshape(Array(ws.counts_dev), size(counts)))
    return nothing
end

"""Workspace for six-invariant-type single-pass batch: `(6, NB, B)` outputs."""
mutable struct GPUBatchSP1DWorkspace{FT, S, C, P}
    base::GPUBatchWorkspace{FT, S, C, P}
    sums_dev::S
    counts_dev::C
end

function GPUBatchSP1DWorkspace(
    backend::KA.Backend,
    ::Type{FT},
    N::Int,
    B::Int,
    NB::Int;
    fixed_x::Bool = true,
) where {FT}
    sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB, B)
    counts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB, B)
    base = GPUBatchWorkspace(backend, FT, N, B, NB; fixed_x = fixed_x)
    return GPUBatchSP1DWorkspace{FT, typeof(sums_dev), typeof(counts_dev), typeof(base.partial_dev)}(
        base, sums_dev, counts_dev,
    )
end

function reset_batch_sp1d_output!(ws::GPUBatchSP1DWorkspace{FT}) where {FT}
    fill!(ws.sums_dev, zero(FT))
    fill!(ws.counts_dev, zero(UInt32))
    return ws
end

"""Workspace for six-invariant-type SP2D batch: `(6, n_dist, n_val, B)` outputs."""
mutable struct GPUBatchSP2DWorkspace{FT, S, C}
    N::Int
    B::Int
    n_dist::Int
    n_val::Int
    fixed_x::Bool
    sums_dev::S
    counts_dev::C
    x_dev::Union{AbstractArray{FT, 2}, Nothing}
    u_dev::Union{AbstractArray{FT, 3}, Nothing}
    partial_sums_dev
    partial_cnts_dev
    n_tile_blocks::Int
end

function GPUBatchSP2DWorkspace(
    backend::KA.Backend,
    ::Type{FT},
    N::Int,
    B::Int,
    n_dist::Int,
    n_val::Int;
    fixed_x::Bool = true,
) where {FT}
    n_tiles = cld(N, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_dist, n_val, B)
    counts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, n_dist, n_val, B)
    return GPUBatchSP2DWorkspace{FT, typeof(sums_dev), typeof(counts_dev)}(
        N, B, n_dist, n_val, fixed_x,
        sums_dev, counts_dev, nothing, nothing,
        nothing, nothing, n_tile_blocks,
    )
end

function reset_batch_sp2d_output!(ws::GPUBatchSP2DWorkspace{FT}) where {FT}
    fill!(ws.sums_dev, zero(FT))
    fill!(ws.counts_dev, zero(UInt32))
    return ws
end
