# GPUSFWorkspace — device-resident histogram buffers and cached bin edges for GPU SF paths.
# Included from StructureFunctionsKernelAbstractionsExt.jl (uses _gpu_normalize_bins, etc.).


"""Log-spaced and linear distance bins carry host-side FMA params only; no device edge upload."""
_workspace_dist_edges(::KA.Backend, ::LogBinEdges, ::Int) = nothing
_workspace_dist_edges(::KA.Backend, ::LinearBinEdges, ::Int) = nothing

function _workspace_dist_edges(backend::KA.Backend, bins::Vector{FT}, n_edges::Int) where {FT}
    bins_dev = KA.allocate(backend, FT, n_edges)
    copyto!(bins_dev, bins)
    return bins_dev
end

function _workspace_check_nb!(n_bins::Int)
    NB = n_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("GPUSFWorkspace: at most $SF_GPU_MAX_BINS distance bins (got NB=$NB)")
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
    n_bins = _gpu_n_edges(distance_bins)
    dist_bins = _gpu_normalize_bins(distance_bins)
    NB, n_bins = _workspace_check_nb!(n_bins)

    if kind == :sf1d
        out_sums_dev = KA.zeros(backend, FT, NB)
        out_cnts_dev = KA.zeros(backend, UInt32, NB)
    else
        out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB)
        out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB)
    end
    dist_edges_dev = _workspace_dist_edges(backend, dist_bins, n_bins)

    return GPUSFWorkspace{kind, FT, typeof(backend), typeof(dist_bins), Nothing,
        typeof(out_sums_dev), typeof(out_cnts_dev), Nothing, typeof(dist_edges_dev),
        Nothing, Nothing, Nothing, GPUSFLazyBuffers}(
        backend, dist_bins, nothing,
        out_sums_dev, out_cnts_dev,
        nothing, dist_edges_dev,
        NB, n_bins, NB, 0, 0,
        Vector{FT}(undef, NB), Vector{UInt32}(undef, NB),
        nothing, nothing, nothing,
        0, 0, GPUSFLazyBuffers(),
    )
end

"""
    GPUSFWorkspace(backend, distance_bins, value_bins; kind=:joint2d, ...)

Workspace for 2D histograms. Routes on `kind`:

- `kind=:joint2d` — single distance × value joint histogram (see [`joint2d_smem_max`](@ref))
- `kind=:single_pass_2d` — six-invariant-type single-pass 2D (see [`gpu/SP2D_HTP_EJ.md`](../gpu/SP2D_HTP_EJ.md))

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
    distance_bins::Union{AbstractVector{FT1}, LinearBinEdges, LogBinEdges, InfPaddedBinEdges},
    value_bins::Union{AbstractVector{FT2}, LinearBinEdges, LogBinEdges, InfPaddedBinEdges};
    joint2d_compile_cells::Union{Nothing, Int} = nothing,
) where {FT1, FT2}
    FT = promote_type(FT1, FT2)
    n_dist_edges = _gpu_n_edges(distance_bins)
    n_val_edges = _gpu_n_edges(value_bins)
    dist_bins = _gpu_normalize_bins(distance_bins)
    val_bins = _gpu_normalize_bins(value_bins)
    NB, n_bins = _workspace_check_nb!(n_dist_edges)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    n_dist > 0 && n_val > 0 ||
        throw(ArgumentError("distance_bins and value_bins must each have at least two edges"))
    nb2 = n_dist * n_val
    compile_cells = _joint2d_resolve_compile_cells(nb2, joint2d_compile_cells)
    val_plan = _joint2d_build_val_plan(backend, value_bins)

    value_host = _gpu_host_edge_vector(value_bins)
    value_edges_dev = KA.allocate(backend, FT, n_val_edges)
    copyto!(value_edges_dev, value_host)

    out_sums_dev = KA.zeros(backend, FT, n_dist, n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, n_dist, n_val)

    dist_edges_dev = _workspace_dist_edges(backend, dist_bins, n_dist_edges)

    return GPUSFWorkspace{:joint2d, FT, typeof(backend), typeof(dist_bins), typeof(val_bins),
        typeof(out_sums_dev), typeof(out_cnts_dev), typeof(value_edges_dev),
        typeof(dist_edges_dev), typeof(val_plan), Nothing, Nothing, GPUSFLazyBuffers}(
        backend, dist_bins, val_bins,
        out_sums_dev, out_cnts_dev,
        value_edges_dev, dist_edges_dev,
        NB, n_bins, n_dist, n_val, n_val_edges,
        Vector{FT}(undef, n_dist * n_val), Vector{UInt32}(undef, n_dist * n_val),
        val_plan, nothing, nothing,
        nb2, compile_cells, GPUSFLazyBuffers(),
    )
end

"""
Build a `:single_pass_2d` workspace (six invariant distance × value joint histograms).
Pass one shared edge object or `NTuple{6,...}` when columns may differ.
"""
function _sp2d_value_eltype(value_bins::LinearBinEdges, FT3)
    return promote_type(FT3, eltype(value_bins.edges))
end
function _sp2d_value_eltype(value_bins::LogBinEdges, FT3)
    return promote_type(FT3, eltype(value_bins.log_edges))
end
function _sp2d_value_eltype(value_bins::InfPaddedBinEdges, FT3)
    return _sp2d_value_eltype(value_bins.edges, FT3)
end
function _sp2d_value_eltype(value_bins::Tuple, FT3)
    return promote_type(FT3, (_sp2d_value_eltype(value_bins[t], FT3) for t in eachindex(value_bins))...)
end
function _sp2d_value_eltype(v::AbstractVector{FT}, FT3) where {FT <: Number}
    return promote_type(FT3, FT)
end

function _gpusf_workspace_sp2d!(
    backend::KA.Backend,
    distance_bins::Union{AbstractVector{FT3}, LinearBinEdges, LogBinEdges},
    value_bins::SFC.SinglePass2DValueBins;
    n_val::Union{Nothing, Int} = nothing,
) where {FT3}
    n_dist_edges = _gpu_n_edges(distance_bins)
    dist_bins = _gpu_normalize_bins(distance_bins)
    NB, n_bins = _workspace_check_nb!(n_dist_edges)
    edge_n_val = _sp2d_n_val_edges(value_bins) - 1
    hist_n_val = n_val === nothing ? edge_n_val : n_val
    _validate_gpu_value_bins!(value_bins, hist_n_val)
    n_val_edges = _sp2d_n_val_edges(value_bins)
    FT = _sp2d_value_eltype(value_bins, FT3)
    val_plan = _gpu_build_value_digitize_plan(backend, value_bins)
    value_edges_dev = _gpu_build_value_vector_cols_plan(backend, value_bins).edges_dev

    out_sums_dev = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB, hist_n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB, hist_n_val)
    strategy = _sp2d_accumulation_strategy(NB, hist_n_val, FT, SFC.gpu_device_caps(backend))

    dist_edges_dev = _workspace_dist_edges(backend, dist_bins, n_dist_edges)
    # The pair kernel is keyed only on (dist variant, value variant, accum mode), all fixed here,
    # so it is resolved once and held immutably.
    pair_kernel = _sp2d_partition_kernel_fn(
        _sp2d_dist_variant(dist_bins), _sp2d_val_variant(val_plan),
        strategy.accum_mode, backend, SF_GPU_TILED_WS,
    )

    return GPUSFWorkspace{:single_pass_2d, FT, typeof(backend), typeof(dist_bins),
        typeof(value_bins), typeof(out_sums_dev), typeof(out_cnts_dev),
        typeof(value_edges_dev), typeof(dist_edges_dev), typeof(val_plan),
        typeof(strategy), typeof(pair_kernel), GPUSFLazyBuffers}(
        backend, dist_bins, value_bins,
        out_sums_dev, out_cnts_dev,
        value_edges_dev, dist_edges_dev,
        NB, n_bins, NB, hist_n_val, n_val_edges,
        Vector{FT}(undef, SF_GPU_SINGLE_PASS_N * NB * hist_n_val),
        Vector{UInt32}(undef, SF_GPU_SINGLE_PASS_N * NB * hist_n_val),
        val_plan, strategy, pair_kernel,
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

"""Zero device histogram buffers in `ws` (call before each kernel launch)."""
function SFC.reset_histogram!(ws::GPUSFWorkspace{<:Any, FT}) where {FT}
    fill!(ws.out_sums_dev, zero(FT))
    fill!(ws.out_cnts_dev, zero(UInt32))
    return ws
end

function SFC.reset_histogram!(ws::GPUSFWorkspace{:single_pass_2d, FT}) where {FT}
    fill!(ws.out_sums_dev, zero(FT))
    fill!(ws.out_cnts_dev, zero(UInt32))
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
    lazy.active = nothing
    lazy.cull = nothing
    return nothing
end

function _validate_gpu_workspace!(
    ws::GPUSFWorkspace{kind},
    backend::KA.Backend,
    requested_kind::Symbol,
    NB::Int;
    n_val::Union{Nothing, Int} = nothing,
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
    return ws
end

"""Return cached general distance edge device buffer for tiled launches."""
function _workspace_dist_edge_bufs(ws::Union{GPUSFWorkspace, Nothing})
    ws === nothing && return nothing, nothing, nothing
    return nothing, nothing, ws.dist_general_edges_dev
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
