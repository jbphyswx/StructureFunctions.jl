# GPUSFWorkspace — device-resident histogram buffers and cached bin edges for GPU SF paths.
# Included from StructureFunctionsGPUExt.jl (uses _gpu_normalize_bins, etc.).

"""
    GPUSFWorkspace

Reusable device histogram buffers and cached distance-bin edge arrays for GPU
structure-function launches. Construct once per `(backend, distance_bins[, value_bins])`
configuration; pass to `gpu_calculate_structure_function(!)` or slice drivers to avoid
per-call `KA.zeros` allocation and repeated edge uploads.

# Kinds (`kind` field)
- `:sf1d` — 1D distance histogram (`out_dev`, `cnt_dev` vectors)
- `:joint2d` — distance × value joint histogram; caches compiled tiled kernel in
  `joint2d_kernel` (default exact `n_dist × n_val` smem; override via `joint2d_compile_cells`).
- `:single_pass` — eight 1D distance histograms `(8, NB)`
- `:single_pass_2d` — eight distance × value joint histograms `(8, NB, n_val)`;
  on-chip modes (`:shared`, `:typeplane`) flush shared histograms directly to `out_*`;
  `:direct` uses `priv_sums_dev` / `priv_cnts_dev` plus merge.
  Caches compiled pair kernel in `sp2d_pair_kernel` (typed `LinearBinEdges` / `LogBinEdges` dist).

Use the matching constructor overload; `reset_histogram!(ws)` zeroes device outputs
before each launch. See [`gpu/SP2D_HTP_EJ.md`](../gpu/SP2D_HTP_EJ.md).
"""
mutable struct GPUSFWorkspace
    backend::KA.Backend
    FT::Type
    kind::Symbol
    dist_bins
    val_bins::Union{Nothing, Any}
    out_dev
    cnt_dev
    out_sums_dev
    out_cnts_dev
    value_edges_dev
    value_edges_sp2d_dev
    dist_general_edges_dev
    NB::Int
    n_bins::Int
    n_dist::Int
    n_val::Int
    n_val_edges::Int
    host_sums_scratch
    host_counts_scratch
    x_dev_cache
    u_dev_cache
    x_dev_3d_cache
    u_dev_3d_cache
    val_plan::Union{Nothing, GPUValueDigitizePlan}
    sp2d_priv_config::Union{Nothing, SP2DPrivConfig}
    sp2d_pair_kernel
    priv_sums_dev
    priv_cnts_dev
    priv_n_tile_blocks::Int
    joint2d_nb2::Int
    joint2d_compile_cells::Int
    joint2d_kernel
end

"""Log-spaced distance bins use host-side FMA params only; no device edge upload."""
function _workspace_upload_dist_edges!(ws::GPUSFWorkspace, ::LogBinEdges, ::Int)
    ws.dist_general_edges_dev = nothing
    return ws
end

function _workspace_upload_dist_edges!(ws::GPUSFWorkspace, ::LinearBinEdges, ::Int)
    ws.dist_general_edges_dev = nothing
    return ws
end

function _workspace_upload_dist_edges!(ws::GPUSFWorkspace, bins::Vector{FT}, n_edges::Int) where {FT}
    ws.dist_general_edges_dev = nothing
    bins_dev = KA.allocate(ws.backend, FT, n_edges)
    copyto!(bins_dev, bins)
    ws.dist_general_edges_dev = bins_dev
    return ws
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
eight-type single-pass distance histograms.
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
        out_dev = KA.zeros(backend, FT, NB)
        cnt_dev = KA.zeros(backend, UInt32, NB)
        out_sums_dev = nothing
        out_cnts_dev = nothing
    else
        out_dev = nothing
        cnt_dev = nothing
        out_sums_dev = KA.zeros(backend, FT, 8, NB)
        out_cnts_dev = KA.zeros(backend, UInt32, 8, NB)
    end

    ws = GPUSFWorkspace(
        backend, FT, kind, dist_bins, nothing,
        out_dev, cnt_dev, out_sums_dev, out_cnts_dev,
        nothing, nothing, nothing,
        NB, n_bins, NB, 0, 0,
        Vector{FT}(undef, NB), Vector{UInt32}(undef, NB),
        nothing, nothing, nothing, nothing,
        nothing,
        nothing, nothing, nothing, nothing, 0,
        0, 0, nothing,
    )
    return _workspace_upload_dist_edges!(ws, dist_bins, n_bins)
end

"""
    GPUSFWorkspace(backend, distance_bins, value_bins; kind=:joint2d, joint2d_compile_cells=nothing)

Workspace for 2D joint (distance × SF value) histograms.

Pass `joint2d_compile_cells` to override compile-time shared-histogram width (default
exact `n_dist × n_val`). See [`joint2d_smem_max`](@ref), [`joint2d_smem_align256`](@ref).
"""
function SFC.GPUSFWorkspace(
    backend::KA.Backend,
    distance_bins::Union{AbstractVector{FT1}, LinearBinEdges, LogBinEdges, InfPaddedBinEdges},
    value_bins::Union{AbstractVector{FT2}, LinearBinEdges, LogBinEdges, InfPaddedBinEdges};
    kind::Symbol = :joint2d,
    joint2d_compile_cells::Union{Nothing, Int} = nothing,
) where {FT1, FT2}
    kind == :joint2d ||
        throw(ArgumentError("three-argument GPUSFWorkspace expects kind=:joint2d (got $kind)"))
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

    ws = GPUSFWorkspace(
        backend, FT, :joint2d, dist_bins, val_bins,
        nothing, nothing, out_sums_dev, out_cnts_dev,
        value_edges_dev, nothing, nothing,
        NB, n_bins, n_dist, n_val, n_val_edges,
        Vector{FT}(undef, n_dist * n_val), Vector{UInt32}(undef, n_dist * n_val),
        nothing, nothing, nothing, nothing,
        val_plan,
        nothing, nothing, nothing, nothing, 0,
        nb2, compile_cells, nothing,
    )
    ws = _workspace_upload_dist_edges!(ws, dist_bins, n_dist_edges)
    if _gpu_joint_2d_tiled_eligible(n_dist, n_val)
        ws.joint2d_kernel = _joint2d_resolve_tiled_kernel!(
            ws, backend, dist_bins, val_plan, compile_cells,
        )
    end
    return ws
end

"""
    GPUSFWorkspace(backend, distance_bins, value_bins; kind=:single_pass_2d)

Workspace for eight distance × value joint histograms (single-pass 2D).
Pass one shared edge object or `NTuple{8,...}` when columns may differ.
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
function _sp2d_value_eltype(value_bins::NTuple{8}, FT3)
    return promote_type(FT3, (_sp2d_value_eltype(value_bins[t], FT3) for t in 1:8)...)
end
function _sp2d_value_eltype(v::Vector{FT}, FT3) where {FT}
    return promote_type(FT3, FT)
end

function SFC.GPUSFWorkspace(
    backend::KA.Backend,
    distance_bins::AbstractVector{FT3},
    value_bins::Union{
        LinearBinEdges,
        LogBinEdges,
        InfPaddedBinEdges,
        NTuple{8, LinearBinEdges},
        NTuple{8, LogBinEdges},
        NTuple{8, InfPaddedBinEdges},
        NTuple{8, Vector{FT3}},
    };
    kind::Symbol = :single_pass_2d,
    n_val::Union{Nothing, Int} = nothing,
) where {FT3}
    kind == :single_pass_2d ||
        throw(ArgumentError("single-pass 2D GPUSFWorkspace expects kind=:single_pass_2d (got $kind)"))
    n_dist_edges = _gpu_n_edges(distance_bins)
    dist_bins = _gpu_normalize_bins(distance_bins)
    NB, n_bins = _workspace_check_nb!(n_dist_edges)
    edge_n_val = _sp2d_n_val_edges(value_bins) - 1
    hist_n_val = n_val === nothing ? edge_n_val : n_val
    _validate_gpu_value_bins!(value_bins, hist_n_val)
    n_val_edges = _sp2d_n_val_edges(value_bins)
    FT = _sp2d_value_eltype(value_bins, FT3)
    val_plan = _gpu_build_value_digitize_plan(backend, value_bins)
    value_edges_sp2d_dev = val_plan isa GPUValueVectorCols ? val_plan.edges_dev : nothing

    out_sums_dev = KA.zeros(backend, FT, 8, NB, hist_n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, 8, NB, hist_n_val)
    priv_config = _sp2d_priv_config(NB, hist_n_val, FT)

    ws = GPUSFWorkspace(
        backend, FT, :single_pass_2d, dist_bins, value_bins,
        nothing, nothing, out_sums_dev, out_cnts_dev,
        nothing, value_edges_sp2d_dev, nothing,
        NB, n_bins, NB, hist_n_val, n_val_edges,
        Vector{FT}(undef, 8 * NB * hist_n_val), Vector{UInt32}(undef, 8 * NB * hist_n_val),
        nothing, nothing, nothing, nothing,
        val_plan,
        priv_config, nothing, nothing, nothing, 0,
        0, 0, nothing,
    )
    ws = _workspace_upload_dist_edges!(ws, dist_bins, n_dist_edges)
    if dist_bins isa LinearBinEdges || dist_bins isa LogBinEdges
        ws.sp2d_pair_kernel = _sp2d_resolve_pair_kernel(ws, backend, dist_bins, val_plan, priv_config)
    end
    return ws
end

"""
Ensure block-private HTP-EJ slabs are allocated for `n_tile_blocks` CUDA tile blocks.
Reallocates when `N_points` (hence tile-block count) grows.
"""
function _ensure_sp2d_priv_bufs!(
    ws::GPUSFWorkspace,
    n_tile_blocks::Int,
)
    ws.kind == :single_pass_2d ||
        throw(ArgumentError("_ensure_sp2d_priv_bufs! requires kind=:single_pass_2d"))
    cfg = ws.sp2d_priv_config
    cfg === nothing && throw(ArgumentError("single_pass_2d workspace missing sp2d_priv_config"))
    cfg.needs_priv_merge ||
        throw(ArgumentError("_ensure_sp2d_priv_bufs! requires needs_priv_merge (direct mode)"))
    if ws.priv_sums_dev === nothing ||
       ws.priv_cnts_dev === nothing ||
       ws.priv_n_tile_blocks < n_tile_blocks
        FT = ws.FT
        ws.priv_sums_dev = KA.zeros(ws.backend, FT, 8, ws.n_dist, ws.n_val, n_tile_blocks)
        ws.priv_cnts_dev = KA.zeros(ws.backend, UInt32, 8, ws.n_dist, ws.n_val, n_tile_blocks)
        ws.priv_n_tile_blocks = n_tile_blocks
    end
    return ws.priv_sums_dev, ws.priv_cnts_dev
end

"""Allocate ephemeral privatization slabs when no workspace is provided."""
function _alloc_sp2d_priv_bufs(
    backend::KA.Backend,
    FT::Type,
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int,
)
  priv_sums = KA.zeros(backend, FT, 8, n_dist, n_val, n_tile_blocks)
  priv_cnts = KA.zeros(backend, UInt32, 8, n_dist, n_val, n_tile_blocks)
  return priv_sums, priv_cnts
end

"""Zero device histogram buffers in `ws` (call before each kernel launch)."""
function SFC.reset_histogram!(ws::GPUSFWorkspace)
    FT = ws.FT
    if ws.kind == :sf1d
        fill!(ws.out_dev, zero(FT))
        fill!(ws.cnt_dev, zero(UInt32))
    else
        fill!(ws.out_sums_dev, zero(FT))
        fill!(ws.out_cnts_dev, zero(UInt32))
        if ws.kind == :single_pass_2d &&
           ws.sp2d_priv_config !== nothing &&
           ws.sp2d_priv_config.needs_priv_merge &&
           ws.priv_sums_dev !== nothing
            fill!(ws.priv_sums_dev, zero(FT))
            fill!(ws.priv_cnts_dev, zero(UInt32))
        end
    end
    return ws
end

"""Optional explicit release of device buffers (fields are cleared; GC reclaims memory)."""
function SFC.release!(ws::GPUSFWorkspace)
    for f in fieldnames(typeof(ws))
        if f ∉ (
            :backend, :FT, :kind, :dist_bins, :val_bins, :NB, :n_bins, :n_dist, :n_val,
            :n_val_edges, :sp2d_priv_config, :priv_n_tile_blocks,
            :joint2d_nb2, :joint2d_compile_cells,
        )
            setfield!(ws, f, nothing)
        end
    end
    return nothing
end

function _validate_gpu_workspace!(
    ws::GPUSFWorkspace,
    backend::KA.Backend,
    kind::Symbol,
    NB::Int;
    n_val::Union{Nothing, Int} = nothing,
)
    ws.backend == backend ||
        throw(ArgumentError("GPUSFWorkspace belongs to a different backend"))
    ws.kind == kind ||
        throw(ArgumentError("GPUSFWorkspace kind $(ws.kind) incompatible with requested $kind"))
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
