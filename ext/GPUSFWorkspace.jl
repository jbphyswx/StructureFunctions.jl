# GPUSFWorkspace — device-resident histogram buffers and cached bin edges for GPU SF paths.
# Included from StructureFunctionsGPUExt.jl (uses _GPUBinLayout, _resolve_gpu_bin_layout, etc.).

"""
    GPUSFWorkspace

Reusable device histogram buffers and cached distance-bin edge arrays for GPU
structure-function launches. Construct once per `(backend, distance_bins[, value_bins])`
configuration; pass to `gpu_calculate_structure_function(!)` or slice drivers to avoid
per-call `KA.zeros` allocation and repeated edge uploads.

# Kinds (`kind` field)
- `:sf1d` — 1D distance histogram (`out_dev`, `cnt_dev` vectors)
- `:joint2d` — distance × value joint histogram
- `:single_pass` — eight 1D distance histograms `(8, NB)`
- `:single_pass_2d` — eight distance × value joint histograms `(8, NB, n_val)`

Use the matching constructor overload; `reset_histogram!(ws)` zeroes device outputs
before each launch.
"""
mutable struct GPUSFWorkspace
    backend::KA.Backend
    FT::Type
    kind::Symbol
    dist_layout::_GPUBinLayout
    val_layout::Union{_GPUBinLayout, Nothing}
    out_dev
    cnt_dev
    out_sums_dev
    out_cnts_dev
    value_edges_dev
    value_edges_sp2d_dev
    dist_log_edges_dev
    dist_log_lut_dev
    dist_general_edges_dev
    NB::Int
    n_bins::Int
    n_dist::Int
    n_val::Int
    n_val_edges::Int
    host_sums_scratch
    host_counts_scratch
end

"""Upload distance-bin edge tables into workspace fields (log LUT or general edges)."""
function _workspace_upload_dist_edges!(ws::GPUSFWorkspace, layout::_GPUBinLayout, n_edges::Int)
    ws.dist_log_edges_dev = nothing
    ws.dist_log_lut_dev = nothing
    ws.dist_general_edges_dev = nothing
    if layout.log !== nothing
        lbe = layout.log
        FT = eltype(lbe.edges)
        edges_dev = KA.allocate(ws.backend, FT, n_edges)
        lut_dev = KA.allocate(ws.backend, Int32, length(lbe.lut))
        copyto!(edges_dev, collect(lbe.edges))
        copyto!(lut_dev, Int32.(lbe.lut))
        ws.dist_log_edges_dev = edges_dev
        ws.dist_log_lut_dev = lut_dev
    elseif layout.general_edges !== nothing
        edges = layout.general_edges
        FT = eltype(edges)
        bins_dev = KA.allocate(ws.backend, FT, n_edges)
        copyto!(bins_dev, edges)
        ws.dist_general_edges_dev = bins_dev
    end
    return ws
end

function _workspace_check_nb!(layout::_GPUBinLayout, n_bins::Int)
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
    layout = _resolve_gpu_bin_layout(distance_bins)
    NB, n_bins = _workspace_check_nb!(layout, length(_layout_edge_vector(layout)))

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
        backend, FT, kind, layout, nothing,
        out_dev, cnt_dev, out_sums_dev, out_cnts_dev,
        nothing, nothing, nothing, nothing, nothing,
        NB, n_bins, NB, 0, 0,
        Vector{FT}(undef, NB), Vector{UInt32}(undef, NB),
    )
    return _workspace_upload_dist_edges!(ws, layout, n_bins)
end

"""
    GPUSFWorkspace(backend, distance_bins, value_bins; kind=:joint2d)

Workspace for 2D joint (distance × SF value) histograms.
"""
function SFC.GPUSFWorkspace(
    backend::KA.Backend,
    distance_bins::AbstractVector{FT1},
    value_bins::AbstractVector{FT2};
    kind::Symbol = :joint2d,
) where {FT1, FT2}
    kind == :joint2d ||
        throw(ArgumentError("three-argument GPUSFWorkspace expects kind=:joint2d (got $kind)"))
    FT = promote_type(FT1, FT2)
    dist_layout = _resolve_gpu_bin_layout(distance_bins)
    val_layout = _resolve_gpu_bin_layout(value_bins)
    n_dist_edges = length(_layout_edge_vector(dist_layout))
    n_val_edges = length(_layout_edge_vector(val_layout))
    NB, n_bins = _workspace_check_nb!(dist_layout, n_dist_edges)
    n_dist = n_dist_edges - 1
    n_val = n_val_edges - 1
    n_dist > 0 && n_val > 0 ||
        throw(ArgumentError("distance_bins and value_bins must each have at least two edges"))

    value_host = _layout_edge_vector(val_layout)
    value_edges_dev = KA.allocate(backend, FT, n_val_edges)
    copyto!(value_edges_dev, value_host)

    out_sums_dev = KA.zeros(backend, FT, n_dist, n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, n_dist, n_val)

    ws = GPUSFWorkspace(
        backend, FT, :joint2d, dist_layout, val_layout,
        nothing, nothing, out_sums_dev, out_cnts_dev,
        value_edges_dev, nothing, nothing, nothing, nothing,
        NB, n_bins, n_dist, n_val, n_val_edges,
        Vector{FT}(undef, n_dist * n_val), Vector{UInt32}(undef, n_dist * n_val),
    )
    return _workspace_upload_dist_edges!(ws, dist_layout, n_dist_edges)
end

"""
    GPUSFWorkspace(backend, distance_bins, value_bins_by_type; kind=:single_pass_2d)

Workspace for eight distance × value joint histograms (single-pass 2D).
"""
function SFC.GPUSFWorkspace(
    backend::KA.Backend,
    distance_bins::AbstractVector{FT3},
    value_bins_by_type::AbstractVector{<:AbstractVector};
    kind::Symbol = :single_pass_2d,
) where {FT3}
    kind == :single_pass_2d ||
        throw(ArgumentError("vector-of-vectors GPUSFWorkspace expects kind=:single_pass_2d (got $kind)"))
    dist_layout = _resolve_gpu_bin_layout(distance_bins)
    n_dist_edges = length(_layout_edge_vector(dist_layout))
    NB, n_bins = _workspace_check_nb!(dist_layout, n_dist_edges)
    n_val = length(value_bins_by_type[1]) - 1
    SFC._validate_value_bins_by_type(value_bins_by_type, n_val)
    n_val_edges = n_val + 1
    FT = promote_type(FT3, eltype.(value_bins_by_type)...)

    value_host = _gpu_pack_value_edges(value_bins_by_type)
    value_edges_sp2d_dev = KA.allocate(backend, FT, n_val_edges, 8)
    copyto!(value_edges_sp2d_dev, value_host)

    out_sums_dev = KA.zeros(backend, FT, 8, NB, n_val)
    out_cnts_dev = KA.zeros(backend, UInt32, 8, NB, n_val)

    ws = GPUSFWorkspace(
        backend, FT, :single_pass_2d, dist_layout, nothing,
        nothing, nothing, out_sums_dev, out_cnts_dev,
        nothing, value_edges_sp2d_dev, nothing, nothing, nothing,
        NB, n_bins, NB, n_val, n_val_edges,
        Vector{FT}(undef, 8 * NB * n_val), Vector{UInt32}(undef, 8 * NB * n_val),
    )
    return _workspace_upload_dist_edges!(ws, dist_layout, n_dist_edges)
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
    end
    return ws
end

"""Optional explicit release of device buffers (fields are cleared; GC reclaims memory)."""
function SFC.release!(ws::GPUSFWorkspace)
    for f in fieldnames(typeof(ws))
        if f ∉ (:backend, :FT, :kind, :dist_layout, :val_layout, :NB, :n_bins, :n_dist, :n_val, :n_val_edges)
            setfield!(ws, f, nothing)
        end
    end
    return nothing
end

function _validate_gpu_workspace!(
    ws::GPUSFWorkspace,
    backend::KA.Backend,
    kind::Symbol,
    NB::Int,
)
    ws.backend == backend ||
        throw(ArgumentError("GPUSFWorkspace belongs to a different backend"))
    ws.kind == kind ||
        throw(ArgumentError("GPUSFWorkspace kind $(ws.kind) incompatible with requested $kind"))
    ws.NB == NB ||
        throw(ArgumentError("GPUSFWorkspace NB=$(ws.NB) incompatible with requested NB=$NB"))
    return ws
end

"""Return cached log/general distance edge device buffers for tiled launches."""
function _workspace_dist_edge_bufs(ws::Union{GPUSFWorkspace, Nothing})
    ws === nothing && return nothing, nothing, nothing
    return ws.dist_log_edges_dev, ws.dist_log_lut_dev, ws.dist_general_edges_dev
end
