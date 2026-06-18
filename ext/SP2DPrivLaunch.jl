# Host launch routing for HTP-EJ single-pass 2D kernels.
#
# Entry: _launch_single_pass_2d_priv! → needs_priv_merge ?
#   _launch_sp2d_onchip!     (pair → out_*, no merge)
#   _launch_sp2d_direct_priv! (priv slab + merge)
#
# Kernel resolution cached on GPUSFWorkspace.sp2d_pair_kernel when using a workspace.
# See gpu/SP2D_HTP_EJ.md

function _sp2d_val_variant(::GPUValueLinearShared)
    return :linear_shared
end
function _sp2d_val_variant(::GPUValueLinearCols)
    return :linear_cols
end
function _sp2d_val_variant(::GPUValueInfLinearShared)
    return :inflinear_shared
end
function _sp2d_val_variant(::GPUValueInfLinearCols)
    return :inflinear_cols
end
function _sp2d_val_variant(::GPUValueLogLinearShared)
    return :log_linear_shared
end
function _sp2d_val_variant(::GPUValueLogLinearCols)
    return :log_linear_cols
end
function _sp2d_val_variant(::GPUValueVectorCols)
    return :vector_cols
end

"""Trailing kernel args after tile launch params (`C, plane, types_per_pass, n_type_passes`)."""
@inline function _sp2d_priv_kernel_tail_args(config::SP2DPrivConfig)
    return (
        config.n_joint_cells,
        config.plane_cells,
        config.types_per_pass,
        config.n_type_passes,
    )
end

"""Resolve compiled pair kernel; cache on workspace when provided."""
function _sp2d_resolve_pair_kernel(
    workspace::Union{GPUSFWorkspace, Nothing},
    backend::KA.Backend,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    config::SP2DPrivConfig,
    ws::Int = SF_GPU_TILED_WS,
)
    if workspace !== nothing && workspace.sp2d_pair_kernel !== nothing
        return workspace.sp2d_pair_kernel
    end
    dist_bins isa LinearBinEdges || dist_bins isa LogBinEdges ||
        throw(ArgumentError(
            "HTP-EJ sp2d pair kernel requires LinearBinEdges or LogBinEdges distance bins (got $(typeof(dist_bins)))",
        ))
    dist_sym = _sp2d_dist_variant(dist_bins)
    val_sym = _sp2d_val_variant(val_plan)
    kernel! = _sp2d_priv_kernel_fn(dist_sym, val_sym, config.accum_mode, backend, ws)
    if workspace !== nothing && workspace.kind == :single_pass_2d
        workspace.sp2d_pair_kernel = kernel!
    end
    return kernel!
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueInfLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueInfLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLogLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLogLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueInfLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueInfLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLogLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLogLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueVectorCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.edges_dev,
        n_tiles, n_tile_blocks, ws,
        _sp2d_priv_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    priv_sums_dev,
    priv_cnts_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    throw(ArgumentError(
        "unsupported HTP-EJ single-pass 2D pair (dist=$(typeof(dist_bins)), value=$(typeof(val_plan)))",
    ))
end

function _launch_single_pass_2d_priv!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    if config.needs_priv_merge
        return _launch_sp2d_direct_priv!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
            workspace = workspace,
        )
    end
    return _launch_sp2d_onchip!(
        backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
end

"""On-chip path: pair kernel flushes shared histogram directly to `out_*` (no priv, no merge)."""
function _launch_sp2d_onchip!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    _sp2d_pair_launch_kernel!(
        backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
    return nothing
end

"""Direct path: block-private slab during pair traversal, then merge into `out_*`."""
function _launch_sp2d_direct_priv!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    config.needs_priv_merge ||
        throw(ArgumentError("_launch_sp2d_direct_priv! requires needs_priv_merge"))
    priv_sums, priv_cnts, n_tb = _sp2d_priv_pair_bufs_and_launch!(
        backend, out_sums_dev, x_dev, u_dev, dist_bins, val_plan,
        N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
    _launch_merge_sp2d_priv!(
        backend, out_sums_dev, out_cnts_dev, priv_sums, priv_cnts,
        n_dist, n_val_edges - 1, n_tb,
    )
    return nothing
end

"""Allocate/zero priv slabs and run the direct pair kernel; returns `(priv_sums, priv_cnts, n_tile_blocks)`."""
function _sp2d_priv_pair_bufs_and_launch!(
    backend::KA.Backend,
    out_sums_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DPrivConfig;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    config.needs_priv_merge ||
        throw(ArgumentError("_sp2d_priv_pair_bufs_and_launch! requires needs_priv_merge (direct mode)"))
    _, n_tile_blocks, _, _ = _tiled_launch_params(N_points)
    if workspace === nothing
        priv_sums, priv_cnts = _alloc_sp2d_priv_bufs(
            backend, eltype(out_sums_dev), n_dist, n_val_edges - 1, n_tile_blocks,
        )
        fill!(priv_sums, zero(eltype(out_sums_dev)))
        fill!(priv_cnts, zero(UInt32))
    else
        priv_sums, priv_cnts = _ensure_sp2d_priv_bufs!(workspace, n_tile_blocks)
        fill!(priv_sums, zero(eltype(out_sums_dev)))
        fill!(priv_cnts, zero(UInt32))
    end
    n_tb = _sp2d_pair_launch_kernel!(
        backend, priv_sums, priv_cnts, x_dev, u_dev,
        dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
    return priv_sums, priv_cnts, n_tb
end
