# Host launch routing for joint 2D tiled kernels (distance × value digitize plans).

"""Resolve and optionally cache tiled joint kernel on workspace."""
function _joint2d_resolve_tiled_kernel!(
    workspace::Union{GPUSFWorkspace, Nothing},
    backend::KA.Backend,
    dist_bins,
    val_plan::Union{GPUValueDigitizePlan, Nothing},
    compile_cells::Int,
    ws::Int = SF_GPU_TILED_WS,
)
    if workspace !== nothing && workspace.joint2d_kernel !== nothing
        return workspace.joint2d_kernel
    end
    dist_route = _joint2d_dist_route(dist_bins)
    val_route = _joint2d_val_route(val_plan)
    kernel! = _joint2d_tiled_kernel_fn(dist_route, val_route, compile_cells, backend, ws)
    if workspace !== nothing && workspace.kind == :joint2d
        workspace.joint2d_kernel = kernel!
    end
    return kernel!
end

function _launch_joint_2d_tiled_kernel!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    value_edges_dev,
    sf_type,
    dist_bins,
    val_plan::Union{GPUValueDigitizePlan, Nothing},
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    n_val::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    NB2 = n_dist * n_val
    compile_cells = workspace === nothing ? NB2 : workspace.joint2d_compile_cells
    kernel! = _joint2d_resolve_tiled_kernel!(workspace, backend, dist_bins, val_plan, compile_cells, ws)

    if dist_bins isa LinearBinEdges && val_plan isa GPUValueLinearShared
        lbe, vp = dist_bins, val_plan
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa LinearBinEdges && val_plan isa GPUValueInfLinearShared
        lbe, vp = dist_bins, val_plan
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            vp.n_inner_edges, vp.inner_last,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa LinearBinEdges && val_plan isa GPUValueLogLinearShared
        lbe, vp = dist_bins, val_plan
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa LinearBinEdges && val_plan === nothing
        lbe = dist_bins
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa LogBinEdges && val_plan isa GPUValueLinearShared
        lbe, vp = dist_bins, val_plan
        d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            d_f, d_l, d_inv, d_off, d_st,
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa LogBinEdges && val_plan isa GPUValueInfLinearShared
        lbe, vp = dist_bins, val_plan
        d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            d_f, d_l, d_inv, d_off, d_st,
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            vp.n_inner_edges, vp.inner_last,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa LogBinEdges && val_plan isa GPUValueLogLinearShared
        lbe, vp = dist_bins, val_plan
        d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            d_f, d_l, d_inv, d_off, d_st,
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa LogBinEdges && val_plan === nothing
        lbe = dist_bins
        d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            d_f, d_l, d_inv, d_off, d_st,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa Vector && val_plan isa GPUValueLinearShared
        edges, vp = dist_bins, val_plan
        _, _, gen_e = _workspace_dist_edge_bufs(workspace)
        dist_dev = gen_e === nothing ? begin
            d = KA.allocate(backend, eltype(edges), n_dist_edges)
            copyto!(d, edges)
            d
        end : gen_e
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            edges[1],
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa Vector && val_plan isa GPUValueInfLinearShared
        edges, vp = dist_bins, val_plan
        _, _, gen_e = _workspace_dist_edge_bufs(workspace)
        dist_dev = gen_e === nothing ? begin
            d = KA.allocate(backend, eltype(edges), n_dist_edges)
            copyto!(d, edges)
            d
        end : gen_e
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            edges[1],
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            vp.n_inner_edges, vp.inner_last,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa Vector && val_plan isa GPUValueLogLinearShared
        edges, vp = dist_bins, val_plan
        _, _, gen_e = _workspace_dist_edge_bufs(workspace)
        dist_dev = gen_e === nothing ? begin
            d = KA.allocate(backend, eltype(edges), n_dist_edges)
            copyto!(d, edges)
            d
        end : gen_e
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            edges[1],
            vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    elseif dist_bins isa Vector && val_plan === nothing
        edges = dist_bins
        FT = eltype(edges)
        _, _, gen_e = _workspace_dist_edge_bufs(workspace)
        dist_dev = gen_e === nothing ? begin
            d = KA.allocate(backend, FT, n_dist_edges)
            copyto!(d, edges)
            d
        end : gen_e
        _joint2d_invoke_kernel!(
            kernel!,
            out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_dev, value_edges_dev, sf_type,
            N_points, n_dist_edges, n_val_edges, n_val, NB2,
            edges[1], n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
    else
        throw(ArgumentError(
            "unsupported joint2d tiled launch: dist=$(typeof(dist_bins)), val_plan=$(typeof(val_plan))",
        ))
    end
    return nothing
end
# Host launch routing for HTP-EJ single-pass 2D kernels.
#
# Entry: _launch_single_pass_2d_strategy! → needs_partition_merge ?
#   _launch_sp2d_onchip!     (pair → out_*, no merge)
#   _launch_sp2d_direct_partitioned! (private partition + merge)
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
@inline function _sp2d_strategy_kernel_tail_args(config::SP2DAccumulationStrategy)
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
    config::SP2DAccumulationStrategy,
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
    kernel! = _sp2d_partition_kernel_fn(dist_sym, val_sym, config.accum_mode, backend, ws)
    if workspace !== nothing && workspace.kind == :single_pass_2d
        workspace.sp2d_pair_kernel = kernel!
    end
    return kernel!
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueInfLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueInfLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLogLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLogLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueInfLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueInfLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLogLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLogLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueVectorCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    kernel! = _sp2d_resolve_pair_kernel(workspace, backend, dist_bins, val_plan, config, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.edges_dev,
        n_tiles, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    throw(ArgumentError(
        "unsupported HTP-EJ single-pass 2D pair (dist=$(typeof(dist_bins)), value=$(typeof(val_plan)))",
    ))
end

function _launch_single_pass_2d_strategy!(
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
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    if config.needs_partition_merge
        return _launch_sp2d_direct_partitioned!(
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

"""On-chip path: pair kernel flushes shared histogram directly to `out_*` (no partition, no merge)."""
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
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    _sp2d_pair_launch_kernel!(
        backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
    return nothing
end

"""Direct path: block-private partition during pair traversal, then merge into `out_*`."""
function _launch_sp2d_direct_partitioned!(
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
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    config.needs_partition_merge ||
        throw(ArgumentError("_launch_sp2d_direct_partitioned! requires needs_partition_merge"))
    partition_sums, partition_counts, n_tb = _sp2d_partition_pair_bufs_and_launch!(
        backend, out_sums_dev, x_dev, u_dev, dist_bins, val_plan,
        N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
    _launch_merge_sp2d_partitions!(
        backend, out_sums_dev, out_cnts_dev, partition_sums, partition_counts,
        n_dist, n_val_edges - 1, n_tb,
    )
    return nothing
end

"""Allocate/zero private partitions and run the direct pair kernel; returns `(partition_sums, partition_counts, n_tile_blocks)`."""
function _sp2d_partition_pair_bufs_and_launch!(
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
    config::SP2DAccumulationStrategy;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    config.needs_partition_merge ||
        throw(ArgumentError("_sp2d_partition_pair_bufs_and_launch! requires needs_partition_merge (direct mode)"))
    _, n_tile_blocks, _, _ = _tiled_launch_params(N_points)
    if workspace === nothing
        partition_sums, partition_counts = _alloc_sp2d_partition_bufs(
            backend, eltype(out_sums_dev), n_dist, n_val_edges - 1, n_tile_blocks,
        )
        fill!(partition_sums, zero(eltype(out_sums_dev)))
        fill!(partition_counts, zero(UInt32))
    else
        partition_sums, partition_counts = _ensure_sp2d_partition_bufs!(workspace, n_tile_blocks)
        fill!(partition_sums, zero(eltype(out_sums_dev)))
        fill!(partition_counts, zero(UInt32))
    end
    n_tb = _sp2d_pair_launch_kernel!(
        backend, partition_sums, partition_counts, x_dev, u_dev,
        dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
    return partition_sums, partition_counts, n_tb
end
# Host launch routing for single-pass 2D tiled kernels (distance × value digitize plan).

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    kernel! = _sf6_single_pass_2d_kernel_tiled128_linear_linear_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    kernel! = _sf6_single_pass_2d_kernel_tiled128_linear_linear_val_cols_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueInfLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    kernel! = _sf6_single_pass_2d_kernel_tiled128_linear_inflinear_val_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueInfLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    kernel! = _sf6_single_pass_2d_kernel_tiled128_linear_inflinear_val_cols_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf6_single_pass_2d_kernel_tiled128_log_linear_val_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf6_single_pass_2d_kernel_tiled128_log_linear_val_cols_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueInfLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf6_single_pass_2d_kernel_tiled128_log_inflinear_val_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueInfLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf6_single_pass_2d_kernel_tiled128_log_inflinear_val_cols_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLogLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf6_single_pass_2d_kernel_tiled128_log_log_val_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueVectorCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf6_single_pass_2d_kernel_tiled128_log_vector_val_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        vp.edges_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLogLinearShared,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    kernel! = _sf6_single_pass_2d_kernel_tiled128_linear_log_val_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLogLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    kernel! = _sf6_single_pass_2d_kernel_tiled128_linear_log_val_cols_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueLogLinearCols,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf6_single_pass_2d_kernel_tiled128_log_log_val_cols_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    return nothing
end

function _launch_single_pass_2d_tiled!(
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
    n_dist::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    throw(ArgumentError(
        "unsupported single-pass 2D GPU pair (dist=$(typeof(dist_bins)), value=$(typeof(val_plan)))",
    ))
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueLinearShared,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    lbe = dist_bins
    vp = val_plan
    value_edges_dev = workspace === nothing ? nothing : workspace.value_edges_sp2d_dev
    value_edges_dev === nothing && throw(ArgumentError("naive single-pass 2D linear+linear requires vector value workspace"))
    kernel! = _sf_single_pass_2d_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, value_edges_dev,
        N_points, Val(N_dims), n_dist_edges, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LinearBinEdges,
    val_plan::GPUValueVectorCols,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    lbe = dist_bins
    kernel! = _sf_single_pass_2d_kernel_linear!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, val_plan.edges_dev,
        N_points, Val(N_dims), n_dist_edges, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::LogBinEdges,
    val_plan::GPUValueVectorCols,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    lbe = dist_bins
    vp = val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel! = _sf_single_pass_2d_kernel_log!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        vp.edges_dev,
        N_points, Val(N_dims), n_dist_edges, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins::Vector{FT},
    val_plan::GPUValueVectorCols,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
) where {FT}
    _, _, gen_e = _workspace_dist_edge_bufs(workspace)
    bins_dev = gen_e === nothing ? begin
        b = KA.allocate(backend, FT, n_dist_edges)
        copyto!(b, dist_bins)
        b
    end : gen_e
    kernel! = _sf_single_pass_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        bins_dev, val_plan.edges_dev, N_points, Val(N_dims), n_dist_edges, n_val_edges;
        ndrange = (N_points, N_points),
    )
    return nothing
end

function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    if N_dims == 2 && _gpu_single_pass_2d_use_tiled(dist_bins, val_plan, n_dist_edges - 1)
        return _launch_single_pass_2d_tiled!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist_edges - 1;
            workspace = workspace,
        )
    end
    throw(ArgumentError(
        "single-pass 2D global-atomic path unsupported for (dist=$(typeof(dist_bins)), value=$(typeof(val_plan)))",
    ))
end

function _launch_single_pass_2d!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    dist_bins,
    val_plan::GPUValueDigitizePlan,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    force_global_atomic::Bool = false,
)
    n_dist = n_dist_edges - 1
    if force_global_atomic
        return _launch_single_pass_2d_kernel!(
            backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_bins, val_plan, N_points, N_dims, n_dist_edges, n_val_edges;
            workspace = workspace,
        )
    end
    if N_dims == 2 && !force_global_atomic && _gpu_single_pass_2d_use_tiled(dist_bins, val_plan, n_dist)
        config = workspace === nothing ?
            _sp2d_accumulation_strategy(n_dist, n_val_edges - 1, eltype(out_sums_dev)) :
            workspace.sp2d_accumulation_strategy
        return _launch_single_pass_2d_strategy!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
            workspace = workspace,
        )
    end
    return _launch_single_pass_2d_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, val_plan, N_points, N_dims, n_dist_edges, n_val_edges;
        workspace = workspace,
    )
end
# Production batch launch drivers — fixed-x and varying-x.
# Included from StructureFunctionsGPUExt.jl after BatchTiledKernels.jl.
#
# Fixed-x launch invariant (do not regress):
# - `ndrange = n_tile_blocks * workgroup_size` only — never multiply by `cld(B, strip_w)`.
# - Host strip loop over `BATCH_USMEM_STRIP_W` (16) via `_batch_fixed_x_usmem_priv!`.
# - `KA.CPU`: serial merge; other backends: grouped merge. Same kernel on both.
# - Varying-x routes use `(tile, auxiliary)` grid scaling (`ndrange * B`), not host strips.

function _launch_batch_fixed_x_sf!(
    backend::KA.CPU,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    workspace::Union{GPUBatchWorkspace{FT}, Nothing} = nothing,
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("batch tiled128 supports at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    n_priv = _batch_usmem_n_priv(n_tile_blocks)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_fixed_x_sf_kernel(backend, ws)
    merge_sums! = _batch_merge_usmem_sums!(backend, ws)
    merge_cnts! = _batch_merge_usmem_cnts!(backend, ws)
    partial_sums = KA.zeros(backend, FT, NB, BATCH_USMEM_STRIP_W, n_priv)
    partial_cnts = KA.zeros(backend, UInt32, NB, n_priv)
    b_base = 1
    while b_base <= B
        bw = min(BATCH_USMEM_STRIP_W, B - b_base + 1)
        fill!(partial_sums, zero(FT))
        fill!(partial_cnts, zero(UInt32))
        kernel!(
            partial_sums, partial_cnts, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        merge_sums!(
            @view(sums_dev[:, b_base:b_base + bw - 1]), partial_sums,
            NB, bw, n_priv, NB * bw;
            ndrange = NB * bw,
        )
        if b_base == 1
            merge_cnts!(
                @view(counts_dev[:, 1]), partial_cnts, NB, n_priv, NB;
                ndrange = NB,
            )
        end
        b_base += bw
    end
    if B > 1
        counts_dev[:, 2:end] .= @view counts_dev[:, 1]
    end
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_fixed_x_sf!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    workspace::Union{GPUBatchWorkspace{FT}, Nothing} = nothing,
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("batch tiled128 supports at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    n_priv = _batch_usmem_n_priv(n_tile_blocks)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_fixed_x_sf_kernel(backend, ws)
    merge_sums! = _batch_merge_usmem_sums_grouped!(backend, ws)
    merge_cnts! = _batch_merge_usmem_cnts_grouped!(backend, ws)
    partial_sums = KA.zeros(backend, FT, NB, BATCH_USMEM_STRIP_W, n_priv)
    partial_cnts = KA.zeros(backend, UInt32, NB, n_priv)
    b_base = 1
    while b_base <= B
        bw = min(BATCH_USMEM_STRIP_W, B - b_base + 1)
        fill!(partial_sums, zero(FT))
        fill!(partial_cnts, zero(UInt32))
        kernel!(
            partial_sums, partial_cnts, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        merge_sums!(
            @view(sums_dev[:, b_base:b_base + bw - 1]), partial_sums,
            NB, bw, n_priv, ws;
            ndrange = NB * bw * ws,
        )
        if b_base == 1
            merge_cnts!(
                @view(counts_dev[:, 1]), partial_cnts, NB, n_priv, ws;
                ndrange = NB * ws,
            )
        end
        b_base += bw
    end
    if B > 1
        counts_dev[:, 2:end] .= @view counts_dev[:, 1]
    end
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_fixed_x_sp1d!(
    backend::KA.CPU,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("batch SP1D tiled128 supports at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_fixed_x_sp1d_usmem_priv!(backend, ws)
    merge_sums! = _batch_merge_sp1d_sums!(backend, ws)
    merge_cnts! = _batch_merge_sp1d_cnts!(backend, ws)
    partial_sums = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB, BATCH_SP1D_USMEM_STRIP_W, n_tile_blocks)
    partial_cnts = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB, n_tile_blocks)
    b_base = 1
    while b_base <= B
        bw = min(BATCH_SP1D_USMEM_STRIP_W, B - b_base + 1)
        fill!(partial_sums, zero(FT))
        fill!(partial_cnts, zero(UInt32))
        kernel!(
            partial_sums, partial_cnts, x_dev, u_dev,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        merge_sums!(
            @view(sums_dev[:, :, b_base:b_base + bw - 1]), partial_sums,
            NB, bw, n_tile_blocks, SF_GPU_SINGLE_PASS_N * NB * bw;
            ndrange = SF_GPU_SINGLE_PASS_N * NB * bw,
        )
        if b_base == 1
            merge_cnts!(
                @view(counts_dev[:, :, 1]), partial_cnts, NB, n_tile_blocks, SF_GPU_SINGLE_PASS_N * NB;
                ndrange = SF_GPU_SINGLE_PASS_N * NB,
            )
        end
        b_base += bw
    end
    if B > 1
        @views counts_dev[:, :, 2:end] .= counts_dev[:, :, 1:1]
    end
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_fixed_x_sp1d!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SF_GPU_MAX_BINS &&
        error("batch SP1D tiled128 supports at most $SF_GPU_MAX_BINS bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_fixed_x_sp1d_usmem_priv!(backend, ws)
    merge_sums! = _batch_merge_sp1d_sums_grouped!(backend, ws)
    merge_cnts! = _batch_merge_sp1d_cnts_grouped!(backend, ws)
    partial_sums = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, NB, BATCH_SP1D_USMEM_STRIP_W, n_tile_blocks)
    partial_cnts = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, NB, n_tile_blocks)
    b_base = 1
    while b_base <= B
        bw = min(BATCH_SP1D_USMEM_STRIP_W, B - b_base + 1)
        fill!(partial_sums, zero(FT))
        fill!(partial_cnts, zero(UInt32))
        kernel!(
            partial_sums, partial_cnts, x_dev, u_dev,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        merge_sums!(
            @view(sums_dev[:, :, b_base:b_base + bw - 1]), partial_sums,
            NB, bw, n_tile_blocks, ws;
            ndrange = SF_GPU_SINGLE_PASS_N * NB * bw * ws,
        )
        if b_base == 1
            merge_cnts!(
                @view(counts_dev[:, :, 1]), partial_cnts, NB, n_tile_blocks, ws;
                ndrange = SF_GPU_SINGLE_PASS_N * NB * ws,
            )
        end
        b_base += bw
    end
    if B > 1
        @views counts_dev[:, :, 2:end] .= counts_dev[:, :, 1:1]
    end
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_varying_x_sf!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT},
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_varying_x_sf!(backend, ws)
    kernel!(
        sums_dev, counts_dev, x_dev, u_dev, sf_type,
        N, n_bins, NB, fe, le, is_, off, sv,
        n_tiles, n_tile_blocks, ws, B;
        ndrange = ndrange * B,
    )
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_varying_x_sp1d!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT},
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_varying_x_sp1d!(backend, ws)
    kernel!(
        sums_dev, counts_dev, x_dev, u_dev,
        N, n_bins, NB, fe, le, is_, off, sv,
        n_tiles, n_tile_blocks, ws, B;
        ndrange = ndrange * B,
    )
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_fixed_x_sp2d!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    N::Int,
    B::Int,
    dist_lbe::LinearBinEdges{FT},
    val_plan::GPUValueDigitizePlan,
    n_dist::Int,
    n_val::Int,
) where {FT}
    n_bins = length(dist_lbe.edges)
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    fe, le, is_, off, sv = dist_lbe.first_edge, dist_lbe.last_edge, dist_lbe.inv_step,
        dist_lbe.offset, dist_lbe.step_val
    kernel! = _batch_varying_x_sp2d_fixed_x!(backend, ws)
    partial = KA.zeros(backend, FT, SF_GPU_SINGLE_PASS_N, n_dist, n_val, BATCH_USMEM_STRIP_W, n_tile_blocks)
    partial_cnt = KA.zeros(backend, UInt32, SF_GPU_SINGLE_PASS_N, n_dist, n_val, BATCH_USMEM_STRIP_W, n_tile_blocks)
    b_base = 1
    while b_base <= B
        bw = min(BATCH_USMEM_STRIP_W, B - b_base + 1)
        fill!(partial, zero(FT))
        fill!(partial_cnt, zero(UInt32))
        kernel!(
            partial, partial_cnt, x_dev, u_dev,
            N, n_bins, n_dist, n_val, b_base, bw,
            fe, le, is_, off, sv, val_plan,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        _merge_batch_sp2d_partial!(
            backend, sums_dev, counts_dev, partial, partial_cnt,
            n_dist, n_val, b_base, bw, n_tile_blocks, ws,
        )
        b_base += bw
    end
    KA.synchronize(backend)
    return nothing
end

@inline function _merge_batch_sp2d_partial!(
    backend::KA.CPU,
    sums_dev,
    counts_dev,
    partial,
    partial_cnt,
    n_dist::Int,
    n_val::Int,
    b_base::Int,
    bw::Int,
    n_tile_blocks::Int,
    ws::Int,
)
    merge_s! = _batch_merge_sp2d_sums!(backend, ws)
    merge_c! = _batch_merge_sp2d_cnts!(backend, ws)
    n_out = SF_GPU_SINGLE_PASS_N * n_dist * n_val * bw
    merge_s!(
        @view(sums_dev[:, :, :, b_base:(b_base + bw - 1)]), partial,
        n_dist, n_val, bw, n_tile_blocks, n_out;
        ndrange = n_out,
    )
    merge_c!(
        @view(counts_dev[:, :, :, b_base:(b_base + bw - 1)]), partial_cnt,
        n_dist, n_val, bw, n_tile_blocks, n_out;
        ndrange = n_out,
    )
    return nothing
end

@inline function _merge_batch_sp2d_partial!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    partial,
    partial_cnt,
    n_dist::Int,
    n_val::Int,
    b_base::Int,
    bw::Int,
    n_tile_blocks::Int,
    ws::Int,
)
    merge_s! = _batch_merge_sp2d_sums_grouped!(backend, ws)
    merge_c! = _batch_merge_sp2d_cnts_grouped!(backend, ws)
    n_out = SF_GPU_SINGLE_PASS_N * n_dist * n_val * bw
    merge_s!(
        @view(sums_dev[:, :, :, b_base:(b_base + bw - 1)]), partial,
        n_dist, n_val, bw, n_tile_blocks, ws;
        ndrange = n_out * ws,
    )
    merge_c!(
        @view(counts_dev[:, :, :, b_base:(b_base + bw - 1)]), partial_cnt,
        n_dist, n_val, bw, n_tile_blocks, ws;
        ndrange = n_out * ws,
    )
    return nothing
end

function _launch_batch_varying_x_sp2d!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    N::Int,
    B::Int,
    dist_lbe::LinearBinEdges{FT},
    val_plan::GPUValueDigitizePlan,
    n_dist::Int,
    n_val::Int,
) where {FT}
    n_bins = length(dist_lbe.edges)
    n_tiles, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    fe, le, is_, off, sv = dist_lbe.first_edge, dist_lbe.last_edge, dist_lbe.inv_step,
        dist_lbe.offset, dist_lbe.step_val
    kernel! = _batch_varying_x_sp2d!(backend, ws)
    kernel!(
        sums_dev, counts_dev, x_dev, u_dev,
        N, n_bins, n_dist, n_val, fe, le, is_, off, sv, val_plan,
        n_tiles, n_tile_blocks, ws, B;
        ndrange = ndrange * B,
    )
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_varying_x_joint2d!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    distance_bins,
    value_bins,
    n_dist::Int,
    n_val::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    workgroup_size::Int = 64,
)
    FT = eltype(sums_dev)
    dist_bins    = _gpu_normalize_bins(distance_bins)
    n_dist_edges = _gpu_n_edges(distance_bins)
    n_val_edges  = _gpu_n_edges(value_bins)
    val_plan     = _joint2d_build_val_plan(backend, value_bins)

    value_host      = _gpu_host_edge_vector(value_bins)
    value_edges_dev = KA.allocate(backend, FT, n_val_edges)
    copyto!(value_edges_dev, value_host)

    out_sums_dev = KA.allocate(backend, FT, n_dist, n_val)
    out_cnts_dev = KA.allocate(backend, UInt32, n_dist, n_val)

    for b in 1:B
        fill!(out_sums_dev, zero(FT))
        fill!(out_cnts_dev, zero(UInt32))
        _launch_joint_2d_kernel!(
            backend, workgroup_size,
            out_sums_dev, out_cnts_dev,
            @view(x_dev[:, :, b]), @view(u_dev[:, :, b]),
            value_edges_dev,
            sf_type, dist_bins, N, n_dist_edges, n_val_edges;
            val_plan = val_plan,
        )
        copyto!(@view(sums_dev[:, :, b]), out_sums_dev)
        copyto!(@view(counts_dev[:, :, b]), out_cnts_dev)
    end
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_fixed_x_joint2d!(
    backend::KA.Backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    distance_bins,
    value_bins,
    n_dist::Int,
    n_val::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    workgroup_size::Int = 64,
)
    FT = eltype(sums_dev)
    dist_bins    = _gpu_normalize_bins(distance_bins)
    n_dist_edges = _gpu_n_edges(distance_bins)
    n_val_edges  = _gpu_n_edges(value_bins)
    val_plan     = _joint2d_build_val_plan(backend, value_bins)

    value_host      = _gpu_host_edge_vector(value_bins)
    value_edges_dev = KA.allocate(backend, FT, n_val_edges)
    copyto!(value_edges_dev, value_host)

    out_sums_dev = KA.allocate(backend, FT, n_dist, n_val)
    out_cnts_dev = KA.allocate(backend, UInt32, n_dist, n_val)

    # u_dev layout from _stage_batch_device: (B, N, N_dims)
    # pre-transpose to (N_dims, N, B) once so each slice @view(u_t[:, :, b]) is (N_dims, N)
    u_t = permutedims(u_dev, (3, 2, 1))

    for b in 1:B
        fill!(out_sums_dev, zero(FT))
        fill!(out_cnts_dev, zero(UInt32))
        _launch_joint_2d_kernel!(
            backend, workgroup_size,
            out_sums_dev, out_cnts_dev,
            x_dev, @view(u_t[:, :, b]),
            value_edges_dev,
            sf_type, dist_bins, N, n_dist_edges, n_val_edges;
            val_plan = val_plan,
        )
        copyto!(@view(sums_dev[:, :, b]), out_sums_dev)
        copyto!(@view(counts_dev[:, :, b]), out_cnts_dev)
    end
    KA.synchronize(backend)
    return nothing
end
