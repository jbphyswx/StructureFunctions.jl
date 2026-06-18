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
