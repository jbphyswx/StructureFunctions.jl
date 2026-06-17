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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_linear_linear_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_linear_linear_val_cols_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_linear_inflinear_val_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_linear_inflinear_val_cols_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_log_linear_val_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_log_linear_val_cols_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_log_inflinear_val_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_log_inflinear_val_cols_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_log_log_val_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_log_vector_val_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_linear_log_val_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_linear_log_val_cols_u32!(backend, ws)
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
    kernel! = _sf8_single_pass_2d_kernel_tiled128_log_log_val_cols_u32!(backend, ws)
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
        N_points, n_dist_edges, n_val_edges,
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
        N_points, n_dist_edges, n_val_edges,
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
        bins_dev, val_plan.edges_dev, N_points, n_dist_edges, n_val_edges;
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
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
)
    if _gpu_single_pass_2d_use_tiled(dist_bins, val_plan, n_dist_edges - 1)
        return _launch_single_pass_2d_tiled!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist_edges - 1;
            workspace = workspace,
        )
    end
    throw(ArgumentError(
        "naive single-pass 2D fallback unsupported for (dist=$(typeof(dist_bins)), value=$(typeof(val_plan)))",
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
    n_dist_edges::Int,
    n_val_edges::Int;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    force_legacy::Bool = false,
)
    n_dist = n_dist_edges - 1
    if !force_legacy && _gpu_single_pass_2d_use_tiled(dist_bins, val_plan, n_dist)
        config = workspace === nothing ?
            _sp2d_priv_config(n_dist, n_val_edges - 1, eltype(out_sums_dev)) :
            workspace.sp2d_priv_config
        return _launch_single_pass_2d_priv!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
            workspace = workspace,
        )
    end
    if _gpu_single_pass_2d_use_tiled(dist_bins, val_plan, n_dist)
        return _launch_single_pass_2d_tiled!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist;
            workspace = workspace,
        )
    end
    return _launch_single_pass_2d_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        dist_bins, val_plan, N_points, n_dist_edges, n_val_edges;
        workspace = workspace,
    )
end
