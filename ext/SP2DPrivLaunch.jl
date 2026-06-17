# Host launch routing for HTP-EJ privatized single-pass 2D kernels.

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

function _sp2d_dist_variant(::LinearBinEdges)
    return :linear
end
function _sp2d_dist_variant(::LogBinEdges)
    return :log_linear
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:linear, :linear_shared, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:linear, :linear_cols, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:linear, :inflinear_shared, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:linear, :inflinear_cols, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:linear, :log_linear_shared, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:linear, :log_linear_cols, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:log_linear, :linear_shared, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:log_linear, :linear_cols, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:log_linear, :inflinear_shared, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        vp.n_inner_edges, vp.inner_last,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:log_linear, :inflinear_cols, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step, vp.inner_last,
        vp.n_inner_edges,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:log_linear, :log_linear_shared, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:log_linear, :log_linear_cols, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.first, vp.last, vp.inv_step, vp.offset, vp.step,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    bucket = config.strip_bucket
    kernel! = _sp2d_priv_kernel_fn(:log_linear, :vector_cols, bucket, backend, ws)
    lbe, vp = dist_bins, val_plan
    d_f, d_l, d_inv, d_off, d_st = _dist_log_linear_fields(lbe)
    kernel!(
        priv_sums_dev, priv_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        d_f, d_l, d_inv, d_off, d_st,
        vp.edges_dev,
        n_tiles, n_tile_blocks, ws,
        config.n_joint_cells, config.cells_per_strip, config.n_strips;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _sp2d_priv_launch_kernel!(
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
    n_tb = _sp2d_priv_launch_kernel!(
        backend, priv_sums, priv_cnts, x_dev, u_dev,
        dist_bins, val_plan, N_points, n_dist_edges, n_val_edges, n_dist, config;
        workspace = workspace,
    )
    _launch_merge_sp2d_priv!(
        backend, out_sums_dev, out_cnts_dev, priv_sums, priv_cnts,
        n_dist, n_val_edges - 1, n_tb,
    )
    return nothing
end
