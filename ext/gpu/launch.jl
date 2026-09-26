# Host launch routing for the joint 2D tiled kernel, the HTP-EJ single-pass 2D kernels, and the
# fixed-x batch kernel. Every launch takes device digitizers.

function _launch_joint_2d_tiled_kernel!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    sf_type,
    ddig,
    vdig,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    n_val::Int,
    hist::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, workspace)
    W = SFC._val_int(SFH.coordinate_width(geom))
    kernel! = _sf2d_kernel_tiled128_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, _sf_weights_to_device(backend, weights),
        ddig, vdig, sf_type,
        N_points, n_dist_edges, n_val_edges, n_val, n_dist * n_val,
        sched, n_tile_blocks, ws,
        Val(W), Val(hist), Val(eltype(out_cnts_dev)), geom, second_axis;
        ndrange = ndrange,
    )
    return nothing
end

# Host launch routing for HTP-EJ single-pass 2D kernels.
#
# Entry: _launch_single_pass_2d_strategy! → needs_partition_merge ?
#   _launch_sp2d_onchip!     (pair → out_*, no merge)
#   _launch_sp2d_direct_partitioned! (private partition + merge)

"""
Trailing kernel args after tile launch params: `C, plane, types_per_pass, n_type_passes`, then the
compile-time shared-histogram width as a `Val`. Every launch site splats this, so the width reaches
all of them from one place.

`C` and `plane` are the **padded** extents, because the kernel uses them to bound its zeroing and
flush loops over the shared layout — not the logical cell counts, which are what the histogram
actually contains. `D` follows as a `Val` so the kernel can size its tile staging and build its
coordinate vectors at compile time.
"""
@inline function _sp2d_strategy_kernel_tail_args(config::SP2DAccumulationStrategy, D::Int, geom,
                                                 cnt_eltype::Type, weights, backend)
    wts = _sf_weights_to_device(backend, weights)
    return (
        config.shared_cells,
        config.plane_shared_cells,
        config.types_per_pass,
        config.n_type_passes,
        Val(_sp2d_sharedhist_compile_cells(config)),
        D == 2 ? Val(2) : D == 3 ? Val(3) : error(
            "the SP2D strategy kernels stage D components from `Val{D}` at D ∈ {2,3}; a caller " *
            "must route another width elsewhere rather than reach here (got D=$D)"),
        Val(cnt_eltype),
        wts,
        geom,
    )
end

"""The HTP-EJ pair kernel for the strategy's accumulation mode."""
function _sp2d_pair_kernel(backend::KA.Backend, config::SP2DAccumulationStrategy, ws::Int)
    config.accum_mode === :shared && return _sf6_sp2d_sharedhist_tiled128_u32!(backend, ws)
    config.accum_mode === :typeplane && return _sf6_sp2d_typeplane_tiled128_u32!(backend, ws)
    return _sf6_sp2d_directpartition_tiled128_u32!(backend, ws)
end

function _sp2d_pair_launch_kernel!(
    backend::KA.Backend,
    partition_sums_dev,
    partition_counts_dev,
    x_dev,
    u_dev,
    ddig,
    vplan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, workspace)
    kernel! = _sp2d_pair_kernel(backend, config, ws)
    kernel!(
        partition_sums_dev, partition_counts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        ddig, vplan,
        sched, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config, size(x_dev, 1), geom,
                                        eltype(partition_counts_dev), weights, backend)...;
        ndrange = ndrange,
    )
    return n_tile_blocks
end

function _launch_single_pass_2d_strategy!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    ddig,
    vplan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    if config.needs_partition_merge
        return _launch_sp2d_direct_partitioned!(
            backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
            ddig, vplan, N_points, n_dist_edges, n_val_edges, n_dist, config, geom;
            workspace = workspace, weights = weights,
        )
    end
    return _launch_sp2d_onchip!(
        backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        ddig, vplan, N_points, n_dist_edges, n_val_edges, n_dist, config, geom;
        workspace = workspace, weights = weights,
    )
end

"""On-chip path: pair kernel flushes shared histogram directly to `out_*` (no partition, no merge)."""
function _launch_sp2d_onchip!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    ddig,
    vplan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    _sp2d_pair_launch_kernel!(
        backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        ddig, vplan, N_points, n_dist_edges, n_val_edges, n_dist, config, geom;
        workspace = workspace, weights = weights,
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
    ddig,
    vplan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    config.needs_partition_merge ||
        throw(ArgumentError("_launch_sp2d_direct_partitioned! requires needs_partition_merge"))
    partition_sums, partition_counts, n_tb = _sp2d_partition_pair_bufs_and_launch!(
        backend, out_sums_dev, out_cnts_dev, x_dev, u_dev, ddig, vplan,
        N_points, n_dist_edges, n_val_edges, n_dist, config, geom;
        workspace = workspace, weights = weights,
    )
    _launch_merge_sp2d_partitions!(
        backend, out_sums_dev, out_cnts_dev, partition_sums, partition_counts,
        n_dist, n_val_edges - 1, n_tb, config.merge,
    )
    return nothing
end

"""Allocate/zero private partitions and run the direct pair kernel; returns `(partition_sums, partition_counts, n_tile_blocks)`."""
function _sp2d_partition_pair_bufs_and_launch!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    ddig,
    vplan,
    N_points::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    n_dist::Int,
    config::SP2DAccumulationStrategy,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    config.needs_partition_merge ||
        throw(ArgumentError("_sp2d_partition_pair_bufs_and_launch! requires needs_partition_merge (direct mode)"))
    _, n_tile_blocks, _, _ = _tiled_launch_params(N_points, workspace)
    # One tile block holds at most `SF_GPU_TILE^2` pairs, so an unweighted partition counts in `UInt32`
    # whatever the output count type; the merge widens.
    CST = _sf_count_type(weights, eltype(out_cnts_dev), SF_GPU_TILE^2)
    partition_sums, partition_counts = if workspace === nothing
        _alloc_sp2d_partition_bufs(backend, eltype(out_sums_dev), CST, n_dist, n_val_edges - 1,
                                   n_tile_blocks)
    else
        _ensure_sp2d_partition_bufs!(workspace, n_tile_blocks, CST)
    end
    n_tb = _sp2d_pair_launch_kernel!(
        backend, partition_sums, partition_counts, x_dev, u_dev,
        ddig, vplan, N_points, n_dist_edges, n_val_edges, n_dist, config, geom;
        workspace = workspace, weights = weights,
    )
    return partition_sums, partition_counts, n_tb
end

"""Global-atomic single-pass 2D: one work item per ordered pair, any width, any bins."""
function _launch_single_pass_2d_kernel!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    ddig,
    vplan,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    kernel! = _sf_single_pass_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, ddig, vplan,
        _sf_weights_to_device(backend, weights),
        N_points, Val(N_dims), n_dist_edges, n_val_edges, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

"""Launch a point list's six single-pass invariant joint histograms `(6, n_dist, n_val)`: the native
kernel with its plan for the call ([`SFC.gpu_native_2d_plan`](@ref)), or
[`_launch_single_pass_2d_portable!`](@ref) when there is none."""
function _launch_single_pass_2d!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    ddig,
    vplan,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    n_dist, n_val = n_dist_edges - 1, n_val_edges - 1
    plan = SFC.gpu_native_2d_plan(backend, eltype(x_dev), eltype(u_dev), eltype(out_sums_dev),
                                  eltype(out_cnts_dev), weights, geom, SF_GPU_SINGLE_PASS_N, n_dist, n_val)
    if plan === nothing
        _launch_single_pass_2d_portable!(backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
                                         ddig, vplan, N_points, N_dims, n_dist_edges, n_val_edges, geom;
                                         workspace, weights)
    else
        SFC.gpu_native_launch_2d!(plan, reshape(out_sums_dev, SF_GPU_SINGLE_PASS_N, n_dist, n_val, 1),
                                  reshape(out_cnts_dev, SF_GPU_SINGLE_PASS_N, n_dist, n_val, 1), x_dev, u_dev,
                                  weights, nothing, ddig, vplan, N_points, n_dist, n_val, 1, true, geom,
                                  SFC.InvariantValueAxis(), _active_cull(workspace))
    end
    return nothing
end

"""Launch a point list's six single-pass invariant joint histograms on the portable kernels: the
strategy kernels at `D ∈ {2, 3}` while their histogram mode keeps up, the global-atomic kernel
otherwise."""
function _launch_single_pass_2d_portable!(
    backend::KA.Backend,
    workgroup_size::Int,
    out_sums_dev,
    out_cnts_dev,
    x_dev,
    u_dev,
    ddig,
    vplan,
    N_points::Int,
    N_dims::Int,
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    workspace::Union{GPUSFWorkspace, Nothing} = nothing,
    weights = SFC.NoWeights(),
)
    n_dist = n_dist_edges - 1
    # The strategy kernels stage `D ∈ {2, 3}` components; past `SP2D_GLOBAL_ATOMIC_HIST_BYTES` a
    # `:direct` histogram goes to the global-atomic kernel.
    caps = SFC.gpu_device_caps(backend)
    FT, OT, CST = eltype(x_dev), eltype(out_sums_dev), eltype(out_cnts_dev)
    if (N_dims == 2 || N_dims == 3) && _gpu_single_pass_2d_tiled_eligible(n_dist) &&
       SFC.gpu_static_smem_fits(caps, _sp2d_direct_smem_bytes(FT, N_dims))
        config = _sp2d_accumulation_strategy(caps, n_dist, n_val_edges - 1, N_dims, FT, OT, CST)
        _sp2d_prefers_global_atomics(config, OT, CST) ||
            return _launch_single_pass_2d_strategy!(
                backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
                ddig, vplan, N_points, n_dist_edges, n_val_edges, n_dist, config, geom;
                workspace = workspace, weights = weights,
            )
    end
    return _launch_single_pass_2d_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        ddig, vplan, N_points, N_dims, n_dist_edges, n_val_edges, geom;
        workspace = workspace, weights = weights,
    )
end

# Production batch launch drivers — fixed-x.
#
# Fixed-x launch invariant (do not regress):
# - `ndrange = n_tile_blocks * workgroup_size` only — never multiply by `cld(B, strip_w)`.
# - Host strip loop over `_batch_usmem_strip_w(caps, FT)` via `_batch_fixed_x_usmem_priv!`.
# - `KA.CPU`: serial merge; other backends: grouped merge. Same kernel on both.

function _launch_batch_fixed_x_sf!(
    backend::KA.CPU,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    ddig,
    NB::Int,
    geom,
)
    FT = eltype(sums_dev)
    caps = SFC.gpu_device_caps(backend)
    strip_w = _batch_usmem_strip_w(caps, FT)
    sched, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    n_priv = _batch_usmem_n_priv(n_tile_blocks, ws, caps.warp)
    kernel! = _batch_fixed_x_sf_kernel(backend, ws)
    merge_sums! = _batch_merge_usmem_sums!(backend, ws)
    merge_cnts! = _batch_merge_usmem_cnts!(backend, ws)
    partial_sums = KA.zeros(backend, FT, NB, strip_w, n_priv)
    partial_cnts = KA.zeros(backend, UInt32, NB, n_priv)
    b_base = 1
    while b_base <= B
        bw = min(strip_w, B - b_base + 1)
        fill!(partial_sums, zero(FT))
        fill!(partial_cnts, zero(UInt32))
        kernel!(
            partial_sums, partial_cnts, x_dev, u_dev, sf_type,
            N, NB + 1, NB, b_base, bw, ddig,
            sched, n_tile_blocks, ws, Val(strip_w), Val(caps.warp), geom;
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
    ddig,
    NB::Int,
    geom,
)
    FT = eltype(sums_dev)
    caps = SFC.gpu_device_caps(backend)
    strip_w = _batch_usmem_strip_w(caps, FT)
    sched, n_tile_blocks, ws, ndrange = _batch_tiled_launch_params(N)
    n_priv = _batch_usmem_n_priv(n_tile_blocks, ws, caps.warp)
    kernel! = _batch_fixed_x_sf_kernel(backend, ws)
    merge_sums! = _batch_merge_usmem_sums_grouped!(backend, ws)
    merge_cnts! = _batch_merge_usmem_cnts_grouped!(backend, ws)
    partial_sums = KA.zeros(backend, FT, NB, strip_w, n_priv)
    partial_cnts = KA.zeros(backend, UInt32, NB, n_priv)
    b_base = 1
    while b_base <= B
        bw = min(strip_w, B - b_base + 1)
        fill!(partial_sums, zero(FT))
        fill!(partial_cnts, zero(UInt32))
        kernel!(
            partial_sums, partial_cnts, x_dev, u_dev, sf_type,
            N, NB + 1, NB, b_base, bw, ddig,
            sched, n_tile_blocks, ws, Val(strip_w), Val(caps.warp), geom;
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
