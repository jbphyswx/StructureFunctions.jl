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
    cull = nothing,
    weights = SFC.NoWeights(),
    second_axis = SFC.InvariantValueAxis(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, cull)
    kernel! = _sf2d_kernel_tiled128_u32!(backend, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, _sf_weights_to_device(backend, weights),
        ddig, vdig, sf_type,
        N_points, n_dist_edges, n_val_edges, n_val, n_dist * n_val,
        sched, n_tile_blocks, ws,
        SFH.coordinate_width(geom), SFH.field_width(geom), Val(hist), Val(eltype(out_cnts_dev)), geom,
        second_axis;
        ndrange = ndrange,
    )
    return nothing
end

"""
Trailing kernel args after tile launch params: `C, plane, types_per_pass, n_type_passes`, then the
compile-time shared-histogram width as a `Val`. Every launch site splats this, so the width reaches
all of them from one place.

`C` and `plane` are the **padded** extents, because the kernel uses them to bound its zeroing and
flush loops over the shared layout — not the logical cell counts, which are what the histogram
actually contains. The geometry's coordinate and field widths follow as `Val`s.
"""
@inline function _sp2d_strategy_kernel_tail_args(config::SP2DAccumulationStrategy, geom,
                                                 cnt_eltype::Type, weights, backend)
    wts = _sf_weights_to_device(backend, weights)
    return (
        config.shared_cells,
        config.plane_shared_cells,
        config.types_per_pass,
        config.n_type_passes,
        Val(_sp2d_sharedhist_compile_cells(config)),
        SFH.coordinate_width(geom),
        SFH.field_width(geom),
        Val(cnt_eltype),
        wts,
        geom,
    )
end

"""The HTP-EJ pair kernel for the strategy's accumulation mode."""
_sp2d_pair_kernel(backend::KA.Backend, config::SP2DAccumulationStrategy, ws::Int) =
    config.accum_mode === :shared ? _sf6_sp2d_sharedhist_tiled128_u32!(backend, ws) :
                                    _sf6_sp2d_typeplane_tiled128_u32!(backend, ws)

"""Launch the HTP-EJ pair kernel of the on-chip strategy `config`, which flushes its shared histogram into
`out_sums_dev`/`out_cnts_dev` `(6, n_dist, n_val)`."""
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
    cull = nothing,
    weights = SFC.NoWeights(),
)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N_points, cull)
    kernel! = _sp2d_pair_kernel(backend, config, ws)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev,
        N_points, n_dist_edges, n_dist, n_val_edges,
        ddig, vplan,
        sched, n_tile_blocks, ws,
        _sp2d_strategy_kernel_tail_args(config, geom, eltype(out_cnts_dev), weights, backend)...;
        ndrange = ndrange,
    )
    return nothing
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
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    weights = SFC.NoWeights(),
)
    kernel! = _sf_single_pass_2d_kernel!(backend, workgroup_size)
    kernel!(
        out_sums_dev, out_cnts_dev, x_dev, u_dev, ddig, vplan,
        _sf_weights_to_device(backend, weights),
        N_points, n_dist_edges, n_val_edges, geom;
        ndrange = (N_points, N_points),
    )
    return nothing
end

"""Launch a point list's six single-pass invariant joint histograms `(6, n_dist, n_val)`: the backend's native plan
for the call ([`SFC.gpu_native_2d_plan`](@ref)), handed the portable launch as a candidate, or
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
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    cull = nothing,
    weights = SFC.NoWeights(),
)
    n_dist, n_val = n_dist_edges - 1, n_val_edges - 1
    plan = SFC.gpu_native_2d_plan(backend, eltype(x_dev), eltype(u_dev), eltype(out_sums_dev),
                                  eltype(out_cnts_dev), weights, geom, SFT.SinglePassInvariants(), n_dist, n_val,
                                  vplan)
    portable!(o, c) = _launch_single_pass_2d_portable!(backend, workgroup_size,
                                                       reshape(o, SINGLE_PASS_N, n_dist, n_val),
                                                       reshape(c, SINGLE_PASS_N, n_dist, n_val), x_dev, u_dev, ddig,
                                                       vplan, N_points, n_dist_edges, n_val_edges, geom; cull, weights)
    if plan === nothing
        portable!(out_sums_dev, out_cnts_dev)
    else
        SFC.gpu_native_launch_2d!(plan, reshape(out_sums_dev, SINGLE_PASS_N, n_dist, n_val, 1),
                                  reshape(out_cnts_dev, SINGLE_PASS_N, n_dist, n_val, 1), x_dev, u_dev,
                                  weights, SFT.SinglePassInvariants(), ddig, vplan, N_points, n_dist, n_val, 1, true,
                                  geom, SFC.InvariantValueAxis(), cull, portable!)
    end
    return nothing
end

"""Launch a point list's six single-pass invariant joint histograms on the portable kernels: the
on-chip strategy kernel while one of its histograms fits, the global-atomic kernel otherwise."""
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
    n_dist_edges::Int,
    n_val_edges::Int,
    geom;
    cull = nothing,
    weights = SFC.NoWeights(),
)
    n_dist = n_dist_edges - 1
    caps = SFC.gpu_device_caps(backend)
    W, F = SFC._val_int(SFH.coordinate_width(geom)), SFC._val_int(SFH.field_width(geom))
    FT, OT, CST = eltype(x_dev), eltype(out_sums_dev), eltype(out_cnts_dev)
    if _gpu_single_pass_2d_tiled_eligible(n_dist)
        config = _sp2d_accumulation_strategy(caps, n_dist, n_val_edges - 1, W, F, FT, OT, CST)
        config === nothing ||
            return _launch_single_pass_2d_strategy!(
                backend, out_sums_dev, out_cnts_dev, x_dev, u_dev,
                ddig, vplan, N_points, n_dist_edges, n_val_edges, n_dist, config, geom;
                cull = cull, weights = weights,
            )
    end
    return _launch_single_pass_2d_kernel!(
        backend, workgroup_size, out_sums_dev, out_cnts_dev, x_dev, u_dev,
        ddig, vplan, N_points, n_dist_edges, n_val_edges, geom;
        weights = weights,
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
    geom;
    cull = nothing,
)
    FT = eltype(sums_dev)
    caps = SFC.gpu_device_caps(backend)
    strip_w = _batch_usmem_strip_w(caps, FT)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N, cull)
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
            merge_cnts!(counts_dev, partial_cnts, NB, n_priv, NB; ndrange = NB)
        end
        b_base += bw
    end
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
    geom;
    cull = nothing,
)
    FT = eltype(sums_dev)
    caps = SFC.gpu_device_caps(backend)
    strip_w = _batch_usmem_strip_w(caps, FT)
    sched, n_tile_blocks, ws, ndrange = _tiled_launch_params(N, cull)
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
            merge_cnts!(counts_dev, partial_cnts, NB, n_priv, ws; ndrange = NB * ws)
        end
        b_base += bw
    end
    return nothing
end
