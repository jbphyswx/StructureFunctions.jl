# Additional tiled128 single-pass 2D kernels for typed value-axis digitize plans.
# Pair loop is inlined in each kernel (KA @index cannot live inside a user macro).

"""
Stage tile `ti` (and `tj` when off-diagonal) into shared memory, component-major as
`(d-1)*SF_GPU_TILE + k`. `D` defaults to 2 so the two-dimensional kernels here call it unchanged.
"""
function _sp2d_tiled_load_tile!(
    shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat,
    ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, ::Val{D} = Val(2),
) where {D}
    if ni > 0 && nj > 0
        k = lid
        while k <= ni
            gi = i0 + k - 1
            @inbounds for d in 1:D
                shared_xi[(d - 1) * SF_GPU_TILE + k] = x_mat[d, gi]
                shared_ui[(d - 1) * SF_GPU_TILE + k] = u_mat[d, gi]
            end
            k += workgroup_size
        end
        if ti < tj
            k = lid
            while k <= nj
                gj = j0 + k - 1
                @inbounds for d in 1:D
                    shared_xj[(d - 1) * SF_GPU_TILE + k] = x_mat[d, gj]
                    shared_uj[(d - 1) * SF_GPU_TILE + k] = u_mat[d, gj]
                end
                k += workgroup_size
            end
        end
    end
    return nothing
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_linear_linear_val_cols_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first, val_last, val_inv_step, val_step,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_linear(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_linear_val_cols!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step,
                        bin, du_L, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_log_linear_val_cols_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first, val_last, val_inv_step, val_step,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_linear_val_cols!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step,
                        bin, du_L, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_linear_inflinear_val_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_linear(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_inflinear_val!(
                        output_sums, output_counts, bin, du_L, du_L2, du_T2, N_val_edges,
                        val_first, val_last, val_inv_step, val_step, n_inner_edges, inner_last,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_log_inflinear_val_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_inflinear_val!(
                        output_sums, output_counts, bin, du_L, du_L2, du_T2, N_val_edges,
                        val_first, val_last, val_inv_step, val_step, n_inner_edges, inner_last,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_linear_inflinear_val_cols_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first, val_last, val_inv_step, val_step, inner_last,
    n_inner_edges::Int,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_linear(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_inflinear_val_cols!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step, inner_last,
                        bin, du_L, du_L2, du_T2, n_inner_edges, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_log_inflinear_val_cols_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first, val_last, val_inv_step, val_step, inner_last,
    n_inner_edges::Int,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_inflinear_val_cols!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step, inner_last,
                        bin, du_L, du_L2, du_T2, n_inner_edges, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_log_log_val_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_log_val!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step, bin,
                        du_L, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_log_vector_val_u32!(
    output_sums, output_counts, x_mat, u_mat,
    @Const(value_edges),
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global!(
                        output_sums, output_counts, value_edges, bin,
                        du_L, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_linear_log_val_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_linear(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_log_val!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step, bin,
                        du_L, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_linear_log_val_cols_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first, val_last, val_inv_step, val_step,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_linear(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_log_val_cols!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step,
                        bin, du_L, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_2d_kernel_tiled128_log_log_val_cols_u32!(
    output_sums, output_counts, x_mat, u_mat,
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    dist_first::FT, dist_last::FT, dist_inv_step::FT, dist_step::FT,
    val_first, val_last, val_inv_step, val_step,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    geom,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat, ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size)
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    du_L2 = du_L * du_L
                    du_T2 = du_n2 - du_L2
                    _gpu_accumulate_single_pass_2d_pair_global_log_val_cols!(
                        output_sums, output_counts, val_first, val_last, val_inv_step, val_step,
                        bin, du_L, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end
