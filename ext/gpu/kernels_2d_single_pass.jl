# Tiled128 six-invariant-type single-pass 2D joint histogram kernels (global-atomic path).
# Production fast path when eligible is HTP-EJ in kernels_2d_direct.jl (see gpu/SP2D_HTP_EJ.md).
# This file: tile schedule matches 1D tiled128; non-HTP-EJ route uses global atomics.

@inline function _gpu_accumulate_single_pass_2d_pair_global!(
    output_sums,
    output_counts,
    value_edges,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_general_col(vals[t], value_edges, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

@inline function _gpu_accumulate_single_pass_2d_pair_global_linear_val!(
    output_sums,
    output_counts,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    N_val_edges::Int,
    val_first::FT,
    val_last::FT,
    val_inv_step::FT,
    val_offset::FT,
    val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

@inline function _gpu_accumulate_single_pass_2d_pair_global_linear_val_cols!(
    output_sums,
    output_counts,
    val_first,
    val_last,
    val_inv_step,
    val_offset,
    val_step,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    N_val_edges::Int,
)
    FT = eltype(output_sums)
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

@inline function _gpu_accumulate_single_pass_2d_pair_global_inflinear_val!(
    output_sums,
    output_counts,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    N_val_edges::Int,
    val_first::FT,
    val_last::FT,
    val_inv_step::FT,
    val_offset::FT,
    val_step::FT,
    n_inner_edges::Int,
    inner_last::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step,
            n_inner_edges, inner_last,
        )
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

@inline function _gpu_accumulate_single_pass_2d_pair_global_inflinear_val_cols!(
    output_sums,
    output_counts,
    val_first,
    val_last,
    val_inv_step,
    val_offset,
    val_step,
    inner_last,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    n_inner_edges::Int,
    N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            n_inner_edges, inner_last[t],
        )
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

@inline function _gpu_accumulate_single_pass_2d_pair_global_log_val!(
    output_sums,
    output_counts,
    val_first::FT,
    val_last::FT,
    val_inv_step::FT,
    val_offset::FT,
    val_step::FT,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    N_val_edges::Int,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced(vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges)
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

@inline function _gpu_accumulate_single_pass_2d_pair_global_log_val_cols!(
    output_sums,
    output_counts,
    val_first,
    val_last,
    val_inv_step,
    val_offset,
    val_step,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2,
        du_L2,
        du_T2,
        du_L * (du_L2 + du_T2),
        du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced_col(vals[t], val_first, val_last, val_inv_step, val_offset, val_step, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            @atomic output_sums[t, bin, vbin] += vals[t]
            @atomic output_counts[t, bin, vbin] += one(eltype(output_counts))
        end
    end
    return nothing
end

KA.@kernel function _sf6_single_pass_2d_kernel_tiled128_linear_linear_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    N_val_edges::Int,
    dist_first::FT,
    dist_last::FT,
    dist_inv_step::FT,
    dist_offset::FT,
    dist_step::FT,
    val_first::FT,
    val_last::FT,
    val_inv_step::FT,
    val_offset::FT,
    val_step::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SF_GPU_TILE + k] = x_mat[2, gi]
                    shared_ui[k] = u_mat[1, gi]
                    shared_ui[SF_GPU_TILE + k] = u_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SF_GPU_TILE + k] = x_mat[2, gj]
                        shared_uj[k] = u_mat[1, gj]
                        shared_uj[SF_GPU_TILE + k] = u_mat[2, gj]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
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
                dX = X2 - X1
                dist = sqrt(dX[1]^2 + dX[2]^2)
                bin = _gpu_digitize_linear(
                    dist, dist_first, dist_last, dist_inv_step, dist_offset, dist_step, N_bins,
                )
                if 1 <= bin < N_bins
                    dU = U2 - U1
                    r̂ = dX / dist
                    n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                    du_L = SA.dot(dU, r̂)
                    du_T = SA.dot(dU, n̂)
                    du_L2 = du_L * du_L
                    du_T2 = du_T * du_T
                    _gpu_accumulate_single_pass_2d_pair_global_linear_val!(
                        output_sums, output_counts, bin,
                        du_L, du_T, du_L2, du_T2, N_val_edges,
                        val_first, val_last, val_inv_step, val_offset, val_step,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel function _sf6_single_pass_2d_kernel_tiled128_log_general_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    @Const(value_edges),
    N_points::Int,
    N_bins::Int,
    NB::Int,
    N_val_edges::Int,
    dist_first::FT,
    dist_last::FT,
    dist_inv_step::FT,
    dist_offset::FT,
    dist_step::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SF_GPU_TILE + k] = x_mat[2, gi]
                    shared_ui[k] = u_mat[1, gi]
                    shared_ui[SF_GPU_TILE + k] = u_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SF_GPU_TILE + k] = x_mat[2, gj]
                        shared_uj[k] = u_mat[1, gj]
                        shared_uj[SF_GPU_TILE + k] = u_mat[2, gj]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
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
                dX = X2 - X1
                dist = sqrt(dX[1]^2 + dX[2]^2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_offset, dist_step, N_bins)
                if 1 <= bin < N_bins
                    dU = U2 - U1
                    r̂ = dX / dist
                    n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                    du_L = SA.dot(dU, r̂)
                    du_T = SA.dot(dU, n̂)
                    du_L2 = du_L * du_L
                    du_T2 = du_T * du_T
                    _gpu_accumulate_single_pass_2d_pair_global!(
                        output_sums, output_counts, value_edges, bin,
                        du_L, du_T, du_L2, du_T2, N_val_edges,
                    )
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel function _sf6_single_pass_2d_kernel_tiled128_log_linear_val_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    N_val_edges::Int,
    dist_first::FT,
    dist_last::FT,
    dist_inv_step::FT,
    dist_offset::FT,
    dist_step::FT,
    val_first::FT,
    val_last::FT,
    val_inv_step::FT,
    val_offset::FT,
    val_step::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SF_GPU_TILE + k] = x_mat[2, gi]
                    shared_ui[k] = u_mat[1, gi]
                    shared_ui[SF_GPU_TILE + k] = u_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SF_GPU_TILE + k] = x_mat[2, gj]
                        shared_uj[k] = u_mat[1, gj]
                        shared_uj[SF_GPU_TILE + k] = u_mat[2, gj]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
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
                dX = X2 - X1
                dist = sqrt(dX[1]^2 + dX[2]^2)
                bin = _gpu_digitize_log_spaced(dist, dist_first, dist_last, dist_inv_step, dist_offset, dist_step, N_bins)
                if 1 <= bin < N_bins
                    dU = U2 - U1
                    r̂ = dX / dist
                    n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                    du_L = SA.dot(dU, r̂)
                    du_T = SA.dot(dU, n̂)
                    du_L2 = du_L * du_L
                    du_T2 = du_T * du_T
                    _gpu_accumulate_single_pass_2d_pair_global_linear_val!(
                        output_sums, output_counts, bin,
                        du_L, du_T, du_L2, du_T2, N_val_edges,
                        val_first, val_last, val_inv_step, val_offset, val_step,
                    )
                end
                p += workgroup_size
            end
        end
    end
end
