# Tiled128 joint 2D SF histogram kernels (distance × value, linear/log/general distance).
# Included from StructureFunctionsGPUExt.jl — block-local flat histogram in
# `@localmem` (max `SF_GPU_MAX_2D_HIST` cells), same tile schedule as 1D tiled128.

KA.@kernel function _sf2d_kernel_tiled128_linear_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_edges::Int,
    N_val_edges::Int,
    NV::Int,
    NB2::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_MAX_2D_HIST,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_2D_HIST,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB2
            shared_sums[b] = zero(FT)
            shared_cnts[b] = UInt32(0)
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
                dbin = _gpu_digitize_linear(
                    dist, first_edge, last_edge, inv_step, offset, step_val, N_dist_edges,
                )
                if 1 <= dbin < N_dist_edges
                    r̂ = dX / dist
                    val = sf_type(U2 - U1, r̂)
                    vbin = _gpu_digitize_general(val, value_edges, N_val_edges)
                    if 1 <= vbin < N_val_edges
                        idx = (dbin - 1) * NV + vbin
                        @atomic shared_sums[idx] += val
                        @atomic shared_cnts[idx] += UInt32(1)
                    end
                end
                p += workgroup_size
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        b = lid
        while b <= NB2
            dbin = (b - 1) ÷ NV + 1
            vbin = b - (dbin - 1) * NV
            @atomic output_sums[dbin, vbin] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic output_counts[dbin, vbin] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf2d_kernel_tiled128_log_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    @Const(edges),
    @Const(lut),
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_edges::Int,
    N_val_edges::Int,
    NV::Int,
    NB2::Int,
    edge_anchor::FT,
    e_min::Int,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_MAX_2D_HIST,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_2D_HIST,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB2
            shared_sums[b] = zero(FT)
            shared_cnts[b] = UInt32(0)
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
                dbin = _gpu_digitize_log(dist, edges, lut, e_min, N_dist_edges)
                if 1 <= dbin < N_dist_edges
                    r̂ = dX / dist
                    val = sf_type(U2 - U1, r̂)
                    vbin = _gpu_digitize_general(val, value_edges, N_val_edges)
                    if 1 <= vbin < N_val_edges
                        idx = (dbin - 1) * NV + vbin
                        @atomic shared_sums[idx] += val
                        @atomic shared_cnts[idx] += UInt32(1)
                    end
                end
                p += workgroup_size
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        b = lid
        while b <= NB2
            dbin = (b - 1) ÷ NV + 1
            vbin = b - (dbin - 1) * NV
            @atomic output_sums[dbin, vbin] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic output_counts[dbin, vbin] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf2d_kernel_tiled128_general_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    @Const(distance_edges),
    @Const(value_edges),
    sf_type,
    N_points::Int,
    N_dist_edges::Int,
    N_val_edges::Int,
    NV::Int,
    NB2::Int,
    edge_anchor::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_MAX_2D_HIST,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_2D_HIST,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB2
            shared_sums[b] = zero(FT)
            shared_cnts[b] = UInt32(0)
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
                dbin = _gpu_digitize_general(dist, distance_edges, N_dist_edges)
                if 1 <= dbin < N_dist_edges
                    r̂ = dX / dist
                    val = sf_type(U2 - U1, r̂)
                    vbin = _gpu_digitize_general(val, value_edges, N_val_edges)
                    if 1 <= vbin < N_val_edges
                        idx = (dbin - 1) * NV + vbin
                        @atomic shared_sums[idx] += val
                        @atomic shared_cnts[idx] += UInt32(1)
                    end
                end
                p += workgroup_size
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        b = lid
        while b <= NB2
            dbin = (b - 1) ÷ NV + 1
            vbin = b - (dbin - 1) * NV
            @atomic output_sums[dbin, vbin] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic output_counts[dbin, vbin] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end
