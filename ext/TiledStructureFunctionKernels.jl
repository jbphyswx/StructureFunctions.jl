# Six production tiled128 SF histogram kernels (2D/3D × linear/log/general).
# Included from StructureFunctionsGPUExt.jl — no macro codegen.
#
# KA CPU lowering splits the body at each @synchronize; index variables and tile
# metadata must be re-established in every segment (see ext module docstring).

# Shared layout: SF_GPU_TILE=128 → 2D shared coords length 256, 3D length 384.

KA.@kernel function _sf_kernel_tiled128_2d_linear_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
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
    shared_sums = @localmem FT (SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB
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
                dist_sq = dX[1]^2 + dX[2]^2
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_linear(
                    dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                )
                if 1 <= bin < N_bins
                    r̂ = dX / dist
                    val = sf_type(U2 - U1, r̂)
                    @atomic shared_sums[bin] += val
                    @atomic shared_cnts[bin] += UInt32(1)
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
        while b <= NB
            @atomic output[b] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic counts[b] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf_kernel_tiled128_3d_linear_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (384,)
    shared_ui = @localmem FT (384,)
    shared_xj = @localmem FT (384,)
    shared_uj = @localmem FT (384,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB
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
                    shared_xi[2 * SF_GPU_TILE + k] = x_mat[3, gi]
                    shared_ui[k] = u_mat[1, gi]
                    shared_ui[SF_GPU_TILE + k] = u_mat[2, gi]
                    shared_ui[2 * SF_GPU_TILE + k] = u_mat[3, gi]
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
                        shared_xj[2 * SF_GPU_TILE + k] = x_mat[3, gj]
                        shared_uj[k] = u_mat[1, gj]
                        shared_uj[SF_GPU_TILE + k] = u_mat[2, gj]
                        shared_uj[2 * SF_GPU_TILE + k] = u_mat[3, gj]
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
                    X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia], shared_xi[2 * SF_GPU_TILE + ia])
                    X2 = SA.SVector{3, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb], shared_xj[2 * SF_GPU_TILE + jb])
                    U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia], shared_ui[2 * SF_GPU_TILE + ia])
                    U2 = SA.SVector{3, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb], shared_uj[2 * SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia], shared_xi[2 * SF_GPU_TILE + ia])
                    X2 = SA.SVector{3, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb], shared_xi[2 * SF_GPU_TILE + jb])
                    U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia], shared_ui[2 * SF_GPU_TILE + ia])
                    U2 = SA.SVector{3, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb], shared_ui[2 * SF_GPU_TILE + jb])
                end
                dX = X2 - X1
                dist_sq = dX[1]^2 + dX[2]^2 + dX[3]^2
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_linear(
                    dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                )
                if 1 <= bin < N_bins
                    r̂ = SFH.r̂(X1, X2)
                    val = sf_type(U2 - U1, r̂)
                    @atomic shared_sums[bin] += val
                    @atomic shared_cnts[bin] += UInt32(1)
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
        while b <= NB
            @atomic output[b] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic counts[b] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf_kernel_tiled128_2d_log_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    edge_anchor::FT,
    @Const(edges),
    @Const(lut),
    e_min::Int,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB
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
                dist_sq = dX[1]^2 + dX[2]^2
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_log(dist, edges, lut, e_min, N_bins)
                if 1 <= bin < N_bins
                    r̂ = dX / dist
                    val = sf_type(U2 - U1, r̂)
                    @atomic shared_sums[bin] += val
                    @atomic shared_cnts[bin] += UInt32(1)
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
        while b <= NB
            @atomic output[b] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic counts[b] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf_kernel_tiled128_3d_log_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    edge_anchor::FT,
    @Const(edges),
    @Const(lut),
    e_min::Int,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (384,)
    shared_ui = @localmem FT (384,)
    shared_xj = @localmem FT (384,)
    shared_uj = @localmem FT (384,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB
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
                    shared_xi[2 * SF_GPU_TILE + k] = x_mat[3, gi]
                    shared_ui[k] = u_mat[1, gi]
                    shared_ui[SF_GPU_TILE + k] = u_mat[2, gi]
                    shared_ui[2 * SF_GPU_TILE + k] = u_mat[3, gi]
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
                        shared_xj[2 * SF_GPU_TILE + k] = x_mat[3, gj]
                        shared_uj[k] = u_mat[1, gj]
                        shared_uj[SF_GPU_TILE + k] = u_mat[2, gj]
                        shared_uj[2 * SF_GPU_TILE + k] = u_mat[3, gj]
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
                    X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia], shared_xi[2 * SF_GPU_TILE + ia])
                    X2 = SA.SVector{3, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb], shared_xj[2 * SF_GPU_TILE + jb])
                    U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia], shared_ui[2 * SF_GPU_TILE + ia])
                    U2 = SA.SVector{3, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb], shared_uj[2 * SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia], shared_xi[2 * SF_GPU_TILE + ia])
                    X2 = SA.SVector{3, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb], shared_xi[2 * SF_GPU_TILE + jb])
                    U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia], shared_ui[2 * SF_GPU_TILE + ia])
                    U2 = SA.SVector{3, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb], shared_ui[2 * SF_GPU_TILE + jb])
                end
                dX = X2 - X1
                dist_sq = dX[1]^2 + dX[2]^2 + dX[3]^2
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_log(dist, edges, lut, e_min, N_bins)
                if 1 <= bin < N_bins
                    r̂ = SFH.r̂(X1, X2)
                    val = sf_type(U2 - U1, r̂)
                    @atomic shared_sums[bin] += val
                    @atomic shared_cnts[bin] += UInt32(1)
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
        while b <= NB
            @atomic output[b] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic counts[b] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf_kernel_tiled128_2d_general_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    edge_anchor::FT,
    @Const(distance_bins),
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB
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
                dist_sq = dX[1]^2 + dX[2]^2
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_general(dist, distance_bins, N_bins)
                if 1 <= bin < N_bins
                    r̂ = dX / dist
                    val = sf_type(U2 - U1, r̂)
                    @atomic shared_sums[bin] += val
                    @atomic shared_cnts[bin] += UInt32(1)
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
        while b <= NB
            @atomic output[b] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic counts[b] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf_kernel_tiled128_3d_general_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    edge_anchor::FT,
    @Const(distance_bins),
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (384,)
    shared_ui = @localmem FT (384,)
    shared_xj = @localmem FT (384,)
    shared_uj = @localmem FT (384,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for b in 1:NB
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
                    shared_xi[2 * SF_GPU_TILE + k] = x_mat[3, gi]
                    shared_ui[k] = u_mat[1, gi]
                    shared_ui[SF_GPU_TILE + k] = u_mat[2, gi]
                    shared_ui[2 * SF_GPU_TILE + k] = u_mat[3, gi]
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
                        shared_xj[2 * SF_GPU_TILE + k] = x_mat[3, gj]
                        shared_uj[k] = u_mat[1, gj]
                        shared_uj[SF_GPU_TILE + k] = u_mat[2, gj]
                        shared_uj[2 * SF_GPU_TILE + k] = u_mat[3, gj]
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
                    X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia], shared_xi[2 * SF_GPU_TILE + ia])
                    X2 = SA.SVector{3, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb], shared_xj[2 * SF_GPU_TILE + jb])
                    U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia], shared_ui[2 * SF_GPU_TILE + ia])
                    U2 = SA.SVector{3, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb], shared_uj[2 * SF_GPU_TILE + jb])
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia], shared_xi[2 * SF_GPU_TILE + ia])
                    X2 = SA.SVector{3, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb], shared_xi[2 * SF_GPU_TILE + jb])
                    U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia], shared_ui[2 * SF_GPU_TILE + ia])
                    U2 = SA.SVector{3, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb], shared_ui[2 * SF_GPU_TILE + jb])
                end
                dX = X2 - X1
                dist_sq = dX[1]^2 + dX[2]^2 + dX[3]^2
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_general(dist, distance_bins, N_bins)
                if 1 <= bin < N_bins
                    r̂ = SFH.r̂(X1, X2)
                    val = sf_type(U2 - U1, r̂)
                    @atomic shared_sums[bin] += val
                    @atomic shared_cnts[bin] += UInt32(1)
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
        while b <= NB
            @atomic output[b] += shared_sums[b]
            if shared_cnts[b] != UInt32(0)
                @atomic counts[b] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end
