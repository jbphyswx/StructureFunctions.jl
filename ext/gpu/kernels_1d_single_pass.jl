# Tiled128 six-invariant-type single-pass 1D distance histogram kernels (linear/log/general).
# Included from StructureFunctionsGPUExt.jl — block-local (6, NB) sums + one count row.

"""Accumulate six invariant native 1D SF types into flat `@localmem` sums `(6*NB,)`."""
@inline function _gpu_accumulate_single_pass_1d_shared!(
    shared_sums,
    shared_cnts,
    bin::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    NB::Int,
)
    @atomic shared_sums[bin] += du_L2 + du_T2
    @atomic shared_sums[NB + bin] += du_L2
    @atomic shared_sums[2NB + bin] += du_T2
    @atomic shared_sums[3NB + bin] += du_L * (du_L2 + du_T2)
    @atomic shared_sums[4NB + bin] += du_L * du_L2
    @atomic shared_sums[5NB + bin] += du_L * du_T2
    @atomic shared_cnts[bin] += UInt32(1)
    return nothing
end

KA.@kernel function _sf6_single_pass_kernel_tiled128_linear_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    N_points::Int,
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
    shared_sums = @localmem FT (SF_GPU_SINGLE_PASS_N * SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for k in 1:(SF_GPU_SINGLE_PASS_N * NB)
            shared_sums[k] = zero(FT)
        end
        @inbounds for b in 1:NB
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
                bin = _gpu_digitize_linear(
                    dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                )
                if 1 <= bin < N_bins
                    dU = U2 - U1
                    r̂ = dX / dist
                    n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                    du_L = SA.dot(dU, r̂)
                    du_T = SA.dot(dU, n̂)
                    du_L2 = du_L * du_L
                    du_T2 = du_T * du_T
                    _gpu_accumulate_single_pass_1d_shared!(
                        shared_sums, shared_cnts, bin, du_L, du_T, du_L2, du_T2, NB,
                    )
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
        k = lid
        while k <= SF_GPU_SINGLE_PASS_N * NB
            s = shared_sums[k]
            if s != zero(FT)
                t = (k - 1) ÷ NB + 1
                b = (k - 1) % NB + 1
                @atomic output_sums[t, b] += s
            end
            k += workgroup_size
        end
        b = lid
        while b <= NB
            c = shared_cnts[b]
            if c != UInt32(0)
                for t in 1:SF_GPU_SINGLE_PASS_N
                    @atomic output_counts[t, b] += c
                end
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf6_single_pass_kernel_tiled128_log_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_SINGLE_PASS_N * SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for k in 1:(SF_GPU_SINGLE_PASS_N * NB)
            shared_sums[k] = zero(FT)
        end
        @inbounds for b in 1:NB
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
                bin = _gpu_digitize_log_spaced(dist, first_edge, last_edge, inv_step, offset, step_val, N_bins)
                if 1 <= bin < N_bins
                    dU = U2 - U1
                    r̂ = dX / dist
                    n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                    du_L = SA.dot(dU, r̂)
                    du_T = SA.dot(dU, n̂)
                    du_L2 = du_L * du_L
                    du_T2 = du_T * du_T
                    _gpu_accumulate_single_pass_1d_shared!(
                        shared_sums, shared_cnts, bin, du_L, du_T, du_L2, du_T2, NB,
                    )
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
        k = lid
        while k <= SF_GPU_SINGLE_PASS_N * NB
            s = shared_sums[k]
            if s != zero(FT)
                t = (k - 1) ÷ NB + 1
                b = (k - 1) % NB + 1
                @atomic output_sums[t, b] += s
            end
            k += workgroup_size
        end
        b = lid
        while b <= NB
            c = shared_cnts[b]
            if c != UInt32(0)
                for t in 1:SF_GPU_SINGLE_PASS_N
                    @atomic output_counts[t, b] += c
                end
            end
            b += workgroup_size
        end
    end
end

KA.@kernel function _sf6_single_pass_kernel_tiled128_general_u32!(
    output_sums,
    output_counts,
    x_mat,
    u_mat,
    edge_anchor::FT,
    @Const(bins),
    N_points::Int,
    N_bins::Int,
    NB::Int,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_uj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_SINGLE_PASS_N * SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    if lid == 1
        @inbounds for k in 1:(SF_GPU_SINGLE_PASS_N * NB)
            shared_sums[k] = zero(FT)
        end
        @inbounds for b in 1:NB
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
                bin = _gpu_digitize_general(dist, bins, N_bins)
                if 1 <= bin < N_bins
                    dU = U2 - U1
                    r̂ = dX / dist
                    n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                    du_L = SA.dot(dU, r̂)
                    du_T = SA.dot(dU, n̂)
                    du_L2 = du_L * du_L
                    du_T2 = du_T * du_T
                    _gpu_accumulate_single_pass_1d_shared!(
                        shared_sums, shared_cnts, bin, du_L, du_T, du_L2, du_T2, NB,
                    )
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
        k = lid
        while k <= SF_GPU_SINGLE_PASS_N * NB
            s = shared_sums[k]
            if s != zero(FT)
                t = (k - 1) ÷ NB + 1
                b = (k - 1) % NB + 1
                @atomic output_sums[t, b] += s
            end
            k += workgroup_size
        end
        b = lid
        while b <= NB
            c = shared_cnts[b]
            if c != UInt32(0)
                for t in 1:SF_GPU_SINGLE_PASS_N
                    @atomic output_counts[t, b] += c
                end
            end
            b += workgroup_size
        end
    end
end
