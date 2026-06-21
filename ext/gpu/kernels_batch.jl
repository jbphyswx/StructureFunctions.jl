# Production batch tiled128 kernels — fixed-x u-smem strips + block-private merge.
# Included from StructureFunctionsGPUExt.jl after GPUBatchWorkspace.jl.

"""u-smem strip width for individual 1D batch (fits 48 KiB with xi/xj + ui/uj staging)."""
const BATCH_USMEM_STRIP_W = 16

"""
SP1D batch strip width — smaller than `BATCH_USMEM_STRIP_W` because on-chip hist is
`(6, NB, strip_w)` not `(NB, strip_w)`.
"""
const BATCH_SP1D_USMEM_STRIP_W = 8

function _batch_tiled_launch_params(N_points::Int)
    n_tiles = cld(N_points, SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SF_GPU_TILED_WS
    return n_tiles, n_tile_blocks, ws, n_tile_blocks * ws
end

@inline function _batch_usmem_idx(c::Int, k::Int, col::Int)
    return c + 2 * (k - 1) + SF_GPU_TILE * 2 * (col - 1)
end

@inline function _pair_bin_rhat_from_smem!(
    shared_xi,
    shared_xj,
    ia::Int,
    jb::Int,
    off_diag::Bool,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    N_bins::Int,
    ::Type{FT},
) where {FT}
    if off_diag
        X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
        X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
    else
        X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
        X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
    end
    dX = X2 - X1
    dist_sq = dX[1]^2 + dX[2]^2
    dist = sqrt(dist_sq)
    bin = _gpu_digitize_linear(dist, first_edge, last_edge, inv_step, offset, step_val, N_bins)
    if 1 <= bin < N_bins
        return (true, bin, dX / dist)
    end
    return (false, bin, SA.SVector{2, FT}(zero(FT), zero(FT)))
end

"""Stage host batch inputs. Fixed-x: `u` → `(B, N, N_dims)` batch-major."""
function _stage_batch_device(backend::KA.Backend, x::AbstractArray{FT}, u::AbstractArray{FT}; fixed_x::Bool) where {FT}
    if fixed_x
        ndims(x) == 2 || throw(ArgumentError("fixed-x batch expects matrix x"))
        ndims(u) >= 3 || throw(ArgumentError("fixed-x batch expects trailing batch dims on u"))
        size(x)[1:2] == size(u)[1:2] || throw(ArgumentError("x and u leading dims must match"))
        B = SFC.batch_size(u)
        u_flat = reshape(u, size(u, 1), size(u, 2), B)
        u_batchmajor = permutedims(u_flat, (3, 2, 1))
        return KA.adapt(backend, x), KA.adapt(backend, u_batchmajor)
    else
        ndims(x) >= 3 || throw(ArgumentError("varying-x batch expects ndims >= 3"))
        size(x) == size(u) || throw(ArgumentError("varying-x requires x and u same shape"))
        B = SFC.batch_size(u)
        x_flat = reshape(x, size(x, 1), size(x, 2), B)
        u_flat = reshape(u, size(u, 1), size(u, 2), B)
        return KA.adapt(backend, x_flat), KA.adapt(backend, u_flat)
    end
end

KA.@kernel function _batch_merge_usmem_sums!(
    output,
    @Const(partial_sums),
    NB::Int,
    bw::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    n_out = NB * bw
    t = worker
    while t <= n_out
        rem0 = t - 1
        bin = rem0 % NB + 1
        col = rem0 ÷ NB + 1
        acc_s = zero(eltype(output))
        @inbounds for blk in 1:n_priv
            acc_s += partial_sums[bin, col, blk]
        end
        @inbounds output[bin, col] = acc_s
        t += nworkers
    end
end

KA.@kernel function _batch_merge_usmem_cnts!(
    output_cnts,
    @Const(partial_cnts),
    NB::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    t = worker
    while t <= NB
        bin = t
        acc_c = UInt32(0)
        @inbounds for blk in 1:n_priv
            acc_c += partial_cnts[bin, blk]
        end
        @inbounds output_cnts[bin] = acc_c
        t += nworkers
    end
end

KA.@kernel function _batch_merge_sp1d_sums!(
    output,
    @Const(partial_sums),
    NB::Int,
    bw::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    n_out = SF_GPU_SINGLE_PASS_N * NB * bw
    t = worker
    while t <= n_out
        rem0 = t - 1
        bin = rem0 % NB + 1
        rem1 = rem0 ÷ NB
        ty = rem1 % SF_GPU_SINGLE_PASS_N + 1
        col = rem1 ÷ SF_GPU_SINGLE_PASS_N + 1
        acc_s = zero(eltype(output))
        @inbounds for blk in 1:n_priv
            acc_s += partial_sums[ty, bin, col, blk]
        end
        @inbounds output[ty, bin, col] = acc_s
        t += nworkers
    end
end

KA.@kernel function _batch_merge_sp1d_cnts!(
    output_cnts,
    @Const(partial_cnts),
    NB::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    n_out = SF_GPU_SINGLE_PASS_N * NB
    t = worker
    while t <= n_out
        rem0 = t - 1
        bin = rem0 % NB + 1
        ty = rem0 ÷ NB + 1
        acc_c = UInt32(0)
        @inbounds for blk in 1:n_priv
            acc_c += partial_cnts[ty, bin, blk]
        end
        @inbounds output_cnts[ty, bin] = acc_c
        t += nworkers
    end
end

KA.@kernel function _batch_merge_sp2d_cnts!(
    output_cnts,
    @Const(partial_cnts),
    n_dist::Int,
    n_val::Int,
    bw::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    n_out = SF_GPU_SINGLE_PASS_N * n_dist * n_val * bw
    t = worker
    while t <= n_out
        rem0 = t - 1
        vbin = rem0 % n_val + 1
        rem1 = rem0 ÷ n_val
        dbin = rem1 % n_dist + 1
        rem2 = rem1 ÷ n_dist
        ty = rem2 % SF_GPU_SINGLE_PASS_N + 1
        col = rem2 ÷ SF_GPU_SINGLE_PASS_N + 1
        acc_c = UInt32(0)
        @inbounds for blk in 1:n_priv
            acc_c += partial_cnts[ty, dbin, vbin, col, blk]
        end
        @inbounds output_cnts[ty, dbin, vbin, col] = acc_c
        t += nworkers
    end
end

KA.@kernel function _batch_merge_sp2d_sums!(
    output,
    @Const(partial_sums),
    n_dist::Int,
    n_val::Int,
    bw::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    n_out = SF_GPU_SINGLE_PASS_N * n_dist * n_val * bw
    t = worker
    while t <= n_out
        rem0 = t - 1
        vbin = rem0 % n_val + 1
        rem1 = rem0 ÷ n_val
        dbin = rem1 % n_dist + 1
        rem2 = rem1 ÷ n_dist
        ty = rem2 % SF_GPU_SINGLE_PASS_N + 1
        col = rem2 ÷ SF_GPU_SINGLE_PASS_N + 1
        acc_s = zero(eltype(output))
        @inbounds for blk in 1:n_priv
            acc_s += partial_sums[ty, dbin, vbin, col, blk]
        end
        @inbounds output[ty, dbin, vbin, col] = acc_s
        t += nworkers
    end
end

@inline function _batch_sp1d_accum_shared!(
    shared_sums,
    shared_cnts,
    bin::Int,
    col::Int,
    du_L,
    du_T,
    du_L2,
    du_T2,
    NB::Int,
)
    base = (col - 1) * SF_GPU_SINGLE_PASS_N * NB
    @atomic shared_sums[base + bin] += du_L2 + du_T2
    @atomic shared_sums[base + NB + bin] += du_L2
    @atomic shared_sums[base + 2NB + bin] += du_T2
    @atomic shared_sums[base + 3NB + bin] += du_L * (du_L2 + du_T2)
    @atomic shared_sums[base + 4NB + bin] += du_L * du_L2
     shared_sums[base + 5NB + bin] += du_L * du_T2
    if col == 1
        @atomic shared_cnts[bin] += UInt32(1)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Fixed-x individual SF — u-smem priv strip kernel
# ---------------------------------------------------------------------------

KA.@kernel function _batch_fixed_x_usmem_priv!(
    partial_sums,
    partial_cnts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    b_base::Int,
    bw::Int,
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
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_uj = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS * BATCH_USMEM_STRIP_W,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid

    slot = lid
    while slot <= NB * bw
        @inbounds shared_sums[slot] = zero(FT)
        slot += workgroup_size
    end
    if b_base == 1
        slot = lid
        while slot <= NB
            @inbounds shared_cnts[slot] = UInt32(0)
            slot += workgroup_size
        end
    end

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
                    end
                    k += workgroup_size
                end
            end
            elem = lid
            while elem <= 256 * bw
                col = (elem - 1) ÷ 256 + 1
                rem = (elem - 1) % 256
                k = rem ÷ 2 + 1
                c = rem % 2 + 1
                gi = i0 + k - 1
                b_idx = b_base + col - 1
                if k <= ni
                    @inbounds shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    @inbounds shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end
            if ti < tj
                elem = lid
                while elem <= 256 * bw
                    col = (elem - 1) ÷ 256 + 1
                    rem = (elem - 1) % 256
                    k = rem ÷ 2 + 1
                    c = rem % 2 + 1
                    gj = j0 + k - 1
                    b_idx = b_base + col - 1
                    if k <= nj
                        @inbounds shared_uj[elem] = u_batch[b_idx, gj, c]
                    else
                        @inbounds shared_uj[elem] = zero(FT)
                    end
                    elem += workgroup_size
                end
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            if ti < tj
                n_pairs = ni * nj
                p = lid
                while p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, true,
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            U1 = SA.SVector{2, FT}(
                                shared_ui[_batch_usmem_idx(1, ia, col)],
                                shared_ui[_batch_usmem_idx(2, ia, col)],
                            )
                            U2 = SA.SVector{2, FT}(
                                shared_uj[_batch_usmem_idx(1, jb, col)],
                                shared_uj[_batch_usmem_idx(2, jb, col)],
                            )
                            val = sf_type(U2 - U1, r̂)
                            hist_slot = bin + (col - 1) * NB
                            @atomic shared_sums[hist_slot] += val
                        end
                        if b_base == 1
                            @atomic shared_cnts[bin] += UInt32(1)
                        end
                    end
                    p += workgroup_size
                end
            else
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = _pair_from_linear(p, ni)
                    pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, false,
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            U1 = SA.SVector{2, FT}(
                                shared_ui[_batch_usmem_idx(1, ia, col)],
                                shared_ui[_batch_usmem_idx(2, ia, col)],
                            )
                            U2 = SA.SVector{2, FT}(
                                shared_ui[_batch_usmem_idx(1, jb, col)],
                                shared_ui[_batch_usmem_idx(2, jb, col)],
                            )
                            val = sf_type(U2 - U1, r̂)
                            hist_slot = bin + (col - 1) * NB
                            @atomic shared_sums[hist_slot] += val
                        end
                        if b_base == 1
                            @atomic shared_cnts[bin] += UInt32(1)
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks
        slot = lid
        while slot <= NB * bw
            bin = (slot - 1) % NB + 1
            col = (slot - 1) ÷ NB + 1
            @inbounds partial_sums[bin, col, block_id] = shared_sums[slot]
            slot += workgroup_size
        end
        if b_base == 1
            slot = lid
            while slot <= NB
                @inbounds partial_cnts[slot, block_id] = shared_cnts[slot]
                slot += workgroup_size
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Fixed-x SP1D (8 types) — u-smem priv strip kernel
# ---------------------------------------------------------------------------

KA.@kernel function _batch_fixed_x_sp1d_usmem_priv!(
    partial_sums,
    partial_cnts,
    @Const(x_mat),
    @Const(u_batch),
    N_points::Int,
    N_bins::Int,
    NB::Int,
    b_base::Int,
    bw::Int,
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
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (256 * BATCH_SP1D_USMEM_STRIP_W,)
    shared_uj = @localmem FT (256 * BATCH_SP1D_USMEM_STRIP_W,)
    shared_sums = @localmem FT (SF_GPU_SINGLE_PASS_N * SF_GPU_MAX_BINS * BATCH_SP1D_USMEM_STRIP_W,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid

    slot = lid
    while slot <= SF_GPU_SINGLE_PASS_N * NB * bw
        @inbounds shared_sums[slot] = zero(FT)
        slot += workgroup_size
    end
    if b_base == 1
        slot = lid
        while slot <= NB
            @inbounds shared_cnts[slot] = UInt32(0)
            slot += workgroup_size
        end
    end

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
                    end
                    k += workgroup_size
                end
            end
            elem = lid
            while elem <= 256 * bw
                col = (elem - 1) ÷ 256 + 1
                rem = (elem - 1) % 256
                k = rem ÷ 2 + 1
                c = rem % 2 + 1
                gi = i0 + k - 1
                b_idx = b_base + col - 1
                if k <= ni
                    @inbounds shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    @inbounds shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end
            if ti < tj
                elem = lid
                while elem <= 256 * bw
                    col = (elem - 1) ÷ 256 + 1
                    rem = (elem - 1) % 256
                    k = rem ÷ 2 + 1
                    c = rem % 2 + 1
                    gj = j0 + k - 1
                    b_idx = b_base + col - 1
                    if k <= nj
                        @inbounds shared_uj[elem] = u_batch[b_idx, gj, c]
                    else
                        @inbounds shared_uj[elem] = zero(FT)
                    end
                    elem += workgroup_size
                end
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            if ti < tj
                n_pairs = ni * nj
                p = lid
                while p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, true,
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            du_x = shared_uj[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_uj[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            du_L = r̂[1] * du_x + r̂[2] * du_y
                            du_T = r̂[2] * du_x - r̂[1] * du_y
                            du_L2 = du_L * du_L
                            du_T2 = du_T * du_T
                            _batch_sp1d_accum_shared!(
                                shared_sums, shared_cnts, bin, col,
                                du_L, du_T, du_L2, du_T2, NB,
                            )
                        end
                    end
                    p += workgroup_size
                end
            else
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = _pair_from_linear(p, ni)
                    pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, false,
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            du_x = shared_ui[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_ui[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            du_L = r̂[1] * du_x + r̂[2] * du_y
                            du_T = r̂[2] * du_x - r̂[1] * du_y
                            du_L2 = du_L * du_L
                            du_T2 = du_T * du_T
                            _batch_sp1d_accum_shared!(
                                shared_sums, shared_cnts, bin, col,
                                du_L, du_T, du_L2, du_T2, NB,
                            )
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks
        slot = lid
        while slot <= SF_GPU_SINGLE_PASS_N * NB * bw
            rem0 = slot - 1
            bin = rem0 % NB + 1
            rem1 = rem0 ÷ NB
            ty = rem1 % SF_GPU_SINGLE_PASS_N + 1
            col = rem1 ÷ SF_GPU_SINGLE_PASS_N + 1
            @inbounds partial_sums[ty, bin, col, block_id] = shared_sums[slot]
            slot += workgroup_size
        end
        if b_base == 1
            slot = lid
            while slot <= SF_GPU_SINGLE_PASS_N * NB
                rem0 = slot - 1
                bin = rem0 % NB + 1
                ty = rem0 ÷ NB + 1
                @inbounds partial_cnts[ty, bin, block_id] = shared_cnts[bin]
                slot += workgroup_size
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Varying geometry — one launch, geometry per (pair, b)
# ---------------------------------------------------------------------------

KA.@kernel function _batch_varying_x_sf!(
    output,
    counts,
    @Const(x_batch),
    @Const(u_batch),
    sf_type,
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
    B::Int,
) where {FT}
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
                    gi = i0 + ia - 1
                    gj = j0 + jb - 1
                else
                    ia, jb = _pair_from_linear(p, ni)
                    gi = i0 + ia - 1
                    gj = i0 + jb - 1
                end
                @inbounds for b in 1:B
                    X1 = SA.SVector{2, FT}(x_batch[1, gi, b], x_batch[2, gi, b])
                    X2 = SA.SVector{2, FT}(x_batch[1, gj, b], x_batch[2, gj, b])
                    U1 = SA.SVector{2, FT}(u_batch[1, gi, b], u_batch[2, gi, b])
                    U2 = SA.SVector{2, FT}(u_batch[1, gj, b], u_batch[2, gj, b])
                    dX = X2 - X1
                    dist_sq = dX[1]^2 + dX[2]^2
                    dist = sqrt(dist_sq)
                    bin = _gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= bin < N_bins
                        r̂ = dX / dist
                        val = sf_type(U2 - U1, r̂)
                        @atomic output[bin, b] += val
                        @atomic counts[bin, b] += UInt32(1)
                    end
                end
                p += workgroup_size
            end
        end
    end
end

KA.@kernel function _batch_varying_x_sp1d!(
    output_sums,
    output_counts,
    @Const(x_batch),
    @Const(u_batch),
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
    B::Int,
) where {FT}
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
                    gi = i0 + ia - 1
                    gj = j0 + jb - 1
                else
                    ia, jb = _pair_from_linear(p, ni)
                    gi = i0 + ia - 1
                    gj = i0 + jb - 1
                end
                @inbounds for b in 1:B
                    X1 = SA.SVector{2, FT}(x_batch[1, gi, b], x_batch[2, gi, b])
                    X2 = SA.SVector{2, FT}(x_batch[1, gj, b], x_batch[2, gj, b])
                    U1 = SA.SVector{2, FT}(u_batch[1, gi, b], u_batch[2, gi, b])
                    U2 = SA.SVector{2, FT}(u_batch[1, gj, b], u_batch[2, gj, b])
                    dX = X2 - X1
                    dist_sq = dX[1]^2 + dX[2]^2
                    dist = sqrt(dist_sq)
                    bin = _gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= bin < N_bins
                        r̂ = dX / dist
                        du = U2 - U1
                        du_L = r̂[1] * du[1] + r̂[2] * du[2]
                        du_T = r̂[2] * du[1] - r̂[1] * du[2]
                        du_L2 = du_L * du_L
                        du_T2 = du_T * du_T
                        @atomic output_sums[1, bin, b] += du_L2 + du_T2
                        @atomic output_sums[2, bin, b] += du_L2
                        @atomic output_sums[3, bin, b] += du_T2
                        @atomic output_sums[4, bin, b] += du_L * (du_L2 + du_T2)
                        @atomic output_sums[5, bin, b] += du_L * du_L2
                         output_sums[6, bin, b] += du_L * du_T2
                        for t in 1:SF_GPU_SINGLE_PASS_N
                            @atomic output_counts[t, bin, b] += UInt32(1)
                        end
                    end
                end
                p += workgroup_size
            end
        end
    end
end
