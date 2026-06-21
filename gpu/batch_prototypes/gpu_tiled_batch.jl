# Production-aligned tiled128 batch kernels (fork of ext/TiledStructureFunctionKernels.jl).
# CUDA default: strip-outer u-smem (`_batch_tiled128_2d_linear_fixed_x_fused_strip_outer!`, ~23 s).
# Measured correct+fast: §4 priv ~14 s, §5 atomic ~16 s (benchmark §4/§5). BATCH_WARP_B=1 broken experiment.

"""Max pair index per tile block (upper bound for VRAM geometry cache third dimension)."""
const BATCH_GEOM_CACHE_PAIRS = SFGE.SF_GPU_TILE * SFGE.SF_GPU_TILE

"""Batch strip width for fixed-x block-local `(NB, strip)` histogram columns."""
const BATCH_TILED_STRIP_W = 32

"""
Strip width when staging `u` in smem `(128, strip_w)` per tile block.
Fits default 48 KiB smem with `ui`+`uj` staging (16 cols × 256 × 2 × 4 B ≈ 32 KiB).
"""
const BATCH_FUSED_USMEM_STRIP_W = 16

"""Warp width and strip size for fused fixed-x integration (CUDA)."""
const BATCH_WARP_WIDTH = 32
const BATCH_FUSED_WARP_STRIP_W = 32

const _BATCH_SHMEM_HIST = SFGE.SF_GPU_MAX_BINS * BATCH_TILED_STRIP_W
const _BATCH_FUSED_USMEM_SHMEM = SFGE.SF_GPU_TILE * 2 * BATCH_FUSED_USMEM_STRIP_W
const _BATCH_FUSED_WARP_HIST = SFGE.SF_GPU_MAX_BINS * BATCH_FUSED_WARP_STRIP_W

"""Smem slots for per-wave geometry cache (`bin`, `r̂_x`, `r̂_y`) — one row per thread."""
const _BATCH_GEOM_WAVE_SMEM = 3 * SFGE.SF_GPU_TILED_WS

@inline function _geom_cache_save!(
    cache,
    p::Int,
    block_id::Int,
    pair_ok::Bool,
    bin::Int,
    r̂,
    ::Type{FT},
) where {FT}
    @inbounds if pair_ok
        cache[1, p, block_id] = FT(bin)
        cache[2, p, block_id] = r̂[1]
        cache[3, p, block_id] = r̂[2]
    else
        cache[1, p, block_id] = zero(FT)
    end
    return nothing
end

@inline function _geom_cache_load(cache, p::Int, block_id::Int, N_bins::Int, ::Type{FT}) where {FT}
    @inbounds bin = Int(cache[1, p, block_id])
    if 1 <= bin < N_bins
        r̂ = SA.SVector{2, FT}(cache[2, p, block_id], cache[3, p, block_id])
        return (true, bin, r̂)
    end
    return (false, 0, SA.SVector{2, FT}(zero(FT), zero(FT)))
end

@inline function _geom_wave_store!(buf, lid::Int, pair_ok::Bool, bin::Int, r̂, ::Type{FT}) where {FT}
    base = 3 * (lid - 1)
    if pair_ok
        @inbounds begin
            buf[base + 1] = FT(bin)
            buf[base + 2] = r̂[1]
            buf[base + 3] = r̂[2]
        end
    else
        @inbounds buf[base + 1] = zero(FT)
    end
    return nothing
end

@inline function _geom_wave_load(buf, lid::Int, N_bins::Int, ::Type{FT}) where {FT}
    base = 3 * (lid - 1)
    @inbounds bin = Int(buf[base + 1])
    if 1 <= bin < N_bins
        r̂ = SA.SVector{2, FT}(buf[base + 2], buf[base + 3])
        return (true, bin, r̂)
    end
    return (false, 0, SA.SVector{2, FT}(zero(FT), zero(FT)))
end

@inline function _batch_warp_ids(lid::Int)
    warp_id = (lid - 1) ÷ BATCH_WARP_WIDTH
    lane = (lid - 1) % BATCH_WARP_WIDTH
    return warp_id, lane
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
        X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
        X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SFGE.SF_GPU_TILE + jb])
    else
        X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
        X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SFGE.SF_GPU_TILE + jb])
    end
    dX = X2 - X1
    dist_sq = dX[1]^2 + dX[2]^2
    dist = sqrt(dist_sq)
    bin = SFGE._gpu_digitize_linear(
        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
    )
    if 1 <= bin < N_bins
        return (true, bin, dX / dist)
    end
    return (false, bin, SA.SVector{2, FT}(zero(FT), zero(FT)))
end

@inline function _batch_hist_idx(bin::Int, col::Int, NB::Int)
    return bin + (col - 1) * NB
end

@inline function _batch_usmem_idx(c::Int, k::Int, col::Int)
    return c + 2 * (k - 1) + SFGE.SF_GPU_TILE * 2 * (col - 1)
end

@inline function _ka_is_cpu_backend(backend)
    # KA singleton is `KernelAbstractions.CPU` (not `CPUBackend`).
    return backend isa KA.CPU || nameof(typeof(backend)) === :CPUBackend
end

function _tiled_batch_launch_params(N_points::Int)
    TILE = SFGE.SF_GPU_TILE
    n_tiles = cld(N_points, TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ws = SFGE.SF_GPU_TILED_WS
    return n_tiles, n_tile_blocks, ws, n_tile_blocks * ws
end

"""
One fixed-x batch strip: smem `x` tile, geometry once per pair, inner `bw` velocity columns
in block-local histogram, flush to `output[:, b_base:b_base+bw-1]`.
"""
@kernel function _batch_tiled128_2d_linear_fixed_x_strip!(
    output,
    counts,
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
    shared_sums = @localmem FT (_BATCH_SHMEM_HIST,)
    shared_cnts = @localmem UInt32 (_BATCH_SHMEM_HIST,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    slot = lid
    while slot <= NB * bw
        @inbounds begin
            shared_sums[slot] = zero(FT)
            shared_cnts[slot] = UInt32(0)
        end
        slot += workgroup_size
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SFGE.SF_GPU_TILE + k] = x_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SFGE.SF_GPU_TILE + k] = x_mat[2, gj]
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
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    gi = i0 + ia - 1
                    gj = j0 + jb - 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SFGE.SF_GPU_TILE + jb])
                else
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    gi = i0 + ia - 1
                    gj = i0 + jb - 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SFGE.SF_GPU_TILE + jb])
                end
                dX = X2 - X1
                dist_sq = dX[1]^2 + dX[2]^2
                dist = sqrt(dist_sq)
                bin = SFGE._gpu_digitize_linear(
                    dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                )
                if 1 <= bin < N_bins
                    r̂ = dX / dist
                    @inbounds for col in 1:bw
                        b = b_base + col - 1
                        U1 = _batch_u_at(u_batch, b, gi)
                        U2 = _batch_u_at(u_batch, b, gj)
                        val = sf_type(U2 - U1, r̂)
                        idx = _batch_hist_idx(bin, col, NB)
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
        slot = lid
        n_flush = NB * bw
        while slot <= n_flush
            col = (slot - 1) ÷ NB + 1
            bin = (slot - 1) % NB + 1
            b = b_base + col - 1
            idx = _batch_hist_idx(bin, col, NB)
            @atomic output[bin, b] += shared_sums[idx]
            if shared_cnts[idx] != UInt32(0)
                @atomic counts[bin, b] += shared_cnts[idx]
            end
            slot += workgroup_size
        end
    end
end

"""
One launch, fixed-x (experimental — **do not use as default**): warp lanes map to `b`
but pair loop uses `p += n_warps` (8 pairs/block), collapsing pair parallelism vs
`p += workgroup_size` (256). Measured ~28 s vs ~13 s for smem fused at N=20k B=8064.
"""
@kernel function _batch_tiled128_2d_linear_fixed_x_fused_warp!(
    output,
    counts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    B::Int,
    strip_w::Int,
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
    shared_sums = @localmem FT (_BATCH_FUSED_WARP_HIST,)
    shared_cnts = @localmem UInt32 (SFGE.SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SFGE.SF_GPU_TILE + k] = x_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SFGE.SF_GPU_TILE + k] = x_mat[2, gj]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    n_warps = workgroup_size ÷ BATCH_WARP_WIDTH

    b_base = 1
    while b_base <= B
        bw = min(strip_w, B - b_base + 1)

        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        bid = (g - 1) ÷ workgroup_size + 1
        warp_id, lane = _batch_warp_ids(lid)
        col = lane + 1

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
        @synchronize

        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        bid = (g - 1) ÷ workgroup_size + 1
        warp_id, lane = _batch_warp_ids(lid)
        col = lane + 1

        if bid <= n_tile_blocks && col <= bw
            ti, tj = SFGE._tile_from_linear(bid, n_tiles)
            i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
            j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
            ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
            nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
            b = b_base + col - 1
            if ni > 0 && nj > 0
                if ti < tj
                    n_pairs = ni * nj
                    p = warp_id + 1
                    while p <= n_pairs
                        ia = (p - 1) ÷ nj + 1
                        jb = (p - 1) - (ia - 1) * nj + 1
                        gi = i0 + ia - 1
                        gj = j0 + jb - 1
                        pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                            shared_xi, shared_xj, ia, jb, true,
                            first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                        )
                        if pair_ok
                            U1 = _batch_u_at(u_batch, b, gi)
                            U2 = _batch_u_at(u_batch, b, gj)
                            val = sf_type(U2 - U1, r̂)
                            hist_slot = _batch_hist_idx(bin, col, NB)
                            @atomic shared_sums[hist_slot] += val
                            if lane == 0 && b_base == 1
                                @atomic shared_cnts[bin] += UInt32(1)
                            end
                        end
                        p += n_warps
                    end
                else
                    n_pairs = ni * (ni - 1) ÷ 2
                    p = warp_id + 1
                    while p <= n_pairs
                        ia, jb = SFGE._pair_from_linear(p, ni)
                        gi = i0 + ia - 1
                        gj = i0 + jb - 1
                        pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                            shared_xi, shared_xj, ia, jb, false,
                            first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                        )
                        if pair_ok
                            U1 = _batch_u_at(u_batch, b, gi)
                            U2 = _batch_u_at(u_batch, b, gj)
                            val = sf_type(U2 - U1, r̂)
                            hist_slot = _batch_hist_idx(bin, col, NB)
                            @atomic shared_sums[hist_slot] += val
                            if lane == 0 && b_base == 1
                                @atomic shared_cnts[bin] += UInt32(1)
                            end
                        end
                        p += n_warps
                    end
                end
            end
        end
        @synchronize

        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        bid = (g - 1) ÷ workgroup_size + 1
        if bid <= n_tile_blocks
            slot = lid
            n_flush = NB * bw
            while slot <= n_flush
                col_f = (slot - 1) ÷ NB + 1
                bin = (slot - 1) % NB + 1
                b = b_base + col_f - 1
                idx = _batch_hist_idx(bin, col_f, NB)
                s_val = shared_sums[idx]
                if s_val != zero(FT)
                    @atomic output[bin, b] += s_val
                end
                slot += workgroup_size
            end
            if b_base == 1
                slot = lid
                while slot <= NB
                    c_val = shared_cnts[slot]
                    if c_val != UInt32(0)
                        @atomic counts[slot, 1] += c_val
                    end
                    slot += workgroup_size
                end
            end
        end
        @synchronize

        b_base += strip_w
    end
end

"""
One launch, fixed-x (broken — do not wire): pair-wave **outer** × strip **inner** replays
`u` staging `n_waves×` per block (~47 s vs ~13 s). Kept for post-mortem only.
"""
@kernel function _batch_tiled128_2d_linear_fixed_x_fused_onpass!(
    output,
    counts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    B::Int,
    strip_w::Int,
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
    shared_ui = @localmem FT (_BATCH_FUSED_USMEM_SHMEM,)
    shared_uj = @localmem FT (_BATCH_FUSED_USMEM_SHMEM,)
    shared_sums = @localmem FT (SFGE.SF_GPU_MAX_BINS * BATCH_FUSED_USMEM_STRIP_W,)
    shared_cnts = @localmem UInt32 (SFGE.SF_GPU_MAX_BINS,)
    shared_geom = @localmem FT (_BATCH_GEOM_WAVE_SMEM,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SFGE.SF_GPU_TILE + k] = x_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SFGE.SF_GPU_TILE + k] = x_mat[2, gj]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2

            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            slot = lid
            while slot <= NB
                @inbounds shared_cnts[slot] = UInt32(0)
                slot += workgroup_size
            end
            @synchronize

            p_wave = 1
            while p_wave <= n_pairs
                g = @index(Global, Linear)
                lid = (g - 1) % workgroup_size + 1
                p = p_wave + lid - 1
                if p <= n_pairs
                    if ti < tj
                        ia = (p - 1) ÷ nj + 1
                        jb = (p - 1) - (ia - 1) * nj + 1
                        pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                            shared_xi, shared_xj, ia, jb, true,
                            first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                        )
                    else
                        ia, jb = SFGE._pair_from_linear(p, ni)
                        pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                            shared_xi, shared_xj, ia, jb, false,
                            first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                        )
                    end
                    _geom_wave_store!(shared_geom, lid, pair_ok, bin, r̂, FT)
                else
                    _geom_wave_store!(shared_geom, lid, false, 0, SA.SVector{2, FT}(zero(FT), zero(FT)), FT)
                end
                @synchronize

                b_base = 1
                while b_base <= B
                    bw = min(strip_w, B - b_base + 1)

                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1

                    slot = lid
                    while slot <= NB * bw
                        @inbounds shared_sums[slot] = zero(FT)
                        slot += workgroup_size
                    end

                    elem = lid
                    while elem <= SFGE.SF_GPU_TILE * 2 * bw
                        col = (elem - 1) ÷ (SFGE.SF_GPU_TILE * 2) + 1
                        rem = (elem - 1) % (SFGE.SF_GPU_TILE * 2)
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
                        while elem <= SFGE.SF_GPU_TILE * 2 * bw
                            col = (elem - 1) ÷ (SFGE.SF_GPU_TILE * 2) + 1
                            rem = (elem - 1) % (SFGE.SF_GPU_TILE * 2)
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
                    @synchronize

                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    p = p_wave + lid - 1
                    if p <= n_pairs
                        pair_ok, bin, r̂ = _geom_wave_load(shared_geom, lid, N_bins, FT)
                        if pair_ok
                            if ti < tj
                                ia = (p - 1) ÷ nj + 1
                                jb = (p - 1) - (ia - 1) * nj + 1
                            else
                                ia, jb = SFGE._pair_from_linear(p, ni)
                            end
                            @inbounds for col in 1:bw
                                idx_a1 = _batch_usmem_idx(1, ia, col)
                                idx_a2 = _batch_usmem_idx(2, ia, col)
                                U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                if ti < tj
                                    idx_b1 = _batch_usmem_idx(1, jb, col)
                                    idx_b2 = _batch_usmem_idx(2, jb, col)
                                    U2 = SA.SVector{2, FT}(shared_uj[idx_b1], shared_uj[idx_b2])
                                else
                                    idx_b1 = _batch_usmem_idx(1, jb, col)
                                    idx_b2 = _batch_usmem_idx(2, jb, col)
                                    U2 = SA.SVector{2, FT}(shared_ui[idx_b1], shared_ui[idx_b2])
                                end
                                val = sf_type(U2 - U1, r̂)
                                hist_slot = _batch_hist_idx(bin, col, NB)
                                @atomic shared_sums[hist_slot] += val
                            end
                            if b_base == 1
                                @atomic shared_cnts[bin] += UInt32(1)
                            end
                        end
                    end
                    @synchronize

                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    slot = lid
                    n_flush = NB * bw
                    while slot <= n_flush
                        col_f = (slot - 1) ÷ NB + 1
                        bin_f = (slot - 1) % NB + 1
                        b = b_base + col_f - 1
                        idx = _batch_hist_idx(bin_f, col_f, NB)
                        s_val = shared_sums[idx]
                        if s_val != zero(FT)
                            @atomic output[bin_f, b] += s_val
                        end
                        slot += workgroup_size
                    end
                    @synchronize

                    b_base += strip_w
                end

                p_wave += workgroup_size
            end

            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            slot = lid
            while slot <= NB
                c_val = shared_cnts[slot]
                if c_val != UInt32(0)
                    @atomic counts[slot, 1] += c_val
                end
                slot += workgroup_size
            end
        end
    end
end

"""
One launch, fixed-x: stage `x` once per tile block (smem); inner batch strips with `u` in smem,
block-local `(NB, strip_w)` histogram, strip flush to global output (atomics on flush only).

`x` stays in smem across strips; each strip recomputes `(bin, r̂)` from smem (cheap).
Do **not** spill geometry to VRAM — that added ~3 global reads/pair/strip and regressed §8.
"""
# Accumulate one pair into output[:, b] after geometry is fixed.
# strip_w chunks the inner batch axis; counts only update counts[bin, 1] (fixed-x).
@inline function _accumulate_pair_batch_inner!(
    output,
    counts,
    u_batch,
    sf_type,
    bin::Int,
    r̂,
    gi::Int,
    gj::Int,
    B::Int,
    strip_w::Int,
    ::Type{FT},
) where {FT}
    b_base = 1
    while b_base <= B
        bw = min(strip_w, B - b_base + 1)
        @inbounds for col in 1:bw
            b = b_base + col - 1
            U1 = _batch_u_at(u_batch, b, gi)
            U2 = _batch_u_at(u_batch, b, gj)
            val = sf_type(U2 - U1, r̂)
            @atomic output[bin, b] += val
        end
        b_base += strip_w
    end
    @atomic counts[bin, 1] += UInt32(1)
    return nothing
end

"""
Pair-outer / batch-inner fixed-x kernel (integration default).

One tile schedule; `x` staged once per tile block; **geometry once per pair**; inner `b`
chunks load batch-major `u(B, N, 2)` at the pair endpoints. Replaces strip-outer fused
(`_batch_tiled128_2d_linear_fixed_x_fused_strip_outer!`) which replayed the pair loop
`ceil(B / strip_w)` times.
"""
@kernel function _batch_tiled128_2d_linear_fixed_x_pair_outer!(
    output,
    counts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    B::Int,
    strip_w::Int,
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

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SFGE.SF_GPU_TILE + k] = x_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SFGE.SF_GPU_TILE + k] = x_mat[2, gj]
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
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            if ti < tj
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                            shared_xi, shared_xj, ia, jb, true,
                            first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                        )
                        if pair_ok
                            gi = i0 + ia - 1
                            gj = j0 + jb - 1
                            _accumulate_pair_batch_inner!(
                                output, counts, u_batch, sf_type,
                                bin, r̂, gi, gj, B, strip_w, FT,
                            )
                        end
                        p += workgroup_size
                        jb += workgroup_size
                        while jb > nj
                            jb -= nj
                            ia += 1
                        end
                    end
                end
            else
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, false,
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                    )
                    if pair_ok
                        gi = i0 + ia - 1
                        gj = i0 + jb - 1
                        _accumulate_pair_batch_inner!(
                            output, counts, u_batch, sf_type,
                            bin, r̂, gi, gj, B, strip_w, FT,
                        )
                    end
                    p += workgroup_size
                end
            end
        end
    end
end

"""
One launch, fixed-x: `x` staged once; **pair loop once**; geometry once per pair; each
thread owns one pair and loads `u(b, gi/gj)` from global VRAM over batch chunks (no shared
`u` smem — parallel pairs must not share staging buffers).

Replaces strip-outer (504× pair-loop replay). Measured ~13 s via priv reference @ N=20k B=8064;
target correctness vs §4 priv before trusting sub-second timings.
"""
@kernel function _batch_tiled128_2d_linear_fixed_x_warp_b!(
    output,
    counts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    B::Int,
    chunk_w::Int,
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

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SFGE.SF_GPU_TILE + k] = x_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SFGE.SF_GPU_TILE + k] = x_mat[2, gj]
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
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            if ti < tj
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                            shared_xi, shared_xj, ia, jb, true,
                            first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                        )
                        if pair_ok
                            gi = i0 + ia - 1
                            gj = j0 + jb - 1
                            _accumulate_pair_warp_b!(
                                output, u_batch, sf_type, bin, r̂, gi, gj, B, chunk_w, FT,
                            )
                            @atomic counts[bin, 1] += UInt32(1)
                        end
                        p += workgroup_size
                        jb += workgroup_size
                        while jb > nj
                            jb -= nj
                            ia += 1
                        end
                    end
                end
            else
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, false,
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                    )
                    if pair_ok
                        gi = i0 + ia - 1
                        gj = i0 + jb - 1
                        _accumulate_pair_warp_b!(
                            output, u_batch, sf_type, bin, r̂, gi, gj, B, chunk_w, FT,
                        )
                        @atomic counts[bin, 1] += UInt32(1)
                    end
                    p += workgroup_size
                end
            end
        end
    end
end

@inline function _accumulate_pair_warp_b!(
    output,
    u_batch,
    sf_type,
    bin::Int,
    r̂,
    gi::Int,
    gj::Int,
    B::Int,
    chunk_w::Int,
    ::Type{FT},
) where {FT}
    b0 = 1
    while b0 <= B
        bw = min(chunk_w, B - b0 + 1)
        col = 1
        while col <= bw
            b_idx = b0 + col - 1
            U1 = _batch_u_at(u_batch, b_idx, gi)
            U2 = _batch_u_at(u_batch, b_idx, gj)
            val = sf_type(U2 - U1, r̂)
            @atomic output[bin, b_idx] += val
            col += 1
        end
        b0 += chunk_w
    end
    return nothing
end

"""Strip-outer fused kernel — smem hist → block-private partial or global atomics on flush."""
@kernel function _batch_tiled128_2d_linear_fixed_x_fused_strip_outer!(
    output,
    counts,
    partial,
    geom_cache,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    B::Int,
    strip_w::Int,
    use_geom_cache::Bool,
    use_block_priv::Bool,
    b_col_offset::Int,
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
    shared_ui = @localmem FT (_BATCH_FUSED_USMEM_SHMEM,)
    shared_uj = @localmem FT (_BATCH_FUSED_USMEM_SHMEM,)
    shared_sums = @localmem FT (SFGE.SF_GPU_MAX_BINS * BATCH_FUSED_USMEM_STRIP_W,)
    shared_cnts = @localmem UInt32 (SFGE.SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # Stage x once per tile block (amortized over all batch strips).
    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_mat[1, gi]
                    shared_xi[SFGE.SF_GPU_TILE + k] = x_mat[2, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_mat[1, gj]
                        shared_xj[SFGE.SF_GPU_TILE + k] = x_mat[2, gj]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    b_base = 1
    while b_base <= B
        bw = min(strip_w, B - b_base + 1)

        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        bid = (g - 1) ÷ workgroup_size + 1

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
            ti, tj = SFGE._tile_from_linear(bid, n_tiles)
            i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
            j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
            ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
            nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
            if ni > 0 && nj > 0
                elem = lid
                while elem <= SFGE.SF_GPU_TILE * 2 * bw
                    col = (elem - 1) ÷ (SFGE.SF_GPU_TILE * 2) + 1
                    rem = (elem - 1) % (SFGE.SF_GPU_TILE * 2)
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
                    while elem <= SFGE.SF_GPU_TILE * 2 * bw
                        col = (elem - 1) ÷ (SFGE.SF_GPU_TILE * 2) + 1
                        rem = (elem - 1) % (SFGE.SF_GPU_TILE * 2)
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
        if bid <= n_tile_blocks
            ti, tj = SFGE._tile_from_linear(bid, n_tiles)
            i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
            j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
            ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
            nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
            if ni > 0 && nj > 0
                if ti < tj
                    n_pairs = ni * nj
                    p = lid
                    if p <= n_pairs
                        ia = (p - 1) ÷ nj + 1
                        jb = (p - 1) - (ia - 1) * nj + 1
                        while p <= n_pairs
                            if use_geom_cache && b_base > 1
                                pair_ok, bin, r̂ = _geom_cache_load(
                                    geom_cache, p, bid, N_bins, FT,
                                )
                            else
                                pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                                    shared_xi, shared_xj, ia, jb, true,
                                    first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                                )
                                if use_geom_cache && b_base == 1
                                    _geom_cache_save!(geom_cache, p, bid, pair_ok, bin, r̂, FT)
                                end
                            end
                            if pair_ok
                                @inbounds for col in 1:bw
                                    idx_a1 = _batch_usmem_idx(1, ia, col)
                                    idx_a2 = _batch_usmem_idx(2, ia, col)
                                    U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                    idx_b1 = _batch_usmem_idx(1, jb, col)
                                    idx_b2 = _batch_usmem_idx(2, jb, col)
                                    U2 = SA.SVector{2, FT}(shared_uj[idx_b1], shared_uj[idx_b2])
                                    val = sf_type(U2 - U1, r̂)
                                    hist_slot = _batch_hist_idx(bin, col, NB)
                                    @atomic shared_sums[hist_slot] += val
                                end
                                if b_base == 1
                                    @atomic shared_cnts[bin] += UInt32(1)
                                end
                            end
                            p += workgroup_size
                            jb += workgroup_size
                            while jb > nj
                                jb -= nj
                                ia += 1
                            end
                        end
                    end
                else
                    n_pairs = ni * (ni - 1) ÷ 2
                    p = lid
                    while p <= n_pairs
                        ia, jb = SFGE._pair_from_linear(p, ni)
                        if use_geom_cache && b_base > 1
                            pair_ok, bin, r̂ = _geom_cache_load(
                                geom_cache, p, bid, N_bins, FT,
                            )
                        else
                            pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                                shared_xi, shared_xj, ia, jb, false,
                                first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                            )
                            if use_geom_cache && b_base == 1
                                _geom_cache_save!(geom_cache, p, bid, pair_ok, bin, r̂, FT)
                            end
                        end
                        if pair_ok
                            @inbounds for col in 1:bw
                                idx_a1 = _batch_usmem_idx(1, ia, col)
                                idx_a2 = _batch_usmem_idx(2, ia, col)
                                U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                idx_b1 = _batch_usmem_idx(1, jb, col)
                                idx_b2 = _batch_usmem_idx(2, jb, col)
                                U2 = SA.SVector{2, FT}(shared_ui[idx_b1], shared_ui[idx_b2])
                                val = sf_type(U2 - U1, r̂)
                                hist_slot = _batch_hist_idx(bin, col, NB)
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
        if bid <= n_tile_blocks
            slot = lid
            n_flush = NB * bw
            while slot <= n_flush
                col = (slot - 1) ÷ NB + 1
                bin = (slot - 1) % NB + 1
                b = b_base + col - 1
                idx = _batch_hist_idx(bin, col, NB)
                s_val = shared_sums[idx]
                if s_val != zero(FT)
                    if use_block_priv
                        @inbounds partial[bin, b, bid] += s_val
                    else
                        @atomic output[bin, b] += s_val
                    end
                end
                slot += workgroup_size
            end
            if b_base == 1
                slot = lid
                while slot <= NB
                    c_val = shared_cnts[slot]
                    if c_val != UInt32(0)
                        if use_block_priv
                            @inbounds partial[NB + slot, 1 + b_col_offset, bid] += FT(c_val)
                        else
                            @atomic counts[slot, 1 + b_col_offset] += c_val
                        end
                    end
                    slot += workgroup_size
                end
            end
        end
        @synchronize

        b_base += strip_w
    end
end

"""
One varying-x batch slice `b`: identical to production `_sf_kernel_tiled128_2d_linear_u32!`
with `(x_batch, u_batch)` at batch index `b`, output column `output[:, b]`.
"""
@kernel function _batch_tiled128_2d_linear_varying_x_slice!(
    output,
    counts,
    @Const(x_batch),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    b::Int,
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
    shared_sums = @localmem FT (SFGE.SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SFGE.SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    hb = lid
    while hb <= NB
        @inbounds begin
            shared_sums[hb] = zero(FT)
            shared_cnts[hb] = UInt32(0)
        end
        hb += workgroup_size
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k] = x_batch[1, gi, b]
                    shared_xi[SFGE.SF_GPU_TILE + k] = x_batch[2, gi, b]
                    shared_ui[k] = u_batch[1, gi, b]
                    shared_ui[SFGE.SF_GPU_TILE + k] = u_batch[2, gi, b]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k] = x_batch[1, gj, b]
                        shared_xj[SFGE.SF_GPU_TILE + k] = x_batch[2, gj, b]
                        shared_uj[k] = u_batch[1, gj, b]
                        shared_uj[SFGE.SF_GPU_TILE + k] = u_batch[2, gj, b]
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
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SFGE.SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SFGE.SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SFGE.SF_GPU_TILE + jb])
                else
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SFGE.SF_GPU_TILE + jb])
                    U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SFGE.SF_GPU_TILE + ia])
                    U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SFGE.SF_GPU_TILE + jb])
                end
                dX = X2 - X1
                dist_sq = dX[1]^2 + dX[2]^2
                dist = sqrt(dist_sq)
                bin = SFGE._gpu_digitize_linear(
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
        hb = lid
        while hb <= NB
            @atomic output[hb] += shared_sums[hb]
            if shared_cnts[hb] != UInt32(0)
                @atomic counts[hb] += shared_cnts[hb]
            end
            hb += workgroup_size
        end
    end
end

"""Copy production-style 1D slice histogram into batch column `b` (host/device)."""
function _copy_batch_column!(sums_dev, counts_dev, col_sums, col_counts, b::Int)
    # `copyto!(selectdim(...))` does not reliably update CuArray columns on CUDA.
    sums_dev[:, b] .= col_sums
    counts_dev[:, b] .= col_counts
    return nothing
end

function _launch_batch_tiled128_2d_linear_varying_x!(
    backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    col_sums = nothing,
    col_counts = nothing,
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SFGE.SF_GPU_MAX_BINS &&
        error("batch tiled128 supports at most $(SFGE.SF_GPU_MAX_BINS) bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_tiled128_2d_linear_varying_x_slice!(backend, ws)
    if col_sums === nothing
        col_sums = KA.adapt(backend, zeros(FT, NB))
        col_counts = KA.adapt(backend, zeros(UInt32, NB))
    end
    for b in 1:B
        fill!(col_sums, zero(FT))
        fill!(col_counts, zero(UInt32))
        kernel!(
            col_sums, col_counts, x_dev, u_dev, sf_type,
            N, n_bins, NB, b, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        KA.synchronize(backend)
        _copy_batch_column!(sums_dev, counts_dev, col_sums, col_counts, b)
        KA.synchronize(backend)
    end
    KA.synchronize(backend)
    return nothing
end

function _launch_batch_tiled128_2d_linear_fixed_x!(
    backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT},
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SFGE.SF_GPU_MAX_BINS &&
        error("batch tiled128 supports at most $(SFGE.SF_GPU_MAX_BINS) bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_tiled128_2d_linear_fixed_x_strip!(backend, ws)
    b_base = 1
    while b_base <= B
        bw = min(BATCH_TILED_STRIP_W, B - b_base + 1)
        kernel!(
            sums_dev, counts_dev, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        b_base += BATCH_TILED_STRIP_W
    end
    KA.synchronize(backend)
    return nothing
end

function _resolve_partial_dev!(
    backend,
    workspace::Union{Nothing, BatchGPUWorkspace{FT}},
    sums_dev,
    n_tile_blocks::Int,
    NB::Int,
    B::Int,
) where {FT}
    if workspace !== nothing
        return ensure_partial_dev!(workspace, backend)
    end
    return KA.zeros(backend, FT, 2 * NB, B, n_tile_blocks)
end

function _ensure_geom_cache!(
    backend,
    ws::Union{Nothing, BatchGPUWorkspace{FT}},
    n_tile_blocks::Int,
    ::Type{FT},
) where {FT}
    if ws !== nothing
        return ensure_geom_cache!(ws, backend, FT)
    end
    n_pairs = SFGE.SF_GPU_TILE * SFGE.SF_GPU_TILE
    return KA.zeros(backend, FT, 3, n_pairs, n_tile_blocks)
end

"""Single launch: warp-B pair-once (default CUDA), strip-outer, pair-outer, or host strips."""
function _launch_batch_tiled128_2d_linear_fixed_x_fused!(
    backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    strip_outer::Bool = true,
    pair_outer::Bool = false,
    warp_b::Bool = false,
    use_geom_cache::Bool = false,
    use_block_priv::Bool = false,
    host_strips::Bool = false,
    max_vram::Int = 0,
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
    profile::Bool = false,
) where {FT}
    pair_outer && strip_outer &&
        error("pair_outer and strip_outer are mutually exclusive")
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SFGE.SF_GPU_MAX_BINS &&
        error("batch tiled128 supports at most $(SFGE.SF_GPU_MAX_BINS) bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val

    if pair_outer
        W = BATCH_FUSED_USMEM_STRIP_W
        t_kern = @elapsed begin
            kern = _batch_tiled128_2d_linear_fixed_x_pair_outer!(backend, ws)
            kern(
                sums_dev, counts_dev, x_dev, u_dev, sf_type,
                N, n_bins, NB, B, W, fe, le, is_, off, sv,
                n_tiles, n_tile_blocks, ws;
                ndrange = ndrange,
            )
            KA.synchronize(backend)
        end
        profile && @printf("  [profile] pair_outer kernel: %.4fs\n", t_kern)
        if B > 1
            counts_dev[:, 2:end] .= @view counts_dev[:, 1]
        end
        KA.synchronize(backend)
        return profile ? (kernel_s = t_kern,) : nothing
    end

    if warp_b && !_ka_is_cpu_backend(backend)
        W = BATCH_WARP_WIDTH
        t_kern = @elapsed begin
            kern = _batch_tiled128_2d_linear_fixed_x_warp_b!(backend, ws)
            kern(
                sums_dev, counts_dev, x_dev, u_dev, sf_type,
                N, n_bins, NB, B, W, fe, le, is_, off, sv,
                n_tiles, n_tile_blocks, ws;
                ndrange = ndrange,
            )
            KA.synchronize(backend)
        end
        if B > 1
            counts_dev[:, 2:end] .= @view counts_dev[:, 1]
        end
        KA.synchronize(backend)
        if profile
            n_chunks = cld(B, W)
            _batch_profile_log!(
                @sprintf(
                    "  [profile] path=warp_b_pair_once n_b_chunks=%d chunk_w=%d kernel=%.4fs merge=0.0000s",
                    n_chunks, W, t_kern,
                ),
            )
            return (path = "warp_b_pair_once", n_b_chunks = n_chunks, chunk_w = W,
                kernel_s = t_kern, merge_s = 0.0)
        end
        return nothing
    end

    if strip_outer && (_ka_is_cpu_backend(backend) || host_strips)
        t_cpu = @elapsed begin
            b_base = 1
            n_launches = 0
            while b_base <= B
                bw = min(BATCH_TILED_STRIP_W, B - b_base + 1)
                kernel! = _batch_tiled128_2d_linear_fixed_x_strip!(backend, ws)
                kernel!(
                    sums_dev, counts_dev, x_dev, u_dev, sf_type,
                    N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
                    n_tiles, n_tile_blocks, ws;
                    ndrange = ndrange,
                )
                b_base += BATCH_TILED_STRIP_W
                n_launches += 1
            end
            KA.synchronize(backend)
        end
        if B > 1
            counts_dev[:, 2:end] .= @view counts_dev[:, 1]
        end
        if profile
            path = _ka_is_cpu_backend(backend) ? "cpu_host_strip" : "cuda_host_strip"
            _batch_profile_log!(
                @sprintf(
                    "  [profile] path=%s n_launches=%d strip_w=%d geom_replays=%d kernel=%.4fs merge=0.0000s",
                    path, n_launches, BATCH_TILED_STRIP_W, n_launches, t_cpu,
                ),
            )
            return (path = path, n_launches = n_launches, strip_w = BATCH_TILED_STRIP_W,
                kernel_s = t_cpu, merge_s = 0.0)
        end
        return nothing
    end

    W = BATCH_FUSED_USMEM_STRIP_W
    geom_on = use_geom_cache && strip_outer
    t_geom = 0.0
    geom_dev = KA.zeros(backend, FT, 0, 0, 0)
    if geom_on
        t_geom = @elapsed geom_dev = _ensure_geom_cache!(backend, workspace, n_tile_blocks, FT)
    end
    partial_dev = if use_block_priv
        _resolve_partial_dev!(backend, workspace, sums_dev, n_tile_blocks, NB, B)
    else
        KA.zeros(backend, FT, 0, 0, 0)
    end
    t_fill = 0.0
    if use_block_priv
        t_fill = @elapsed begin
            fill!(partial_dev, zero(eltype(partial_dev)))
            KA.synchronize(backend)
        end
    end
    t_kern = @elapsed begin
        kern = _batch_tiled128_2d_linear_fixed_x_fused_strip_outer!(backend, ws)
        kern(
            sums_dev, counts_dev, partial_dev, geom_dev, x_dev, u_dev, sf_type,
            N, n_bins, NB, B, W, geom_on, use_block_priv, 0, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        KA.synchronize(backend)
    end
    t_merge = 0.0
    if use_block_priv
        t_merge = @elapsed begin
            _merge_batch_partial!(sums_dev, counts_dev, partial_dev, NB, B)
            KA.synchronize(backend)
        end
    end
    if B > 1
        counts_dev[:, 2:end] .= @view counts_dev[:, 1]
    end
    KA.synchronize(backend)
    if profile
        n_strips = cld(B, W)
        _batch_profile_log!(
            @sprintf(
                "  [profile] path=inkernel_strip_outer n_strips=%d strip_w=%d block_syncs≈%d geom_alloc=%.4fs fill=%.4fs kernel=%.4fs merge=%.4fs (block_priv=%s geom_cache=%s)",
                n_strips, W, 3 * n_strips, t_geom, t_fill, t_kern, t_merge, use_block_priv, geom_on,
            ),
        )
        return (path = "inkernel_strip_outer", n_strips = n_strips, strip_w = W,
            geom_alloc_s = t_geom, fill_s = t_fill, kernel_s = t_kern, merge_s = t_merge,
            use_block_priv = use_block_priv, geom_cache = geom_on)
    end
    return nothing
end

"""Stage host `(N_dims, N)` / `(N_dims, N, batch...)` to device. Fixed-x uses `(B, N, 2)` for `u`."""
function stage_batch_device(backend, x::AbstractArray{FT}, u::AbstractArray{FT}; fixed_x::Bool) where {FT}
    if fixed_x
        ndims(x) == 2 || throw(ArgumentError("fixed-x batch tiled expects matrix x, got ndims=$(ndims(x))"))
        _require_trailing_batch(u)
        size(x)[1:2] == size(u)[1:2] ||
            throw(ArgumentError("x and u leading dims must match"))
        B = batch_size(u)
        u_flat = reshape(u, size(u, 1), size(u, 2), B)
        # Batch-contiguous `(B, N, 2)` — inner-b stride 1 at fixed grid point.
        u_batchmajor = permutedims(u_flat, (3, 2, 1))
        return KA.adapt(backend, x), KA.adapt(backend, u_batchmajor)
    else
        _require_trailing_batch(x)
        size(x) == size(u) || throw(ArgumentError("varying-x requires x and u same shape"))
        B = batch_size(u)
        x_flat = reshape(x, size(x, 1), size(x, 2), B)
        u_flat = reshape(u, size(u, 1), size(u, 2), B)
        return KA.adapt(backend, x_flat), KA.adapt(backend, u_flat)
    end
end

function gpu_batch_tiled_fixed_x!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace = nothing,
    download::Bool = true,
) where {FT}
    B = batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_mat, 2)
    if workspace === nothing
        sums_dev = KA.adapt(backend, zeros(FT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev, u_dev = stage_batch_device(backend, x_mat, u_batch; fixed_x = true)
    else
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x for tiled fixed-x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        workspace.x_dev === nothing && upload_batch!(workspace, backend, x_mat, u_batch)
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end
    _launch_batch_tiled128_2d_linear_fixed_x!(
        backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, bin_edges,
    )
    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

function gpu_batch_tiled_varying_x!(
    backend,
    sums,
    counts,
    x_batch::AbstractArray{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace = nothing,
    download::Bool = true,
) where {FT}
    B = batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_batch, 2)
    if workspace === nothing
        sums_dev = KA.adapt(backend, zeros(FT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev, u_dev = stage_batch_device(backend, x_batch, u_batch; fixed_x = false)
    else
        !workspace.fixed_x || error("BatchGPUWorkspace must be varying_x for tiled varying-x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        workspace.x_dev === nothing && upload_batch!(workspace, backend, x_batch, u_batch)
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end
    _launch_batch_tiled128_2d_linear_varying_x!(
        backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, bin_edges;
        col_sums = workspace === nothing ? nothing : workspace.col_sums_dev,
        col_counts = workspace === nothing ? nothing : workspace.col_counts_dev,
    )
    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end
