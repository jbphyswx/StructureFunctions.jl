# Fused tiled128 batch kernels — integration candidate for Phase 1.
#
# Case A (fixed-x): one tile schedule, geometry once per pair, inner b loop,
# block-private VRAM partial (hist, B, n_tile_blocks) + merge.
#
# Case B (varying-x): one tile schedule, geometry per (pair, b), inner b loop,
# direct global atomics to output (bounded memory; no B host relaunches).

"""Default soft VRAM budget when `max_vram=0` (user should set `max_vram` on L40/Metal)."""
const BATCH_DEFAULT_VRAM_BUDGET = 60 * 1024^3

"""Read `ENV[name] == "1"` for benchmark toggles."""
function _env_flag(name::String)
    return get(ENV, name, "0") == "1"
end

"""
    batch_accum_plan(N, B, NB, ::Type{FT}; max_vram=0)

VRAM ladder for fixed-x batch integration:

1. **Smem `(NB, strip_w)` → flush** — default for 1D L2SF: final histogram is small
   (`NB×B`); per-block atomics to `output[bin,b]` match the NVIDIA two-level pattern
   (local smem hist, then global merge). See [NVIDIA histogram blog](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/).
2. **Block-private partial** `(2·NB, B, n_tile_blocks)` — opt-in for **wide** histograms
   (SP2D) when `use_block_priv=true`; **not** default at NB=20 (15 GiB partial + RMW
   dominated §8 at N=20k B=8064).
3. **Geom cache** — opt-in only (`use_geom_cache=true`); disabled by default until profiled.
4. **`batch_slab_ranges`** — when partial does not fit `max_vram`.

Set **`max_vram`** on L40/Metal. Use **`BATCH_PROFILE=1`** for phase timings.
"""
function batch_accum_plan(
    N_points::Int,
    B::Int,
    NB::Int,
    ::Type{FT};
    max_vram::Int = 0,
    use_block_priv::Bool = false,
    use_geom_cache::Bool = false,
) where {FT}
    est = estimate_batch_priv_bytes(N_points, B, NB, FT)
    geom_bytes = estimate_geom_cache_bytes(est.n_priv, FT)
    budget = max_vram > 0 ? max_vram : BATCH_DEFAULT_VRAM_BUDGET
    priv_ok = est.partial_bytes <= budget
    use_block_priv = use_block_priv && priv_ok
    remain = budget - (use_block_priv ? est.partial_bytes : 0) - est.output_bytes
    use_geom_cache = use_geom_cache && (geom_bytes <= max(remain, 0))
    b_slabs = if use_block_priv
        [1:B]
    else
        slab_budget = max(1, (budget - est.output_bytes) ÷ 2)
        batch_slab_ranges(B, slab_budget, N_points, NB, FT)
    end
    return (
        use_block_priv = use_block_priv,
        use_geom_cache = use_geom_cache,
        b_slabs = b_slabs,
        partial_bytes = est.partial_bytes,
        geom_bytes = geom_bytes,
        budget = budget,
        n_priv = est.n_priv,
        priv_fits = priv_ok,
    )
end

"""Estimate block-private partial VRAM for fused tiled batch (sums + counts)."""
function estimate_batch_priv_bytes(
    N_points::Int,
    B::Int,
    NB::Int,
    ::Type{FT};
    tile::Int = SFGE.SF_GPU_TILE,
) where {FT}
    n_tiles = cld(N_points, tile)
    n_priv = n_tiles * (n_tiles + 1) ÷ 2
    slab = 2 * NB * B * sizeof(FT)
    partial_bytes = n_priv * slab
    return (
        partial_bytes = partial_bytes,
        n_priv = n_priv,
        n_tiles = n_tiles,
        per_block_slab_bytes = slab,
        output_bytes = 2 * NB * B * sizeof(FT),
    )
end

"""
    batch_slab_ranges(B, max_partial_bytes, N_points, NB, ::Type{FT}) -> Vector{UnitRange{Int}}

Split linear batch axis `1:B` into sub-ranges so each slab's partial buffer fits
`max_partial_bytes` (0 = no splitting → single range `1:B`).
"""
function batch_slab_ranges(
    B::Int,
    max_partial_bytes::Int,
    N_points::Int,
    NB::Int,
    ::Type{FT},
) where {FT}
    if max_partial_bytes <= 0 || B <= 0
        return [1:B]
    end
    est = estimate_batch_priv_bytes(N_points, 1, NB, FT)
    per_b_partial = est.n_priv * 2 * NB * sizeof(FT)
    per_b_partial <= 0 && return [1:B]
    chunk = max(1, max_partial_bytes ÷ per_b_partial)
    ranges = UnitRange{Int}[]
    b0 = 1
    while b0 <= B
        b1 = min(B, b0 + chunk - 1)
        push!(ranges, b0:b1)
        b0 = b1 + 1
    end
    return ranges
end

function _make_fused_partial!(backend, NB::Int, B::Int, n_priv::Int, ::Type{FT}) where {FT}
    return KA.zeros(backend, FT, 2 * NB, B, n_priv)
end

@inline function _fused_priv_idx(bin::Int, col::Int, NB::Int, block_id::Int, n_priv::Int)
    return bin + (col - 1) * NB + (block_id - 1) * NB * col
end

# ---------------------------------------------------------------------------
# fixed-x fused — one launch, geometry once per pair, inner b, direct output atomics
# (no block-private VRAM partial, no 15 GiB fill, no merge pass)
# ---------------------------------------------------------------------------

@kernel function _batch_fused_fixed_x_direct!(
    output,
    counts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
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
                        U1 = _batch_u_at(u_batch, col, gi)
                        U2 = _batch_u_at(u_batch, col, gj)
                        val = sf_type(U2 - U1, r̂)
                        @atomic output[bin, col] += val
                        @atomic counts[bin, col] += UInt32(1)
                    end
                end
                p += workgroup_size
            end
        end
    end
end

function _launch_fused_fixed_x_direct!(
    backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    b_range::UnitRange{Int} = 1:B,
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SFGE.SF_GPU_MAX_BINS &&
        error("fused batch tiled128 supports at most $(SFGE.SF_GPU_MAX_BINS) bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    bw = length(b_range)
    kernel! = _batch_fused_fixed_x_direct!(backend, ws)
    kernel!(
        sums_dev, counts_dev, x_dev, u_dev, sf_type,
        N, n_bins, NB, bw, fe, le, is_, off, sv,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    KA.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# fixed-x VRAM partial + merge (experimental — superseded by direct path above)
# ---------------------------------------------------------------------------

@kernel function _batch_fused_fixed_x_priv!(
    partial,
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
    block_id = bid
    if bid <= n_tile_blocks && block_id <= n_tile_blocks
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
                        @atomic partial[bin, col, block_id] += val
                        @atomic partial[NB + bin, col, block_id] += FT(1)
                    end
                end
                p += workgroup_size
            end
        end
    end
end

"""
Parallel merge of block-private partial `(2·NB, B, n_tile_blocks)` along the block axis.

Uses `sum(..., dims=3)` (GPU reduction when `partial` is a `CuArray`). The old
`_batch_fused_merge_priv!` grid-stride kernel did `O(NB·B·n_priv)` serial work per output
cell and regressed §8 to ~23 s at N=20k B=8064.
"""
function _merge_batch_partial!(sums_dev, counts_dev, partial_dev, NB::Int, B::Int)
    FT = eltype(sums_dev)
    s_red = dropdims(sum(view(partial_dev, 1:NB, :, :), dims=3), dims=3)
    copy!(sums_dev, s_red)
    c_red = dropdims(sum(view(partial_dev, (NB + 1):(2 * NB), 1:1, :), dims=3), dims=3)
    @inbounds @views counts_dev[:, 1] .= UInt32.(round.(c_red[:, 1]))
    return nothing
end

@kernel function _batch_fused_merge_priv!(
    output,
    counts,
    @Const(partial),
    NB::Int,
    B::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    n_out = NB * B
    t = worker
    while t <= n_out
        rem0 = t - 1
        bin = rem0 % NB + 1
        b = rem0 ÷ NB + 1
        acc_s = zero(eltype(output))
        acc_c = zero(eltype(partial))
        @inbounds for blk in 1:n_priv
            acc_s += partial[bin, b, blk]
            acc_c += partial[NB + bin, b, blk]
        end
        @inbounds begin
            output[bin, b] = acc_s
            counts[bin, b] = UInt32(acc_c)
        end
        t += nworkers
    end
end

@kernel function _batch_fused_merge_usmem_sums!(
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
        @inbounds begin
            output[bin, col] = acc_s
        end
        t += nworkers
    end
end

@kernel function _batch_fused_merge_usmem_cnts!(
    output_cnts,
    @Const(partial_cnts),
    NB::Int,
    n_priv::Int,
    nworkers::Int,
)
    worker = @index(Global, Linear)
    n_out = NB
    t = worker
    while t <= n_out
        bin = t
        acc_c = UInt32(0)
        @inbounds for blk in 1:n_priv
            acc_c += partial_cnts[bin, blk]
        end
        @inbounds begin
            output_cnts[bin] = acc_c
        end
        t += nworkers
    end
end

function _launch_fused_fixed_x_priv!(
    backend,
    sums_dev,
    counts_dev,
    partial_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    b_range::UnitRange{Int} = 1:B,
    merge_nworkers::Int = 262_144,
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    NB > SFGE.SF_GPU_MAX_BINS &&
        error("fused batch tiled128 supports at most $(SFGE.SF_GPU_MAX_BINS) bins (got NB=$NB)")
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    bw = length(b_range)
    b_base = first(b_range)
    fill!(partial_dev, zero(FT))
    KA.synchronize(backend)
    acc! = _batch_fused_fixed_x_priv!(backend, ws)
    acc!(
        partial_dev, x_dev, u_dev, sf_type,
        N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
        n_tiles, n_tile_blocks, ws;
        ndrange = ndrange,
    )
    KA.synchronize(backend)
    merge! = _batch_fused_merge_priv!(backend, 256)
    merge!(
        sums_dev, counts_dev, partial_dev, NB, bw, n_tile_blocks, merge_nworkers;
        ndrange = merge_nworkers, workgroupsize = 256,
    )
    KA.synchronize(backend)
    return nothing
end

"""Return `(fill_s, accum_s, merge_s)` for VRAM partial path profiling (CUDA events when available)."""
function _launch_fused_fixed_x_priv_timed!(
    backend,
    sums_dev,
    counts_dev,
    partial_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    b_range::UnitRange{Int} = 1:B,
    merge_nworkers::Int = 262_144,
) where {FT}
    n_bins = length(lbe.edges)
    NB = n_bins - 1
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    bw = length(b_range)
    b_base = first(b_range)
    t_fill = @elapsed begin
        fill!(partial_dev, zero(FT))
        KA.synchronize(backend)
    end
    t_accum = @elapsed begin
        acc! = _batch_fused_fixed_x_priv!(backend, ws)
        acc!(
            partial_dev, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        KA.synchronize(backend)
    end
    t_merge = @elapsed begin
        merge! = _batch_fused_merge_priv!(backend, 256)
        merge!(
            sums_dev, counts_dev, partial_dev, NB, bw, n_tile_blocks, merge_nworkers;
            ndrange = merge_nworkers, workgroupsize = 256,
        )
        KA.synchronize(backend)
    end
    return (fill_s = t_fill, accum_s = t_accum, merge_s = t_merge)
end

function _launch_fused_fixed_x_in_kernel_timed!(
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
    t_accum = @elapsed begin
        _launch_batch_tiled128_2d_linear_fixed_x_fused!(
            backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, lbe,
        )
    end
    return (accum_s = t_accum,)
end

function _launch_fused_fixed_x_direct_timed!(
    backend,
    sums_dev,
    counts_dev,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    B::Int,
    lbe::LinearBinEdges{FT};
    b_range::UnitRange{Int} = 1:B,
) where {FT}
    t_accum = @elapsed begin
        _launch_fused_fixed_x_direct!(
            backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, lbe;
            b_range = b_range,
        )
    end
    return (accum_s = t_accum,)
end


@kernel function _batch_fused_varying_x!(
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
                else
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    gi = i0 + ia - 1
                    gj = i0 + jb - 1
                end
                @inbounds for b in 1:B
                    X1 = SA.SVector{2, FT}(
                        x_batch[1, gi, b], x_batch[2, gi, b],
                    )
                    X2 = SA.SVector{2, FT}(
                        x_batch[1, gj, b], x_batch[2, gj, b],
                    )
                    U1 = SA.SVector{2, FT}(
                        u_batch[1, gi, b], u_batch[2, gi, b],
                    )
                    U2 = SA.SVector{2, FT}(
                        u_batch[1, gj, b], u_batch[2, gj, b],
                    )
                    dX = X2 - X1
                    dist_sq = dX[1]^2 + dX[2]^2
                    dist = sqrt(dist_sq)
                    bin = SFGE._gpu_digitize_linear(
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

function _launch_fused_varying_x!(
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
    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.offset, lbe.step_val
    kernel! = _batch_fused_varying_x!(backend, ws)
    kernel!(
        sums_dev, counts_dev, x_dev, u_dev, sf_type,
        N, n_bins, NB, fe, le, is_, off, sv,
        n_tiles, n_tile_blocks, ws, B;
        ndrange = ndrange,
    )
    KA.synchronize(backend)
    return nothing
end

"""
    gpu_batch_fused_tiled_fixed_x!(backend, sums, counts, x_mat, u_batch, sf_type, bin_edges;
        max_vram=0, workspace=nothing, download=true)

Fixed-x integration default on CUDA: `gpu_batch_fused_tiled_fixed_x_usmem_priv!` with
**host strips + smem-x geometry + priv merge** (~14 s @ N=20k B=8064). Opt-in experiments:
`BATCH_GEOM_CACHE=1`, `BATCH_TILE_SLABS=1`, `BATCH_FUSED_SINGLE=1`, `BATCH_STRIP_OUTER=1`,
`BATCH_WARP_B=1`, `BATCH_HOST_STRIPS=1`.
"""
function gpu_batch_fused_tiled_fixed_x!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    max_vram::Int = 0,
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
    download::Bool = true,
    use_block_priv::Bool = _env_flag("BATCH_BLOCK_PRIV"),
    use_geom_cache::Bool = true,
    profile::Bool = _env_flag("BATCH_PROFILE"),
) where {FT}
    B = batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_mat, 2)
    is_cpu = backend isa KA.CPU
    use_legacy_fused = !is_cpu && (
        _env_flag("BATCH_STRIP_OUTER") || _env_flag("BATCH_WARP_B") ||
        (_env_flag("BATCH_HOST_STRIPS") && !_env_flag("BATCH_LEGACY_PRIV_STRIPS"))
    )
    if !use_legacy_fused
        return gpu_batch_fused_tiled_fixed_x_usmem_priv!(
            backend, sums, counts, x_mat, u_batch, sf_type, bin_edges;
            workspace = workspace, download = download, profile = profile,
        )
    end
    plan = batch_accum_plan(
        N, B, NB, FT; max_vram = max_vram,
        use_block_priv = use_block_priv, use_geom_cache = use_geom_cache,
    )
    if profile
        @printf(
            "  [plan] block_priv=%s geom_cache=%s partial=%.2f GiB geom=%.2f GiB n_priv=%d\n",
            plan.use_block_priv, plan.use_geom_cache,
            plan.partial_bytes / 1024^3, plan.geom_bytes / 1024^3, plan.n_priv,
        )
        flush(stdout)
        _batch_profile_log!(
            @sprintf(
                "[plan] N=%d B=%d block_priv=%s geom_cache=%s host_strips=%s strip_outer=%s partial=%.2f GiB",
                N, B, plan.use_block_priv, plan.use_geom_cache,
                get(ENV, "BATCH_HOST_STRIPS", "0") == "1",
                get(ENV, "BATCH_STRIP_OUTER", "0") == "1",
                plan.partial_bytes / 1024^3,
            ),
        )
    end
    if max_vram > 0 && !plan.use_block_priv && length(plan.b_slabs) > 1
        return gpu_batch_tiled_fixed_x!(
            backend, sums, counts, x_mat, u_batch, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    end
    if workspace === nothing
        sums_dev = KA.adapt(backend, zeros(FT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev, u_dev = stage_batch_device(backend, x_mat, u_batch; fixed_x = true)
    else
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x for fused fixed-x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end
    is_cpu = backend isa KA.CPU
    use_warp_b = _env_flag("BATCH_WARP_B") && !is_cpu
    use_host_strips = _env_flag("BATCH_HOST_STRIPS") || is_cpu
    # CUDA default: strip-outer u-smem (~23 s @ N=20k B=8064). §4 priv ~14 s; §5 atomic ~16 s (parity PASS).
    # BATCH_WARP_B=1 pair-once global-u is ~33 s and was briefly wrong at 0.95 s (smem race).
    use_strip_outer = !use_warp_b && !(_env_flag("BATCH_HOST_STRIPS") && !is_cpu)
    _launch_batch_tiled128_2d_linear_fixed_x_fused!(
        backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, bin_edges;
        workspace = workspace,
        use_geom_cache = plan.use_geom_cache,
        use_block_priv = plan.use_block_priv,
        warp_b = use_warp_b,
        strip_outer = use_strip_outer || is_cpu,
        host_strips = use_host_strips,
        max_vram = max_vram,
        profile = profile,
    )
    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

"""
    gpu_batch_fused_tiled_varying_x!(...)

One tiled128 launch: one tile schedule, geometry per `(pair, b)` inside inner `b`,
direct `@atomic` to `output[bin, b]` (no B× host slice relaunches).
"""
function gpu_batch_fused_tiled_varying_x!(
    backend,
    sums,
    counts,
    x_batch::AbstractArray{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
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
        !workspace.fixed_x || error("BatchGPUWorkspace must be varying_x for fused varying-x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_batch, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end
    _launch_fused_varying_x!(
        backend, sums_dev, counts_dev, x_dev, u_dev, sf_type, N, B, bin_edges,
    )
    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

# Legacy VRAM partial + merge path (benchmark / experiments only).
function gpu_batch_fused_tiled_fixed_x_vram!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    max_vram::Int = 0,
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
    download::Bool = true,
) where {FT}
    B = batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_mat, 2)
    slabs = batch_slab_ranges(B, max_vram, N, NB, FT)
    if workspace === nothing
        sums_dev = KA.adapt(backend, zeros(FT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev, u_dev = stage_batch_device(backend, x_mat, u_batch; fixed_x = true)
    else
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x for fused fixed-x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end
    n_tiles, n_tile_blocks, _, _ = _tiled_batch_launch_params(N)
    for b_range in slabs
        bw = length(b_range)
        partial_dev = if workspace === nothing
            _make_fused_partial!(backend, NB, bw, n_tile_blocks, FT)
        elseif length(b_range) == B && first(b_range) == 1
            ensure_partial_dev!(workspace, backend)
        else
            _make_fused_partial!(backend, NB, bw, n_tile_blocks, FT)
        end
        if workspace === nothing || length(b_range) == B
            sums_out = sums_dev
            counts_out = counts_dev
            u_sl = u_dev
        else
            sums_out = @view sums_dev[:, b_range]
            counts_out = @view counts_dev[:, b_range]
            u_sl = ndims(u_dev) == 3 ? (@view u_dev[:, :, b_range]) : u_dev
        end
        _launch_fused_fixed_x_priv!(
            backend, sums_out, counts_out, partial_dev, x_dev, u_sl, sf_type,
            N, B, bin_edges; b_range = 1:bw,
        )
    end
    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

function gpu_batch_fused_tiled!(
    backend,
    sums,
    counts,
    x,
    u,
    sf_type,
    bin_edges::LinearBinEdges{FT};
    max_vram::Int = 0,
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
    download::Bool = true,
) where {FT}
    geo = classify_batch_geometry(x, u)
    if geo === FixedX
        gpu_batch_fused_tiled_fixed_x!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            max_vram = max_vram, workspace = workspace, download = download,
        )
    else
        max_vram > 0 && @info "max_vram ignored for varying-x fused (per-b tiled path)"
        gpu_batch_fused_tiled_varying_x!(
            backend, sums, counts, x, u, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    end
    return nothing
end

# Do not override gpu_batch_tiled_varying_x! — production per-b path lives in gpu_tiled_batch.jl.

# ===========================================================================
# U-SMEM Strip-based Kernels (Phase 0 Prototypes)
# ===========================================================================

const BATCH_USMEM_STRIP_W = 16

# Helper to index into u_smem layout: component c (1:2), grid index k (1:128), batch column col (1:strip_w)
@inline function _usmem_idx(c::Int, k::Int, col::Int)
    return c + 2 * (k - 1) + 256 * (col - 1)
end

@kernel function _batch_tiled128_fixed_x_usmem!(
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
    shared_ui = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_uj = @localmem FT (256 * BATCH_USMEM_STRIP_W,)

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

            # Load shared_ui
            elem = lid
            while elem <= 256 * bw
                col = (elem - 1) ÷ 256 + 1
                rem = (elem - 1) % 256
                k = rem ÷ 2 + 1
                c = rem % 2 + 1
                gi = i0 + k - 1
                b_idx = b_base + col - 1
                if k <= ni
                    shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end

            # Load shared_uj (only if ti < tj)
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
                        shared_uj[elem] = u_batch[b_idx, gj, c]
                    else
                        shared_uj[elem] = zero(FT)
                    end
                    elem += workgroup_size
                end
            end
        end
    end
    @synchronize

    # KA CPU lowering requires re-establishing index vars after @synchronize
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
                # Cross tile block
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        gi = i0 + ia - 1
                        gj = j0 + jb - 1
                        X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                        X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SFGE.SF_GPU_TILE + jb])
                        dX = X2 - X1
                        dist_sq = dX[1]^2 + dX[2]^2
                        dist = sqrt(dist_sq)
                        bin = SFGE._gpu_digitize_linear(
                            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                        )
                        if 1 <= bin < N_bins
                            r̂ = dX / dist
                            @inbounds for col in 1:bw
                                idx_a1 = _usmem_idx(1, ia, col)
                                idx_a2 = _usmem_idx(2, ia, col)
                                U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                
                                idx_b1 = _usmem_idx(1, jb, col)
                                idx_b2 = _usmem_idx(2, jb, col)
                                U2 = SA.SVector{2, FT}(shared_uj[idx_b1], shared_uj[idx_b2])
                                
                                val = sf_type(U2 - U1, r̂)
                                b_global = b_base + col - 1
                                @atomic output[bin, b_global] += val
                                @atomic counts[bin, b_global] += UInt32(1)
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
                # Diagonal tile block
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    gi = i0 + ia - 1
                    gj = i0 + jb - 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SFGE.SF_GPU_TILE + jb])
                    dX = X2 - X1
                    dist_sq = dX[1]^2 + dX[2]^2
                    dist = sqrt(dist_sq)
                    bin = SFGE._gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= bin < N_bins
                        r̂ = dX / dist
                        @inbounds for col in 1:bw
                            idx_a1 = _usmem_idx(1, ia, col)
                            idx_a2 = _usmem_idx(2, ia, col)
                            U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                            
                            idx_b1 = _usmem_idx(1, jb, col)
                            idx_b2 = _usmem_idx(2, jb, col)
                            U2 = SA.SVector{2, FT}(shared_ui[idx_b1], shared_ui[idx_b2])
                            
                            val = sf_type(U2 - U1, r̂)
                            b_global = b_base + col - 1
                            @atomic output[bin, b_global] += val
                            @atomic counts[bin, b_global] += UInt32(1)
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
end

"""
One launch, fixed-x batch optimum for 1D L2SF:

- `x` staged once per tile block (smem)
- **Geometry once per pair** via VRAM `geom_cache` (strip 1 compute + save; strips 2+ load)
- Inner `b` strips: tile `u` in smem, smem histogram, flush sums with cross-block atomics
- Counts accumulated on strip 1 only (fixed-x); host broadcasts column 1 → all `b`

Replaces strip-outer host relaunches that replayed digitize/`r̂` `ceil(B/strip_w)` times.
"""
@kernel function _batch_tiled128_fixed_x_usmem_fused_opt!(
    output,
    counts,
    geom_cache,
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
    shared_ui = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_uj = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_sums = @localmem FT (SFGE.SF_GPU_MAX_BINS * BATCH_USMEM_STRIP_W,)
    shared_cnts = @localmem UInt32 (SFGE.SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # Stage x once (batch-independent geometry source).
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
                            if b_base == 1
                                pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                                    shared_xi, shared_xj, ia, jb, true,
                                    first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                                )
                                _geom_cache_save!(geom_cache, p, bid, pair_ok, bin, r̂, FT)
                            else
                                pair_ok, bin, r̂ = _geom_cache_load(geom_cache, p, bid, N_bins, FT)
                            end
                            if pair_ok
                                @inbounds for col in 1:bw
                                    idx_a1 = _usmem_idx(1, ia, col)
                                    idx_a2 = _usmem_idx(2, ia, col)
                                    U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                    idx_b1 = _usmem_idx(1, jb, col)
                                    idx_b2 = _usmem_idx(2, jb, col)
                                    U2 = SA.SVector{2, FT}(shared_uj[idx_b1], shared_uj[idx_b2])
                                    val = sf_type(U2 - U1, r̂)
                                    hist_slot = bin + (col - 1) * NB
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
                        if b_base == 1
                            pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                                shared_xi, shared_xj, ia, jb, false,
                                first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                            )
                            _geom_cache_save!(geom_cache, p, bid, pair_ok, bin, r̂, FT)
                        else
                            pair_ok, bin, r̂ = _geom_cache_load(geom_cache, p, bid, N_bins, FT)
                        end
                        if pair_ok
                            @inbounds for col in 1:bw
                                idx_a1 = _usmem_idx(1, ia, col)
                                idx_a2 = _usmem_idx(2, ia, col)
                                U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                idx_b1 = _usmem_idx(1, jb, col)
                                idx_b2 = _usmem_idx(2, jb, col)
                                U2 = SA.SVector{2, FT}(shared_ui[idx_b1], shared_ui[idx_b2])
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
        if bid <= n_tile_blocks
            slot = lid
            while slot <= NB * bw
                bin = (slot - 1) % NB + 1
                col = (slot - 1) ÷ NB + 1
                b = b_base + col - 1
                s_val = shared_sums[slot]
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

"""Per-strip u-smem + block partial; optional VRAM geometry cache (strip 1 save, strip 2+ load)."""
@kernel function _batch_tiled128_fixed_x_usmem_priv!(
    partial_sums,
    partial_cnts,
    geom_cache,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    b_base::Int,
    bw::Int,
    use_geom_cache::Bool,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    tile_block_offset::Int,
    n_tile_blocks_run::Int,
    workgroup_size::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_uj = @localmem FT (256 * BATCH_USMEM_STRIP_W,)

    # Smem histogram for this block
    shared_sums = @localmem FT (SFGE.SF_GPU_MAX_BINS * BATCH_USMEM_STRIP_W,)
    shared_cnts = @localmem UInt32 (SFGE.SF_GPU_MAX_BINS,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid

    # 1. Parallel smem histogram initialization
    slot = lid
    while slot <= NB * bw
        @inbounds begin
            shared_sums[slot] = zero(FT)
        end
        slot += workgroup_size
    end

    if b_base == 1
        slot = lid
        while slot <= NB
            @inbounds begin
                shared_cnts[slot] = UInt32(0)
            end
            slot += workgroup_size
        end
    end

    # 2. Load x and u into smem
    if bid <= n_tile_blocks_run
        global_bid = tile_block_offset + bid
        ti, tj = SFGE._tile_from_linear(global_bid, n_tiles)
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

            # Load shared_ui
            elem = lid
            while elem <= 256 * bw
                col = (elem - 1) ÷ 256 + 1
                rem = (elem - 1) % 256
                k = rem ÷ 2 + 1
                c = rem % 2 + 1
                gi = i0 + k - 1
                b_idx = b_base + col - 1
                if k <= ni
                    shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end

            # Load shared_uj (only if ti < tj)
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
                        shared_uj[elem] = u_batch[b_idx, gj, c]
                    else
                        shared_uj[elem] = zero(FT)
                    end
                    elem += workgroup_size
                end
            end
        end
    end
    @synchronize

    # KA CPU lowering requires re-establishing index vars after @synchronize
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks_run
        global_bid = tile_block_offset + bid
        ti, tj = SFGE._tile_from_linear(global_bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            if ti < tj
                # Cross tile block
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        if use_geom_cache && b_base > 1
                            pair_ok, bin, r̂ = _geom_cache_load(geom_cache, p, block_id, N_bins, FT)
                        else
                            ia = (p - 1) ÷ nj + 1
                            jb = (p - 1) - (ia - 1) * nj + 1
                            pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                                shared_xi, shared_xj, ia, jb, true,
                                first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                            )
                            if use_geom_cache && b_base == 1
                                _geom_cache_save!(geom_cache, p, block_id, pair_ok, bin, r̂, FT)
                            end
                        end
                        if pair_ok
                            ia = (p - 1) ÷ nj + 1
                            jb = (p - 1) - (ia - 1) * nj + 1
                            @inbounds for col in 1:bw
                                idx_a1 = _usmem_idx(1, ia, col)
                                idx_a2 = _usmem_idx(2, ia, col)
                                U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                
                                idx_b1 = _usmem_idx(1, jb, col)
                                idx_b2 = _usmem_idx(2, jb, col)
                                U2 = SA.SVector{2, FT}(shared_uj[idx_b1], shared_uj[idx_b2])
                                
                                val = sf_type(U2 - U1, r̂)
                                hist_slot = bin + (col - 1) * NB
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
                # Diagonal tile block
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    if use_geom_cache && b_base > 1
                        pair_ok, bin, r̂ = _geom_cache_load(geom_cache, p, block_id, N_bins, FT)
                    else
                        pair_ok, bin, r̂ = _pair_bin_rhat_from_smem!(
                            shared_xi, shared_xj, ia, jb, false,
                            first_edge, last_edge, inv_step, offset, step_val, N_bins, FT,
                        )
                        if use_geom_cache && b_base == 1
                            _geom_cache_save!(geom_cache, p, block_id, pair_ok, bin, r̂, FT)
                        end
                    end
                    if pair_ok
                        @inbounds for col in 1:bw
                            idx_a1 = _usmem_idx(1, ia, col)
                            idx_a2 = _usmem_idx(2, ia, col)
                            U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                            
                            idx_b1 = _usmem_idx(1, jb, col)
                            idx_b2 = _usmem_idx(2, jb, col)
                            U2 = SA.SVector{2, FT}(shared_ui[idx_b1], shared_ui[idx_b2])
                            
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

    # 4. Flush smem histogram to global VRAM partials (non-atomic!)
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks_run
        slot = lid
        while slot <= NB * bw
            bin = (slot - 1) % NB + 1
            col = (slot - 1) ÷ NB + 1
            @inbounds begin
                partial_sums[bin, col, block_id] = shared_sums[slot]
            end
            slot += workgroup_size
        end
        if b_base == 1
            slot = lid
            while slot <= NB
                @inbounds begin
                    partial_cnts[slot, block_id] = shared_cnts[slot]
                end
                slot += workgroup_size
            end
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_usmem!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
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
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x for usmem")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end

    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = bin_edges.first_edge, bin_edges.last_edge, bin_edges.inv_step, bin_edges.offset, bin_edges.step_val
    n_bins = NB + 1
    
    kernel! = _batch_tiled128_fixed_x_usmem!(backend, ws)
    b_base = 1
    while b_base <= B
        bw = min(BATCH_USMEM_STRIP_W, B - b_base + 1)
        kernel!(
            sums_dev, counts_dev, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        b_base += bw
    end
    KA.synchronize(backend)

    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

function gpu_batch_fused_tiled_fixed_x_usmem_priv!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
    download::Bool = true,
    profile::Bool = _env_flag("BATCH_PROFILE"),
    max_geom_cache_bytes::Int = BATCH_GEOM_CACHE_MAX_BYTES,
) where {FT}
    B = batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_mat, 2)

    if workspace === nothing
        sums_dev = KA.adapt(backend, zeros(FT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev, u_dev = stage_batch_device(backend, x_mat, u_batch; fixed_x = true)
    else
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x for usmem")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end

    n_tiles, n_tile_blocks, ws, _ = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = bin_edges.first_edge, bin_edges.last_edge, bin_edges.inv_step, bin_edges.offset, bin_edges.step_val
    n_bins = NB + 1

    use_geom_cache = _env_flag("BATCH_GEOM_CACHE")
    use_slabs = _env_flag("BATCH_TILE_SLABS") ||
        (use_geom_cache && !geom_cache_fits(n_tile_blocks, FT; max_bytes = max_geom_cache_bytes))
    use_single_fused = _env_flag("BATCH_FUSED_SINGLE") && !(backend isa KA.CPU)

    if use_single_fused && use_geom_cache && geom_cache_fits(n_tile_blocks, FT; max_bytes = max_geom_cache_bytes)
        geom_cache = allocate_geom_cache_slab(backend, n_tile_blocks, FT)
        ndrange = n_tile_blocks * ws
        t_kern = @elapsed begin
            kernel! = _batch_tiled128_fixed_x_usmem_fused_opt!(backend, ws)
            kernel!(
                sums_dev, counts_dev, geom_cache, x_dev, u_dev, sf_type,
                N, n_bins, NB, B, BATCH_USMEM_STRIP_W, fe, le, is_, off, sv,
                n_tiles, n_tile_blocks, ws;
                ndrange = ndrange,
            )
            KA.synchronize(backend)
        end
        profile && @printf(
            "  [profile] path=usmem_fused_single n_strips=%d strip_w=%d kernel=%.4fs\n",
            cld(B, BATCH_USMEM_STRIP_W), BATCH_USMEM_STRIP_W, t_kern,
        )
    elseif use_slabs
        slabs = tile_block_geom_slab_ranges(n_tile_blocks, max_geom_cache_bytes, FT)
        max_slab = maximum(length, slabs)
        geom_cache = use_geom_cache ? allocate_geom_cache_slab(backend, max_slab, FT) :
            KA.zeros(backend, FT, 3, 1, 1)
        use_cache = use_geom_cache
        kernel! = _batch_tiled128_fixed_x_usmem_priv!(backend, ws)
        merge_sums! = _batch_fused_merge_usmem_sums!(backend, ws)
        merge_cnts! = _batch_fused_merge_usmem_cnts!(backend, ws)
        partial_sums = KA.zeros(backend, FT, NB, BATCH_USMEM_STRIP_W, max_slab)
        partial_cnts = KA.zeros(backend, UInt32, NB, max_slab)
        strip_acc = KA.zeros(backend, FT, NB, BATCH_USMEM_STRIP_W)
        cnt_acc = KA.zeros(backend, UInt32, NB)
        t_total = @elapsed begin
            for slab in slabs
                slab_len = length(slab)
                tile_off = first(slab) - 1
                ndrange = slab_len * ws
                b_base = 1
                while b_base <= B
                    bw = min(BATCH_USMEM_STRIP_W, B - b_base + 1)
                    kernel!(
                        partial_sums, partial_cnts, geom_cache, x_dev, u_dev, sf_type,
                        N, n_bins, NB, b_base, bw, use_cache, fe, le, is_, off, sv,
                        n_tiles, n_tile_blocks, tile_off, slab_len, ws;
                        ndrange = ndrange,
                    )
                    strip_view = @view strip_acc[:, 1:bw]
                    merge_sums!(
                        strip_view, partial_sums, NB, bw, slab_len, NB * bw;
                        ndrange = NB * bw,
                    )
                    @view(sums_dev[:, b_base:b_base+bw-1]) .+= strip_view
                    if b_base == 1
                        merge_cnts!(cnt_acc, partial_cnts, NB, slab_len, NB; ndrange = NB)
                        counts_dev[:, 1] .+= cnt_acc
                    end
                    b_base += bw
                end
            end
            KA.synchronize(backend)
        end
        profile && @printf(
            "  [profile] path=usmem_priv_slabbed n_slabs=%d n_strips=%d total=%.4fs\n",
            length(slabs), cld(B, BATCH_USMEM_STRIP_W), t_total,
        )
    else
        # Default: host strip + smem-x geometry + priv merge (~14 s @ N=20k B=8064).
        # Geometry from smem x each strip is cheap; VRAM geom cache regressed to ~18 s.
        kernel! = _batch_tiled128_fixed_x_usmem_priv!(backend, ws)
        merge_sums! = _batch_fused_merge_usmem_sums!(backend, ws)
        merge_cnts! = _batch_fused_merge_usmem_cnts!(backend, ws)
        geom_dummy = KA.zeros(backend, FT, 3, 1, 1)
        partial_sums = KA.zeros(backend, FT, NB, BATCH_USMEM_STRIP_W, n_tile_blocks)
        partial_cnts = KA.zeros(backend, UInt32, NB, n_tile_blocks)
        ndrange = n_tile_blocks * ws
        t_total = @elapsed begin
            b_base = 1
            while b_base <= B
                bw = min(BATCH_USMEM_STRIP_W, B - b_base + 1)
                kernel!(
                    partial_sums, partial_cnts, geom_dummy, x_dev, u_dev, sf_type,
                    N, n_bins, NB, b_base, bw, false, fe, le, is_, off, sv,
                    n_tiles, n_tile_blocks, 0, n_tile_blocks, ws;
                    ndrange = ndrange,
                )
                merge_sums!(
                    @view(sums_dev[:, b_base:b_base+bw-1]), partial_sums,
                    NB, bw, n_tile_blocks, NB * bw;
                    ndrange = NB * bw,
                )
                if b_base == 1
                    merge_cnts!(
                        @view(counts_dev[:, 1]), partial_cnts, NB, n_tile_blocks, NB;
                        ndrange = NB,
                    )
                end
                b_base += bw
            end
            KA.synchronize(backend)
        end
        profile && @printf(
            "  [profile] path=usmem_priv_strips n_strips=%d strip_w=%d total=%.4fs\n",
            cld(B, BATCH_USMEM_STRIP_W), BATCH_USMEM_STRIP_W, t_total,
        )
    end

    if B > 1
        counts_dev[:, 2:end] .= @view counts_dev[:, 1]
    end
    KA.synchronize(backend)

    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

# ---------------------------------------------------------------------------
# Coalesced Vectorized Batch Kernel (No redundant geometry loops)
# ---------------------------------------------------------------------------

@kernel function _batch_tiled128_fixed_x_coalesced!(
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
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    col = lid

    shared_sums = @localmem FT (32 * 128,)
    shared_cnts = @localmem UInt32 (32 * 128,)

    # Initialize shared memory to zero
    @inbounds for bin in 1:NB
        idx = bin + (col - 1) * 32
        shared_sums[idx] = zero(FT)
        shared_cnts[idx] = UInt32(0)
    end

    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)

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
    col = lid

    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)

        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2

            for p in 1:n_pairs
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
                    if col <= bw
                        b = b_base + col - 1
                        U1 = SA.SVector{2, FT}(u_batch[b, gi, 1], u_batch[b, gi, 2])
                        U2 = SA.SVector{2, FT}(u_batch[b, gj, 1], u_batch[b, gj, 2])
                        val = sf_type(U2 - U1, r̂)
                        idx = bin + (col - 1) * 32
                        @inbounds begin
                            shared_sums[idx] += val
                            shared_cnts[idx] += UInt32(1)
                        end
                    end
                end
            end

            if col <= bw
                b = b_base + col - 1
                @inbounds for bin in 1:NB
                    idx = bin + (col - 1) * 32
                    s_val = shared_sums[idx]
                    c_val = shared_cnts[idx]
                    if c_val > 0
                        @atomic output[bin, b] += s_val
                        @atomic counts[bin, b] += c_val
                    end
                end
            end
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_coalesced!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
    download::Bool = true,
) where {FT}
    B = batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_mat, 2)

    if NB > 32
        # Fallback to private kernel if number of bins exceeds the coalesced shared mem layout
        return gpu_batch_fused_tiled_fixed_x_usmem_priv!(
            backend, sums, counts, x_mat, u_batch, sf_type, bin_edges;
            workspace = workspace, download = download,
        )
    end

    if workspace === nothing
        sums_dev = KA.adapt(backend, zeros(FT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev, u_dev = stage_batch_device(backend, x_mat, u_batch; fixed_x = true)
    else
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end

    ws = 128
    TILE = SFGE.SF_GPU_TILE
    n_tiles = cld(N, TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    ndrange = n_tile_blocks * ws

    fe, le, is_, off, sv = bin_edges.first_edge, bin_edges.last_edge, bin_edges.inv_step, bin_edges.offset, bin_edges.step_val
    n_bins = NB + 1

    kernel! = _batch_tiled128_fixed_x_coalesced!(backend, ws)

    b_base = 1
    while b_base <= B
        bw = min(ws, B - b_base + 1)

        kernel!(
            sums_dev, counts_dev, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )

        b_base += ws
    end
    KA.synchronize(backend)

    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

# ---------------------------------------------------------------------------
# u-smem with Block-level Atomic Flush (No partial VRAM buffer / no merge kernel)
# ---------------------------------------------------------------------------

@kernel function _batch_tiled128_fixed_x_usmem_atomic_flush!(
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
    shared_ui = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_uj = @localmem FT (256 * BATCH_USMEM_STRIP_W,)

    # Smem histogram for this block
    shared_sums = @localmem FT (SFGE.SF_GPU_MAX_BINS * BATCH_USMEM_STRIP_W,)
    shared_cnts = @localmem UInt32 (SFGE.SF_GPU_MAX_BINS * BATCH_USMEM_STRIP_W,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # 1. Parallel smem histogram initialization
    slot = lid
    while slot <= NB * bw
        @inbounds begin
            shared_sums[slot] = zero(FT)
            shared_cnts[slot] = UInt32(0)
        end
        slot += workgroup_size
    end

    # 2. Load x and u into smem
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

            # Load shared_ui
            elem = lid
            while elem <= 256 * bw
                col = (elem - 1) ÷ 256 + 1
                rem = (elem - 1) % 256
                k = rem ÷ 2 + 1
                c = rem % 2 + 1
                gi = i0 + k - 1
                b_idx = b_base + col - 1
                if k <= ni
                    shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end

            # Load shared_uj (only if ti < tj)
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
                        shared_uj[elem] = u_batch[b_idx, gj, c]
                    else
                        shared_uj[elem] = zero(FT)
                    end
                    elem += workgroup_size
                end
            end
        end
    end
    @synchronize

    # 3. Main pair loop (flat 1D loop with additive/subtractive coordinate updates)
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
                # Cross tile block
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        gi = i0 + ia - 1
                        gj = j0 + jb - 1
                        X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                        X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SFGE.SF_GPU_TILE + jb])
                        dX = X2 - X1
                        dist_sq = dX[1]^2 + dX[2]^2
                        dist = sqrt(dist_sq)
                        bin = SFGE._gpu_digitize_linear(
                            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                        )
                        if 1 <= bin < N_bins
                            r̂ = dX / dist
                            @inbounds for col in 1:bw
                                idx_a1 = _usmem_idx(1, ia, col)
                                idx_a2 = _usmem_idx(2, ia, col)
                                U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                
                                idx_b1 = _usmem_idx(1, jb, col)
                                idx_b2 = _usmem_idx(2, jb, col)
                                U2 = SA.SVector{2, FT}(shared_uj[idx_b1], shared_uj[idx_b2])
                                
                                val = sf_type(U2 - U1, r̂)
                                hist_slot = bin + (col - 1) * NB
                                @atomic shared_sums[hist_slot] += val
                                @atomic shared_cnts[hist_slot] += UInt32(1)
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
                # Diagonal tile block
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    gi = i0 + ia - 1
                    gj = i0 + jb - 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SFGE.SF_GPU_TILE + jb])
                    dX = X2 - X1
                    dist_sq = dX[1]^2 + dX[2]^2
                    dist = sqrt(dist_sq)
                    bin = SFGE._gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= bin < N_bins
                        r̂ = dX / dist
                        @inbounds for col in 1:bw
                            idx_a1 = _usmem_idx(1, ia, col)
                            idx_a2 = _usmem_idx(2, ia, col)
                            U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                            
                            idx_b1 = _usmem_idx(1, jb, col)
                            idx_b2 = _usmem_idx(2, jb, col)
                            U2 = SA.SVector{2, FT}(shared_ui[idx_b1], shared_ui[idx_b2])
                            
                            val = sf_type(U2 - U1, r̂)
                            hist_slot = bin + (col - 1) * NB
                            @atomic shared_sums[hist_slot] += val
                            @atomic shared_cnts[hist_slot] += UInt32(1)
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
    @synchronize

    # 4. Flush directly to global output with block-level atomics
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        slot = lid
        while slot <= NB * bw
            bin = (slot - 1) % NB + 1
            col = (slot - 1) ÷ NB + 1
            b_global = b_base + col - 1
            @inbounds begin
                s_val = shared_sums[slot]
                c_val = shared_cnts[slot]
                if c_val > 0
                    @atomic output[bin, b_global] += s_val
                    @atomic counts[bin, b_global] += c_val
                end
            end
            slot += workgroup_size
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_usmem_atomic_flush!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
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
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end

    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = bin_edges.first_edge, bin_edges.last_edge, bin_edges.inv_step, bin_edges.offset, bin_edges.step_val
    n_bins = NB + 1

    kernel! = _batch_tiled128_fixed_x_usmem_atomic_flush!(backend, ws)
    
    b_base = 1
    while b_base <= B
        bw = min(BATCH_USMEM_STRIP_W, B - b_base + 1)
        
        kernel!(
            sums_dev, counts_dev, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws;
            ndrange = ndrange,
        )
        
        b_base += BATCH_USMEM_STRIP_W
    end
    KA.synchronize(backend)

    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

# ---------------------------------------------------------------------------
# Strategy D: Fully Vectorized Single-Launch GPU Batch Kernel (Optimized Block-Local u-smem with Dynamic W)
# ---------------------------------------------------------------------------

@kernel function _batch_tiled128_fixed_x_vectorized!(
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
    ::Val{NB_VAL},
    ::Val{W},
) where {NB_VAL, W, FT}
    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (256 * W,)
    shared_uj = @localmem FT (256 * W,)

    # Cache sums block-locally in shared memory sized dynamically by type parameters
    shared_sums = @localmem FT (NB_VAL * W,)
    shared_cnts = @localmem UInt32 (NB_VAL,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # Stage 1: Parallel initialization of shared memory histograms
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

    # Stage 2: Load coordinates and velocities into shared memory (once per block)
    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            # Load coordinates
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

            # Load velocities (shared_ui) - fully coalesced across threads
            total_ui = 2 * ni * W
            elem = lid
            while elem <= total_ui
                col = (elem - 1) % W + 1
                rem = (elem - 1) ÷ W
                k = rem % ni + 1
                c = rem ÷ ni + 1
                
                b_idx = b_base + col - 1
                gi = i0 + k - 1
                
                val = (col <= bw && k <= ni) ? u_batch[b_idx, gi, c] : zero(FT)
                @inbounds shared_ui[col + W * (k - 1) + W * ni * (c - 1)] = val
                elem += workgroup_size
            end

            # Load velocities (shared_uj) - fully coalesced across threads
            if ti < tj
                total_uj = 2 * nj * W
                elem = lid
                while elem <= total_uj
                    col = (elem - 1) % W + 1
                    rem = (elem - 1) ÷ W
                    k = rem % nj + 1
                    c = rem ÷ nj + 1
                    
                    b_idx = b_base + col - 1
                    gj = j0 + k - 1
                    
                    val = (col <= bw && k <= nj) ? u_batch[b_idx, gj, c] : zero(FT)
                    @inbounds shared_uj[col + W * (k - 1) + W * nj * (c - 1)] = val
                    elem += workgroup_size
                end
            end
        end
    end
    @synchronize

    # Stage 3: Pair loop & local accumulation
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
                # Cross tile block
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        gi = i0 + ia - 1
                        gj = j0 + jb - 1
                        X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                        X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SFGE.SF_GPU_TILE + jb])
                        dX = X2 - X1
                        dist_sq = dX[1]^2 + dX[2]^2
                        dist = sqrt(dist_sq)
                        bin = SFGE._gpu_digitize_linear(
                            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                        )
                        if 1 <= bin < N_bins
                            r̂ = dX / dist
                            if b_base == 1
                                @atomic shared_cnts[bin] += UInt32(1)
                            end
                            @inbounds for col in 1:W
                                if col <= bw
                                    idx_a1 = col + W * (ia - 1)
                                    idx_a2 = idx_a1 + W * ni
                                    idx_b1 = col + W * (jb - 1)
                                    idx_b2 = idx_b1 + W * nj
                                    U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                    U2 = SA.SVector{2, FT}(shared_uj[idx_b1], shared_uj[idx_b2])
                                    
                                    val = sf_type(U2 - U1, r̂)
                                    hist_slot = bin + (col - 1) * NB
                                    @atomic shared_sums[hist_slot] += val
                                end
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
                # Diagonal tile block
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = SFGE._pair_from_linear(p, ni)
                    gi = i0 + ia - 1
                    gj = i0 + jb - 1
                    X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SFGE.SF_GPU_TILE + ia])
                    X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SFGE.SF_GPU_TILE + jb])
                    dX = X2 - X1
                    dist_sq = dX[1]^2 + dX[2]^2
                    dist = sqrt(dist_sq)
                    bin = SFGE._gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= bin < N_bins
                        r̂ = dX / dist
                        if b_base == 1
                            @atomic shared_cnts[bin] += UInt32(1)
                        end
                        @inbounds for col in 1:W
                            if col <= bw
                                idx_a1 = col + W * (ia - 1)
                                idx_a2 = idx_a1 + W * ni
                                idx_b1 = col + W * (jb - 1)
                                idx_b2 = idx_b1 + W * ni
                                U1 = SA.SVector{2, FT}(shared_ui[idx_a1], shared_ui[idx_a2])
                                U2 = SA.SVector{2, FT}(shared_ui[idx_b1], shared_ui[idx_b2])
                                
                                val = sf_type(U2 - U1, r̂)
                                hist_slot = bin + (col - 1) * NB
                                @atomic shared_sums[hist_slot] += val
                            end
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
    @synchronize

    # Stage 4: Flush shared memory to global VRAM block-private arrays (no atomics!)
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks
        ti, tj = SFGE._tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SFGE.SF_GPU_TILE + 1
        j0 = (tj - 1) * SFGE.SF_GPU_TILE + 1
        ni = min(SFGE.SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SFGE.SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            slot = lid
            while slot <= NB * bw
                bin = (slot - 1) % NB + 1
                col = (slot - 1) ÷ NB + 1
                @inbounds begin
                    partial_sums[bin, col, block_id] = shared_sums[slot]
                end
                slot += workgroup_size
            end
            
            if b_base == 1
                slot = lid
                while slot <= NB
                    @inbounds begin
                        partial_cnts[slot, block_id] = shared_cnts[slot]
                    end
                    slot += workgroup_size
                end
            end
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_vectorized!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    workspace::Union{Nothing, BatchGPUWorkspace{FT}} = nothing,
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
        workspace.fixed_x || error("BatchGPUWorkspace must be fixed_x")
        (workspace.N == N && workspace.B == B && workspace.NB == NB) ||
            error("BatchGPUWorkspace size mismatch")
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end

    n_tiles, n_tile_blocks, ws, ndrange = _tiled_batch_launch_params(N)
    fe, le, is_, off, sv = bin_edges.first_edge, bin_edges.last_edge, bin_edges.inv_step, bin_edges.offset, bin_edges.step_val
    n_bins = NB + 1

    # W is fixed to 16 for static shared memory layout compatibility (keeps shared memory under 48KB limit)
    W = 16

    kernel! = _batch_tiled128_fixed_x_vectorized!(backend, ws)
    merge_sums! = _batch_fused_merge_usmem_sums!(backend, ws)
    merge_cnts! = _batch_fused_merge_usmem_cnts!(backend, ws)

    partial_sums = KA.zeros(backend, FT, NB, W, n_tile_blocks)
    partial_cnts = KA.zeros(backend, UInt32, NB, n_tile_blocks)
    
    b_base = 1
    while b_base <= B
        bw = min(W, B - b_base + 1)
        kernel!(
            partial_sums, partial_cnts, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws, Val(NB), Val(W);
            ndrange = ndrange,
        )

        sums_view = @view sums_dev[:, b_base:b_base+bw-1]
        nworkers_sums = NB * bw
        merge_sums!(
            sums_view, partial_sums, NB, bw, n_tile_blocks, nworkers_sums;
            ndrange = nworkers_sums,
        )

        if b_base == 1
            counts_view = @view counts_dev[:, 1]
            nworkers_cnts = NB
            merge_cnts!(
                counts_view, partial_cnts, NB, n_tile_blocks, nworkers_cnts;
                ndrange = nworkers_cnts,
            )
        end
        
        b_base += W
    end
    
    if B > 1
        # Broadcast counts from column 1 to all other columns
        counts_dev[:, 2:end] .= @view counts_dev[:, 1]
    end
    KA.synchronize(backend)

    download || return nothing
    bd = batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end
