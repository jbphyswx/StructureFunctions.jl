#!/usr/bin/env julia
# DEPRECATED: use `gpu/benchmark_batch_usmem.jl` (§8 `gpu_batch_fused_tiled_fixed_x!`).
# This file duplicated kernels with host-strip hacks; kept only for historical comparison.

using CUDA
using KernelAbstractions: KernelAbstractions as KA, @kernel, @localmem, @index, @synchronize, @atomic, @Const
using StructureFunctions
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges
include(joinpath(@__DIR__, "batch_prototypes", "BatchPrototypes.jl"))
using .BatchPrototypes: BatchPrototypes as BP

# ---------------------------------------------------------------------------
# Direct inline helper definitions
# ---------------------------------------------------------------------------
@inline function _gpu_digitize_linear(
    x::T,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    offset::T,
    step_val::T,
    n_edges::Int,
) where {T}
    if x <= first_edge
        return 0
    end
    if x > last_edge
        return n_edges
    end
    t = muladd(x, inv_step, -first_edge * inv_step)
    idx = clamp(floor(Int, t) + 1, 1, n_edges)
    edge_val = muladd(step_val, T(idx - 1), first_edge)
    search_idx = edge_val < x ? idx + 1 : idx
    return search_idx - 1
end

@inline function _tile_from_linear(k, n_tiles)
    ti = one(k)
    rleft = k - one(k)
    while ti < n_tiles && rleft >= n_tiles - ti + one(k)
        rleft -= n_tiles - ti + one(k)
        ti += one(k)
    end
    tj = ti + rleft
    return ti, tj
end

@inline function _pair_from_linear(k, N)
    term = Float32(4 * N * N - 4 * N + 1 - 8 * (k - 1))
    i_float = (Float32(2 * N - 1) - sqrt(max(0.0f0, term))) * 0.5f0
    i = floor(Int, i_float) + 1
    j = k - (i - 1) * N + (i - 1) * i ÷ 2 + i
    return i, j
end

const SF_GPU_TILE = 128
const SA = BP.SA

# ---------------------------------------------------------------------------
# Method 1: Batched 1D Structure Function (Fixed-x & Varying-x)
# ---------------------------------------------------------------------------
@kernel function _batch_1d_fixed_x_kernel!(
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
    ::Val{NDIMS},
    ::Val{NB_VAL},
    ::Val{W_VAL},
) where {NDIMS, NB_VAL, W_VAL, FT}
    shared_xi = @localmem FT (128 * NDIMS,)
    shared_xj = @localmem FT (128 * NDIMS,)
    
    stride_u = 128 * NDIMS
    shared_ui = @localmem FT (128 * NDIMS * W_VAL,)
    shared_uj = @localmem FT (128 * NDIMS * W_VAL,)

    shared_sums = @localmem FT (NB_VAL * W_VAL,)
    shared_cnts = @localmem UInt32 (NB_VAL,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # Stage 1: Zero shared memory histograms
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

    # Stage 2: Load coordinates and fields into shared memory
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        
        if ni > 0 && nj > 0
            # Load x_mat
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    for d in 1:NDIMS
                        shared_xi[(d - 1) * 128 + k] = x_mat[d, gi]
                    end
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        for d in 1:NDIMS
                            shared_xj[(d - 1) * 128 + k] = x_mat[d, gj]
                        end
                    end
                    k += workgroup_size
                end
            end

            # Load u_batch into shared_ui
            elem = lid
            while elem <= stride_u * bw
                col = (elem - 1) ÷ stride_u + 1
                rem = (elem - 1) % stride_u
                pt = rem ÷ NDIMS + 1
                c = rem % NDIMS + 1
                gi = i0 + pt - 1
                b_idx = b_base + col - 1
                if pt <= ni
                    shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end

            # Load u_batch into shared_uj (cross-tile only)
            if ti < tj
                elem = lid
                while elem <= stride_u * bw
                    col = (elem - 1) ÷ stride_u + 1
                    rem = (elem - 1) % stride_u
                    pt = rem ÷ NDIMS + 1
                    c = rem % NDIMS + 1
                    gj = j0 + pt - 1
                    b_idx = b_base + col - 1
                    if pt <= nj
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

    # Stage 3: Pair loop & local accumulation
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
            if ti < tj
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        X1 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                        else
                            SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                        end
                        X2 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xj[jb], shared_xj[128 + jb])
                        else
                            SA.SVector{3, FT}(shared_xj[jb], shared_xj[128 + jb], shared_xj[256 + jb])
                        end
                        dX = X2 - X1
                        dist_sq = zero(FT)
                        for d in 1:NDIMS
                            dist_sq += dX[d]^2
                        end
                        dist = sqrt(dist_sq)
                        bin = _gpu_digitize_linear(
                            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                        )
                        if 1 <= bin < N_bins
                            r̂ = dX / dist
                            if b_base == 1
                                @atomic shared_cnts[bin] += UInt32(1)
                            end
                            @inbounds for col in 1:bw
                                U1 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia],
                                        shared_ui[(col - 1) * stride_u + 256 + ia]
                                    )
                                end
                                U2 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb],
                                        shared_uj[(col - 1) * stride_u + 256 + jb]
                                    )
                                end
                                
                                val = sf_type(U2 - U1, r̂)
                                hist_slot = bin + (col - 1) * NB
                                @atomic shared_sums[hist_slot] += val
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
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                    else
                        SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                    end
                    X2 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[jb], shared_xi[128 + jb])
                    else
                        SA.SVector{3, FT}(shared_xi[jb], shared_xi[128 + jb], shared_xi[256 + jb])
                    end
                    dX = X2 - X1
                    dist_sq = zero(FT)
                    for d in 1:NDIMS
                        dist_sq += dX[d]^2
                    end
                    dist = sqrt(dist_sq)
                    bin = _gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= bin < N_bins
                        r̂ = dX / dist
                        if b_base == 1
                            @atomic shared_cnts[bin] += UInt32(1)
                        end
                        @inbounds for col in 1:bw
                            U1 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia],
                                    shared_ui[(col - 1) * stride_u + 256 + ia]
                                )
                            end
                            U2 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb],
                                    shared_ui[(col - 1) * stride_u + 256 + jb]
                                )
                            end
                            
                            val = sf_type(U2 - U1, r̂)
                            hist_slot = bin + (col - 1) * NB
                            @atomic shared_sums[hist_slot] += val
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
    @synchronize

    # Stage 4: Flush shared memory to global VRAM
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        ni = min(SF_GPU_TILE, N_points - (ti - 1) * SF_GPU_TILE)
        nj = min(SF_GPU_TILE, N_points - (tj - 1) * SF_GPU_TILE)
        if ni > 0 && nj > 0
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
                    if c_val > 0
                        @atomic counts[slot, 1] += c_val
                    end
                    slot += workgroup_size
                end
            end
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_optimal!(
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
    B = BP.batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_mat, 2)
    NDIMS = size(x_mat, 1)
    
    if workspace === nothing
        sums_dev = KA.adapt(backend, zeros(FT, NB, B))
        counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
        x_dev = KA.adapt(backend, x_mat)
        u_dev = KA.adapt(backend, u_batch)
    else
        sums_dev = workspace.sums_dev
        counts_dev = workspace.counts_dev
        if workspace.x_dev === nothing
            BP.upload_batch!(workspace, backend, x_mat, u_batch)
        end
        x_dev, u_dev = workspace.x_dev, workspace.u_dev
    end

    n_tiles, n_tile_blocks, ws, ndrange = BP._tiled_batch_launch_params(N)
    fe, le, is_, off, sv = bin_edges.first_edge, bin_edges.last_edge, bin_edges.inv_step, bin_edges.offset, bin_edges.step_val
    n_bins = NB + 1

    # Choose W dynamically based on SMEM limits
    # max_smem = 48 KB (48000 bytes). Total SMEM per block:
    # 2 * 128 * NDIMS * sizeof(FT) coords
    # + 2 * 128 * NDIMS * W * sizeof(FT) (velocities: shared_ui and shared_uj)
    # + NB * W * sizeof(FT) (histogram: shared_sums)
    # + NB * sizeof(UInt32) (counts: shared_cnts)
    # 2 * 128 * NDIMS * sizeof(FT) + NB * sizeof(UInt32) + W * ((2 * 128 * NDIMS + NB) * sizeof(FT)) <= 48000
    constant_smem = 2 * 128 * NDIMS * sizeof(FT) + NB * sizeof(UInt32)
    smem_per_W = (2 * 128 * NDIMS + NB) * sizeof(FT)
    W = floor(Int, (48000 - constant_smem) / smem_per_W)
    W = clamp(W, 1, 32)

    kernel! = _batch_1d_fixed_x_kernel!(backend, ws)
    
    b_base = 1
    while b_base <= B
        bw = min(W, B - b_base + 1)
        kernel!(
            sums_dev, counts_dev, x_dev, u_dev, sf_type,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws, Val(NDIMS), Val(NB), Val(W);
            ndrange = ndrange,
        )
        b_base += W
    end
    
    if B > 1
        counts_dev[:, 2:end] .= @view counts_dev[:, 1]
    end
    KA.synchronize(backend)

    download || return nothing
    bd = BP.batch_dims(u_batch)
    copy!(sums, reshape(Array(sums_dev), NB, bd...))
    copy!(counts, reshape(Array(counts_dev), NB, bd...))
    return nothing
end

# ---------------------------------------------------------------------------
# Method 2: Batched 2D Joint Structure Function (Fixed-x) using Direct Global Atomics
# ---------------------------------------------------------------------------
@kernel function _batch_2d_fixed_x_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_dist_edges::Int,
    N_val_edges::Int,
    NV::Int,
    b_base::Int,
    bw::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    val_first::FT,
    val_last::FT,
    val_inv_step::FT,
    val_offset::FT,
    val_step::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
    ::Val{NDIMS},
    ::Val{W_VAL},
) where {NDIMS, W_VAL, FT}
    shared_xi = @localmem FT (128 * NDIMS,)
    shared_xj = @localmem FT (128 * NDIMS,)
    
    stride_u = 128 * NDIMS
    shared_ui = @localmem FT (128 * NDIMS * W_VAL,)
    shared_uj = @localmem FT (128 * NDIMS * W_VAL,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # Stage 1: Load coordinates and fields into shared memory
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
                    for d in 1:NDIMS
                        shared_xi[(d - 1) * 128 + k] = x_mat[d, gi]
                    end
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        for d in 1:NDIMS
                            shared_xj[(d - 1) * 128 + k] = x_mat[d, gj]
                        end
                    end
                    k += workgroup_size
                end
            end

            elem = lid
            while elem <= stride_u * bw
                col = (elem - 1) ÷ stride_u + 1
                rem = (elem - 1) % stride_u
                pt = rem ÷ NDIMS + 1
                c = rem % NDIMS + 1
                gi = i0 + pt - 1
                b_idx = b_base + col - 1
                if pt <= ni
                    shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end

            if ti < tj
                elem = lid
                while elem <= stride_u * bw
                    col = (elem - 1) ÷ stride_u + 1
                    rem = (elem - 1) % stride_u
                    pt = rem ÷ NDIMS + 1
                    c = rem % NDIMS + 1
                    gj = j0 + pt - 1
                    b_idx = b_base + col - 1
                    if pt <= nj
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

    # Stage 2: Pair loop & direct global atomics
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
            if ti < tj
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        X1 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                        else
                            SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                        end
                        X2 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xj[jb], shared_xj[128 + jb])
                        else
                            SA.SVector{3, FT}(shared_xj[jb], shared_xj[128 + jb], shared_xj[256 + jb])
                        end
                        dX = X2 - X1
                        dist_sq = zero(FT)
                        for d in 1:NDIMS
                            dist_sq += dX[d]^2
                        end
                        dist = sqrt(dist_sq)
                        bin = _gpu_digitize_linear(
                            dist, first_edge, last_edge, inv_step, offset, step_val, N_dist_edges,
                        )
                        if 1 <= bin < N_dist_edges
                            r̂ = dX / dist
                            @inbounds for col in 1:bw
                                U1 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia],
                                        shared_ui[(col - 1) * stride_u + 256 + ia]
                                    )
                                end
                                U2 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb],
                                        shared_uj[(col - 1) * stride_u + 256 + jb]
                                    )
                                end
                                
                                val = sf_type(U2 - U1, r̂)
                                vbin = _gpu_digitize_linear(
                                    val, val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
                                )
                                if 1 <= vbin < N_val_edges
                                    b_idx = b_base + col - 1
                                    @atomic output_sums[bin, vbin, b_idx] += val
                                    @atomic output_counts[bin, vbin, b_idx] += UInt32(1)
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
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                    else
                        SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                    end
                    X2 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[jb], shared_xi[128 + jb])
                    else
                        SA.SVector{3, FT}(shared_xi[jb], shared_xi[128 + jb], shared_xi[256 + jb])
                    end
                    dX = X2 - X1
                    dist_sq = zero(FT)
                    for d in 1:NDIMS
                        dist_sq += dX[d]^2
                    end
                    dist = sqrt(dist_sq)
                    bin = _gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_dist_edges,
                    )
                    if 1 <= bin < N_dist_edges
                        r̂ = dX / dist
                        @inbounds for col in 1:bw
                            U1 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia],
                                    shared_ui[(col - 1) * stride_u + 256 + ia]
                                )
                            end
                            U2 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb],
                                    shared_ui[(col - 1) * stride_u + 256 + jb]
                                )
                            end
                            
                            val = sf_type(U2 - U1, r̂)
                            vbin = _gpu_digitize_linear(
                                val, val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
                            )
                            if 1 <= vbin < N_val_edges
                                b_idx = b_base + col - 1
                                @atomic output_sums[bin, vbin, b_idx] += val
                                @atomic output_counts[bin, vbin, b_idx] += UInt32(1)
                            end
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_optimal_2d!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    dist_edges::LinearBinEdges{FT},
    val_edges::LinearBinEdges{FT};
    download::Bool = true,
) where {FT}
    B = BP.batch_size(u_batch)
    n_dist = length(dist_edges.edges) - 1
    n_val = length(val_edges.edges) - 1
    N = size(x_mat, 2)
    NDIMS = size(x_mat, 1)

    sums_dev = KA.adapt(backend, zeros(FT, n_dist, n_val, B))
    counts_dev = KA.adapt(backend, zeros(UInt32, n_dist, n_val, B))
    x_dev = KA.adapt(backend, x_mat)
    u_dev = KA.adapt(backend, u_batch)

    n_tiles, n_tile_blocks, ws, ndrange = BP._tiled_batch_launch_params(N)
    fe, le, is_, off, sv = dist_edges.first_edge, dist_edges.last_edge, dist_edges.inv_step, dist_edges.offset, dist_edges.step_val
    v_fe, v_le, v_is, v_off, v_sv = val_edges.first_edge, val_edges.last_edge, val_edges.inv_step, val_edges.offset, val_edges.step_val

    # Size W safely for SMEM: coords (2 * 128 * NDIMS * sizeof(FT)) + W * velocities (2 * 128 * NDIMS * sizeof(FT)) <= 48000
    constant_smem = 2 * 128 * NDIMS * sizeof(FT)
    smem_per_W = (2 * 128 * NDIMS) * sizeof(FT)
    W = floor(Int, (48000 - constant_smem) / smem_per_W)
    W = clamp(W, 1, 32)
    kernel! = _batch_2d_fixed_x_kernel!(backend, ws)

    b_base = 1
    while b_base <= B
        bw = min(W, B - b_base + 1)
        kernel!(
            sums_dev, counts_dev, x_dev, u_dev, sf_type,
            N, n_dist + 1, n_val + 1, n_val, b_base, bw,
            fe, le, is_, off, sv,
            v_fe, v_le, v_is, v_off, v_sv,
            n_tiles, n_tile_blocks, ws, Val(NDIMS), Val(W);
            ndrange = ndrange,
        )
        b_base += W
    end
    KA.synchronize(backend)

    download || return nothing
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

# ---------------------------------------------------------------------------
# Method 3: Batched Single-Pass 1D (Fixed-x)
# ---------------------------------------------------------------------------
@kernel function _batch_sp1d_fixed_x_kernel!(
    output_sums,
    output_counts,
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
    ::Val{NDIMS},
    ::Val{NB_VAL},
    ::Val{W_VAL},
) where {NDIMS, NB_VAL, W_VAL, FT}
    shared_xi = @localmem FT (128 * NDIMS,)
    shared_xj = @localmem FT (128 * NDIMS,)
    
    stride_u = 128 * NDIMS
    shared_ui = @localmem FT (128 * NDIMS * W_VAL,)
    shared_uj = @localmem FT (128 * NDIMS * W_VAL,)

    shared_sums = @localmem FT (8 * NB_VAL * W_VAL,)
    shared_cnts = @localmem UInt32 (NB_VAL * W_VAL,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # Stage 1: Zero shared memory histograms
    slot = lid
    while slot <= 8 * NB * bw
        @inbounds shared_sums[slot] = zero(FT)
        slot += workgroup_size
    end
    slot = lid
    while slot <= NB * bw
        @inbounds shared_cnts[slot] = UInt32(0)
        slot += workgroup_size
    end
    @synchronize

    # Stage 2: Load coordinates and fields into shared memory
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
                    for d in 1:NDIMS
                        shared_xi[(d - 1) * 128 + k] = x_mat[d, gi]
                    end
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        for d in 1:NDIMS
                            shared_xj[(d - 1) * 128 + k] = x_mat[d, gj]
                        end
                    end
                    k += workgroup_size
                end
            end

            elem = lid
            while elem <= stride_u * bw
                col = (elem - 1) ÷ stride_u + 1
                rem = (elem - 1) % stride_u
                pt = rem ÷ NDIMS + 1
                c = rem % NDIMS + 1
                gi = i0 + pt - 1
                b_idx = b_base + col - 1
                if pt <= ni
                    shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end

            if ti < tj
                elem = lid
                while elem <= stride_u * bw
                    col = (elem - 1) ÷ stride_u + 1
                    rem = (elem - 1) % stride_u
                    pt = rem ÷ NDIMS + 1
                    c = rem % NDIMS + 1
                    gj = j0 + pt - 1
                    b_idx = b_base + col - 1
                    if pt <= nj
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

    # Stage 3: Pair loop & local accumulation
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
            if ti < tj
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        X1 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                        else
                            SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                        end
                        X2 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xj[jb], shared_xj[128 + jb])
                        else
                            SA.SVector{3, FT}(shared_xj[jb], shared_xj[128 + jb], shared_xj[256 + jb])
                        end
                        dX = X2 - X1
                        dist_sq = zero(FT)
                        for d in 1:NDIMS
                            dist_sq += dX[d]^2
                        end
                        dist = sqrt(dist_sq)
                        bin = _gpu_digitize_linear(
                            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                        )
                        if 1 <= bin < N_bins
                            r̂ = dX / dist
                            n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                            @inbounds for col in 1:bw
                                U1 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia],
                                        shared_ui[(col - 1) * stride_u + 256 + ia]
                                    )
                                end
                                U2 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb],
                                        shared_uj[(col - 1) * stride_u + 256 + jb]
                                    )
                                end
                                
                                dU = U2 - U1
                                du_L = SA.dot(dU, r̂)
                                du_T = SA.dot(dU, n̂)
                                du_L2 = du_L * du_L
                                du_T2 = du_T * du_T
                                
                                base_idx = (col - 1) * 8 * NB + (bin - 1) * 8
                                @atomic shared_sums[base_idx + 1] += du_L2 + du_T2
                                @atomic shared_sums[base_idx + 2] += du_L2
                                @atomic shared_sums[base_idx + 3] += du_T2
                                @atomic shared_sums[base_idx + 4] += du_L * (du_L2 + du_T2)
                                @atomic shared_sums[base_idx + 5] += du_L * du_L2
                                @atomic shared_sums[base_idx + 6] += du_L2 * du_T
                                @atomic shared_sums[base_idx + 7] += du_L * du_T2
                                @atomic shared_sums[base_idx + 8] += du_T * du_T2
                                
                                @atomic shared_cnts[(col - 1) * NB + bin] += UInt32(1)
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
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                    else
                        SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                    end
                    X2 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[jb], shared_xi[128 + jb])
                    else
                        SA.SVector{3, FT}(shared_xi[jb], shared_xi[128 + jb], shared_xi[256 + jb])
                    end
                    dX = X2 - X1
                    dist_sq = zero(FT)
                    for d in 1:NDIMS
                        dist_sq += dX[d]^2
                    end
                    dist = sqrt(dist_sq)
                    bin = _gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= bin < N_bins
                        r̂ = dX / dist
                        n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                        @inbounds for col in 1:bw
                            U1 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia],
                                    shared_ui[(col - 1) * stride_u + 256 + ia]
                                )
                            end
                            U2 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb],
                                    shared_ui[(col - 1) * stride_u + 256 + jb]
                                )
                            end
                            
                            dU = U2 - U1
                            du_L = SA.dot(dU, r̂)
                            du_T = SA.dot(dU, n̂)
                            du_L2 = du_L * du_L
                            du_T2 = du_T * du_T
                            
                            base_idx = (col - 1) * 8 * NB + (bin - 1) * 8
                            @atomic shared_sums[base_idx + 1] += du_L2 + du_T2
                            @atomic shared_sums[base_idx + 2] += du_L2
                            @atomic shared_sums[base_idx + 3] += du_T2
                            @atomic shared_sums[base_idx + 4] += du_L * (du_L2 + du_T2)
                            @atomic shared_sums[base_idx + 5] += du_L * du_L2
                            @atomic shared_sums[base_idx + 6] += du_L2 * du_T
                            @atomic shared_sums[base_idx + 7] += du_L * du_T2
                            @atomic shared_sums[base_idx + 8] += du_T * du_T2
                            
                            @atomic shared_cnts[(col - 1) * NB + bin] += UInt32(1)
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
    @synchronize

    # Stage 4: Flush shared memory to global VRAM
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    if bid <= n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        ni = min(SF_GPU_TILE, N_points - (ti - 1) * SF_GPU_TILE)
        nj = min(SF_GPU_TILE, N_points - (tj - 1) * SF_GPU_TILE)
        if ni > 0 && nj > 0
            slot = lid
            while slot <= 8 * NB * bw
                t = (slot - 1) % 8 + 1
                rem_slot = (slot - 1) ÷ 8
                bin = rem_slot % NB + 1
                col = rem_slot ÷ NB + 1
                b = b_base + col - 1
                s_val = shared_sums[slot]
                if s_val != zero(FT)
                    @atomic output_sums[t, bin, b] += s_val
                end
                slot += workgroup_size
            end
            
            slot = lid
            while slot <= NB * bw
                bin = (slot - 1) % NB + 1
                col = (slot - 1) ÷ NB + 1
                b = b_base + col - 1
                c_val = shared_cnts[slot]
                if c_val > 0
                    for t in 1:8
                        @atomic output_counts[t, bin, b] += c_val
                    end
                end
                slot += workgroup_size
            end
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_optimal_sp1d!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    bin_edges::LinearBinEdges{FT};
    download::Bool = true,
) where {FT}
    B = BP.batch_size(u_batch)
    NB = length(bin_edges.edges) - 1
    N = size(x_mat, 2)
    NDIMS = size(x_mat, 1)

    sums_dev = KA.adapt(backend, zeros(FT, 8, NB, B))
    counts_dev = KA.adapt(backend, zeros(UInt32, 8, NB, B))
    x_dev = KA.adapt(backend, x_mat)
    u_dev = KA.adapt(backend, u_batch)

    n_tiles, n_tile_blocks, ws, ndrange = BP._tiled_batch_launch_params(N)
    fe, le, is_, off, sv = bin_edges.first_edge, bin_edges.last_edge, bin_edges.inv_step, bin_edges.offset, bin_edges.step_val
    n_bins = NB + 1

    # Choose W safely for SMEM: coords (2 * 128 * NDIMS * sizeof(FT)) + W * (velocities (2 * 128 * NDIMS * sizeof(FT)) + sums (8 * NB * sizeof(FT)) + counts (NB * sizeof(UInt32))) <= 48000
    constant_smem = 2 * 128 * NDIMS * sizeof(FT)
    smem_per_W = (2 * 128 * NDIMS + 8 * NB + NB) * sizeof(FT)
    W = floor(Int, (48000 - constant_smem) / smem_per_W)
    W = clamp(W, 1, 32)

    kernel! = _batch_sp1d_fixed_x_kernel!(backend, ws)

    b_base = 1
    while b_base <= B
        bw = min(W, B - b_base + 1)
        kernel!(
            sums_dev, counts_dev, x_dev, u_dev,
            N, n_bins, NB, b_base, bw, fe, le, is_, off, sv,
            n_tiles, n_tile_blocks, ws, Val(NDIMS), Val(NB), Val(W);
            ndrange = ndrange,
        )
        b_base += W
    end
    KA.synchronize(backend)

    download || return nothing
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

# ---------------------------------------------------------------------------
# Method 4: Batched Single-Pass 2D (Fixed-x) using Direct Global Atomics
# ---------------------------------------------------------------------------
@kernel function _batch_sp2d_fixed_x_kernel!(
    output_sums,
    output_counts,
    @Const(x_mat),
    @Const(u_batch),
    N_points::Int,
    N_dist_edges::Int,
    N_val_edges::Int,
    NV::Int,
    b_base::Int,
    bw::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    val_first::FT,
    val_last::FT,
    val_inv_step::FT,
    val_offset::FT,
    val_step::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
    ::Val{NDIMS},
    ::Val{W_VAL},
) where {NDIMS, W_VAL, FT}
    shared_xi = @localmem FT (128 * NDIMS,)
    shared_xj = @localmem FT (128 * NDIMS,)
    
    stride_u = 128 * NDIMS
    shared_ui = @localmem FT (128 * NDIMS * W_VAL,)
    shared_uj = @localmem FT (128 * NDIMS * W_VAL,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1

    # Stage 1: Load coordinates and fields into shared memory
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
                    for d in 1:NDIMS
                        shared_xi[(d - 1) * 128 + k] = x_mat[d, gi]
                    end
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        for d in 1:NDIMS
                            shared_xj[(d - 1) * 128 + k] = x_mat[d, gj]
                        end
                    end
                    k += workgroup_size
                end
            end

            elem = lid
            while elem <= stride_u * bw
                col = (elem - 1) ÷ stride_u + 1
                rem = (elem - 1) % stride_u
                pt = rem ÷ NDIMS + 1
                c = rem % NDIMS + 1
                gi = i0 + pt - 1
                b_idx = b_base + col - 1
                if pt <= ni
                    shared_ui[elem] = u_batch[b_idx, gi, c]
                else
                    shared_ui[elem] = zero(FT)
                end
                elem += workgroup_size
            end

            if ti < tj
                elem = lid
                while elem <= stride_u * bw
                    col = (elem - 1) ÷ stride_u + 1
                    rem = (elem - 1) % stride_u
                    pt = rem ÷ NDIMS + 1
                    c = rem % NDIMS + 1
                    gj = j0 + pt - 1
                    b_idx = b_base + col - 1
                    if pt <= nj
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

    # Stage 2: Pair loop & direct global atomics
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
            if ti < tj
                n_pairs = ni * nj
                p = lid
                if p <= n_pairs
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    while p <= n_pairs
                        X1 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                        else
                            SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                        end
                        X2 = if NDIMS == 2
                            SA.SVector{2, FT}(shared_xj[jb], shared_xj[128 + jb])
                        else
                            SA.SVector{3, FT}(shared_xj[jb], shared_xj[128 + jb], shared_xj[256 + jb])
                        end
                        dX = X2 - X1
                        dist_sq = zero(FT)
                        for d in 1:NDIMS
                            dist_sq += dX[d]^2
                        end
                        dist = sqrt(dist_sq)
                        bin = _gpu_digitize_linear(
                            dist, first_edge, last_edge, inv_step, offset, step_val, N_dist_edges,
                        )
                        if 1 <= bin < N_dist_edges
                            r̂ = dX / dist
                            n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                            @inbounds for col in 1:bw
                                U1 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_ui[(col - 1) * stride_u + ia],
                                        shared_ui[(col - 1) * stride_u + 128 + ia],
                                        shared_ui[(col - 1) * stride_u + 256 + ia]
                                    )
                                end
                                U2 = if NDIMS == 2
                                    SA.SVector{2, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb]
                                    )
                                else
                                    SA.SVector{3, FT}(
                                        shared_uj[(col - 1) * stride_u + jb],
                                        shared_uj[(col - 1) * stride_u + 128 + jb],
                                        shared_uj[(col - 1) * stride_u + 256 + jb]
                                    )
                                end
                                
                                dU = U2 - U1
                                du_L = SA.dot(dU, r̂)
                                du_T = SA.dot(dU, n̂)
                                du_L2 = du_L * du_L
                                du_T2 = du_T * du_T

                                # 8 computed types
                                val1 = du_L2 + du_T2
                                val2 = du_L2
                                val3 = du_T2
                                val4 = du_L * (du_L2 + du_T2)
                                val5 = du_L * du_L2
                                val6 = du_L2 * du_T
                                val7 = du_L * du_T2
                                val8 = du_T * du_T2

                                # Digitize val1 (used as the value bin)
                                vbin = _gpu_digitize_linear(
                                    val1, val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
                                )
                                if 1 <= vbin < N_val_edges
                                    b_idx = b_base + col - 1
                                    
                                    @atomic output_sums[1, bin, vbin, b_idx] += val1
                                    @atomic output_sums[2, bin, vbin, b_idx] += val2
                                    @atomic output_sums[3, bin, vbin, b_idx] += val3
                                    @atomic output_sums[4, bin, vbin, b_idx] += val4
                                    @atomic output_sums[5, bin, vbin, b_idx] += val5
                                    @atomic output_sums[6, bin, vbin, b_idx] += val6
                                    @atomic output_sums[7, bin, vbin, b_idx] += val7
                                    @atomic output_sums[8, bin, vbin, b_idx] += val8

                                    @atomic output_counts[1, bin, vbin, b_idx] += UInt32(1)
                                    @atomic output_counts[2, bin, vbin, b_idx] += UInt32(1)
                                    @atomic output_counts[3, bin, vbin, b_idx] += UInt32(1)
                                    @atomic output_counts[4, bin, vbin, b_idx] += UInt32(1)
                                    @atomic output_counts[5, bin, vbin, b_idx] += UInt32(1)
                                    @atomic output_counts[6, bin, vbin, b_idx] += UInt32(1)
                                    @atomic output_counts[7, bin, vbin, b_idx] += UInt32(1)
                                    @atomic output_counts[8, bin, vbin, b_idx] += UInt32(1)
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
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[ia], shared_xi[128 + ia])
                    else
                        SA.SVector{3, FT}(shared_xi[ia], shared_xi[128 + ia], shared_xi[256 + ia])
                    end
                    X2 = if NDIMS == 2
                        SA.SVector{2, FT}(shared_xi[jb], shared_xi[128 + jb])
                    else
                        SA.SVector{3, FT}(shared_xi[jb], shared_xi[128 + jb], shared_xi[256 + jb])
                    end
                    dX = X2 - X1
                    dist_sq = zero(FT)
                    for d in 1:NDIMS
                        dist_sq += dX[d]^2
                    end
                    dist = sqrt(dist_sq)
                    bin = _gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_dist_edges,
                    )
                    if 1 <= bin < N_dist_edges
                        r̂ = dX / dist
                        n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                        @inbounds for col in 1:bw
                            U1 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + ia],
                                    shared_ui[(col - 1) * stride_u + 128 + ia],
                                    shared_ui[(col - 1) * stride_u + 256 + ia]
                                )
                            end
                            U2 = if NDIMS == 2
                                SA.SVector{2, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb]
                                )
                            else
                                SA.SVector{3, FT}(
                                    shared_ui[(col - 1) * stride_u + jb],
                                    shared_ui[(col - 1) * stride_u + 128 + jb],
                                    shared_ui[(col - 1) * stride_u + 256 + jb]
                                )
                            end
                            
                            dU = U2 - U1
                            du_L = SA.dot(dU, r̂)
                            du_T = SA.dot(dU, n̂)
                            du_L2 = du_L * du_L
                            du_T2 = du_T * du_T

                            val1 = du_L2 + du_T2
                            val2 = du_L2
                            val3 = du_T2
                            val4 = du_L * (du_L2 + du_T2)
                            val5 = du_L * du_L2
                            val6 = du_L2 * du_T
                            val7 = du_L * du_T2
                            val8 = du_T * du_T2

                            vbin = _gpu_digitize_linear(
                                val1, val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
                            )
                            if 1 <= vbin < N_val_edges
                                b_idx = b_base + col - 1
                                
                                @atomic output_sums[1, bin, vbin, b_idx] += val1
                                @atomic output_sums[2, bin, vbin, b_idx] += val2
                                @atomic output_sums[3, bin, vbin, b_idx] += val3
                                @atomic output_sums[4, bin, vbin, b_idx] += val4
                                @atomic output_sums[5, bin, vbin, b_idx] += val5
                                @atomic output_sums[6, bin, vbin, b_idx] += val6
                                @atomic output_sums[7, bin, vbin, b_idx] += val7
                                @atomic output_sums[8, bin, vbin, b_idx] += val8

                                @atomic output_counts[1, bin, vbin, b_idx] += UInt32(1)
                                @atomic output_counts[2, bin, vbin, b_idx] += UInt32(1)
                                @atomic output_counts[3, bin, vbin, b_idx] += UInt32(1)
                                @atomic output_counts[4, bin, vbin, b_idx] += UInt32(1)
                                @atomic output_counts[5, bin, vbin, b_idx] += UInt32(1)
                                @atomic output_counts[6, bin, vbin, b_idx] += UInt32(1)
                                @atomic output_counts[7, bin, vbin, b_idx] += UInt32(1)
                                @atomic output_counts[8, bin, vbin, b_idx] += UInt32(1)
                            end
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
end

function gpu_batch_fused_tiled_fixed_x_optimal_sp2d!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    dist_edges::LinearBinEdges{FT},
    val_edges::LinearBinEdges{FT};
    download::Bool = true,
) where {FT}
    B = BP.batch_size(u_batch)
    n_dist = length(dist_edges.edges) - 1
    n_val = length(val_edges.edges) - 1
    N = size(x_mat, 2)
    NDIMS = size(x_mat, 1)

    sums_dev = KA.adapt(backend, zeros(FT, 8, n_dist, n_val, B))
    counts_dev = KA.adapt(backend, zeros(UInt32, 8, n_dist, n_val, B))
    x_dev = KA.adapt(backend, x_mat)
    u_dev = KA.adapt(backend, u_batch)

    n_tiles, n_tile_blocks, ws, ndrange = BP._tiled_batch_launch_params(N)
    fe, le, is_, off, sv = dist_edges.first_edge, dist_edges.last_edge, dist_edges.inv_step, dist_edges.offset, dist_edges.step_val
    v_fe, v_le, v_is, v_off, v_sv = val_edges.first_edge, val_edges.last_edge, val_edges.inv_step, val_edges.offset, val_edges.step_val

    # Size W safely for SMEM: coords (2 * 128 * NDIMS * sizeof(FT)) + W * velocities (2 * 128 * NDIMS * sizeof(FT)) <= 48000
    constant_smem = 2 * 128 * NDIMS * sizeof(FT)
    smem_per_W = (2 * 128 * NDIMS) * sizeof(FT)
    W = floor(Int, (48000 - constant_smem) / smem_per_W)
    W = clamp(W, 1, 32)
    kernel! = _batch_sp2d_fixed_x_kernel!(backend, ws)

    b_base = 1
    while b_base <= B
        bw = min(W, B - b_base + 1)
        kernel!(
            sums_dev, counts_dev, x_dev, u_dev,
            N, n_dist + 1, n_val + 1, n_val, b_base, bw,
            fe, le, is_, off, sv,
            v_fe, v_le, v_is, v_off, v_sv,
            n_tiles, n_tile_blocks, ws, Val(NDIMS), Val(W);
            ndrange = ndrange,
        )
        b_base += W
    end
    KA.synchronize(backend)

    download || return nothing
    copy!(sums, Array(sums_dev))
    copy!(counts, Array(counts_dev))
    return nothing
end

# ---------------------------------------------------------------------------
# Main Harness
# ---------------------------------------------------------------------------
function main()
    FT = Float32
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    val_edges = LinearBinEdges(range(FT(0.0), FT(1.0); length = 11))
    NB = length(bin_edges.edges) - 1
    n_val = length(val_edges.edges) - 1
    backend = CUDA.CUDABackend()

    # =======================================================================
    # STAGE 1: Small-scale Validation (N=1000, B=64)
    # =======================================================================
    println("=== STAGE 1: Correctness Validation (Small-Scale: N=1000, B=64) ===")
    N_val = 1000
    B_val = 64
    x_val, u_val = BP.make_random_batch_problem(FT, N_val, (B_val,); fixed_x = true, seed = 1)
    
    ws_val = BP.BatchGPUWorkspace(backend, FT, N_val, B_val, NB; fixed_x = true)
    BP.upload_batch!(ws_val, backend, x_val, u_val)

    # 1. Method 1 validation
    sums_val_ref = zeros(FT, NB, B_val)
    counts_val_ref = zeros(UInt32, NB, B_val)
    BP.gpu_batch_fused_tiled_fixed_x_usmem_priv!(
        backend, sums_val_ref, counts_val_ref, x_val, u_val, sft, bin_edges;
        workspace = ws_val, download = true
    )

    sums_val_opt = zeros(FT, NB, B_val)
    counts_val_opt = zeros(UInt32, NB, B_val)
    gpu_batch_fused_tiled_fixed_x_optimal!(
        backend, sums_val_opt, counts_val_opt, x_val, u_val, sft, bin_edges;
        workspace = ws_val, download = true
    )

    m1_sums_diff = maximum(abs.(sums_val_ref .- sums_val_opt))
    m1_counts_diff = maximum(abs.(counts_val_ref .- counts_val_opt))
    println("  Method 1 vs Reference:")
    println("    Max abs sums diff:   ", m1_sums_diff)
    println("    Max abs counts diff: ", m1_counts_diff)
    println("    Parity passed:       ", m1_sums_diff < 1e-4 && m1_counts_diff == 0)

    # 2. Method 2 validation
    sums_val_2d = zeros(FT, NB, n_val, B_val)
    counts_val_2d = zeros(UInt32, NB, n_val, B_val)
    gpu_batch_fused_tiled_fixed_x_optimal_2d!(
        backend, sums_val_2d, counts_val_2d, x_val, u_val, sft, bin_edges, val_edges;
        download = true
    )
    # Check that Method 2 sums along value dimension equal Method 1 sums
    sums_val_2d_proj = dropdims(sum(sums_val_2d; dims=2); dims=2)
    m2_proj_diff = maximum(abs.(sums_val_opt .- sums_val_2d_proj))
    println("  Method 2 (2D Joint projection) vs Method 1:")
    println("    Max abs sums diff:   ", m2_proj_diff)

    # 3. Method 3 validation
    sums_val_sp1d = zeros(FT, 8, NB, B_val)
    counts_val_sp1d = zeros(UInt32, 8, NB, B_val)
    gpu_batch_fused_tiled_fixed_x_optimal_sp1d!(
        backend, sums_val_sp1d, counts_val_sp1d, x_val, u_val, bin_edges;
        download = true
    )
    # Compare type 1 (L2) of SP1D with Method 1
    m3_sp1d_diff = maximum(abs.(sums_val_opt .- @view(sums_val_sp1d[1, :, :])))
    println("  Method 3 (SP1D type 1 L2) vs Method 1:")
    println("    Max abs sums diff:   ", m3_sp1d_diff)

    # 4. Method 4 validation
    sums_val_sp2d = zeros(FT, 8, NB, n_val, B_val)
    counts_val_sp2d = zeros(UInt32, 8, NB, n_val, B_val)
    gpu_batch_fused_tiled_fixed_x_optimal_sp2d!(
        backend, sums_val_sp2d, counts_val_sp2d, x_val, u_val, bin_edges, val_edges;
        download = true
    )
    # Compare type 1 of SP2D with Method 2
    m4_sp2d_diff = maximum(abs.(sums_val_2d .- @view(sums_val_sp2d[1, :, :, :])))
    println("  Method 4 (SP2D type 1 L2) vs Method 2:")
    println("    Max abs sums diff:   ", m4_sp2d_diff)

    # =======================================================================
    # STAGE 2: Large-scale Benchmark (N=20000, B=8064)
    # =======================================================================
    println("\n=== STAGE 2: Large-Scale Benchmark (N=20000, B=8064) ===")
    N_bench = 20000
    B_bench = 8064
    println("Generating problem (N=$N_bench, B=$B_bench)...")
    x_bench, u_bench = BP.make_random_batch_problem(FT, N_bench, (B_bench,); fixed_x = true, seed = 1)
    
    ws_bench = BP.BatchGPUWorkspace(backend, FT, N_bench, B_bench, NB; fixed_x = true)
    BP.upload_batch!(ws_bench, backend, x_bench, u_bench)

    # 1. Benchmark Method 1
    println("Running optimized Method 1 (1D SF)...")
    sums_opt = zeros(FT, NB, B_bench)
    counts_opt = zeros(UInt32, NB, B_bench)
    # Warmup
    gpu_batch_fused_tiled_fixed_x_optimal!(
        backend, sums_opt, counts_opt, x_bench, u_bench, sft, bin_edges;
        workspace = ws_bench, download = false
    )
    # Benchmark
    t_opt = @elapsed gpu_batch_fused_tiled_fixed_x_optimal!(
        backend, sums_opt, counts_opt, x_bench, u_bench, sft, bin_edges;
        workspace = ws_bench, download = true
    )
    println("  Method 1 Time: ", t_opt, " s")

    # 2. Benchmark Method 2
    println("Running optimized Method 2 (2D Joint SF)...")
    sums_2d = zeros(FT, NB, n_val, B_bench)
    counts_2d = zeros(UInt32, NB, n_val, B_bench)
    # Warmup
    gpu_batch_fused_tiled_fixed_x_optimal_2d!(
        backend, sums_2d, counts_2d, x_bench, u_bench, sft, bin_edges, val_edges;
        download = false
    )
    # Benchmark
    t_2d = @elapsed gpu_batch_fused_tiled_fixed_x_optimal_2d!(
        backend, sums_2d, counts_2d, x_bench, u_bench, sft, bin_edges, val_edges;
        download = true
    )
    println("  Method 2 Time: ", t_2d, " s")

    # 3. Benchmark Method 3
    println("Running optimized Method 3 (SP1D)...")
    sums_sp1d = zeros(FT, 8, NB, B_bench)
    counts_sp1d = zeros(UInt32, 8, NB, B_bench)
    # Warmup
    gpu_batch_fused_tiled_fixed_x_optimal_sp1d!(
        backend, sums_sp1d, counts_sp1d, x_bench, u_bench, bin_edges;
        download = false
    )
    # Benchmark
    t_sp1d = @elapsed gpu_batch_fused_tiled_fixed_x_optimal_sp1d!(
        backend, sums_sp1d, counts_sp1d, x_bench, u_bench, bin_edges;
        download = true
    )
    println("  Method 3 Time: ", t_sp1d, " s")

    # 4. Benchmark Method 4
    println("Running optimized Method 4 (SP2D)...")
    sums_sp2d = zeros(FT, 8, NB, n_val, B_bench)
    counts_sp2d = zeros(UInt32, 8, NB, n_val, B_bench)
    # Warmup
    gpu_batch_fused_tiled_fixed_x_optimal_sp2d!(
        backend, sums_sp2d, counts_sp2d, x_bench, u_bench, bin_edges, val_edges;
        download = false
    )
    # Benchmark
    t_sp2d = @elapsed gpu_batch_fused_tiled_fixed_x_optimal_sp2d!(
        backend, sums_sp2d, counts_sp2d, x_bench, u_bench, bin_edges, val_edges;
        download = true
    )
    println("  Method 4 Time: ", t_sp2d, " s")
end

main()
