# Production batch tiled128 kernels — fixed-x u-smem strips + block-private merge.

"""Static shared bytes of `_batch_fixed_x_usmem_priv!` for staging of `FT` and strips of `SW`
fields."""
@inline _batch_fixed_x_smem_bytes(::Type{FT}, SW::Int) where {FT} =
    2 * SFC.gpu_localmem_bytes(FT, 2 * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(FT, 2 * SF_GPU_TILE * SW)

"""Widest strip of at most 16 fields whose `_batch_fixed_x_usmem_priv!` fits the device `caps`
describes; 0 when a strip of one does not."""
@inline _batch_usmem_strip_w(caps::SFC.GPUDeviceCaps, ::Type{FT}) where {FT} =
    _sf_fitting_width(w -> _batch_fixed_x_smem_bytes(FT, w), caps, 16)

"""`partial_{sums,cnts}` third-axis length: one slot per `(tile_block, warp)` of `warp` threads."""
@inline _batch_usmem_n_priv(n_tile_blocks::Int, workgroup_size::Int, warp::Int) =
    n_tile_blocks * (workgroup_size ÷ warp)

# Thread/block index args are ::Integer, not ::Int: CUDA @index(Local/Group, Linear)
# yields Int32, and ::Int-typed methods fail dispatch inside device code
# (InvalidIRError at kernel compile; see _sp2d_flush_typeplane_to_output!).
"""Private partial slot of thread `lid` of tile block `block_id`: one slot per warp of `WARP`."""
@inline _batch_usmem_priv_idx(block_id::Integer, lid::Integer, workgroup_size::Integer,
                              ::Val{WARP}) where {WARP} =
    (Int(block_id) - 1) * (Int(workgroup_size) ÷ WARP) + (Int(lid) - 1) ÷ WARP + 1

"""Production fixed-x 1D batch kernel (u staged in shared memory, strips of up to 16 fields)."""
_batch_fixed_x_sf_kernel(backend::KA.Backend, ws::Int) =
    _batch_fixed_x_usmem_priv!(backend, ws)

@inline function _batch_usmem_idx(c::Int, k::Int, col::Int)
    return c + 2 * (k - 1) + SF_GPU_TILE * 2 * (col - 1)
end

"""Stage strip `u` into `shared_ui` with coalesced global loads along the N index."""
@inline function _stage_batch_ui_tile!(
    shared_ui,
    u_batch,
    b_base::Int,
    bw::Int,
    i0::Int,
    ni::Int,
    workgroup_size::Integer,
    lid::Integer,
)
    col = 1
    while col <= bw
        b_idx = b_base + col - 1
        k = Int(lid)
        while k <= ni
            gi = i0 + k - 1
            @inbounds begin
                shared_ui[_batch_usmem_idx(1, k, col)] = u_batch[b_idx, gi, 1]
                shared_ui[_batch_usmem_idx(2, k, col)] = u_batch[b_idx, gi, 2]
            end
            k += workgroup_size
        end
        col += 1
    end
    return nothing
end

"""Separation, pair frame and distance bin for one staged pair."""
@inline function _pair_bin_frame_from_smem!(
    shared_xi::AbstractVector{FT},
    shared_xj::AbstractVector{FT},
    ia::Int,
    jb::Int,
    ::Val{OFF_DIAG},
    ddig,
    N_bins::Int,
    geom,
) where {FT, OFF_DIAG}
    if OFF_DIAG
        Xi = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
        Xj = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
    else
        Xi = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
        Xj = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
    end
    ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
    bin = SFH.digitize(dist, ddig)
    return (ok && 1 <= bin < N_bins, bin, Xi, Xj, dist, frame)
end

"""Stage shared positions `x` `(W, N)` and fields `u` `(W, N, B…)` on `backend`, `u` batch-major `(B, N, W)`."""
function _stage_batch_device(backend::KA.Backend, x::AbstractMatrix{FT}, u::AbstractArray{FT}) where {FT}
    ndims(u) >= 3 || throw(ArgumentError("a batch over shared positions expects trailing batch dims on u"))
    size(x) == size(u)[1:2] || throw(ArgumentError("x and u leading dims must match"))
    u_batchmajor = permutedims(reshape(u, size(u, 1), size(u, 2), SFC.batch_size(u)), (3, 2, 1))
    return KA.adapt(backend, x), KA.adapt(backend, u_batchmajor)
end

KA.@kernel unsafe_indices=true function _batch_merge_usmem_sums!(
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
        @inbounds output[bin, col] += acc_s
        t += nworkers
    end
end

"""Parallel workgroup reduction merge for CUDA fixed-x batch routes."""
KA.@kernel unsafe_indices=true function _batch_merge_usmem_sums_grouped!(
    output,
    @Const(partial_sums),
    NB::Int,
    bw::Int,
    n_priv::Int,
    workgroup_size::Int,
)
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    gid = (g - 1) ÷ workgroup_size + 1
    shared_acc = @localmem eltype(output) (SF_GPU_TILED_WS,)
    n_out = NB * bw
    if gid <= n_out
        rem0 = gid - 1
        bin = rem0 % NB + 1
        col = rem0 ÷ NB + 1
        acc_s = zero(eltype(output))
        blk = lid
        @inbounds while blk <= n_priv
            acc_s += partial_sums[bin, col, blk]
            blk += workgroup_size
        end
        shared_acc[lid] = acc_s
        @synchronize
        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        gid = (g - 1) ÷ workgroup_size + 1
        if gid <= n_out && lid == 1
            rem0 = gid - 1
            bin = rem0 % NB + 1
            col = rem0 ÷ NB + 1
            total = zero(eltype(output))
            @inbounds for t in 1:workgroup_size
                total += shared_acc[t]
            end
            @inbounds output[bin, col] += total
        end
    end
end

"""Add each bin's count, summed over the `n_priv` partials, into every column of `output_cnts`
`(NB, B)`: with shared positions every slice has the same pairs."""
KA.@kernel unsafe_indices=true function _batch_merge_usmem_cnts!(
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
        acc_c = zero(eltype(output_cnts))
        @inbounds for blk in 1:n_priv
            acc_c += partial_cnts[bin, blk]
        end
        @inbounds for col in 1:size(output_cnts, 2)
            output_cnts[bin, col] += acc_c
        end
        t += nworkers
    end
end

"""Workgroup form of `_batch_merge_usmem_cnts!`: one workgroup per bin reduces the partials, then its
lanes add the total across the columns."""
KA.@kernel unsafe_indices=true function _batch_merge_usmem_cnts_grouped!(
    output_cnts,
    @Const(partial_cnts),
    NB::Int,
    n_priv::Int,
    workgroup_size::Int,
)
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    gid = (g - 1) ÷ workgroup_size + 1
    shared_acc = @localmem eltype(output_cnts) (SF_GPU_TILED_WS,)
    if gid <= NB
        acc_c = zero(eltype(output_cnts))
        blk = lid
        @inbounds while blk <= n_priv
            acc_c += partial_cnts[gid, blk]
            blk += workgroup_size
        end
        shared_acc[lid] = acc_c
        @synchronize
        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        if lid == 1
            total = zero(eltype(output_cnts))
            @inbounds for t in 1:workgroup_size
                total += shared_acc[t]
            end
            @inbounds shared_acc[1] = total
        end
        @synchronize
        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        gid = (g - 1) ÷ workgroup_size + 1
        total = @inbounds shared_acc[1]
        col = lid
        @inbounds while col <= size(output_cnts, 2)
            output_cnts[gid, col] += total
            col += workgroup_size
        end
    end
end


# Fixed-x individual SF: the pair loop adds atomically into `partial_sums[bin, col, priv_idx]`, one slot per
# warp; the `_batch_merge_usmem_*` kernels sum the `priv_idx` axis into the strip output.

KA.@kernel unsafe_indices=true function _batch_fixed_x_usmem_priv!(
    partial_sums::AbstractArray{FT},
    partial_cnts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    b_base::Int,
    bw::Int,
    ddig,
    sched,
    n_tile_blocks::Int,
    workgroup_size::Int,
    ::Val{SW},
    ::Val{WARP},
    geom,
) where {FT, SW, WARP}
    shared_xi = @localmem FT (2 * SF_GPU_TILE,)
    shared_xj = @localmem FT (2 * SF_GPU_TILE,)
    shared_ui = @localmem FT (2 * SF_GPU_TILE * SW,)
    shared_uj = @localmem FT (2 * SF_GPU_TILE * SW,)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid

    if bid <= n_tile_blocks
        priv_idx = _batch_usmem_priv_idx(block_id, lid, workgroup_size, Val(WARP))
        slot = lid
        while slot <= NB * bw
            bin = (slot - 1) % NB + 1
            col = (slot - 1) ÷ NB + 1
            @inbounds partial_sums[bin, col, priv_idx] = zero(FT)
            slot += workgroup_size
        end
        if b_base == 1
            slot = lid
            while slot <= NB
                @inbounds partial_cnts[slot, priv_idx] = UInt32(0)
                slot += workgroup_size
            end
        end
    end

    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
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
            _stage_batch_ui_tile!(shared_ui, u_batch, b_base, bw, i0, ni, workgroup_size, lid)
            if ti < tj
                _stage_batch_ui_tile!(shared_uj, u_batch, b_base, bw, j0, nj, workgroup_size, lid)
            end
        end
    end
    @synchronize

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
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
                    pair_ok, bin, Xi, Xj, dist, frame = _pair_bin_frame_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(true), ddig, N_bins, geom,
                    )
                    if pair_ok
                        priv_idx = _batch_usmem_priv_idx(block_id, lid, workgroup_size, Val(WARP))
                        # Loop-invariant across the field strip.
                        rhat = SFH.pair_direction(geom, frame, dist)
                        @inbounds for col in 1:bw
                            Ui = SA.SVector{2}(shared_ui[_batch_usmem_idx(1, ia, col)], shared_ui[_batch_usmem_idx(2, ia, col)])
                            Uj = SA.SVector{2}(shared_uj[_batch_usmem_idx(1, jb, col)], shared_uj[_batch_usmem_idx(2, jb, col)])
                            dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                            val = sf_type(dU, rhat)
                            @atomic partial_sums[bin, col, priv_idx] += val
                        end
                        if b_base == 1
                            @atomic partial_cnts[bin, priv_idx] += UInt32(1)
                        end
                    end
                    p += workgroup_size
                end
            else
                n_pairs = ni * (ni - 1) ÷ 2
                p = lid
                while p <= n_pairs
                    ia, jb = _pair_from_linear(p, ni)
                    pair_ok, bin, Xi, Xj, dist, frame = _pair_bin_frame_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(false), ddig, N_bins, geom,
                    )
                    if pair_ok
                        priv_idx = _batch_usmem_priv_idx(block_id, lid, workgroup_size, Val(WARP))
                        # Loop-invariant across the field strip.
                        rhat = SFH.pair_direction(geom, frame, dist)
                        @inbounds for col in 1:bw
                            Ui = SA.SVector{2}(shared_ui[_batch_usmem_idx(1, ia, col)], shared_ui[_batch_usmem_idx(2, ia, col)])
                            Uj = SA.SVector{2}(shared_ui[_batch_usmem_idx(1, jb, col)], shared_ui[_batch_usmem_idx(2, jb, col)])
                            dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                            val = sf_type(dU, rhat)
                            @atomic partial_sums[bin, col, priv_idx] += val
                        end
                        if b_base == 1
                            @atomic partial_cnts[bin, priv_idx] += UInt32(1)
                        end
                    end
                    p += workgroup_size
                end
            end
        end
    end
end

