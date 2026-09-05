# Production batch tiled128 kernels — fixed-x u-smem strips + block-private merge.
# Included from StructureFunctionsKernelAbstractionsExt.jl after GPUBatchWorkspace.jl.

"""
u-smem strip width for `_batch_fixed_x_usmem_priv!`, the widest strip whose static staging fits.

The kernel stages `512 + 512W` elements of `FT`, so `W` is a shared-memory budget question, not a
constant: 16 for `Float32` (34 KB) but at most 11 for `Float64`, which is why a hardcoded 16
requested 68 KB against the 48 KB static cap and failed to compile. Rounded down to a power of two
for a margin under the cap.
"""
@inline function _batch_usmem_strip_w(::Type{FT}) where {FT}
    w = SFC.GPU_SMEM_STATIC_MAX ÷ (512 * sizeof(FT)) - 1
    return w >= 16 ? 16 : w >= 8 ? 8 : w >= 4 ? 4 : w >= 2 ? 2 : 1
end

"""Warps per batch tile block (`SF_GPU_TILED_WS ÷ 32`)."""
const BATCH_USMEM_WARPS = SF_GPU_TILED_WS ÷ 32

# Thread/block index args are ::Integer, not ::Int: CUDA @index(Local/Group, Linear)
# yields Int32, and ::Int-typed methods fail dispatch inside device code
# (InvalidIRError at kernel compile; see _sp2d_flush_typeplane_to_output!).
@inline _batch_warp_id(lid::Integer) = (Int(lid) - 1) >> 5

"""`partial_{sums,cnts}` third-axis length: one slot per `(tile_block, warp)`."""
@inline _batch_usmem_n_priv(n_tile_blocks::Int) = n_tile_blocks * BATCH_USMEM_WARPS

@inline _batch_usmem_priv_idx(block_id::Integer, lid::Integer) =
    (Int(block_id) - 1) * BATCH_USMEM_WARPS + _batch_warp_id(lid) + 1

"""Production fixed-x 1D batch kernel (u staged in shared memory, strip 16)."""
_batch_fixed_x_sf_kernel(backend::KA.Backend, ws::Int) =
    _batch_fixed_x_usmem_priv!(backend, ws)

function _batch_tiled_launch_params(N_points::Int)
    sched = FullUpperTriangle(cld(N_points, SF_GPU_TILE))
    n_tile_blocks = n_pair_blocks(sched)
    ws = SF_GPU_TILED_WS
    return sched, n_tile_blocks, ws, n_tile_blocks * ws
end

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

@inline _gpu_pow_int(x, ::Val{0}) = one(x)
@inline _gpu_pow_int(x, ::Val{1}) = x
@inline _gpu_pow_int(x, ::Val{2}) = x * x
@inline _gpu_pow_int(x, ::Val{3}) = x * x * x
@inline _gpu_pow_int(x, ::Val{N}) where {N} = x^N

@inline function _gpu_sf_value_2d(::SFT.ProjectedStructureFunctionType{NL, NT}, du_x, du_y, rx, ry) where {NL, NT}
    du_L = rx * du_x + ry * du_y
    du_T = ry * du_x - rx * du_y
    val = one(du_L)
    if NL != 0
        val *= _gpu_pow_int(du_L, Val(NL))
    end
    if NT != 0
        val *= _gpu_pow_int(du_T, Val(NT))
    end
    return val
end

@inline function _gpu_sf_value_2d(::SFT.SecondOrderStructureFunctionType, du_x, du_y, rx, ry)
    return du_x * du_x + du_y * du_y
end

@inline function _gpu_sf_value_2d(::SFT.ThirdOrderStructureFunctionType, du_x, du_y, rx, ry)
    du_L = rx * du_x + ry * du_y
    return du_L * (du_x * du_x + du_y * du_y)
end

@inline function _gpu_sf_value_2d(::SFT.FullVectorStructureFunctionType{NF}, du_x, du_y, rx, ry) where {NF}
    n2 = du_x * du_x + du_y * du_y
    NF == 2 && return n2
    return _gpu_pow_int(sqrt(n2), Val(NF))
end

@inline function _gpu_sf_value_2d(::SFT.TransverseComponentSecondOrderStructureFunctionType, du_x, du_y, rx, ry)
    du_L = rx * du_x + ry * du_y
    return du_x * du_x + du_y * du_y - du_L * du_L
end

@inline function _gpu_sf_value_2d(::SFT.LongitudinalTransverseComponentThirdOrderStructureFunctionType, du_x, du_y, rx, ry)
    du_L = rx * du_x + ry * du_y
    du_T2 = du_x * du_x + du_y * du_y - du_L * du_L
    return du_L * du_T2
end

@inline function _batch_dist_bin(
    dist::T, fe::T, le::T, is_::T, sv::T, nb::Int, ::Val{false},
) where {T}
    return _gpu_digitize_linear(dist, fe, le, is_, sv, nb)
end
@inline function _batch_dist_bin(
    dist::T, fe::T, le::T, is_::T, sv::T, nb::Int, ::Val{true},
) where {T}
    return _gpu_digitize_log_spaced(dist, fe, le, is_, sv, nb)
end

"""
Separation and pair frame for one staged pair, plus its bin.

The bin comes from the same O(1) FMA digitizer whatever the geometry: `_fma_distance_bins` is a
statement about the *edges* being linear- or log-spaced, not about the space the separation was
measured in. Only the separation and frame are geometry-dependent, and those come from dispatch on
`geom`.
"""
@inline function _pair_bin_frame_from_smem!(
    shared_xi::AbstractVector{FT},
    shared_xj::AbstractVector{FT},
    ia::Int,
    jb::Int,
    ::Val{OFF_DIAG},
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    step_val::FT,
    N_bins::Int,
    geom,
    ::Val{LOG} = Val(false),
) where {FT, OFF_DIAG, LOG}
    if OFF_DIAG
        Xi = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
        Xj = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
    else
        Xi = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
        Xj = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
    end
    ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
    bin = _batch_dist_bin(dist, first_edge, last_edge, inv_step, step_val, N_bins, Val(LOG))
    return (ok && 1 <= bin < N_bins, bin, Xi, Xj, dist, frame)
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
        @inbounds output[bin, col] = acc_s
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
    shared_acc = @localmem eltype(output) (256,)
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
            @inbounds output[bin, col] = total
        end
    end
end

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
        acc_c = UInt32(0)
        @inbounds for blk in 1:n_priv
            acc_c += partial_cnts[bin, blk]
        end
        @inbounds output_cnts[bin] = acc_c
        t += nworkers
    end
end

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
    shared_acc = @localmem UInt32 (256,)
    if gid <= NB
        bin = gid
        acc_c = UInt32(0)
        blk = lid
        @inbounds while blk <= n_priv
            acc_c += partial_cnts[bin, blk]
            blk += workgroup_size
        end
        shared_acc[lid] = acc_c
        @synchronize
        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        gid = (g - 1) ÷ workgroup_size + 1
        if gid <= NB && lid == 1
            bin = gid
            total = UInt32(0)
            @inbounds for t in 1:workgroup_size
                total += shared_acc[t]
            end
            @inbounds output_cnts[bin] = total
        end
    end
end


# ---------------------------------------------------------------------------
# Fixed-x individual SF — u-smem priv strip kernel
#
# Histogram levels (warp-private partials do not fit in smem on sm_80 — 48 KiB cap):
#   1. Pair loop: `@atomic` into `partial_sums[bin, col, priv_idx]` (32 threads / priv slot).
#   2. Host merge kernel: sum all `priv_idx` → strip output (`_batch_merge_usmem_*`, `n_priv` axis).
# ---------------------------------------------------------------------------

KA.@kernel unsafe_indices=true function _batch_fixed_x_usmem_priv!(
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
    step_val::FT,
    sched,
    n_tile_blocks::Int,
    workgroup_size::Int,
    geom,
    ::Val{LOG} = Val(false),
) where {FT, LOG}
    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (256 * _batch_usmem_strip_w(FT),)
    shared_uj = @localmem FT (256 * _batch_usmem_strip_w(FT),)

    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    bid = (g - 1) ÷ workgroup_size + 1
    block_id = bid

    if bid <= n_tile_blocks
        priv_idx = _batch_usmem_priv_idx(block_id, lid)
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
                        shared_xi, shared_xj, ia, jb, Val(true),
                        first_edge, last_edge, inv_step, step_val, N_bins, geom, Val(LOG),
                    )
                    if pair_ok
                        priv_idx = _batch_usmem_priv_idx(block_id, lid)
                        # Loop-invariant across the field strip.
                        rhat = SFH.pair_direction(geom, frame, dist)
                        @inbounds for col in 1:bw
                            Ui = SA.SVector{2}(shared_ui[_batch_usmem_idx(1, ia, col)], shared_ui[_batch_usmem_idx(2, ia, col)])
                            Uj = SA.SVector{2}(shared_uj[_batch_usmem_idx(1, jb, col)], shared_uj[_batch_usmem_idx(2, jb, col)])
                            dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                            val = _gpu_sf_value_2d(sf_type, dU[1], dU[2], rhat[1], rhat[2])
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
                        shared_xi, shared_xj, ia, jb, Val(false),
                        first_edge, last_edge, inv_step, step_val, N_bins, geom, Val(LOG),
                    )
                    if pair_ok
                        priv_idx = _batch_usmem_priv_idx(block_id, lid)
                        # Loop-invariant across the field strip.
                        rhat = SFH.pair_direction(geom, frame, dist)
                        @inbounds for col in 1:bw
                            Ui = SA.SVector{2}(shared_ui[_batch_usmem_idx(1, ia, col)], shared_ui[_batch_usmem_idx(2, ia, col)])
                            Uj = SA.SVector{2}(shared_ui[_batch_usmem_idx(1, jb, col)], shared_ui[_batch_usmem_idx(2, jb, col)])
                            dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
                            val = _gpu_sf_value_2d(sf_type, dU[1], dU[2], rhat[1], rhat[2])
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

