# Production batch tiled128 kernels — fixed-x u-smem strips + block-private merge.
# Included from StructureFunctionsGPUExt.jl after GPUBatchWorkspace.jl.

"""u-smem strip width for small-strip reference (`_batch_fixed_x_usmem_priv!`)."""
const BATCH_USMEM_STRIP_W = 16

"""Warps per batch tile block (`SF_GPU_TILED_WS ÷ 32`)."""
const BATCH_USMEM_WARPS = SF_GPU_TILED_WS ÷ 32

@inline _batch_warp_id(lid::Int) = (lid - 1) >> 5

"""`partial_{sums,cnts}` third-axis length: one slot per `(tile_block, warp)`."""
@inline _batch_usmem_n_priv(n_tile_blocks::Int) = n_tile_blocks * BATCH_USMEM_WARPS

@inline _batch_usmem_priv_idx(block_id::Int, lid::Int) =
    (block_id - 1) * BATCH_USMEM_WARPS + _batch_warp_id(lid) + 1

"""Production fixed-x 1D batch kernel (u staged in shared memory, strip 16)."""
_batch_fixed_x_sf_kernel(backend::KA.Backend, ws::Int) =
    _batch_fixed_x_usmem_priv!(backend, ws)

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

"""Stage strip `u` into `shared_ui` with coalesced global loads along the N index."""
@inline function _stage_batch_ui_tile!(
    shared_ui,
    u_batch,
    b_base::Int,
    bw::Int,
    i0::Int,
    ni::Int,
    workgroup_size::Int,
    lid::Int,
)
    col = 1
    while col <= bw
        b_idx = b_base + col - 1
        k = lid
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
    dist::T, fe::T, le::T, is_::T, off::T, sv::T, nb::Int, ::Val{false},
) where {T}
    return _gpu_digitize_linear(dist, fe, le, is_, off, sv, nb)
end
@inline function _batch_dist_bin(
    dist::T, fe::T, le::T, is_::T, off::T, sv::T, nb::Int, ::Val{true},
) where {T}
    return _gpu_digitize_log_spaced(dist, fe, le, is_, off, sv, nb)
end

@inline function _pair_bin_rhat_from_smem!(
    shared_xi::AbstractVector{FT},
    shared_xj::AbstractVector{FT},
    ia::Int,
    jb::Int,
    ::Val{OFF_DIAG},
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    N_bins::Int,
    ::Val{LOG} = Val(false),
) where {FT, OFF_DIAG, LOG}
    if OFF_DIAG
        dx = shared_xj[jb] - shared_xi[ia]
        dy = shared_xj[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
    else
        dx = shared_xi[jb] - shared_xi[ia]
        dy = shared_xi[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
    end
    dist_sq = dx * dx + dy * dy
    dist = sqrt(dist_sq)
    bin = _batch_dist_bin(dist, first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG))
    if 1 <= bin < N_bins
        return (true, bin, dx / dist, dy / dist)
    end
    return (false, bin, zero(FT), zero(FT))
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

KA.@kernel unsafe_indices=true function _batch_merge_usmem_cnts_by_col!(
    output_cnts,
    @Const(partial_cnts),
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
        acc_c = UInt32(0)
        @inbounds for blk in 1:n_priv
            acc_c += partial_cnts[bin, col, blk]
        end
        @inbounds output_cnts[bin, col] = acc_c
        t += nworkers
    end
end

KA.@kernel unsafe_indices=true function _batch_merge_sp1d_sums!(
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

KA.@kernel unsafe_indices=true function _batch_merge_sp1d_sums_grouped!(
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
    n_out = SF_GPU_SINGLE_PASS_N * NB * bw
    if gid <= n_out
        rem0 = gid - 1
        bin = rem0 % NB + 1
        rem1 = rem0 ÷ NB
        ty = rem1 % SF_GPU_SINGLE_PASS_N + 1
        col = rem1 ÷ SF_GPU_SINGLE_PASS_N + 1
        acc_s = zero(eltype(output))
        blk = lid
        @inbounds while blk <= n_priv
            acc_s += partial_sums[ty, bin, col, blk]
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
            rem1 = rem0 ÷ NB
            ty = rem1 % SF_GPU_SINGLE_PASS_N + 1
            col = rem1 ÷ SF_GPU_SINGLE_PASS_N + 1
            total = zero(eltype(output))
            @inbounds for t in 1:workgroup_size
                total += shared_acc[t]
            end
            @inbounds output[ty, bin, col] = total
        end
    end
end

KA.@kernel unsafe_indices=true function _batch_merge_sp1d_cnts!(
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

KA.@kernel unsafe_indices=true function _batch_merge_sp1d_cnts_grouped!(
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
    n_out = SF_GPU_SINGLE_PASS_N * NB
    if gid <= n_out
        rem0 = gid - 1
        bin = rem0 % NB + 1
        ty = rem0 ÷ NB + 1
        acc_c = UInt32(0)
        blk = lid
        @inbounds while blk <= n_priv
            acc_c += partial_cnts[ty, bin, blk]
            blk += workgroup_size
        end
        shared_acc[lid] = acc_c
        @synchronize
        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        gid = (g - 1) ÷ workgroup_size + 1
        if gid <= n_out && lid == 1
            rem0 = gid - 1
            bin = rem0 % NB + 1
            ty = rem0 ÷ NB + 1
            total = UInt32(0)
            @inbounds for t in 1:workgroup_size
                total += shared_acc[t]
            end
            @inbounds output_cnts[ty, bin] = total
        end
    end
end

KA.@kernel unsafe_indices=true function _batch_merge_sp2d_cnts!(
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

KA.@kernel unsafe_indices=true function _batch_merge_sp2d_cnts_grouped!(
    output_cnts,
    @Const(partial_cnts),
    n_dist::Int,
    n_val::Int,
    bw::Int,
    n_priv::Int,
    workgroup_size::Int,
)
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    gid = (g - 1) ÷ workgroup_size + 1
    shared_acc = @localmem UInt32 (256,)
    n_out = SF_GPU_SINGLE_PASS_N * n_dist * n_val * bw
    if gid <= n_out
        rem0 = gid - 1
        vbin = rem0 % n_val + 1
        rem1 = rem0 ÷ n_val
        dbin = rem1 % n_dist + 1
        rem2 = rem1 ÷ n_dist
        ty = rem2 % SF_GPU_SINGLE_PASS_N + 1
        col = rem2 ÷ SF_GPU_SINGLE_PASS_N + 1
        acc_c = UInt32(0)
        blk = lid
        @inbounds while blk <= n_priv
            acc_c += partial_cnts[ty, dbin, vbin, col, blk]
            blk += workgroup_size
        end
        shared_acc[lid] = acc_c
        @synchronize
        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        gid = (g - 1) ÷ workgroup_size + 1
        if gid <= n_out && lid == 1
            rem0 = gid - 1
            vbin = rem0 % n_val + 1
            rem1 = rem0 ÷ n_val
            dbin = rem1 % n_dist + 1
            rem2 = rem1 ÷ n_dist
            ty = rem2 % SF_GPU_SINGLE_PASS_N + 1
            col = rem2 ÷ SF_GPU_SINGLE_PASS_N + 1
            total = UInt32(0)
            @inbounds for t in 1:workgroup_size
                total += shared_acc[t]
            end
            @inbounds output_cnts[ty, dbin, vbin, col] = total
        end
    end
end

KA.@kernel unsafe_indices=true function _batch_merge_sp2d_sums!(
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

KA.@kernel unsafe_indices=true function _batch_merge_sp2d_sums_grouped!(
    output,
    @Const(partial_sums),
    n_dist::Int,
    n_val::Int,
    bw::Int,
    n_priv::Int,
    workgroup_size::Int,
)
    g = @index(Global, Linear)
    lid = (g - 1) % workgroup_size + 1
    gid = (g - 1) ÷ workgroup_size + 1
    shared_acc = @localmem eltype(output) (256,)
    n_out = SF_GPU_SINGLE_PASS_N * n_dist * n_val * bw
    if gid <= n_out
        rem0 = gid - 1
        vbin = rem0 % n_val + 1
        rem1 = rem0 ÷ n_val
        dbin = rem1 % n_dist + 1
        rem2 = rem1 ÷ n_dist
        ty = rem2 % SF_GPU_SINGLE_PASS_N + 1
        col = rem2 ÷ SF_GPU_SINGLE_PASS_N + 1
        acc_s = zero(eltype(output))
        blk = lid
        @inbounds while blk <= n_priv
            acc_s += partial_sums[ty, dbin, vbin, col, blk]
            blk += workgroup_size
        end
        shared_acc[lid] = acc_s
        @synchronize
        g = @index(Global, Linear)
        lid = (g - 1) % workgroup_size + 1
        gid = (g - 1) ÷ workgroup_size + 1
        if gid <= n_out && lid == 1
            rem0 = gid - 1
            vbin = rem0 % n_val + 1
            rem1 = rem0 ÷ n_val
            dbin = rem1 % n_dist + 1
            rem2 = rem1 ÷ n_dist
            ty = rem2 % SF_GPU_SINGLE_PASS_N + 1
            col = rem2 ÷ SF_GPU_SINGLE_PASS_N + 1
            total = zero(eltype(output))
            @inbounds for t in 1:workgroup_size
                total += shared_acc[t]
            end
            @inbounds output[ty, dbin, vbin, col] = total
        end
    end
end

@inline function _batch_sp1d_accum_shared!(
    shared_sums,
    shared_cnts,
    bin::Int,
    col::Int,
    du_L,
    du_L2,
    du_T2,
    du_norm2,
    NB::Int;
    track_counts::Bool = true,
)
    base = (col - 1) * SF_GPU_SINGLE_PASS_N * NB
    @atomic shared_sums[base + bin] += du_norm2
    @atomic shared_sums[base + NB + bin] += du_L2
    @atomic shared_sums[base + 2NB + bin] += du_T2
    @atomic shared_sums[base + 3NB + bin] += du_L * du_norm2
    @atomic shared_sums[base + 4NB + bin] += du_L * du_L2
    @atomic shared_sums[base + 5NB + bin] += du_L * du_T2
    if track_counts && col == 1
        @atomic shared_cnts[bin] += UInt32(1)
    end
    return nothing
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
    offset::FT,
    step_val::FT,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
    ::Val{LOG} = Val(false),
) where {FT, LOG}
    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (256 * BATCH_USMEM_STRIP_W,)
    shared_uj = @localmem FT (256 * BATCH_USMEM_STRIP_W,)

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
                    pair_ok, bin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(true),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        priv_idx = _batch_usmem_priv_idx(block_id, lid)
                        @inbounds for col in 1:bw
                            du_x = shared_uj[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_uj[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            val = _gpu_sf_value_2d(sf_type, du_x, du_y, rx, ry)
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
                    pair_ok, bin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(false),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        priv_idx = _batch_usmem_priv_idx(block_id, lid)
                        @inbounds for col in 1:bw
                            du_x = shared_ui[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_ui[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            val = _gpu_sf_value_2d(sf_type, du_x, du_y, rx, ry)
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

# ---------------------------------------------------------------------------
# Fixed-x individual SF — global-`u` strip kernel (WITHDRAWN — not used in launch)
#
# A100 Mar 2026: strip 128 / 63 sweeps measured ~48 s at NB=50–64 vs ~12 s for usmem
# strip 16 / 504 sweeps at NB=50. Fewer geometry replays were canceled by 8× more
# inner-loop atomics per pair and uncached global `u` loads. Do not wire without profiling.
# ---------------------------------------------------------------------------

"""
    _batch_fixed_x_global_u_priv!(partial_sums, partial_cnts, ...)
"""
KA.@kernel unsafe_indices=true function _batch_fixed_x_global_u_priv!(
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
    ::Val{LOG} = Val(false),
) where {FT, LOG}
    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS * 128,)
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
                    pair_ok, bin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(true),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        gi = i0 + ia - 1
                        gj = j0 + jb - 1
                        @inbounds for col in 1:bw
                            b_idx = b_base + col - 1
                            du_x = u_batch[b_idx, gj, 1] - u_batch[b_idx, gi, 1]
                            du_y = u_batch[b_idx, gj, 2] - u_batch[b_idx, gi, 2]
                            val = _gpu_sf_value_2d(sf_type, du_x, du_y, rx, ry)
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
                    pair_ok, bin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(false),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        gi = i0 + ia - 1
                        gj = i0 + jb - 1
                        @inbounds for col in 1:bw
                            b_idx = b_base + col - 1
                            du_x = u_batch[b_idx, gj, 1] - u_batch[b_idx, gi, 1]
                            du_y = u_batch[b_idx, gj, 2] - u_batch[b_idx, gi, 2]
                            val = _gpu_sf_value_2d(sf_type, du_x, du_y, rx, ry)
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
# Fixed-x SP1D (six invariants) — u-smem priv strip kernel
# ---------------------------------------------------------------------------

KA.@kernel unsafe_indices=true function _batch_fixed_x_sp1d_usmem_priv!(
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
    ::Val{LOG} = Val(false),
) where {FT, LOG}
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
                    pair_ok, bin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(true),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            du_x = shared_uj[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_uj[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            du_L = rx * du_x + ry * du_y
                            du_L2 = du_L * du_L
                            du_norm2 = du_x * du_x + du_y * du_y
                            du_T2 = du_norm2 - du_L2
                            _batch_sp1d_accum_shared!(
                                shared_sums, shared_cnts, bin, col,
                                du_L, du_L2, du_T2, du_norm2, NB,
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
                    pair_ok, bin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(false),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            du_x = shared_ui[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_ui[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            du_L = rx * du_x + ry * du_y
                            du_L2 = du_L * du_L
                            du_norm2 = du_x * du_x + du_y * du_y
                            du_T2 = du_norm2 - du_L2
                            _batch_sp1d_accum_shared!(
                                shared_sums, shared_cnts, bin, col,
                                du_L, du_L2, du_T2, du_norm2, NB,
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
# Fixed-x SP2D — shared x, u strip, partial block-private outputs
# ---------------------------------------------------------------------------

KA.@kernel unsafe_indices=true function _batch_varying_x_sp2d_fixed_x!(
    partial_sums,
    partial_cnts,
    @Const(x_mat),
    @Const(u_batch),
    N_points::Int,
    N_bins::Int,
    n_dist::Int,
    n_val::Int,
    b_base::Int,
    bw::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    val_plan::GPUValueDigitizePlan,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
    ::Val{LOG} = Val(false),
) where {FT, LOG}
    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (BATCH_USMEM_STRIP_W * SF_GPU_TILE * 2,)
    shared_uj = @localmem FT (BATCH_USMEM_STRIP_W * SF_GPU_TILE * 2,)

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
            k = lid
            while k <= ni
                gi = i0 + k - 1
                @inbounds begin
                    shared_xi[k]               = x_mat[1, gi]
                    shared_xi[SF_GPU_TILE + k] = x_mat[2, gi]
                end
                k += workgroup_size
            end
            _stage_batch_ui_tile!(shared_ui, u_batch, b_base, bw, i0, ni, workgroup_size, lid)
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k]               = x_mat[1, gj]
                        shared_xj[SF_GPU_TILE + k] = x_mat[2, gj]
                    end
                    k += workgroup_size
                end
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
                    pair_ok, dbin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(true),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            du_x = shared_uj[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_uj[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            du_L     = rx * du_x + ry * du_y
                            du_L2    = du_L * du_L
                            du_norm2 = du_x * du_x + du_y * du_y
                            du_T2    = du_norm2 - du_L2
                            v1 = du_norm2
                            v2 = du_L2
                            v3 = du_T2
                            v4 = du_L * du_norm2
                            v5 = du_L * du_L2
                            v6 = du_L * du_T2
                            @inbounds for t in 1:SF_GPU_SINGLE_PASS_N
                                val_t = t == 1 ? v1 : t == 2 ? v2 : t == 3 ? v3 : t == 4 ? v4 : t == 5 ? v5 : v6
                                vbin = _gpu_digitize_value_plan(val_t, val_plan, t, n_val + 1)
                                if 1 <= vbin <= n_val
                                    @atomic partial_sums[t, dbin, vbin, col, block_id] += val_t
                                    @atomic partial_cnts[t, dbin, vbin, col, block_id] += UInt32(1)
                                end
                            end
                        end
                    end
                else
                    ia, jb = _pair_from_linear(p, ni)
                    pair_ok, dbin, rx, ry = _pair_bin_rhat_from_smem!(
                        shared_xi, shared_xj, ia, jb, Val(false),
                        first_edge, last_edge, inv_step, offset, step_val, N_bins, Val(LOG),
                    )
                    if pair_ok
                        @inbounds for col in 1:bw
                            du_x = shared_ui[_batch_usmem_idx(1, jb, col)] - shared_ui[_batch_usmem_idx(1, ia, col)]
                            du_y = shared_ui[_batch_usmem_idx(2, jb, col)] - shared_ui[_batch_usmem_idx(2, ia, col)]
                            du_L     = rx * du_x + ry * du_y
                            du_L2    = du_L * du_L
                            du_norm2 = du_x * du_x + du_y * du_y
                            du_T2    = du_norm2 - du_L2
                            v1 = du_norm2
                            v2 = du_L2
                            v3 = du_T2
                            v4 = du_L * du_norm2
                            v5 = du_L * du_L2
                            v6 = du_L * du_T2
                            @inbounds for t in 1:SF_GPU_SINGLE_PASS_N
                                val_t = t == 1 ? v1 : t == 2 ? v2 : t == 3 ? v3 : t == 4 ? v4 : t == 5 ? v5 : v6
                                vbin = _gpu_digitize_value_plan(val_t, val_plan, t, n_val + 1)
                                if 1 <= vbin <= n_val
                                    @atomic partial_sums[t, dbin, vbin, col, block_id] += val_t
                                    @atomic partial_cnts[t, dbin, vbin, col, block_id] += UInt32(1)
                                end
                            end
                        end
                    end
                end
                p += workgroup_size
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Varying geometry — one launch per (tile, b)
# ---------------------------------------------------------------------------

"""Tiled varying-x individual SF: stages x and u per tile per batch element into shared memory,
accumulates into a block-local histogram, then flushes to global output.
Replaces the naive version which did all reads/writes from global memory."""
KA.@kernel unsafe_indices=true function _batch_varying_x_sf!(
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
    shared_xi  = @localmem FT (256,)
    shared_xj  = @localmem FT (256,)
    shared_ui  = @localmem FT (256,)
    shared_uj  = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1

    b_init = lid
    while b_init <= NB
        @inbounds shared_sums[b_init] = zero(FT)
        @inbounds shared_cnts[b_init] = UInt32(0)
        b_init += workgroup_size
    end

    if bid <= n_tile_blocks && b <= B
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
                    shared_xi[k]              = x_batch[1, gi, b]
                    shared_xi[SF_GPU_TILE + k] = x_batch[2, gi, b]
                    shared_ui[k]              = u_batch[1, gi, b]
                    shared_ui[SF_GPU_TILE + k] = u_batch[2, gi, b]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k]              = x_batch[1, gj, b]
                        shared_xj[SF_GPU_TILE + k] = x_batch[2, gj, b]
                        shared_uj[k]              = u_batch[1, gj, b]
                        shared_uj[SF_GPU_TILE + k] = u_batch[2, gj, b]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
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
                    dx = shared_xj[jb]              - shared_xi[ia]
                    dy = shared_xj[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
                    U1x = shared_ui[ia]; U1y = shared_ui[SF_GPU_TILE + ia]
                    U2x = shared_uj[jb]; U2y = shared_uj[SF_GPU_TILE + jb]
                else
                    ia, jb = _pair_from_linear(p, ni)
                    dx = shared_xi[jb]              - shared_xi[ia]
                    dy = shared_xi[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
                    U1x = shared_ui[ia]; U1y = shared_ui[SF_GPU_TILE + ia]
                    U2x = shared_ui[jb]; U2y = shared_ui[SF_GPU_TILE + jb]
                end
                dist_sq = dx * dx + dy * dy
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_linear(dist, first_edge, last_edge, inv_step, offset, step_val, N_bins)
                if 1 <= bin < N_bins
                    rx = dx / dist; ry = dy / dist
                    val = _gpu_sf_value_2d(sf_type, U2x - U1x, U2y - U1y, rx, ry)
                    @atomic shared_sums[bin] += val
                    @atomic shared_cnts[bin] += UInt32(1)
                end
                p += workgroup_size
            end
        end
    end
    @synchronize

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        flushed = lid
        while flushed <= NB
            s = shared_sums[flushed]
            @atomic output[flushed, b] += s
            c = shared_cnts[flushed]
            if c != UInt32(0)
                @atomic counts[flushed, b] += c
            end
            flushed += workgroup_size
        end
    end
end

"""Tiled varying-x SP1D: stages x and u per tile per batch element into shared memory,
accumulates into a block-local (6, NB) histogram, then flushes to global output.
Replaces the naive version that issued 12 global atomics per valid pair."""
KA.@kernel unsafe_indices=true function _batch_varying_x_sp1d!(
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
    shared_xi  = @localmem FT (256,)
    shared_xj  = @localmem FT (256,)
    shared_ui  = @localmem FT (256,)
    shared_uj  = @localmem FT (256,)
    shared_sums = @localmem FT (SF_GPU_SINGLE_PASS_N * SF_GPU_MAX_BINS,)
    shared_cnts = @localmem UInt32 (SF_GPU_MAX_BINS,)

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1

    k_init = lid
    while k_init <= SF_GPU_SINGLE_PASS_N * NB
        @inbounds shared_sums[k_init] = zero(FT)
        k_init += workgroup_size
    end
    b_init = lid
    while b_init <= NB
        @inbounds shared_cnts[b_init] = UInt32(0)
        b_init += workgroup_size
    end

    if bid <= n_tile_blocks && b <= B
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
                    shared_xi[k]              = x_batch[1, gi, b]
                    shared_xi[SF_GPU_TILE + k] = x_batch[2, gi, b]
                    shared_ui[k]              = u_batch[1, gi, b]
                    shared_ui[SF_GPU_TILE + k] = u_batch[2, gi, b]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k]              = x_batch[1, gj, b]
                        shared_xj[SF_GPU_TILE + k] = x_batch[2, gj, b]
                        shared_uj[k]              = u_batch[1, gj, b]
                        shared_uj[SF_GPU_TILE + k] = u_batch[2, gj, b]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
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
                    dx = shared_xj[jb]              - shared_xi[ia]
                    dy = shared_xj[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
                    U1x = shared_ui[ia]; U1y = shared_ui[SF_GPU_TILE + ia]
                    U2x = shared_uj[jb]; U2y = shared_uj[SF_GPU_TILE + jb]
                else
                    ia, jb = _pair_from_linear(p, ni)
                    dx = shared_xi[jb]              - shared_xi[ia]
                    dy = shared_xi[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
                    U1x = shared_ui[ia]; U1y = shared_ui[SF_GPU_TILE + ia]
                    U2x = shared_ui[jb]; U2y = shared_ui[SF_GPU_TILE + jb]
                end
                dist_sq = dx * dx + dy * dy
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_linear(dist, first_edge, last_edge, inv_step, offset, step_val, N_bins)
                if 1 <= bin < N_bins
                    rx = dx / dist; ry = dy / dist
                    du_x = U2x - U1x; du_y = U2y - U1y
                    du_L = rx * du_x + ry * du_y
                    du_L2 = du_L * du_L
                    du_norm2 = du_x * du_x + du_y * du_y
                    du_T2 = du_norm2 - du_L2
                    _gpu_accumulate_single_pass_1d_shared!(
                        shared_sums, shared_cnts, bin, du_L, du_L2, du_T2, du_norm2, NB,
                    )
                end
                p += workgroup_size
            end
        end
    end
    @synchronize

    lid = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        k = lid
        while k <= SF_GPU_SINGLE_PASS_N * NB
            s = shared_sums[k]
            if s != zero(FT)
                t   = (k - 1) ÷ NB + 1
                bin = (k - 1) % NB + 1
                @atomic output_sums[t, bin, b] += s
            end
            k += workgroup_size
        end
        cnt_slot = lid
        while cnt_slot <= NB
            c = shared_cnts[cnt_slot]
            if c != UInt32(0)
                for t in 1:SF_GPU_SINGLE_PASS_N
                    @atomic output_counts[t, cnt_slot, b] += c
                end
            end
            cnt_slot += workgroup_size
        end
    end
end

"""Tiled varying-x SP2D: stages tile x/u into shared memory to cut global loads per pair
from 8 down to 0 after the load phase. Output histogram is too large for shared memory
(n_dist×n_val×6 can exceed 48 KB), so global atomics are used for output."""
KA.@kernel unsafe_indices=true function _batch_varying_x_sp2d!(
    output_sums,
    output_counts,
    @Const(x_batch),
    @Const(u_batch),
    N_points::Int,
    N_bins::Int,
    n_dist::Int,
    n_val::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    val_plan::GPUValueDigitizePlan,
    n_tiles::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
    B::Int,
) where {FT}
    shared_xi = @localmem FT (256,)
    shared_xj = @localmem FT (256,)
    shared_ui = @localmem FT (256,)
    shared_uj = @localmem FT (256,)

    lid          = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1

    if bid <= n_tile_blocks && b <= B
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
                    shared_xi[k]               = x_batch[1, gi, b]
                    shared_xi[SF_GPU_TILE + k] = x_batch[2, gi, b]
                    shared_ui[k]               = u_batch[1, gi, b]
                    shared_ui[SF_GPU_TILE + k] = u_batch[2, gi, b]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds begin
                        shared_xj[k]               = x_batch[1, gj, b]
                        shared_xj[SF_GPU_TILE + k] = x_batch[2, gj, b]
                        shared_uj[k]               = u_batch[1, gj, b]
                        shared_uj[SF_GPU_TILE + k] = u_batch[2, gj, b]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize

    lid          = @index(Local, Linear)
    launch_block = @index(Group, Linear)
    bid = (launch_block - 1) % n_tile_blocks + 1
    b   = (launch_block - 1) ÷ n_tile_blocks + 1
    if bid <= n_tile_blocks && b <= B
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            p = lid
            while p <= n_pairs
                @inbounds begin
                    if ti < tj
                        ia = (p - 1) ÷ nj + 1
                        jb = (p - 1) - (ia - 1) * nj + 1
                        dx   = shared_xj[jb]               - shared_xi[ia]
                        dy   = shared_xj[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
                        du_x = shared_uj[jb]               - shared_ui[ia]
                        du_y = shared_uj[SF_GPU_TILE + jb] - shared_ui[SF_GPU_TILE + ia]
                    else
                        ia, jb = _pair_from_linear(p, ni)
                        dx   = shared_xi[jb]               - shared_xi[ia]
                        dy   = shared_xi[SF_GPU_TILE + jb] - shared_xi[SF_GPU_TILE + ia]
                        du_x = shared_ui[jb]               - shared_ui[ia]
                        du_y = shared_ui[SF_GPU_TILE + jb] - shared_ui[SF_GPU_TILE + ia]
                    end
                    dist = sqrt(dx * dx + dy * dy)
                    dbin = _gpu_digitize_linear(
                        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                    )
                    if 1 <= dbin < N_bins
                        rx = dx / dist
                        ry = dy / dist
                        du_L     = rx * du_x + ry * du_y
                        du_L2    = du_L * du_L
                        du_norm2 = du_x * du_x + du_y * du_y
                        du_T2    = du_norm2 - du_L2
                        v1 = du_norm2
                        v2 = du_L2
                        v3 = du_T2
                        v4 = du_L * du_norm2
                        v5 = du_L * du_L2
                        v6 = du_L * du_T2
                        @inbounds for t in 1:SF_GPU_SINGLE_PASS_N
                            val_t = t == 1 ? v1 : t == 2 ? v2 : t == 3 ? v3 : t == 4 ? v4 : t == 5 ? v5 : v6
                            vbin = _gpu_digitize_value_plan(val_t, val_plan, t, n_val + 1)
                            if 1 <= vbin <= n_val
                                @atomic output_sums[t, dbin, vbin, b] += val_t
                                @atomic output_counts[t, dbin, vbin, b] += UInt32(1)
                            end
                        end
                    end
                end
                p += workgroup_size
            end
        end
    end
end
