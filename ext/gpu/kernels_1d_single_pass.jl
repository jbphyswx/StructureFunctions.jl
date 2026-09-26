# Tiled128 six-invariant-type single-pass 1D distance histogram kernel.
# Included from StructureFunctionsKernelAbstractionsExt.jl — block-local (6, NB) sums + one count row.

"""Accumulate the four independent invariants into flat `@localmem` sums `(6*NB,)`.
`du_norm2 = ||dU||²` must be pre-computed by the caller.

Rows 3 (`T2`) and 6 (`L1T2`) are exact differences of the others (`T2 = S2 - L2`,
`L1T2 = S3 - L3`) and a bin is a sum, so they are left zero here and reconstructed once at flush by
[`_sf_flush_moment`](@ref), leaving 4 shared atomics per pair."""
@inline function _gpu_accumulate_single_pass_1d_shared!(
    shared_sums,
    shared_cnts,
    bin::Int,
    du_L,
    du_norm2,
    NB::Int,
    w = true,
)
    vals = SFC.single_pass_invariants(du_L, du_norm2)
    @atomic shared_sums[bin] += w * vals[1]
    @atomic shared_sums[NB + bin] += w * vals[2]
    @atomic shared_sums[3NB + bin] += w * vals[4]
    @atomic shared_sums[4NB + bin] += w * vals[5]
    @atomic shared_cnts[bin] += convert(eltype(shared_cnts), w)
    return nothing
end

KA.@kernel unsafe_indices=true function _sf6_single_pass_kernel_tiled128_u32!(
    output_sums,
    output_counts,
    x_mat::AbstractMatrix{FT},
    u_mat,
    dig,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    sched,
    n_tile_blocks::Int,
    workgroup_size::Int,
    wts,                    # NoWeights(), or one weight per point
    ::Val{W},               # coordinate width
    ::Val{F},               # field width
    ::Val{CST},             # shared count element: UInt32 unweighted, the count type weighted
    geom,
) where {FT, W, F, CST}
    shared_xi = @localmem FT (W * SF_GPU_TILE,)
    shared_ui = @localmem FT (F * SF_GPU_TILE,)
    shared_xj = @localmem FT (W * SF_GPU_TILE,)
    shared_uj = @localmem FT (F * SF_GPU_TILE,)
    shared_sums = @localmem FT (SF_GPU_SINGLE_PASS_N * SF_GPU_MAX_BINS,)
    shared_cnts = @localmem CST (SF_GPU_MAX_BINS,)
    lid = @index(Local, Linear)
    k_init = lid
    while k_init <= SF_GPU_SINGLE_PASS_N * NB
        @inbounds shared_sums[k_init] = zero(FT)
        k_init += workgroup_size
    end
    b_init = lid
    while b_init <= NB
        @inbounds shared_cnts[b_init] = zero(CST)
        b_init += workgroup_size
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
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
                @inbounds for d in 1:W
                    shared_xi[(d - 1) * SF_GPU_TILE + k] = x_mat[d, gi]
                end
                @inbounds for d in 1:F
                    shared_ui[(d - 1) * SF_GPU_TILE + k] = u_mat[d, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for d in 1:W
                        shared_xj[(d - 1) * SF_GPU_TILE + k] = x_mat[d, gj]
                    end
                    @inbounds for d in 1:F
                        shared_uj[(d - 1) * SF_GPU_TILE + k] = u_mat[d, gj]
                    end
                    k += workgroup_size
                end
            end
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        ti, tj = tile_for(sched, bid)
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if ni > 0 && nj > 0
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            jbase = ti < tj ? j0 : i0
            p = lid
            while p <= n_pairs
                if ti < tj
                    ia = (p - 1) ÷ nj + 1
                    jb = (p - 1) - (ia - 1) * nj + 1
                    X1 = _sf_load_pt(Val(W), shared_xi, ia)
                    X2 = _sf_load_pt(Val(W), shared_xj, jb)
                    U1 = _sf_load_pt(Val(F), shared_ui, ia)
                    U2 = _sf_load_pt(Val(F), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = _sf_load_pt(Val(W), shared_xi, ia)
                    X2 = _sf_load_pt(Val(W), shared_xi, jb)
                    U1 = _sf_load_pt(Val(F), shared_ui, ia)
                    U2 = _sf_load_pt(Val(F), shared_ui, jb)
                end
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = SFH.digitize(dist, dig)
                if ok && 1 <= bin < N_bins
                    du_L, du_norm2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    pw = SFC._point_weight(wts, i0 + ia - 1) * SFC._point_weight(wts, jbase + jb - 1)
                    _gpu_accumulate_single_pass_1d_shared!(
                        shared_sums, shared_cnts, bin, du_L, du_norm2, NB, pw,
                    )
                end
                p += workgroup_size
            end
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        k = lid
        while k <= SF_GPU_SINGLE_PASS_N * NB
            t = (k - 1) ÷ NB + 1
            b = (k - 1) % NB + 1
            s = _sf_flush_moment(Val(SF_GPU_SINGLE_PASS_N), shared_sums, NB, t, b)
            if s != zero(FT)
                @atomic output_sums[t, b] += s
            end
            k += workgroup_size
        end
        b = lid
        while b <= NB
            c = shared_cnts[b]
            if c != zero(CST)
                for t in 1:SF_GPU_SINGLE_PASS_N
                    @atomic output_counts[t, b] += c
                end
            end
            b += workgroup_size
        end
    end
end

"""Static shared bytes of `_sf6_single_pass_kernel_tiled128_u32!` for `W`-wide coordinates and `F`-wide
fields staged and summed as `FT`, and counts of `CST`."""
@inline _sp1d_tiled_smem_bytes(::Type{FT}, ::Type{CST}, W::Int, F::Int) where {FT, CST} =
    2 * SFC.gpu_localmem_bytes(FT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(FT, F * SF_GPU_TILE) +
    SFC.gpu_localmem_bytes(FT, SF_GPU_SINGLE_PASS_N * SF_GPU_MAX_BINS) +
    SFC.gpu_localmem_bytes(CST, SF_GPU_MAX_BINS)

"""Whether `_sf6_single_pass_kernel_tiled128_u32!` takes `NB` distance bins, `W`-wide coordinates and
`F`-wide fields of `FT` and counts of `CST` on the device `caps` describes."""
@inline _gpu_single_pass_tiled_eligible(caps, NB::Int, ::Type{FT}, ::Type{CST}, W::Int, F::Int) where {FT, CST} =
    NB <= SF_GPU_MAX_BINS && SFC.gpu_static_smem_fits(caps, _sp1d_tiled_smem_bytes(FT, CST, W, F))
