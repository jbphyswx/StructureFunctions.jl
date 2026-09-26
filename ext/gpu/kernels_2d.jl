# Tiled128 joint 2D SF histogram kernel (distance × value) with a block-local flat histogram.
#
# The compile-time histogram width `HIST` is chosen per [`GPUSFWorkspace`](@ref) (default exact
# `n_dist × n_val`; optional override via `joint2d_compile_cells`); `W` is the coordinate width and `F`
# the field width.

KA.@kernel unsafe_indices=true function _sf2d_kernel_tiled128_u32!(
    output_sums,
    output_counts,
    x_mat::AbstractMatrix{FT},
    u_mat,
    wts,
    ddig,
    vdig,
    sf_type,
    N_points::Int,
    N_dist_edges::Int,
    N_val_edges::Int,
    NV::Int,
    NB2::Int,
    sched,
    n_tile_blocks::Int,
    workgroup_size::Int,
    ::Val{W},
    ::Val{F},
    ::Val{HIST},
    ::Val{CST},
    geom,
    second_axis,
) where {FT, W, F, HIST, CST}
    shared_xi = @localmem FT (W * SF_GPU_TILE,)
    shared_ui = @localmem FT (F * SF_GPU_TILE,)
    shared_xj = @localmem FT (W * SF_GPU_TILE,)
    shared_uj = @localmem FT (F * SF_GPU_TILE,)
    shared_sums = @localmem eltype(output_sums) (HIST,)
    shared_cnts = @localmem CST (HIST,)

    lid = @index(Local, Linear)
    b = lid
    while b <= NB2
        @inbounds begin
            shared_sums[b] = zero(eltype(output_sums))
            shared_cnts[b] = zero(CST)
        end
        b += workgroup_size
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
                @inbounds for c in 1:W
                    shared_xi[(c - 1) * SF_GPU_TILE + k] = x_mat[c, gi]
                end
                @inbounds for c in 1:F
                    shared_ui[(c - 1) * SF_GPU_TILE + k] = u_mat[c, gi]
                end
                k += workgroup_size
            end
            if ti < tj
                k = lid
                while k <= nj
                    gj = j0 + k - 1
                    @inbounds for c in 1:W
                        shared_xj[(c - 1) * SF_GPU_TILE + k] = x_mat[c, gj]
                    end
                    @inbounds for c in 1:F
                        shared_uj[(c - 1) * SF_GPU_TILE + k] = u_mat[c, gj]
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
                dbin = SFH.digitize(dist, ddig)
                if ok && 1 <= dbin < N_dist_edges
                    dU = SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                    val = SFT.pair_value(sf_type, geom, frame, dist, dU)
                    akey = SFC.pair_axis_key(second_axis, val, X1, X2, dist)
                    vbin = SFH.digitize(akey, vdig)
                    if 1 <= vbin < N_val_edges
                        idx = (dbin - 1) * NV + vbin
                        pw = SFC._point_weight(wts, i0 + ia - 1) *
                             SFC._point_weight(wts, jbase + jb - 1)
                        @atomic shared_sums[idx] += pw * val
                        @atomic shared_cnts[idx] += convert(CST, pw)
                    end
                end
                p += workgroup_size
            end
        end
    end
    @synchronize

    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    if bid <= n_tile_blocks
        b = lid
        while b <= NB2
            dbin = (b - 1) ÷ NV + 1
            vbin = b - (dbin - 1) * NV
            @atomic output_sums[dbin, vbin] += shared_sums[b]
            if shared_cnts[b] != zero(CST)
                @atomic output_counts[dbin, vbin] += shared_cnts[b]
            end
            b += workgroup_size
        end
    end
end

"""Static shared bytes of `_sf2d_kernel_tiled128_u32!` for `W`-wide coordinates and `F`-wide fields
staged as `FT`, sums of `OT` and `HIST` histogram cells counted in `CST`."""
@inline _joint2d_tiled_smem_bytes(::Type{FT}, ::Type{OT}, ::Type{CST}, W::Int, F::Int, HIST::Int) where {FT, OT, CST} =
    2 * SFC.gpu_localmem_bytes(FT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(FT, F * SF_GPU_TILE) +
    SFC.gpu_localmem_bytes(OT, HIST) + SFC.gpu_localmem_bytes(CST, HIST)

"""Whether `_sf2d_kernel_tiled128_u32!` compiled at width `HIST` fits the device `caps` describes;
the global-atomic joint kernel takes the call when it does not."""
@inline _gpu_joint_2d_tiled_eligible(caps, W::Int, F::Int, ::Type{FT}, ::Type{OT}, ::Type{CST},
                                     HIST::Int) where {FT, OT, CST} =
    SFC.gpu_static_smem_fits(caps, _joint2d_tiled_smem_bytes(FT, OT, CST, W, F, HIST))
