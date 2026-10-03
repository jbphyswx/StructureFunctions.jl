# HTP-EJ: tiled128 pair traversal for six-invariant-type single-pass 2D.
#
# On-chip (:shared, :typeplane): @localmem histogram during pair loop; block-end flush
#   via _sp2d_flush_*_to_output! (@atomic into out_sums/out_cnts, joint pattern).

"""
    _sp2d_val_stride(n_val)

Row stride of the shared histogram's value axis, forced odd.

Shared memory has 32 banks. With a stride of `n_val = 16`, the flat index
`(dbin-1)*n_val + vbin` puts every row of the value axis in the same 2 banks, so lanes that differ
only in `dbin` serialize up to 16 ways on bank conflicts alone — before any same-address atomic
contention. An odd stride is coprime with 32 and spreads them over all banks. The cost is one unused
column per row.
"""
@inline _sp2d_val_stride(n_val) = isodd(n_val) ? n_val : n_val + 1

"""Cells in the bank-conflict-padded shared histogram."""
@inline _sp2d_shared_cells(n_dist, n_val) =
    SINGLE_PASS_N * n_dist * _sp2d_val_stride(n_val)

"""Flat index into the padded shared histogram."""
@inline function _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
    s = _sp2d_val_stride(n_val)
    return (t - 1) * n_dist * s + (dbin - 1) * s + vbin
end

"""Invert [`_sp2d_shared_index`](@ref); `vbin > n_val` marks a padding hole to skip."""
@inline function _sp2d_shared_decode(g, n_dist, n_val)
    s = _sp2d_val_stride(n_val)
    plane = n_dist * s
    t = (g - 1) ÷ plane + 1
    r = (g - 1) % plane
    return t, r ÷ s + 1, r % s + 1
end

"""Joint-style flush: `@atomic` block-local shared histogram into final output (on-chip path)."""
@inline function _sp2d_flush_shared_to_output!(
    out_sums,
    out_cnts,
    shared_sums,
    shared_cnts,
    C,
    n_dist,
    n_val,
    lid,
    workgroup_size,
)
    g = lid
    while g <= C
        t, dbin, vbin = _sp2d_shared_decode(g, n_dist, n_val)
        if vbin <= n_val
            @inbounds begin
                @atomic out_sums[t, dbin, vbin] += shared_sums[g]
                if shared_cnts[g] != zero(eltype(shared_cnts))
                    @atomic out_cnts[t, dbin, vbin] += shared_cnts[g]
                end
            end
        end
        g += workgroup_size
    end
    return nothing
end

"""Accumulate one pair's six invariants into the padded shared histogram."""
@inline function _gpu_accumulate_sp2d_sharedhist!(
    shared_sums, shared_cnts, n_dist, n_val, vplan,
    dbin::Int, du_L, du_n2, N_val_edges::Int,
    w,
)
    vals = SA.SVector(SFC.single_pass_invariants(du_L, du_n2))
    for t in 1:SINGLE_PASS_N
        vbin = _sf_value_bin(vplan, vals[t], t)
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += w * vals[t]
            @atomic shared_cnts[g] += convert(eltype(shared_cnts), w)
        end
    end
    return nothing
end

# --- type-plane shared accumulation (one SF type per pass; plane = n_dist × n_val) ---

"""Flat index within one padded type plane; same bank argument as [`_sp2d_val_stride`](@ref)."""
@inline function _sp2d_plane_flat_index(dbin::Int, vbin::Int, n_val::Int)
    return (dbin - 1) * _sp2d_val_stride(n_val) + vbin
end

"""Cells in one padded type plane."""
@inline _sp2d_plane_cells(n_dist::Int, n_val::Int) = n_dist * _sp2d_val_stride(n_val)

"""Joint-style flush for type-plane shared histogram into final output (on-chip path)."""
@inline function _sp2d_flush_typeplane_to_output!(
    out_sums,
    out_cnts,
    shared_sums,
    shared_cnts,
    type_pass::Integer,
    types_per_pass::Integer,
    plane::Integer,
    n_dist::Integer,
    n_val::Integer,
    lid::Integer,
    workgroup_size::Integer,
)
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SINGLE_PASS_N, type_pass * types_per_pass)
    stride = _sp2d_val_stride(n_val)
    g = Int(lid)
    while g <= plane
        dbin = (g - 1) ÷ stride + 1
        vbin = (g - 1) % stride + 1
        if vbin <= n_val
            for t in t_lo:t_hi
                slot = t - t_lo
                @inbounds idx = slot * plane + g
                @inbounds @atomic out_sums[t, dbin, vbin] += shared_sums[idx]
                @inbounds if shared_cnts[idx] != UInt32(0)
                    @atomic out_cnts[t, dbin, vbin] += shared_cnts[idx]
                end
            end
        end
        g += workgroup_size
    end
    return nothing
end

"""Accumulate one pair's invariants of this type pass into the type-plane shared histogram."""
@inline function _gpu_accumulate_sp2d_typeplane!(
    shared_sums, shared_cnts, n_val, plane::Int, type_pass::Int, types_per_pass::Int, vplan,
    dbin::Int, du_L, du_n2, N_val_edges::Int,
    w,
)
    vals = SA.SVector(SFC.single_pass_invariants(du_L, du_n2))
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _sf_value_bin(vplan, vals[t], t)
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += w * vals[t]
            @atomic shared_cnts[g] += convert(eltype(shared_cnts), w)
        end
    end
    return nothing
end

"""
Stage tile `ti` (and `tj` when off-diagonal) into shared memory, component-major as
`(d-1)*SF_GPU_TILE + k`.
"""
function _sp2d_tiled_load_tile!(
    shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat,
    ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, ::Val{W}, ::Val{F},
) where {W, F}
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
    return nothing
end

# --- HTP-EJ pair kernels, one per accumulation mode ---
#
# Each stages its tile pair, walks the pairs, and accumulates through the mode's helper. The tile
# coordinates are held in `shared_tile` so every section after a `@synchronize` rereads them.

KA.@kernel unsafe_indices=true function _sf6_sp2d_sharedhist_tiled128_u32!(
    out_sums::AbstractArray{OT}, out_cnts, x_mat::AbstractMatrix{FT}, u_mat::AbstractMatrix{FT},
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    ddig, vplan,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    C::Int, plane::Int, types_per_pass::Int, n_type_passes::Int,
    ::Val{HC}, ::Val{W}, ::Val{F}, ::Val{CST}, wts, geom,
) where {OT, FT, HC, W, F, CST}
    shared_xi = @localmem FT (W * SF_GPU_TILE,)
    shared_ui = @localmem FT (F * SF_GPU_TILE,)
    shared_xj = @localmem FT (W * SF_GPU_TILE,)
    shared_uj = @localmem FT (F * SF_GPU_TILE,)
    shared_sums = @localmem OT (HC,)
    shared_cnts = @localmem CST (HC,)
    shared_block_id = @localmem Int (1,)
    shared_tile = @localmem Int (4,)

    g = @index(Global, Linear)
    if (g - 1) % workgroup_size + 1 == 1
        @inbounds shared_block_id[1] = (g - 1) ÷ workgroup_size + 1
    end
    @synchronize
    lid = @index(Local, Linear)
    if @inbounds(shared_block_id[1]) <= n_tile_blocks
        ti, tj = tile_for(sched, @inbounds(shared_block_id[1]))
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(
            shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat,
            ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, Val(W), Val(F),
        )
    end
    @synchronize
    lid = @index(Local, Linear)
    if @inbounds(shared_block_id[1]) <= n_tile_blocks
        ti, tj = tile_for(sched, @inbounds(shared_block_id[1]))
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if lid == 1
            @inbounds shared_tile[1] = ti
            @inbounds shared_tile[2] = tj
            @inbounds shared_tile[3] = ni
            @inbounds shared_tile[4] = nj
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    if @inbounds(shared_block_id[1]) <= n_tile_blocks
        g_zero = lid
        while g_zero <= C
            @inbounds begin
                shared_sums[g_zero] = zero(OT)
                shared_cnts[g_zero] = zero(CST)
            end
            g_zero += workgroup_size
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    if @inbounds(shared_block_id[1]) <= n_tile_blocks &&
       @inbounds(shared_tile[3]) > 0 && @inbounds(shared_tile[4]) > 0
        @inbounds ti = shared_tile[1]
        @inbounds tj = shared_tile[2]
        @inbounds ni = shared_tile[3]
        @inbounds nj = shared_tile[4]
        n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
        i0w = (ti - 1) * SF_GPU_TILE + 1
        jbw = ti < tj ? (tj - 1) * SF_GPU_TILE + 1 : i0w
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
            bin = SFH.digitize(dist, ddig)
            if ok && 1 <= bin < N_bins
                du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                pw = SFC._point_weight(wts, i0w + ia - 1) * SFC._point_weight(wts, jbw + jb - 1)
                _gpu_accumulate_sp2d_sharedhist!(
                    shared_sums, shared_cnts, NB, N_val_edges - 1, vplan,
                    bin, du_L, du_n2, N_val_edges, pw,
                )
            end
            p += workgroup_size
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    _sp2d_flush_shared_to_output!(
        out_sums, out_cnts, shared_sums, shared_cnts, C, NB, N_val_edges - 1,
        lid, workgroup_size,
    )
end

KA.@kernel unsafe_indices=true function _sf6_sp2d_typeplane_tiled128_u32!(
    out_sums::AbstractArray{OT}, out_cnts, x_mat::AbstractMatrix{FT}, u_mat::AbstractMatrix{FT},
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    ddig, vplan,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    C::Int, plane::Int, types_per_pass::Int, n_type_passes::Int,
    ::Val{HC}, ::Val{W}, ::Val{F}, ::Val{CST}, wts, geom,
) where {OT, FT, HC, W, F, CST}
    shared_xi = @localmem FT (W * SF_GPU_TILE,)
    shared_ui = @localmem FT (F * SF_GPU_TILE,)
    shared_xj = @localmem FT (W * SF_GPU_TILE,)
    shared_uj = @localmem FT (F * SF_GPU_TILE,)
    shared_sums = @localmem OT (HC,)
    shared_cnts = @localmem CST (HC,)
    shared_type_pass = @localmem Int (1,)
    shared_block_id = @localmem Int (1,)
    shared_tile = @localmem Int (4,)

    g = @index(Global, Linear)
    if (g - 1) % workgroup_size + 1 == 1
        @inbounds shared_block_id[1] = (g - 1) ÷ workgroup_size + 1
    end
    @synchronize
    lid = @index(Local, Linear)
    if @inbounds(shared_block_id[1]) <= n_tile_blocks
        ti, tj = tile_for(sched, @inbounds(shared_block_id[1]))
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        _sp2d_tiled_load_tile!(
            shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat,
            ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, Val(W), Val(F),
        )
    end
    @synchronize
    lid = @index(Local, Linear)
    if @inbounds(shared_block_id[1]) <= n_tile_blocks
        ti, tj = tile_for(sched, @inbounds(shared_block_id[1]))
        i0 = (ti - 1) * SF_GPU_TILE + 1
        j0 = (tj - 1) * SF_GPU_TILE + 1
        ni = min(SF_GPU_TILE, N_points - i0 + 1)
        nj = min(SF_GPU_TILE, N_points - j0 + 1)
        if lid == 1
            @inbounds shared_tile[1] = ti
            @inbounds shared_tile[2] = tj
            @inbounds shared_tile[3] = ni
            @inbounds shared_tile[4] = nj
        end
    end
    @synchronize
    lid = @index(Local, Linear)
    if lid == 1
        @inbounds shared_type_pass[1] = 1
    end
    @synchronize
    lid = @index(Local, Linear)
    if @inbounds(shared_block_id[1]) <= n_tile_blocks &&
       @inbounds(shared_tile[3]) > 0 && @inbounds(shared_tile[4]) > 0
        for _ in 1:n_type_passes
            lid = @index(Local, Linear)
            g_zero = lid
            while g_zero <= types_per_pass * plane
                @inbounds begin
                    shared_sums[g_zero] = zero(OT)
                    shared_cnts[g_zero] = zero(CST)
                end
                g_zero += workgroup_size
            end
            @synchronize
            lid = @index(Local, Linear)
            @inbounds ti = shared_tile[1]
            @inbounds tj = shared_tile[2]
            @inbounds ni = shared_tile[3]
            @inbounds nj = shared_tile[4]
            @inbounds type_pass = shared_type_pass[1]
            n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
            i0w = (ti - 1) * SF_GPU_TILE + 1
            jbw = ti < tj ? (tj - 1) * SF_GPU_TILE + 1 : i0w
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
                bin = SFH.digitize(dist, ddig)
                if ok && 1 <= bin < N_bins
                    du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                    pw = SFC._point_weight(wts, i0w + ia - 1) * SFC._point_weight(wts, jbw + jb - 1)
                    _gpu_accumulate_sp2d_typeplane!(
                        shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
                        vplan, bin, du_L, du_n2, N_val_edges, pw,
                    )
                end
                p += workgroup_size
            end
            @synchronize
            lid = @index(Local, Linear)
            @inbounds type_pass = shared_type_pass[1]
            _sp2d_flush_typeplane_to_output!(
                out_sums, out_cnts, shared_sums, shared_cnts, type_pass, types_per_pass, plane,
                NB, N_val_edges - 1, lid, workgroup_size,
            )
            @synchronize
            lid = @index(Local, Linear)
            if lid == 1
                @inbounds shared_type_pass[1] += 1
            end
            @synchronize
        end
    end
end

"""Static shared bytes every HTP-EJ pair kernel stages for `W`-wide coordinates and `F`-wide fields of
`FT`: four point tiles and the block id and tile coordinates, which the kernels index by constants."""
@inline _sp2d_staging_smem_bytes(::Type{FT}, W::Int, F::Int) where {FT} =
    2 * SFC.gpu_localmem_bytes(FT, W * SF_GPU_TILE) + 2 * SFC.gpu_localmem_bytes(FT, F * SF_GPU_TILE) +
    SFC.gpu_localmem_scalar_bytes(Int, 1 + 4)

"""Static shared bytes of `_sf6_sp2d_sharedhist_tiled128_u32!` with an `HC`-cell histogram of sums
`OT` and counts `CST`."""
@inline _sp2d_sharedhist_smem_bytes(::Type{FT}, ::Type{OT}, ::Type{CST}, W::Int, F::Int, HC::Int) where {FT, OT, CST} =
    _sp2d_staging_smem_bytes(FT, W, F) + SFC.gpu_localmem_bytes(OT, HC) + SFC.gpu_localmem_bytes(CST, HC)

"""Static shared bytes of `_sf6_sp2d_typeplane_tiled128_u32!`: the shared-histogram kernel's and the
type pass it holds."""
@inline _sp2d_typeplane_smem_bytes(::Type{FT}, ::Type{OT}, ::Type{CST}, W::Int, F::Int, HC::Int) where {FT, OT, CST} =
    _sp2d_sharedhist_smem_bytes(FT, OT, CST, W, F, HC) + SFC.gpu_localmem_scalar_bytes(Int, 1)
