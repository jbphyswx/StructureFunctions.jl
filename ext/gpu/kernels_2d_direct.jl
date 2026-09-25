# HTP-EJ: tiled128 pair traversal for six-invariant-type single-pass 2D.
#
# On-chip (:shared, :typeplane): @localmem histogram during pair loop; block-end flush
#   via _sp2d_flush_*_to_output! (@atomic into out_sums/out_cnts, joint pattern).
# Direct (:direct): global atomics into block-private partition; merge on host.

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
    SF_GPU_SINGLE_PASS_N * n_dist * _sp2d_val_stride(n_val)

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

@inline function _sp2d_decode_flat_index(g, n_dist, n_val)
    t = (g - 1) ÷ (n_dist * n_val) + 1
    rem = (g - 1) % (n_dist * n_val)
    dbin = rem ÷ n_val + 1
    vbin = rem % n_val + 1
    return t, dbin, vbin
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
    for t in 1:SF_GPU_SINGLE_PASS_N
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
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
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
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
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

# --- direct partitioned accumulation (block-partitioned global atomics) ---

"""Accumulate one pair's six invariants into this block's global partition."""
@inline function _gpu_accumulate_sp2d_partitioned_direct!(
    partition_sums, partition_counts, block_id::Integer, vplan,
    dbin::Int, du_L, du_n2, N_val_edges::Int,
    w,
)
    vals = SA.SVector(SFC.single_pass_invariants(du_L, du_n2))
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _sf_value_bin(vplan, vals[t], t)
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += w * vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += convert(eltype(partition_counts), w)
        end
    end
    return nothing
end

# --- merge: serial (per-cell loop over blocks) and parallel (workgroup tree-reduce) ---

KA.@kernel unsafe_indices=true function _merge_sp2d_partitions_serial_u32!(
    output_sums,
    output_counts,
    partition_sums,
    partition_counts,
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int,
)
    g = @index(Global, Linear)
    C = SF_GPU_SINGLE_PASS_N * n_dist * n_val
    if g <= C
        t, dbin, vbin = _sp2d_decode_flat_index(g, n_dist, n_val)
        FT = eltype(output_sums)
        s = zero(FT)
        c = zero(UInt32)
        @inbounds for block_id in 1:n_tile_blocks
            s += partition_sums[t, dbin, vbin, block_id]
            c += partition_counts[t, dbin, vbin, block_id]
        end
        @inbounds begin
            output_sums[t, dbin, vbin] = s
            output_counts[t, dbin, vbin] = c
        end
    end
end

"""Parallel merge: one workgroup per joint cell (`ndrange = C × workgroup_size`)."""
KA.@kernel unsafe_indices=true function _merge_sp2d_partitions_parallel_u32!(
    output_sums::AbstractArray{FT, 3},
    output_counts::AbstractArray{UInt32, 3},
    partition_sums::AbstractArray{FT, 4},
    partition_counts::AbstractArray{UInt32, 4},
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int,
    workgroup_size::Int,
) where {FT}
    g_global = @index(Global, Linear)
    g = (g_global - 1) ÷ workgroup_size + 1
    lid = (g_global - 1) % workgroup_size + 1

    shared_t = @localmem Int (1,)
    shared_dbin = @localmem Int (1,)
    shared_vbin = @localmem Int (1,)
    shared_stride = @localmem Int (1,)
    shared_s = @localmem FT (256,)
    shared_c = @localmem UInt32 (256,)

    if lid == 1 && g <= SF_GPU_SINGLE_PASS_N * n_dist * n_val
        t, dbin, vbin = _sp2d_decode_flat_index(g, n_dist, n_val)
        @inbounds begin
            shared_t[1] = t
            shared_dbin[1] = dbin
            shared_vbin[1] = vbin
        end
    end
    @synchronize
    g_global = @index(Global, Linear)
    g = (g_global - 1) ÷ workgroup_size + 1
    lid = (g_global - 1) % workgroup_size + 1

    if g <= SF_GPU_SINGLE_PASS_N * n_dist * n_val
        t = @inbounds(shared_t[1])
        dbin = @inbounds(shared_dbin[1])
        vbin = @inbounds(shared_vbin[1])
        partial_s = zero(FT)
        partial_c = zero(UInt32)
        bid = lid
        while bid <= n_tile_blocks
            @inbounds begin
                partial_s += partition_sums[t, dbin, vbin, bid]
                partial_c += partition_counts[t, dbin, vbin, bid]
            end
            bid += workgroup_size
        end
        @inbounds shared_s[lid] = partial_s
        @inbounds shared_c[lid] = partial_c
    else
        @inbounds begin
            shared_s[lid] = zero(FT)
            shared_c[lid] = zero(UInt32)
        end
    end
    @synchronize
    g_global = @index(Global, Linear)
    lid = (g_global - 1) % workgroup_size + 1
    if lid == 1
        @inbounds shared_stride[1] = workgroup_size ÷ 2
    end
    @synchronize

    # 256-lane fixed workgroup tree reduction.
    for _ in 1:8
        g_global = @index(Global, Linear)
        g = (g_global - 1) ÷ workgroup_size + 1
        lid = (g_global - 1) % workgroup_size + 1
        @inbounds stride = shared_stride[1]
        stride > 0 || break
        if lid <= stride
            @inbounds begin
                shared_s[lid] += shared_s[lid + stride]
                shared_c[lid] += shared_c[lid + stride]
            end
        end
        @synchronize
        g_global = @index(Global, Linear)
        lid = (g_global - 1) % workgroup_size + 1
        if lid == 1
            @inbounds shared_stride[1] = shared_stride[1] ÷ 2
        end
        @synchronize
    end

    g_global = @index(Global, Linear)
    g = (g_global - 1) ÷ workgroup_size + 1
    lid = (g_global - 1) % workgroup_size + 1
    if lid == 1 && g <= SF_GPU_SINGLE_PASS_N * n_dist * n_val
        t = @inbounds(shared_t[1])
        dbin = @inbounds(shared_dbin[1])
        vbin = @inbounds(shared_vbin[1])
        @inbounds begin
            output_sums[t, dbin, vbin] = shared_s[1]
            output_counts[t, dbin, vbin] = shared_c[1]
        end
    end
end

const _SP2D_MERGE_MODES = (:parallel, :serial)

function _sp2d_merge_mode()
    sym = Symbol(lowercase(get(ENV, "SP2D_MERGE", "serial")))
    return sym in _SP2D_MERGE_MODES ? sym : :serial
end

"""
Stage tile `ti` (and `tj` when off-diagonal) into shared memory, component-major as
`(d-1)*SF_GPU_TILE + k`.
"""
function _sp2d_tiled_load_tile!(
    shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat,
    ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, ::Val{D},
) where {D}
    if ni > 0 && nj > 0
        k = lid
        while k <= ni
            gi = i0 + k - 1
            @inbounds for d in 1:D
                shared_xi[(d - 1) * SF_GPU_TILE + k] = x_mat[d, gi]
                shared_ui[(d - 1) * SF_GPU_TILE + k] = u_mat[d, gi]
            end
            k += workgroup_size
        end
        if ti < tj
            k = lid
            while k <= nj
                gj = j0 + k - 1
                @inbounds for d in 1:D
                    shared_xj[(d - 1) * SF_GPU_TILE + k] = x_mat[d, gj]
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
    ::Val{HC}, ::Val{D}, ::Val{CST}, wts, geom,
) where {OT, FT, HC, D, CST}
    shared_xi = @localmem FT (D * SF_GPU_TILE,)
    shared_ui = @localmem FT (D * SF_GPU_TILE,)
    shared_xj = @localmem FT (D * SF_GPU_TILE,)
    shared_uj = @localmem FT (D * SF_GPU_TILE,)
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
            ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, Val(D),
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
                X1 = _sf_load_pt(Val(D), shared_xi, ia)
                X2 = _sf_load_pt(Val(D), shared_xj, jb)
                U1 = _sf_load_pt(Val(D), shared_ui, ia)
                U2 = _sf_load_pt(Val(D), shared_uj, jb)
            else
                ia, jb = _pair_from_linear(p, ni)
                X1 = _sf_load_pt(Val(D), shared_xi, ia)
                X2 = _sf_load_pt(Val(D), shared_xi, jb)
                U1 = _sf_load_pt(Val(D), shared_ui, ia)
                U2 = _sf_load_pt(Val(D), shared_ui, jb)
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
    ::Val{HC}, ::Val{D}, ::Val{CST}, wts, geom,
) where {OT, FT, HC, D, CST}
    shared_xi = @localmem FT (D * SF_GPU_TILE,)
    shared_ui = @localmem FT (D * SF_GPU_TILE,)
    shared_xj = @localmem FT (D * SF_GPU_TILE,)
    shared_uj = @localmem FT (D * SF_GPU_TILE,)
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
            ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, Val(D),
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
                    X1 = _sf_load_pt(Val(D), shared_xi, ia)
                    X2 = _sf_load_pt(Val(D), shared_xj, jb)
                    U1 = _sf_load_pt(Val(D), shared_ui, ia)
                    U2 = _sf_load_pt(Val(D), shared_uj, jb)
                else
                    ia, jb = _pair_from_linear(p, ni)
                    X1 = _sf_load_pt(Val(D), shared_xi, ia)
                    X2 = _sf_load_pt(Val(D), shared_xi, jb)
                    U1 = _sf_load_pt(Val(D), shared_ui, ia)
                    U2 = _sf_load_pt(Val(D), shared_ui, jb)
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

KA.@kernel unsafe_indices=true function _sf6_sp2d_directpartition_tiled128_u32!(
    partition_sums::AbstractArray{OT}, partition_counts,
    x_mat::AbstractMatrix{FT}, u_mat::AbstractMatrix{FT},
    N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
    ddig, vplan,
    sched, n_tile_blocks::Int, workgroup_size::Int,
    C::Int, plane::Int, types_per_pass::Int, n_type_passes::Int,
    ::Val{HC}, ::Val{D}, ::Val{CST}, wts, geom,
) where {OT, FT, HC, D, CST}
    shared_xi = @localmem FT (D * SF_GPU_TILE,)
    shared_ui = @localmem FT (D * SF_GPU_TILE,)
    shared_xj = @localmem FT (D * SF_GPU_TILE,)
    shared_uj = @localmem FT (D * SF_GPU_TILE,)
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
            ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size, Val(D),
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
    if @inbounds(shared_block_id[1]) <= n_tile_blocks &&
       @inbounds(shared_tile[3]) > 0 && @inbounds(shared_tile[4]) > 0
        block_id = @inbounds(shared_block_id[1])
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
                X1 = _sf_load_pt(Val(D), shared_xi, ia)
                X2 = _sf_load_pt(Val(D), shared_xj, jb)
                U1 = _sf_load_pt(Val(D), shared_ui, ia)
                U2 = _sf_load_pt(Val(D), shared_uj, jb)
            else
                ia, jb = _pair_from_linear(p, ni)
                X1 = _sf_load_pt(Val(D), shared_xi, ia)
                X2 = _sf_load_pt(Val(D), shared_xi, jb)
                U1 = _sf_load_pt(Val(D), shared_ui, ia)
                U2 = _sf_load_pt(Val(D), shared_ui, jb)
            end
            ok, dist, frame = SFH.pair_frame(geom, X1, X2)
            bin = SFH.digitize(dist, ddig)
            if ok && 1 <= bin < N_bins
                du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                pw = SFC._point_weight(wts, i0w + ia - 1) * SFC._point_weight(wts, jbw + jb - 1)
                _gpu_accumulate_sp2d_partitioned_direct!(
                    partition_sums, partition_counts, block_id, vplan,
                    bin, du_L, du_n2, N_val_edges, pw,
                )
            end
            p += workgroup_size
        end
    end
end

function _launch_merge_sp2d_partitions!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    partition_sums_dev,
    partition_counts_dev,
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int;
    merge_mode::Symbol = _sp2d_merge_mode(),
)
    C = SF_GPU_SINGLE_PASS_N * n_dist * n_val
    if merge_mode == :serial
        kernel! = _merge_sp2d_partitions_serial_u32!(backend, 256)
        kernel!(
            out_sums_dev, out_cnts_dev, partition_sums_dev, partition_counts_dev,
            n_dist, n_val, n_tile_blocks;
            ndrange = C,
        )
    else
        ws = 256
        kernel! = _merge_sp2d_partitions_parallel_u32!(backend, ws)
        kernel!(
            out_sums_dev, out_cnts_dev, partition_sums_dev, partition_counts_dev,
            n_dist, n_val, n_tile_blocks, ws;
            ndrange = C * ws,
            workgroupsize = (ws,),
        )
    end
    KA.synchronize(backend)
    return nothing
end
