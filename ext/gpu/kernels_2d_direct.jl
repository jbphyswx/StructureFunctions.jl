# HTP-EJ: tiled128 pair traversal for six-invariant-type single-pass 2D.
#
# On-chip (:shared, :typeplane): @localmem histogram during pair loop; block-end flush
#   via _sp2d_flush_*_to_output! (@atomic into out_sums/out_cnts, joint pattern).
# Direct (:direct): global atomics into block-private partition; merge on host.
#
# See gpu/SP2D_HTP_EJ.md for strategy, benchmarks, and future perf notes.
# Included from StructureFunctionsKernelAbstractionsExt.jl after TiledSinglePass2DValueKernels.jl.

"""Load point `k`'s D-vector from a tile buffer staged as `(d-1)*SF_GPU_TILE + k`."""
@inline _sp2d_ld_tile(buf, ::Val{2}, k) =
    @inbounds SA.SVector{2}(buf[k], buf[SF_GPU_TILE + k])
@inline _sp2d_ld_tile(buf, ::Val{3}, k) =
    @inbounds SA.SVector{3}(buf[k], buf[SF_GPU_TILE + k], buf[2 * SF_GPU_TILE + k])

@inline function _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
    return (t - 1) * n_dist * n_val + (dbin - 1) * n_val + vbin
end

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

# Index args are ::Integer, not ::Int: on CUDA, @index(Local, Linear) yields Int32
# (threadIdx().x), and ::Int-typed methods would not match — the resulting
# MethodError cannot even be thrown in device code, so the kernel fails to compile
# (InvalidIRError: gpu_gc_pool_alloc / jl_f_throw_methoderror). Coerce once inside.
@inline function _sp2d_zero_shared_hist!(
    shared_sums,
    shared_cnts,
    C::Integer,
    lid::Integer,
    workgroup_size::Integer,
)
    g = Int(lid)
    FT = eltype(shared_sums)
    while g <= C
        @inbounds begin
            shared_sums[g] = zero(FT)
            shared_cnts[g] = zero(UInt32)
        end
        g += workgroup_size
    end
    return nothing
end

"""Flush block-local shared histogram into block-private partition (`:direct` path only)."""
@inline function _sp2d_flush_shared_hist!(
    partition_sums,
    partition_counts,
    shared_sums,
    shared_cnts,
    C,
    n_dist,
    n_val,
    block_id,
    lid,
    workgroup_size,
)
    g = lid
    while g <= C
        t, dbin, vbin = _sp2d_shared_decode(g, n_dist, n_val)
        if vbin <= n_val
            @inbounds begin
                partition_sums[t, dbin, vbin, block_id] += shared_sums[g]
                if shared_cnts[g] != UInt32(0)
                    partition_counts[t, dbin, vbin, block_id] += shared_cnts[g]
                end
            end
        end
        g += workgroup_size
    end
    return nothing
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
                if shared_cnts[g] != UInt32(0)
                    @atomic out_cnts[t, dbin, vbin] += shared_cnts[g]
                end
            end
        end
        g += workgroup_size
    end
    return nothing
end

# --- shared-histogram accumulation (flat index g, no strip filter) ---

@inline function _gpu_accumulate_sp2d_sharedhist_linear_val!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin, du_L, du_L2, du_T2, N_val_edges,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_linear_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    val_first, val_last, val_inv_step, val_step,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_step[t],
            N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_inflinear_val!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first, val_last, val_inv_step, val_step,
            n_inner_edges, inner_last,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_inflinear_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    val_first, val_last, val_inv_step, val_step, inner_last,
    dbin::Int, du_L, du_L2, du_T2, n_inner_edges::Int, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_step[t],
            n_inner_edges, inner_last[t],
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_log_val!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced(
            vals[t], val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_log_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first, val_last, val_inv_step, val_step,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced_col(
            vals[t], val_first, val_last, val_inv_step, val_step, t, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_vector_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    value_edges,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_general_col(vals[t], value_edges, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            g = _sp2d_shared_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
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

"""Flush type-plane shared histogram into block-private partition (`:direct` path only)."""
@inline function _sp2d_flush_typeplane!(
    partition_sums,
    partition_counts,
    shared_sums,
    shared_cnts,
    type_pass::Integer,
    types_per_pass::Integer,
    plane::Integer,
    n_dist::Integer,
    n_val::Integer,
    block_id::Integer,
    lid::Integer,
    workgroup_size::Integer,
)
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    g = Int(lid)
    while g <= plane
        dbin = (g - 1) ÷ n_val + 1
        vbin = (g - 1) % n_val + 1
        for t in t_lo:t_hi
            slot = t - t_lo
            @inbounds idx = slot * plane + g
            @inbounds partition_sums[t, dbin, vbin, block_id] += shared_sums[idx]
            if shared_cnts[idx] != UInt32(0)
                @inbounds partition_counts[t, dbin, vbin, block_id] += shared_cnts[idx]
            end
        end
        g += workgroup_size
    end
    return nothing
end

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

@inline function _gpu_accumulate_sp2d_typeplane_linear_val!(
    shared_sums, shared_cnts, n_val, plane::Int,
    type_pass::Int, types_per_pass::Int,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _gpu_digitize_linear(
            vals[t], val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_typeplane_linear_val_cols!(
    shared_sums, shared_cnts, n_val, plane::Int, type_pass::Int, types_per_pass::Int,
    val_first, val_last, val_inv_step, val_step,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _gpu_digitize_linear(
            vals[t], val_first[t], val_last[t],
            val_inv_step[t], val_step[t],
            N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_typeplane_inflinear_val!(
    shared_sums, shared_cnts, n_val, plane::Int, type_pass::Int, types_per_pass::Int,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first, val_last, val_inv_step, val_step,
            n_inner_edges, inner_last,
        )
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_typeplane_inflinear_val_cols!(
    shared_sums, shared_cnts, n_val, plane::Int, type_pass::Int, types_per_pass::Int,
    val_first, val_last, val_inv_step, val_step, inner_last,
    dbin::Int, du_L, du_L2, du_T2, n_inner_edges::Int, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first[t], val_last[t],
            val_inv_step[t], val_step[t],
            n_inner_edges, inner_last[t],
        )
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_typeplane_log_val!(
    shared_sums, shared_cnts, n_val, plane::Int, type_pass::Int, types_per_pass::Int,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _gpu_digitize_log_spaced(
            vals[t], val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_typeplane_log_val_cols!(
    shared_sums, shared_cnts, n_val, plane::Int, type_pass::Int, types_per_pass::Int,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first, val_last, val_inv_step, val_step,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _gpu_digitize_log_spaced_col(
            vals[t], val_first, val_last, val_inv_step, val_step,
            t, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_typeplane_vector_val_cols!(
    shared_sums, shared_cnts, n_val, plane::Int, type_pass::Int, types_per_pass::Int,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    value_edges,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    for t in t_lo:t_hi
        vbin = _gpu_digitize_general_col(vals[t], value_edges, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            slot = t - t_lo
            g = slot * plane + _sp2d_plane_flat_index(dbin, vbin, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

# --- direct partitioned accumulation (block-partitioned global atomics) ---

@inline function _gpu_accumulate_sp2d_partitioned_direct_linear_val!(
    partition_sums, partition_counts, block_id::Integer, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_partitioned_direct_linear_val_cols!(
    partition_sums, partition_counts, block_id::Integer, n_dist, n_val,
    val_first, val_last, val_inv_step, val_step,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_step[t],
            N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_partitioned_direct_inflinear_val!(
    partition_sums, partition_counts, block_id::Integer, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first, val_last, val_inv_step, val_step,
            n_inner_edges, inner_last,
        )
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_partitioned_direct_inflinear_val_cols!(
    partition_sums, partition_counts, block_id::Integer, n_dist, n_val,
    val_first, val_last, val_inv_step, val_step, inner_last,
    dbin::Int, du_L, du_L2, du_T2, n_inner_edges::Int, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_step[t],
            n_inner_edges, inner_last[t],
        )
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_partitioned_direct_log_val!(
    partition_sums, partition_counts, block_id::Integer, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced(
            vals[t], val_first, val_last, val_inv_step, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_partitioned_direct_log_val_cols!(
    partition_sums, partition_counts, block_id::Integer, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    val_first, val_last, val_inv_step, val_step,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced_col(
            vals[t], val_first, val_last, val_inv_step, val_step, t, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_partitioned_direct_vector_val_cols!(
    partition_sums, partition_counts, block_id::Integer, n_dist, n_val,
    dbin::Int, du_L, du_L2, du_T2, N_val_edges::Int,
    value_edges,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_general_col(vals[t], value_edges, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            @atomic partition_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic partition_counts[t, dbin, vbin, block_id] += UInt32(1)
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

# --- kernel codegen ---

function _sp2d_partition_dist_spec(::Val{:linear})
    params = [
        :(dist_first), :(dist_last), :(dist_inv_step),
        :(dist_step),
    ]
    bin = :(_gpu_digitize_linear(
        dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins,
    ))
    return params, bin
end

function _sp2d_partition_dist_spec(::Val{:log_linear})
    params = [
        :(dist_first), :(dist_last), :(dist_inv_step),
        :(dist_step),
    ]
    bin = :(_gpu_digitize_log_spaced(
        dist, dist_first, dist_last, dist_inv_step, dist_step, N_bins,
    ))
    return params, bin
end

"""Arbitrary distance edges: binary search on device, so no bin spacing is assumed."""
function _sp2d_partition_dist_spec(::Val{:general})
    params = [:(@Const(distance_edges))]
    bin = :(_gpu_digitize_general(dist, distance_edges, N_bins))
    return params, bin
end

function _sp2d_partition_val_accum(::Val{:linear_shared}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_linear_val!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:linear_shared}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_partitioned_direct_linear_val!(
            partition_sums, partition_counts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end

function _sp2d_partition_val_accum(::Val{:linear_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_linear_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_step,
            bin, du_L, du_L2, du_T2, N_val_edges,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:linear_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_partitioned_direct_linear_val_cols!(
            partition_sums, partition_counts, block_id, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_step,
            bin, du_L, du_L2, du_T2, N_val_edges,
        )
    end
end

function _sp2d_partition_val_accum(::Val{:inflinear_shared}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_inflinear_val!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step, n_inner_edges, inner_last,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:inflinear_shared}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_partitioned_direct_inflinear_val!(
            partition_sums, partition_counts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step, n_inner_edges, inner_last,
        )
    end
end

function _sp2d_partition_val_accum(::Val{:inflinear_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_inflinear_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_step, inner_last,
            bin, du_L, du_L2, du_T2, n_inner_edges, N_val_edges,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:inflinear_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_partitioned_direct_inflinear_val_cols!(
            partition_sums, partition_counts, block_id, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_step, inner_last,
            bin, du_L, du_L2, du_T2, n_inner_edges, N_val_edges,
        )
    end
end

function _sp2d_partition_val_accum(::Val{:log_linear_shared}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_log_val!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:log_linear_shared}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_partitioned_direct_log_val!(
            partition_sums, partition_counts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end

function _sp2d_partition_val_accum(::Val{:log_linear_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_log_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:log_linear_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_partitioned_direct_log_val_cols!(
            partition_sums, partition_counts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end

function _sp2d_partition_val_accum(::Val{:vector_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_vector_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges, value_edges,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:vector_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_partitioned_direct_vector_val_cols!(
            partition_sums, partition_counts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_L2, du_T2, N_val_edges, value_edges,
        )
    end
end

# type-plane (one or more SF types per pass; shared hist = types_per_pass × plane)
function _sp2d_partition_val_accum(::Val{:linear_shared}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_linear_val!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:linear_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_linear_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            val_first, val_last, val_inv_step, val_step,
            bin, du_L, du_L2, du_T2, N_val_edges,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:inflinear_shared}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_inflinear_val!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step, n_inner_edges, inner_last,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:inflinear_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_inflinear_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            val_first, val_last, val_inv_step, val_step, inner_last,
            bin, du_L, du_L2, du_T2, n_inner_edges, N_val_edges,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:log_linear_shared}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_log_val!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:log_linear_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_log_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_step,
        )
    end
end
function _sp2d_partition_val_accum(::Val{:vector_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_vector_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_L2, du_T2, N_val_edges, value_edges,
        )
    end
end

function _sp2d_partition_kernel_prefix(::Val{:shared})
    return :_sf6_sp2d_sharedhist_tiled128_
end
function _sp2d_partition_kernel_prefix(::Val{:typeplane})
    return :_sf6_sp2d_typeplane_tiled128_
end
function _sp2d_partition_kernel_prefix(::Val{:direct})
    return :_sf6_sp2d_directpartition_tiled128_
end

function _sp2d_partition_val_params(::Val{:linear_shared})
    return [:(val_first), :(val_last), :(val_inv_step), :(val_step)]
end
function _sp2d_partition_val_params(::Val{:linear_cols})
    return [:(val_first), :(val_last), :(val_inv_step), :(val_step)]
end
function _sp2d_partition_val_params(::Val{:inflinear_shared})
    return [
        :(val_first), :(val_last), :(val_inv_step), :(val_step),
        :(n_inner_edges::Int), :(inner_last),
    ]
end
function _sp2d_partition_val_params(::Val{:inflinear_cols})
    return [
        :(val_first), :(val_last), :(val_inv_step), :(val_step),
        :(inner_last), :(n_inner_edges::Int),
    ]
end
function _sp2d_partition_val_params(::Val{:log_linear_shared})
    return [:(val_first), :(val_last), :(val_inv_step), :(val_step)]
end
function _sp2d_partition_val_params(::Val{:log_linear_cols})
    return [:(val_first), :(val_last), :(val_inv_step), :(val_step)]
end
function _sp2d_partition_val_params(::Val{:vector_cols})
    return [:(@Const(value_edges))]
end

function _sp2d_partition_dist_spec(dist::Symbol)
    return _sp2d_partition_dist_spec(Val(dist))
end

"""Emit one HTP-EJ tiled128 kernel for `accum_mode` (`:shared`, `:typeplane`, or `:direct`)."""
function _sp2d_partition_kernel_def(accum_mode::Symbol, dist::Symbol, val::Symbol)
    prefix = _sp2d_partition_kernel_prefix(Val(accum_mode))
    fname = Symbol(prefix, dist, :_, val, :_u32!)
    dist_params, dist_bin = _sp2d_partition_dist_spec(dist)
    val_params = _sp2d_partition_val_params(Val(val))
    accum = _sp2d_partition_val_accum(Val(val), Val(accum_mode))
    uses_shared = accum_mode in (:shared, :typeplane)
    # Width is a kernel type parameter, so one definition specializes per config instead of every
    # config paying for the largest one the budget allows.
    shared_hist_decl = uses_shared ? quote
        shared_sums = @localmem OT (HC,)
        shared_cnts = @localmem UInt32 (HC,)
    end : quote end
    type_pass_decl = accum_mode == :typeplane ? quote
        shared_type_pass = @localmem Int (1,)
    end : quote end
    kernel_tail_params = [
        :(n_tiles::Int), :(n_tile_blocks::Int), :(workgroup_size::Int),
        :(C::Int), :(plane::Int), :(types_per_pass::Int), :(n_type_passes::Int),
        :(::Val{HC}), :(::Val{D}), :(geom),
    ]
    pair_loop = quote
        p = lid
        while p <= n_pairs
            if ti < tj
                ia = (p - 1) ÷ nj + 1
                jb = (p - 1) - (ia - 1) * nj + 1
                X1 = _sp2d_ld_tile(shared_xi, Val(D), ia)
                X2 = _sp2d_ld_tile(shared_xj, Val(D), jb)
                U1 = _sp2d_ld_tile(shared_ui, Val(D), ia)
                U2 = _sp2d_ld_tile(shared_uj, Val(D), jb)
            else
                ia, jb = _pair_from_linear(p, ni)
                X1 = _sp2d_ld_tile(shared_xi, Val(D), ia)
                X2 = _sp2d_ld_tile(shared_xi, Val(D), jb)
                U1 = _sp2d_ld_tile(shared_ui, Val(D), ia)
                U2 = _sp2d_ld_tile(shared_ui, Val(D), jb)
            end
            ok, dist, frame = SFH.pair_frame(geom, X1, X2)
            bin = $(dist_bin)
            if ok && 1 <= bin < N_bins
                # |du_T|² from the norm rather than a transverse basis vector: the six invariants
                # use only du_L, du_L², du_T², never the signed du_T, and building n̂ = (r̂₂, -r̂₁)
                # is both wasted work and the kernel's only 2D-specific assumption. This is also
                # exactly the CPU's formula, so the two agree to the last bit of the algorithm.
                du_L, du_n2 = SFH.pair_invariants(geom, frame, dist, U1, U2)
                du_L2 = du_L * du_L
                du_T2 = du_n2 - du_L2
                $(accum)
            end
            p += workgroup_size
        end
    end
    accum_body = if accum_mode == :typeplane
        quote
            @synchronize
    lid = @index(Local, Linear)
            if lid == 1
                @inbounds shared_type_pass[1] = 1
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
                for _ in 1:n_type_passes
    lid = @index(Local, Linear)
                    @inbounds type_pass = shared_type_pass[1]
                    let g_zero = lid
                        while g_zero <= types_per_pass * plane
                            @inbounds begin
                                shared_sums[g_zero] = zero(OT)
                                shared_cnts[g_zero] = zero(UInt32)
                            end
                            g_zero += workgroup_size
                        end
                    end
                    @synchronize
    lid = @index(Local, Linear)
                    block_id = @inbounds(shared_block_id[1])
                    @inbounds ti = shared_tile[1]
                    @inbounds tj = shared_tile[2]
                    @inbounds ni = shared_tile[3]
                    @inbounds nj = shared_tile[4]
                    @inbounds type_pass = shared_type_pass[1]
                    n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
                    $(pair_loop)
                    @synchronize
    lid = @index(Local, Linear)
                    @inbounds type_pass = shared_type_pass[1]
                    block_id = @inbounds(shared_block_id[1])
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
    elseif accum_mode == :shared
        quote
            @synchronize
    lid = @index(Local, Linear)
            if @inbounds(shared_block_id[1]) <= n_tile_blocks
                let g_zero = lid
                    while g_zero <= C
                        @inbounds begin
                            shared_sums[g_zero] = zero(OT)
                            shared_cnts[g_zero] = zero(UInt32)
                        end
                        g_zero += workgroup_size
                    end
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
                $(pair_loop)
            end
            @synchronize
    lid = @index(Local, Linear)
            _sp2d_flush_shared_to_output!(
                out_sums, out_cnts, shared_sums, shared_cnts, C, NB, N_val_edges - 1,
                lid, workgroup_size,
            )
        end
    else
        quote
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
                $(pair_loop)
            end
        end
    end
    hist_params = accum_mode == :direct ?
        [:(partition_sums::AbstractArray{OT}), :(partition_counts)] :
        [:(out_sums::AbstractArray{OT}), :(out_cnts)]
    return quote
        KA.@kernel unsafe_indices=true function $(fname)(
            $(hist_params...), x_mat::AbstractMatrix{FT}, u_mat::AbstractMatrix{FT},
            N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
            $(dist_params...),
            $(val_params...),
            $(kernel_tail_params...),
        ) where {OT, FT, HC, D}
            shared_xi = @localmem FT (D * SF_GPU_TILE,)
            shared_ui = @localmem FT (D * SF_GPU_TILE,)
            shared_xj = @localmem FT (D * SF_GPU_TILE,)
            shared_uj = @localmem FT (D * SF_GPU_TILE,)
            $(shared_hist_decl)
            $(type_pass_decl)
            shared_block_id = @localmem Int (1,)
            shared_tile = @localmem Int (4,)

            g = @index(Global, Linear)
            if (g - 1) % workgroup_size + 1 == 1
                @inbounds shared_block_id[1] = (g - 1) ÷ workgroup_size + 1
            end
            @synchronize
    lid = @index(Local, Linear)
            if @inbounds(shared_block_id[1]) <= n_tile_blocks
                ti, tj = _tile_from_linear(@inbounds(shared_block_id[1]), n_tiles)
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
                ti, tj = _tile_from_linear(@inbounds(shared_block_id[1]), n_tiles)
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
            $(accum_body)
        end
    end
end

const _SP2D_PARTITION_DIST_VAL = [
    (:linear, :linear_shared),
    (:linear, :linear_cols),
    (:linear, :inflinear_shared),
    (:linear, :inflinear_cols),
    (:linear, :log_linear_shared),
    (:linear, :log_linear_cols),
    (:linear, :vector_cols),
    (:log_linear, :linear_shared),
    (:log_linear, :linear_cols),
    (:log_linear, :inflinear_shared),
    (:log_linear, :inflinear_cols),
    (:log_linear, :log_linear_shared),
    (:log_linear, :log_linear_cols),
    (:log_linear, :vector_cols),
    (:general, :linear_shared),
    (:general, :linear_cols),
    (:general, :inflinear_shared),
    (:general, :inflinear_cols),
    (:general, :log_linear_shared),
    (:general, :log_linear_cols),
    (:general, :vector_cols),
]

for accum_mode in (:shared, :typeplane, :direct), (dist, val) in _SP2D_PARTITION_DIST_VAL
    ex = _sp2d_partition_kernel_def(accum_mode, dist, val)
    @eval $(ex)
end

"""Resolve compiled partitioned kernel for `(dist, val, accum_mode)`."""
function _sp2d_partition_kernel_fn(dist::Symbol, val::Symbol, accum_mode::Symbol, backend, ws)
    prefix = _sp2d_partition_kernel_prefix(Val(accum_mode))
    fname = Symbol(prefix, dist, :_, val, :_u32!)
    kf = getfield(@__MODULE__, fname)
    return kf(backend, ws)
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
