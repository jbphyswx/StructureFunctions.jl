# HTP-EJ: tiled128 pair traversal for six-invariant-type single-pass 2D.
#
# On-chip (:shared, :typeplane): @localmem histogram during pair loop; block-end flush
#   via _sp2d_flush_*_to_output! (@atomic into out_sums/out_cnts, joint pattern).
# Direct (:direct): global atomics into block-private priv partition; merge on host.
#
# See gpu/SP2D_HTP_EJ.md for policy, benchmarks, and future perf notes.
# Included from StructureFunctionsGPUExt.jl after TiledSinglePass2DValueKernels.jl.

@inline function _sp2d_flat_index(t::Int, dbin::Int, vbin::Int, n_dist::Int, n_val::Int)
    return (t - 1) * n_dist * n_val + (dbin - 1) * n_val + vbin
end

@inline function _sp2d_decode_flat_index(g::Int, n_dist::Int, n_val::Int)
    t = (g - 1) ÷ (n_dist * n_val) + 1
    rem = (g - 1) % (n_dist * n_val)
    dbin = rem ÷ n_val + 1
    vbin = rem % n_val + 1
    return t, dbin, vbin
end

@inline function _sp2d_zero_shared_hist!(
    shared_sums,
    shared_cnts,
    C::Int,
    lid::Int,
    workgroup_size::Int,
    ::Type{FT},
) where {FT}
    g = lid
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
    priv_sums,
    priv_cnts,
    shared_sums,
    shared_cnts,
    C::Int,
    n_dist::Int,
    n_val::Int,
    block_id::Int,
    lid::Int,
    workgroup_size::Int,
)
    g = lid
    while g <= C
        t, dbin, vbin = _sp2d_decode_flat_index(g, n_dist, n_val)
        @inbounds begin
            priv_sums[t, dbin, vbin, block_id] += shared_sums[g]
            if shared_cnts[g] != UInt32(0)
                priv_cnts[t, dbin, vbin, block_id] += shared_cnts[g]
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
    C::Int,
    n_dist::Int,
    n_val::Int,
    lid::Int,
    workgroup_size::Int,
)
    g = lid
    while g <= C
        t, dbin, vbin = _sp2d_decode_flat_index(g, n_dist, n_val)
        @inbounds begin
            @atomic out_sums[t, dbin, vbin] += shared_sums[g]
            if shared_cnts[g] != UInt32(0)
                @atomic out_cnts[t, dbin, vbin] += shared_cnts[g]
            end
        end
        g += workgroup_size
    end
    return nothing
end

# --- shared-histogram accumulation (flat index g, no strip filter) ---

@inline function _gpu_accumulate_sp2d_sharedhist_linear_val!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_linear_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    val_first, val_last, val_inv_step, val_offset, val_step,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_inflinear_val!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step,
            n_inner_edges, inner_last,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_inflinear_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
    dbin::Int, du_L, du_T, du_L2, du_T2, n_inner_edges::Int, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            n_inner_edges, inner_last[t],
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_log_val!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_log_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first, val_last, val_inv_step, val_offset, val_step,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced_col(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, t, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_sharedhist_vector_val_cols!(
    shared_sums, shared_cnts, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
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
            g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
            @atomic shared_sums[g] += vals[t]
            @atomic shared_cnts[g] += UInt32(1)
        end
    end
    return nothing
end

# --- type-plane shared accumulation (one SF type per pass; plane = n_dist × n_val) ---

@inline function _sp2d_plane_flat_index(dbin::Int, vbin::Int, n_val::Int)
    return (dbin - 1) * n_val + vbin
end

"""Flush type-plane shared histogram into block-private partition (`:direct` path only)."""
@inline function _sp2d_flush_typeplane!(
    priv_sums,
    priv_cnts,
    shared_sums,
    shared_cnts,
    type_pass::Int,
    types_per_pass::Int,
    plane::Int,
    n_dist::Int,
    n_val::Int,
    block_id::Int,
    lid::Int,
    workgroup_size::Int,
)
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    g = lid
    while g <= plane
        dbin = (g - 1) ÷ n_val + 1
        vbin = (g - 1) % n_val + 1
        for t in t_lo:t_hi
            slot = t - t_lo
            @inbounds idx = slot * plane + g
            @inbounds priv_sums[t, dbin, vbin, block_id] += shared_sums[idx]
            if shared_cnts[idx] != UInt32(0)
                @inbounds priv_cnts[t, dbin, vbin, block_id] += shared_cnts[idx]
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
    type_pass::Int,
    types_per_pass::Int,
    plane::Int,
    n_dist::Int,
    n_val::Int,
    lid::Int,
    workgroup_size::Int,
)
    t_lo = (type_pass - 1) * types_per_pass + 1
    t_hi = min(SF_GPU_SINGLE_PASS_N, type_pass * types_per_pass)
    g = lid
    while g <= plane
        dbin = (g - 1) ÷ n_val + 1
        vbin = (g - 1) % n_val + 1
        for t in t_lo:t_hi
            slot = t - t_lo
            @inbounds idx = slot * plane + g
            @inbounds @atomic out_sums[t, dbin, vbin] += shared_sums[idx]
            @inbounds if shared_cnts[idx] != UInt32(0)
                @atomic out_cnts[t, dbin, vbin] += shared_cnts[idx]
            end
        end
        g += workgroup_size
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_typeplane_linear_val!(
    shared_sums, shared_cnts, n_val, plane::Int,
    type_pass::Int, types_per_pass::Int,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
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
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
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
    val_first, val_last, val_inv_step, val_offset, val_step,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
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
            val_inv_step[t], val_offset[t], val_step[t],
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
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
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
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step,
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
    val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
    dbin::Int, du_L, du_T, du_L2, du_T2, n_inner_edges::Int, N_val_edges::Int,
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
            val_inv_step[t], val_offset[t], val_step[t],
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
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
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
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
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
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first, val_last, val_inv_step, val_offset, val_step,
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
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step,
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
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
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

# --- direct priv-partition accumulation (block-partitioned global atomics) ---

@inline function _gpu_accumulate_sp2d_priv_direct_linear_val!(
    priv_sums, priv_cnts, block_id::Int, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic priv_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic priv_cnts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_priv_direct_linear_val_cols!(
    priv_sums, priv_cnts, block_id::Int, n_dist, n_val,
    val_first, val_last, val_inv_step, val_offset, val_step,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic priv_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic priv_cnts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_priv_direct_inflinear_val!(
    priv_sums, priv_cnts, block_id::Int, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step,
            n_inner_edges, inner_last,
        )
        if 1 <= vbin < N_val_edges
            @atomic priv_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic priv_cnts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_priv_direct_inflinear_val_cols!(
    priv_sums, priv_cnts, block_id::Int, n_dist, n_val,
    val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
    dbin::Int, du_L, du_T, du_L2, du_T2, n_inner_edges::Int, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            n_inner_edges, inner_last[t],
        )
        if 1 <= vbin < N_val_edges
            @atomic priv_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic priv_cnts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_priv_direct_log_val!(
    priv_sums, priv_cnts, block_id::Int, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic priv_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic priv_cnts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_priv_direct_log_val_cols!(
    priv_sums, priv_cnts, block_id::Int, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first, val_last, val_inv_step, val_offset, val_step,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2,
        du_L * du_T2,
    )
    for t in 1:SF_GPU_SINGLE_PASS_N
        vbin = _gpu_digitize_log_spaced_col(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, t, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            @atomic priv_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic priv_cnts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_priv_direct_vector_val_cols!(
    priv_sums, priv_cnts, block_id::Int, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
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
            @atomic priv_sums[t, dbin, vbin, block_id] += vals[t]
            @atomic priv_cnts[t, dbin, vbin, block_id] += UInt32(1)
        end
    end
    return nothing
end

# --- merge: serial (per-cell loop over blocks) and parallel (workgroup tree-reduce) ---

KA.@kernel function _merge_sp2d_priv_serial_u32!(
    output_sums,
    output_counts,
    priv_sums,
    priv_cnts,
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
            s += priv_sums[t, dbin, vbin, block_id]
            c += priv_cnts[t, dbin, vbin, block_id]
        end
        @inbounds begin
            output_sums[t, dbin, vbin] = s
            output_counts[t, dbin, vbin] = c
        end
    end
end

"""Parallel merge: one workgroup per joint cell (`ndrange = C × workgroup_size`)."""
KA.@kernel function _merge_sp2d_priv_parallel_u32!(
    output_sums::AbstractArray{FT, 3},
    output_counts::AbstractArray{UInt32, 3},
    priv_sums::AbstractArray{FT, 4},
    priv_cnts::AbstractArray{UInt32, 4},
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
                partial_s += priv_sums[t, dbin, vbin, bid]
                partial_c += priv_cnts[t, dbin, vbin, bid]
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

"""Compile-time max shared histogram cells (48 KiB default smem, Float32 cell width)."""
const _SP2D_SHAREDHIST_COMPILE_CELLS = _sp2d_max_shared_cells(SF_GPU_SMEM_DEFAULT, Float32)

function _sp2d_priv_dist_spec(::Val{:linear})
    params = [
        :(dist_first::FT), :(dist_last::FT), :(dist_inv_step::FT),
        :(dist_offset::FT), :(dist_step::FT),
    ]
    bin = :(_gpu_digitize_linear(
        dist, dist_first, dist_last, dist_inv_step, dist_offset, dist_step, N_bins,
    ))
    return params, bin
end

function _sp2d_priv_dist_spec(::Val{:log_linear})
    params = [
        :(dist_first::FT), :(dist_last::FT), :(dist_inv_step::FT),
        :(dist_offset::FT), :(dist_step::FT),
    ]
    bin = :(_gpu_digitize_log_spaced(
        dist, dist_first, dist_last, dist_inv_step, dist_offset, dist_step, N_bins,
    ))
    return params, bin
end

function _sp2d_priv_val_accum(::Val{:linear_shared}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_linear_val!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:linear_shared}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_priv_direct_linear_val!(
            priv_sums, priv_cnts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end

function _sp2d_priv_val_accum(::Val{:linear_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_linear_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_offset, val_step,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:linear_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_priv_direct_linear_val_cols!(
            priv_sums, priv_cnts, block_id, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_offset, val_step,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
        )
    end
end

function _sp2d_priv_val_accum(::Val{:inflinear_shared}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_inflinear_val!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step, n_inner_edges, inner_last,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:inflinear_shared}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_priv_direct_inflinear_val!(
            priv_sums, priv_cnts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step, n_inner_edges, inner_last,
        )
    end
end

function _sp2d_priv_val_accum(::Val{:inflinear_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_inflinear_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
            bin, du_L, du_T, du_L2, du_T2, n_inner_edges, N_val_edges,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:inflinear_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_priv_direct_inflinear_val_cols!(
            priv_sums, priv_cnts, block_id, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
            bin, du_L, du_T, du_L2, du_T2, n_inner_edges, N_val_edges,
        )
    end
end

function _sp2d_priv_val_accum(::Val{:log_linear_shared}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_log_val!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:log_linear_shared}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_priv_direct_log_val!(
            priv_sums, priv_cnts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end

function _sp2d_priv_val_accum(::Val{:log_linear_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_log_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:log_linear_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_priv_direct_log_val_cols!(
            priv_sums, priv_cnts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end

function _sp2d_priv_val_accum(::Val{:vector_cols}, ::Val{:shared})
    return quote
        _gpu_accumulate_sp2d_sharedhist_vector_val_cols!(
            shared_sums, shared_cnts, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges, value_edges,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:vector_cols}, ::Val{:direct})
    return quote
        _gpu_accumulate_sp2d_priv_direct_vector_val_cols!(
            priv_sums, priv_cnts, block_id, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges, value_edges,
        )
    end
end

# type-plane (one or more SF types per pass; shared hist = types_per_pass × plane)
function _sp2d_priv_val_accum(::Val{:linear_shared}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_linear_val!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:linear_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_linear_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            val_first, val_last, val_inv_step, val_offset, val_step,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:inflinear_shared}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_inflinear_val!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step, n_inner_edges, inner_last,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:inflinear_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_inflinear_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
            bin, du_L, du_T, du_L2, du_T2, n_inner_edges, N_val_edges,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:log_linear_shared}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_log_val!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:log_linear_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_log_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
end
function _sp2d_priv_val_accum(::Val{:vector_cols}, ::Val{:typeplane})
    return quote
        _gpu_accumulate_sp2d_typeplane_vector_val_cols!(
            shared_sums, shared_cnts, N_val_edges - 1, plane, type_pass, types_per_pass,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges, value_edges,
        )
    end
end

function _sp2d_priv_kernel_prefix(::Val{:shared})
    return :_sf6_sp2d_sharedhist_tiled128_
end
function _sp2d_priv_kernel_prefix(::Val{:typeplane})
    return :_sf6_sp2d_typeplane_tiled128_
end
function _sp2d_priv_kernel_prefix(::Val{:direct})
    return :_sf6_sp2d_directpriv_tiled128_
end

function _sp2d_priv_val_params(::Val{:linear_shared})
    return [
        :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
        :(val_offset::FT), :(val_step::FT),
    ]
end
function _sp2d_priv_val_params(::Val{:linear_cols})
    return [:(val_first), :(val_last), :(val_inv_step), :(val_offset), :(val_step)]
end
function _sp2d_priv_val_params(::Val{:inflinear_shared})
    return [
        :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
        :(val_offset::FT), :(val_step::FT),
        :(n_inner_edges::Int), :(inner_last::FT),
    ]
end
function _sp2d_priv_val_params(::Val{:inflinear_cols})
    return [
        :(val_first), :(val_last), :(val_inv_step), :(val_offset), :(val_step),
        :(inner_last), :(n_inner_edges::Int),
    ]
end
function _sp2d_priv_val_params(::Val{:log_linear_shared})
    return [
        :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
        :(val_offset::FT), :(val_step::FT),
    ]
end
function _sp2d_priv_val_params(::Val{:log_linear_cols})
    return [:(val_first), :(val_last), :(val_inv_step), :(val_offset), :(val_step)]
end
function _sp2d_priv_val_params(::Val{:vector_cols})
    return [:(@Const(value_edges))]
end

function _sp2d_priv_dist_spec(dist::Symbol)
    return _sp2d_priv_dist_spec(Val(dist))
end

"""Emit one HTP-EJ tiled128 kernel for `accum_mode` (`:shared`, `:typeplane`, or `:direct`)."""
function _sp2d_priv_kernel_def(accum_mode::Symbol, dist::Symbol, val::Symbol)
    hist_cells = _SP2D_SHAREDHIST_COMPILE_CELLS
    prefix = _sp2d_priv_kernel_prefix(Val(accum_mode))
    fname = Symbol(prefix, dist, :_, val, :_u32!)
    dist_params, dist_bin = _sp2d_priv_dist_spec(dist)
    val_params = _sp2d_priv_val_params(Val(val))
    accum = _sp2d_priv_val_accum(Val(val), Val(accum_mode))
    uses_shared = accum_mode in (:shared, :typeplane)
    shared_hist_decl = uses_shared ? quote
        shared_sums = @localmem FT ($(hist_cells),)
        shared_cnts = @localmem UInt32 ($(hist_cells),)
    end : quote end
    type_pass_decl = accum_mode == :typeplane ? quote
        shared_type_pass = @localmem Int (1,)
    end : quote end
    kernel_tail_params = [
        :(n_tiles::Int), :(n_tile_blocks::Int), :(workgroup_size::Int),
        :(C::Int), :(plane::Int), :(types_per_pass::Int), :(n_type_passes::Int),
    ]
    pair_loop = quote
        p = lid
        while p <= n_pairs
            if ti < tj
                ia = (p - 1) ÷ nj + 1
                jb = (p - 1) - (ia - 1) * nj + 1
                X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[SF_GPU_TILE + jb])
                U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[SF_GPU_TILE + jb])
            else
                ia, jb = _pair_from_linear(p, ni)
                X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[SF_GPU_TILE + ia])
                X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[SF_GPU_TILE + jb])
                U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[SF_GPU_TILE + ia])
                U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[SF_GPU_TILE + jb])
            end
            dX = X2 - X1
            dist = sqrt(dX[1]^2 + dX[2]^2)
            bin = $(dist_bin)
            if 1 <= bin < N_bins
                dU = U2 - U1
                r̂ = dX / dist
                n̂ = SA.SVector{2, FT}(r̂[2], -r̂[1])
                du_L = SA.dot(dU, r̂)
                du_T = SA.dot(dU, n̂)
                du_L2 = du_L * du_L
                du_T2 = du_T * du_T
                $(accum)
            end
            p += workgroup_size
        end
    end
    accum_body = if accum_mode == :typeplane
        quote
            @synchronize
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            if lid == 1
                @inbounds shared_type_pass[1] = 1
            end
            @synchronize
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            if @inbounds(shared_block_id[1]) <= n_tile_blocks &&
               @inbounds(shared_tile[3]) > 0 && @inbounds(shared_tile[4]) > 0
                block_id = @inbounds(shared_block_id[1])
                @inbounds ti = shared_tile[1]
                @inbounds tj = shared_tile[2]
                @inbounds ni = shared_tile[3]
                @inbounds nj = shared_tile[4]
                n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
                for _ in 1:n_type_passes
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    @inbounds type_pass = shared_type_pass[1]
                    _sp2d_zero_shared_hist!(
                        shared_sums, shared_cnts, types_per_pass * plane, lid, workgroup_size, FT,
                    )
                    @synchronize
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    block_id = @inbounds(shared_block_id[1])
                    @inbounds ti = shared_tile[1]
                    @inbounds tj = shared_tile[2]
                    @inbounds ni = shared_tile[3]
                    @inbounds nj = shared_tile[4]
                    @inbounds type_pass = shared_type_pass[1]
                    n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
                    $(pair_loop)
                    @synchronize
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    @inbounds type_pass = shared_type_pass[1]
                    block_id = @inbounds(shared_block_id[1])
                    _sp2d_flush_typeplane_to_output!(
                        out_sums, out_cnts, shared_sums, shared_cnts, type_pass, types_per_pass, plane,
                        NB, N_val_edges - 1, lid, workgroup_size,
                    )
                    @synchronize
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
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
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            if @inbounds(shared_block_id[1]) <= n_tile_blocks
                _sp2d_zero_shared_hist!(shared_sums, shared_cnts, C, lid, workgroup_size, FT)
            end
            @synchronize
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
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
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            _sp2d_flush_shared_to_output!(
                out_sums, out_cnts, shared_sums, shared_cnts, C, NB, N_val_edges - 1,
                lid, workgroup_size,
            )
        end
    else
        quote
            @synchronize
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
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
        [:(priv_sums), :(priv_cnts)] :
        [:(out_sums), :(out_cnts)]
    return quote
        KA.@kernel function $(fname)(
            $(hist_params...), x_mat, u_mat,
            N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
            $(dist_params...),
            $(val_params...),
            $(kernel_tail_params...),
        ) where {FT}
            shared_xi = @localmem FT (256,)
            shared_ui = @localmem FT (256,)
            shared_xj = @localmem FT (256,)
            shared_uj = @localmem FT (256,)
            $(shared_hist_decl)
            $(type_pass_decl)
            shared_block_id = @localmem Int (1,)
            shared_tile = @localmem Int (4,)

            g = @index(Global, Linear)
            if (g - 1) % workgroup_size + 1 == 1
                @inbounds shared_block_id[1] = (g - 1) ÷ workgroup_size + 1
            end
            @synchronize
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            if @inbounds(shared_block_id[1]) <= n_tile_blocks
                ti, tj = _tile_from_linear(@inbounds(shared_block_id[1]), n_tiles)
                i0 = (ti - 1) * SF_GPU_TILE + 1
                j0 = (tj - 1) * SF_GPU_TILE + 1
                ni = min(SF_GPU_TILE, N_points - i0 + 1)
                nj = min(SF_GPU_TILE, N_points - j0 + 1)
                _sp2d_tiled_load_tile!(
                    shared_xi, shared_ui, shared_xj, shared_uj, x_mat, u_mat,
                    ti, tj, i0, j0, ni, nj, N_points, lid, workgroup_size,
                )
            end
            @synchronize
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
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

const _SP2D_PRIV_DIST_VAL = [
    (:linear, :linear_shared),
    (:linear, :linear_cols),
    (:linear, :inflinear_shared),
    (:linear, :inflinear_cols),
    (:linear, :log_linear_shared),
    (:linear, :log_linear_cols),
    (:log_linear, :linear_shared),
    (:log_linear, :linear_cols),
    (:log_linear, :inflinear_shared),
    (:log_linear, :inflinear_cols),
    (:log_linear, :log_linear_shared),
    (:log_linear, :log_linear_cols),
    (:log_linear, :vector_cols),
]

for accum_mode in (:shared, :typeplane, :direct), (dist, val) in _SP2D_PRIV_DIST_VAL
    ex = _sp2d_priv_kernel_def(accum_mode, dist, val)
    @eval $(ex)
end

"""Resolve compiled priv kernel for `(dist, val, accum_mode)`."""
function _sp2d_priv_kernel_fn(dist::Symbol, val::Symbol, accum_mode::Symbol, backend, ws)
    prefix = _sp2d_priv_kernel_prefix(Val(accum_mode))
    fname = Symbol(prefix, dist, :_, val, :_u32!)
    kf = getfield(@__MODULE__, fname)
    return kf(backend, ws)
end

function _launch_merge_sp2d_priv!(
    backend::KA.Backend,
    out_sums_dev,
    out_cnts_dev,
    priv_sums_dev,
    priv_cnts_dev,
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int;
    merge_mode::Symbol = _sp2d_merge_mode(),
)
    C = SF_GPU_SINGLE_PASS_N * n_dist * n_val
    if merge_mode == :serial
        kernel! = _merge_sp2d_priv_serial_u32!(backend, 256)
        kernel!(
            out_sums_dev, out_cnts_dev, priv_sums_dev, priv_cnts_dev,
            n_dist, n_val, n_tile_blocks;
            ndrange = C,
        )
    else
        ws = 256
        kernel! = _merge_sp2d_priv_parallel_u32!(backend, ws)
        kernel!(
            out_sums_dev, out_cnts_dev, priv_sums_dev, priv_cnts_dev,
            n_dist, n_val, n_tile_blocks, ws;
            ndrange = C * ws,
            workgroupsize = (ws,),
        )
    end
    KA.synchronize(backend)
    return nothing
end
