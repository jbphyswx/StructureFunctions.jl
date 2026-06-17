# HTP-EJ: tiled128 pair traversal + shared strip histogram + block-private slab.
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

@inline function _sp2d_priv_zero_strip!(
    shared_sums,
    shared_cnts,
    cells_in_strip::Int,
    lid::Int,
    workgroup_size::Int,
    ::Type{FT},
) where {FT}
    k = lid
    while k <= cells_in_strip
        @inbounds begin
            shared_sums[k] = zero(FT)
            shared_cnts[k] = zero(UInt32)
        end
        k += workgroup_size
    end
    return nothing
end

@inline function _sp2d_priv_flush_strip!(
    priv_sums,
    priv_cnts,
    shared_sums,
    shared_cnts,
    strip_base::Int,
    cells_in_strip::Int,
    n_dist::Int,
    n_val::Int,
    block_id::Int,
    lid::Int,
    workgroup_size::Int,
)
    k = lid
    while k <= cells_in_strip
        g = strip_base + k
        t, dbin, vbin = _sp2d_decode_flat_index(g, n_dist, n_val)
        @inbounds begin
            priv_sums[t, dbin, vbin, block_id] += shared_sums[k]
            if shared_cnts[k] != UInt32(0)
                priv_cnts[t, dbin, vbin, block_id] += shared_cnts[k]
            end
        end
        k += workgroup_size
    end
    return nothing
end

@inline function _sp2d_in_strip(
    t::Int,
    dbin::Int,
    vbin::Int,
    strip_base::Int,
    cells_in_strip::Int,
    n_dist::Int,
    n_val::Int,
)
    g = _sp2d_flat_index(t, dbin, vbin, n_dist, n_val)
    local_idx = g - strip_base
    return 1 <= local_idx <= cells_in_strip ? local_idx : 0
end

# --- shared-strip accumulation (mirror global helpers in TiledSinglePass2DKernels.jl) ---

@inline function _gpu_accumulate_sp2d_shared_linear_val!(
    shared_sums, shared_cnts, strip_base, cells_in_strip, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2, du_L2 * du_T,
        du_L * du_T2, du_T * du_T2,
    )
    for t in 1:8
        vbin = _gpu_digitize_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            li = _sp2d_in_strip(t, dbin, vbin, strip_base, cells_in_strip, n_dist, n_val)
            li > 0 || continue
            @atomic shared_sums[li] += vals[t]
            @atomic shared_cnts[li] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_shared_linear_val_cols!(
    shared_sums, shared_cnts, strip_base, cells_in_strip, n_dist, n_val,
    val_first, val_last, val_inv_step, val_offset, val_step,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2, du_L2 * du_T,
        du_L * du_T2, du_T * du_T2,
    )
    for t in 1:8
        vbin = _gpu_digitize_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            li = _sp2d_in_strip(t, dbin, vbin, strip_base, cells_in_strip, n_dist, n_val)
            li > 0 || continue
            @atomic shared_sums[li] += vals[t]
            @atomic shared_cnts[li] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_shared_inflinear_val!(
    shared_sums, shared_cnts, strip_base, cells_in_strip, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
    n_inner_edges::Int, inner_last::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2, du_L2 * du_T,
        du_L * du_T2, du_T * du_T2,
    )
    for t in 1:8
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step,
            n_inner_edges, inner_last,
        )
        if 1 <= vbin < N_val_edges
            li = _sp2d_in_strip(t, dbin, vbin, strip_base, cells_in_strip, n_dist, n_val)
            li > 0 || continue
            @atomic shared_sums[li] += vals[t]
            @atomic shared_cnts[li] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_shared_inflinear_val_cols!(
    shared_sums, shared_cnts, strip_base, cells_in_strip, n_dist, n_val,
    val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
    dbin::Int, du_L, du_T, du_L2, du_T2, n_inner_edges::Int, N_val_edges::Int,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2, du_L2 * du_T,
        du_L * du_T2, du_T * du_T2,
    )
    for t in 1:8
        vbin = _gpu_digitize_inf_padded_linear(
            vals[t], val_first[t], val_last[t], val_inv_step[t], val_offset[t], val_step[t],
            n_inner_edges, inner_last[t],
        )
        if 1 <= vbin < N_val_edges
            li = _sp2d_in_strip(t, dbin, vbin, strip_base, cells_in_strip, n_dist, n_val)
            li > 0 || continue
            @atomic shared_sums[li] += vals[t]
            @atomic shared_cnts[li] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_shared_log_val!(
    shared_sums, shared_cnts, strip_base, cells_in_strip, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first::FT, val_last::FT, val_inv_step::FT, val_offset::FT, val_step::FT,
) where {FT}
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2, du_L2 * du_T,
        du_L * du_T2, du_T * du_T2,
    )
    for t in 1:8
        vbin = _gpu_digitize_log_spaced(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            li = _sp2d_in_strip(t, dbin, vbin, strip_base, cells_in_strip, n_dist, n_val)
            li > 0 || continue
            @atomic shared_sums[li] += vals[t]
            @atomic shared_cnts[li] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_shared_log_val_cols!(
    shared_sums, shared_cnts, strip_base, cells_in_strip, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    val_first, val_last, val_inv_step, val_offset, val_step,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2, du_L2 * du_T,
        du_L * du_T2, du_T * du_T2,
    )
    for t in 1:8
        vbin = _gpu_digitize_log_spaced_col(
            vals[t], val_first, val_last, val_inv_step, val_offset, val_step, t, N_val_edges,
        )
        if 1 <= vbin < N_val_edges
            li = _sp2d_in_strip(t, dbin, vbin, strip_base, cells_in_strip, n_dist, n_val)
            li > 0 || continue
            @atomic shared_sums[li] += vals[t]
            @atomic shared_cnts[li] += UInt32(1)
        end
    end
    return nothing
end

@inline function _gpu_accumulate_sp2d_shared_vector_val_cols!(
    shared_sums, shared_cnts, strip_base, cells_in_strip, n_dist, n_val,
    dbin::Int, du_L, du_T, du_L2, du_T2, N_val_edges::Int,
    value_edges,
)
    vals = SA.SVector(
        du_L2 + du_T2, du_L2, du_T2,
        du_L * (du_L2 + du_T2), du_L * du_L2, du_L2 * du_T,
        du_L * du_T2, du_T * du_T2,
    )
    for t in 1:8
        vbin = _gpu_digitize_general_col(vals[t], value_edges, t, N_val_edges)
        if 1 <= vbin < N_val_edges
            li = _sp2d_in_strip(t, dbin, vbin, strip_base, cells_in_strip, n_dist, n_val)
            li > 0 || continue
            @atomic shared_sums[li] += vals[t]
            @atomic shared_cnts[li] += UInt32(1)
        end
    end
    return nothing
end

KA.@kernel function _merge_sp2d_priv_u32!(
    output_sums,
    output_counts,
    priv_sums,
    priv_cnts,
    n_dist::Int,
    n_val::Int,
    n_tile_blocks::Int,
)
    g = @index(Global, Linear)
    C = 8 * n_dist * n_val
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

# --- kernel codegen (strip bucket × dist × value digitize) ---

"""Emit one HTP-EJ tiled128 kernel for fixed `@localmem` strip bucket."""
function _sp2d_priv_kernel_def(bucket::Int, dist::Symbol, val::Symbol)
    strip = bucket
    fname = Symbol(:_sf8_sp2d_priv_tiled128_, dist, :_, val, :_u32_b, bucket, :!)
    dist_params, dist_extra, dist_bin = _sp2d_priv_dist_spec(dist)
    val_params, val_extra, accum = _sp2d_priv_val_spec(val)
    return quote
        # @synchronize splits the CFG: block/strip indices live in @localmem (lane 1 writes).
        KA.@kernel function $(fname)(
            priv_sums, priv_cnts, x_mat, u_mat,
            N_points::Int, N_bins::Int, NB::Int, N_val_edges::Int,
            $(dist_params...),
            $(val_params...),
            n_tiles::Int, n_tile_blocks::Int, workgroup_size::Int,
            C::Int, cells_per_strip::Int, n_strips::Int,
        ) where {FT}
            shared_xi = @localmem FT (256,)
            shared_ui = @localmem FT (256,)
            shared_xj = @localmem FT (256,)
            shared_uj = @localmem FT (256,)
            shared_sums = @localmem FT ($(strip),)
            shared_cnts = @localmem UInt32 ($(strip),)
            # @synchronize splits the CFG: tile/strip metadata must live in @localmem.
            shared_block_id = @localmem Int (1,)
            shared_tile = @localmem Int (4,)   # ti, tj, ni, nj
            shared_strip_s = @localmem Int (1,)

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
                    @inbounds shared_strip_s[1] = 1
                end
            end
            @synchronize
            g = @index(Global, Linear)
            lid = (g - 1) % workgroup_size + 1
            if @inbounds(shared_block_id[1]) <= n_tile_blocks &&
               @inbounds(shared_tile[3]) > 0 && @inbounds(shared_tile[4]) > 0
                for _ in 1:n_strips
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    @inbounds s = shared_strip_s[1]
                    strip_base = (s - 1) * cells_per_strip
                    cells_in_strip = min(cells_per_strip, C - strip_base)
                    _sp2d_priv_zero_strip!(
                        shared_sums, shared_cnts, cells_in_strip, lid, workgroup_size, FT,
                    )
                    @synchronize
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    if @inbounds(shared_block_id[1]) <= n_tile_blocks &&
                       @inbounds(shared_tile[3]) > 0 && @inbounds(shared_tile[4]) > 0
                        @inbounds s = shared_strip_s[1]
                        @inbounds ti = shared_tile[1]
                        @inbounds tj = shared_tile[2]
                        @inbounds ni = shared_tile[3]
                        @inbounds nj = shared_tile[4]
                        strip_base = (s - 1) * cells_per_strip
                        cells_in_strip = min(cells_per_strip, C - strip_base)
                        n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
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
                    @synchronize
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    @inbounds s = shared_strip_s[1]
                    strip_base = (s - 1) * cells_per_strip
                    cells_in_strip = min(cells_per_strip, C - strip_base)
                    _sp2d_priv_flush_strip!(
                        priv_sums, priv_cnts, shared_sums, shared_cnts,
                        strip_base, cells_in_strip, NB, N_val_edges - 1,
                        @inbounds(shared_block_id[1]), lid, workgroup_size,
                    )
                    @synchronize
                    g = @index(Global, Linear)
                    lid = (g - 1) % workgroup_size + 1
                    if lid == 1
                        @inbounds shared_strip_s[1] += 1
                    end
                    @synchronize
                end
            end
        end
    end
end

function _sp2d_priv_dist_spec(::Val{:linear})
    params = [
        :(dist_first::FT), :(dist_last::FT), :(dist_inv_step::FT),
        :(dist_offset::FT), :(dist_step::FT),
    ]
    extra = ()
    bin = :(_gpu_digitize_linear(
        dist, dist_first, dist_last, dist_inv_step, dist_offset, dist_step, N_bins,
    ))
    return params, extra, bin
end

function _sp2d_priv_dist_spec(::Val{:log_linear})
    params = [
        :(dist_first::FT), :(dist_last::FT), :(dist_inv_step::FT),
        :(dist_offset::FT), :(dist_step::FT),
    ]
    extra = ()
    bin = :(_gpu_digitize_log_spaced(
        dist, dist_first, dist_last, dist_inv_step, dist_offset, dist_step, N_bins,
    ))
    return params, extra, bin
end

function _sp2d_priv_val_spec(::Val{:linear_shared})
    params = [
        :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
        :(val_offset::FT), :(val_step::FT),
    ]
    accum = quote
        _gpu_accumulate_sp2d_shared_linear_val!(
            shared_sums, shared_cnts, strip_base, cells_in_strip, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
    return params, (), accum
end

function _sp2d_priv_val_spec(::Val{:linear_cols})
    params = [
        :(val_first), :(val_last), :(val_inv_step), :(val_offset), :(val_step),
    ]
    accum = quote
        _gpu_accumulate_sp2d_shared_linear_val_cols!(
            shared_sums, shared_cnts, strip_base, cells_in_strip, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_offset, val_step,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
        )
    end
    return params, (), accum
end

function _sp2d_priv_val_spec(::Val{:inflinear_shared})
    params = [
        :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
        :(val_offset::FT), :(val_step::FT),
        :(n_inner_edges::Int), :(inner_last::FT),
    ]
    accum = quote
        _gpu_accumulate_sp2d_shared_inflinear_val!(
            shared_sums, shared_cnts, strip_base, cells_in_strip, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step, n_inner_edges, inner_last,
        )
    end
    return params, (), accum
end

function _sp2d_priv_val_spec(::Val{:inflinear_cols})
    params = [
        :(val_first), :(val_last), :(val_inv_step), :(val_offset), :(val_step),
        :(inner_last), :(n_inner_edges::Int),
    ]
    accum = quote
        _gpu_accumulate_sp2d_shared_inflinear_val_cols!(
            shared_sums, shared_cnts, strip_base, cells_in_strip, NB, N_val_edges - 1,
            val_first, val_last, val_inv_step, val_offset, val_step, inner_last,
            bin, du_L, du_T, du_L2, du_T2, n_inner_edges, N_val_edges,
        )
    end
    return params, (), accum
end

function _sp2d_priv_val_spec(::Val{:log_linear_shared})
    params = [
        :(val_first::FT), :(val_last::FT), :(val_inv_step::FT),
        :(val_offset::FT), :(val_step::FT),
    ]
    accum = quote
        _gpu_accumulate_sp2d_shared_log_val!(
            shared_sums, shared_cnts, strip_base, cells_in_strip, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
    return params, (), accum
end

function _sp2d_priv_val_spec(::Val{:log_linear_cols})
    params = [
        :(val_first), :(val_last), :(val_inv_step), :(val_offset), :(val_step),
    ]
    accum = quote
        _gpu_accumulate_sp2d_shared_log_val_cols!(
            shared_sums, shared_cnts, strip_base, cells_in_strip, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges,
            val_first, val_last, val_inv_step, val_offset, val_step,
        )
    end
    return params, (), accum
end

function _sp2d_priv_val_spec(::Val{:vector_cols})
    params = [:(@Const(value_edges))]
    accum = quote
        _gpu_accumulate_sp2d_shared_vector_val_cols!(
            shared_sums, shared_cnts, strip_base, cells_in_strip, NB, N_val_edges - 1,
            bin, du_L, du_T, du_L2, du_T2, N_val_edges, value_edges,
        )
    end
    return params, (), accum
end

function _sp2d_priv_dist_spec(dist::Symbol)
    return _sp2d_priv_dist_spec(Val(dist))
end

function _sp2d_priv_val_spec(val::Symbol)
    return _sp2d_priv_val_spec(Val(val))
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

for bucket in SP2D_STRIP_BUCKETS, (dist, val) in _SP2D_PRIV_DIST_VAL
    ex = _sp2d_priv_kernel_def(bucket, dist, val)
    @eval $(ex)
end

"""Resolve compiled priv kernel for `(dist, val, strip_bucket)`."""
function _sp2d_priv_kernel_fn(dist::Symbol, val::Symbol, bucket::Int, backend, ws)
    fname = Symbol(:_sf8_sp2d_priv_tiled128_, dist, :_, val, :_u32_b, bucket, :!)
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
    n_tile_blocks::Int,
)
    C = 8 * n_dist * n_val
    kernel! = _merge_sp2d_priv_u32!(backend, 256)
    kernel!(
        out_sums_dev, out_cnts_dev, priv_sums_dev, priv_cnts_dev,
        n_dist, n_val, n_tile_blocks;
        ndrange = C,
    )
    return nothing
end

