# CPU batch kernels over a batch-leading `(B, D, N)` working buffer. The widths come from the geometry's type, so each
# `SVector` has a compile-time size. The batch axis is innermost, so `@simd for b` reads unit stride, and with shared
# positions the bin is constant across `b`, making the accumulation contiguous.

using Distances: Distances as DI

@inline _bl_unwrap(u) = (u, false)              # (array, already_batch_leading)
@inline _bl_unwrap(u::BatchLeading) = (u.data, true)

"""
Component-first view of an input, as `prepare_pair_inputs` reads components from axis 1. Only a
batch-leading `(B, W, N)` array is permuted; a `(W, N, B…)` array is returned as is.
"""
@inline _bl_component_first(a, is_batch_leading::Bool) =
    is_batch_leading ? permutedims(a, (2, 3, 1)) : a

# Stages plain `(D,N,B...)` (transposed once) or `BatchLeading` `(B,D,N)` (zero-copy) inputs; `x` is fixed `(D,N)` or varying.
# Returns (xb, ub, B, D, W, N, Val(fixed_x)), with `D` and `W` the field and coordinate widths of the staged arrays.
function _bl_prepare(x, u, geom, workspace = nothing)
    u_raw, u_bl = _bl_unwrap(u)
    x_raw, x_bl = _bl_unwrap(x)
    fixed_x = ndims(x_raw) == 2
    if !(geom isa SFH.FlatGeometry)
        x_raw, u_raw = SFH.prepare_pair_inputs(
            geom, _bl_component_first(x_raw, x_bl), _bl_component_first(u_raw, u_bl),
        )
        x_bl = false
        u_bl = false
    end
    # `W` is the coordinate width and `D` the field width the kernels load; on a sphere they differ.
    W = x_bl ? size(x_raw, 2) : size(x_raw, 1)
    if u_bl
        B, D, N = size(u_raw)
        _validate_ws_shape(workspace, B, N, D, W)
        ub = u_raw
    else
        D, N = size(u_raw, 1), size(u_raw, 2)
        B = prod(size(u_raw)[3:end])
        _validate_ws_shape(workspace, B, N, D, W)
        ub = _to_batch_leading(reshape(u_raw, D, N, B), _ws_ub(workspace))
    end
    xb = fixed_x ? x_raw :
         (x_bl ? x_raw : _to_batch_leading(reshape(x_raw, W, N, B), _ws_xb(workspace)))
    return xb, ub, B, D, W, N, Val(fixed_x)
end

"""The kernel for the positions' layout: `shared` for positions shared by every slice, `varying` otherwise."""
@inline _bl_by_layout(::Val{true}, shared, varying) = shared
@inline _bl_by_layout(::Val{false}, shared, varying) = varying

# Statically-sized, unchecked loads for the two layouts the batch drivers hold: `(W, N)` shared
# positions and `(B, W, N)` batch-leading, indexed within the shapes `_bl_prepare` validated.
@inline _bl_pt(x::AbstractMatrix, i, ::Val{W}) where {W} =
    SA.SVector{W}(ntuple(d -> @inbounds(x[d, i]), Val(W)))
@inline _bl_pt(xb::AbstractArray{<:Any, 3}, b, i, ::Val{W}) where {W} =
    SA.SVector{W}(ntuple(d -> @inbounds(xb[b, d, i]), Val(W)))
@inline _bl_vel(ub, b, i, ::Val{D}) where {D} =
    SA.SVector{D}(ntuple(d -> @inbounds(ub[b, d, i]), Val(D)))

"""
Throw unless the staged coordinate width `W` and field width `D` are those `geom` loads.

The batch entry points take raw `(D, N, B…)` arrays, so this is where a mismatched `x` or `u` is caught,
before the kernels load them under `@inbounds`.
"""
@inline function _validate_bl_geometry(geom, W::Int, D::Int)
    want = _val_int(SFH.coordinate_width(geom))
    W == want || throw(
        DimensionMismatch(
            "$(nameof(typeof(geom))) locates a point with $want coordinate(s) on axis 1 of x, but " *
            "got $W (velocity dimension D=$D)",
        ),
    )
    D == _val_int(SFH.field_width(geom)) || throw(DimensionMismatch(
        "$(nameof(typeof(geom))) loads $(_val_int(SFH.field_width(geom))) field component(s), but u has $D"))
    return nothing
end

# The workspace's transpose buffers, or `nothing` for the allocate-fresh path.
@inline _ws_ub(::Nothing) = nothing
@inline _ws_xb(::Nothing) = nothing

# (D,N,B) -> (B,D,N), materialized.
@inline _to_batch_leading(u_DNB, ::Nothing) = permutedims(u_DNB, (3, 1, 2))
@inline _to_batch_leading(u_DNB, dest::AbstractArray) = permutedims!(dest, u_DNB, (3, 1, 2))

# Shared positions: x is (D,N), geometry is computed once per pair; ub is (B, D, N); sums_bl, counts_bl are (B, n_bins).
"""The distance plan of a shared-position kernel: squared for a flat metric, whose kernels digitize
`r²` in a vectorized geometry pass; the bins' own plan for any other."""
@inline _bl_shared_plan(::SFH.FlatGeometry, bins) = squared_digitize_plan(bins)
@inline _bl_shared_plan(geom, bins) = digitize_plan(bins)

"""Shared positions as a kernel loads them: contiguous component vectors on a flat metric, the
`(W, N)` matrix otherwise."""
@inline _bl_shared_positions(x::AbstractMatrix, ::SFH.FlatGeometry{D}) where {D} = ntuple(d -> x[d, :], Val(D))
@inline _bl_shared_positions(x, geom) = x

"""
    _bl_kernel_scratch(xs, ::Val{W}) -> scratch

A task's scratch for the batch kernel over the shared positions `xs`: for flat component vectors the pair window
and its buffers — digitize key, approximate bin, direction, compacted slots; `nothing` for a kernel that takes
none.
"""
@inline _bl_kernel_scratch(xc::Tuple{AbstractVector{T}, Vararg{AbstractVector{T}}}, vW::Val) where {T} =
    _bl_kernel_scratch(_pair_window(length(xc[1])), xc, vW)
@inline function _bl_kernel_scratch(window, xc::Tuple{AbstractVector{T}, Vararg{AbstractVector{T}}}, ::Val{W}) where {T, W}
    L = _pair_scratch_length(window, length(xc[1]))
    return (window, Vector{T}(undef, L), Vector{Int32}(undef, L), Matrix{T}(undef, L, W), Vector{Int32}(undef, L))
end
@inline _bl_kernel_scratch(xs, ::Val) = nothing

"""The trailing kernel arguments a scratch supplies: its buffers, or none."""
@inline _bl_scratch_args(::Nothing) = ()
@inline _bl_scratch_args(scratch::Tuple) = scratch

"""
    _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, Xi, js, o, vW)

The digitize key, approximate bin and direction of every pair `(i, j)`, `j ∈ js`, of flat shared
positions `xc`, into slot `j - o` of each buffer.
"""
@inline function _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, Xi, js, o, vW::Val{W}) where {W}
    @inbounds @simd for j in js
        dx = _component_point(xc, j, vW) - Xi
        r2 = SFH.norm2(dx)
        keybuf[j - o] = digitize_key(plan, r2)
        if has_vector_index(plan)
            idxbuf[j - o] = squared_approx_index(plan, r2)
        end
        rh = SFH.pair_direction(SFH.FlatGeometry{W}(), dx, sqrt(r2))
        for d in 1:W
            rhbuf[j - o, d] = rh[d]
        end
    end
    return nothing
end

@inline _bl_direction(rhbuf, k, ::Val{W}) where {W} = SA.SVector{W}(ntuple(d -> @inbounds(rhbuf[k, d]), Val(W)))

function _bl_shared_1d!(
    sums_bl::AbstractMatrix{OT},
    counts_bl::AbstractMatrix{CT},
    xc::NTuple{D, AbstractVector},
    ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    plan::AbstractSquaredDigitizePlan,
    geom::SFH.FlatGeometry,
    ::Val{D},
    blocks,
    brange,
    weights,
    window::PairWindow,
    keybuf, idxbuf, rhbuf, sel,
) where {OT, CT, D}
    nb = size(sums_bl, 2)
    boff = first(brange) - 1
    vD = Val(D)
    chooses = _chooses_compaction(blocks)
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            Xi = _component_point(xc, i, vD)
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, Xi, jlo:last(jr), o, vD)
            ks = (jlo - o):(last(jr) - o)
            if chooses && _compacts(_sample_in_range(plan, keybuf, ks)...)
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    bin = squared_bin_select(plan, keybuf[k], idxbuf[k])
                    j = k + o
                    rh = _bl_direction(rhbuf, k, vD)
                    w = wi * _point_weight(weights, j)
                    @simd ivdep for b in brange
                        du = _bl_vel(ub, b, j, vD) - _bl_vel(ub, b, i, vD)
                        sums_bl[b - boff, bin] += w * sf_type(du, rh)
                        counts_bl[b - boff, bin] += CT(w)
                    end
                end
            else
                for k in ks
                    bin = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    1 <= bin <= nb || continue
                    j = k + o
                    rh = _bl_direction(rhbuf, k, vD)
                    w = wi * _point_weight(weights, j)
                    @simd ivdep for b in brange
                        du = _bl_vel(ub, b, j, vD) - _bl_vel(ub, b, i, vD)
                        sums_bl[b - boff, bin] += w * sf_type(du, rh)
                        counts_bl[b - boff, bin] += CT(w)
                    end
                end
            end
        end
    end
    return nothing
end

function _bl_shared_1d!(
    sums_bl::AbstractMatrix{OT},
    counts_bl::AbstractMatrix{CT},
    x::AbstractMatrix,
    ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    dist_be,
    geom,
    ::Val{D},
    blocks,
    brange,
    weights = NoWeights(),
) where {OT, CT, D}
    nb = size(sums_bl, 2)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        Xi = _bl_pt(x, i, vW)
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            bin = SFH.digitize(dist, dist_be)
            (ok && 1 <= bin <= nb) || continue
            # Frame, direction and pair weight are invariant across b.
            rh = SFH.pair_direction(geom, frame, dist)
            w = wi * _point_weight(weights, j)
            @simd for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                v = sf_type(du, rh)
                sums_bl[b - boff, bin] += w * v
                counts_bl[b - boff, bin] += CT(w)
            end
        end
    end
    return nothing
end

# Varying positions: x is (B,D,N); geometry is computed inside the b loop.
function _bl_varying_1d!(
    sums_bl::AbstractMatrix{OT},
    counts_bl::AbstractMatrix{CT},
    xb::AbstractArray{<:Any, 3},
    ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    dist_be,
    geom,
    ::Val{D},
    blocks,
    brange,
    weights = NoWeights(),
) where {OT, CT, D}
    nb = size(sums_bl, 2)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            w = wi * _point_weight(weights, j)
            @simd for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = SFH.digitize(dist, dist_be)
                if ok && 1 <= bin <= nb
                    du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                    sums_bl[b - boff, bin] += w * SFT.pair_value(sf_type, geom, frame, dist, du)
                    counts_bl[b - boff, bin] += CT(w)
                end
            end
        end
    end
    return nothing
end

# Joint 2D (single SF): accumulator (B, n_dist, n_val); vbin varies per b, dbin is constant across b for shared x.
"""
    _shared_axis_bin(source, Xi, Xj, val_be) -> Int or nothing

The second-axis bin of a pair whose positions every slice shares: the angle bin, the same for every
slice, or `nothing` when the axis is the operator's value, which differs per slice.
"""
@inline _shared_axis_bin(::InvariantValueAxis, Xi, Xj, val_be) = nothing
@inline _shared_axis_bin(s::SeparationAngleAxis, Xi, Xj, val_be) =
    (dx = Xj - Xi; SFH.digitize(axis_quantity(s, dx, SFH.norm2(dx)), val_be))

"""The second-axis bin of one slice's pair value, given the pair's [`_shared_axis_bin`](@ref)."""
@inline _slice_axis_bin(::Nothing, val, val_be) = SFH.digitize(val, val_be)
@inline _slice_axis_bin(abin::Integer, val, val_be) = abin

function _bl_joint2d_shared!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType, dist_be, val_be, geom, ::Val{D}, blocks, brange,
    weights = NoWeights(), second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
) where {OT, CT, D}
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3) - 2
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        Xi = _bl_pt(x, i, vW)
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            dbin = SFH.digitize(dist, dist_be)
            (ok && 1 <= dbin <= n_dist) || continue
            rh = SFH.pair_direction(geom, frame, dist)
            w = wi * _point_weight(weights, j)
            abin = _shared_axis_bin(second_axis, Xi, Xj, val_be)
            for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                val = sf_type(du, rh)
                col = clamp(_slice_axis_bin(abin, val, val_be), 0, n_val + 1) + 1
                sums_bl[b - boff, dbin, col] += w * val
                counts_bl[b - boff, dbin, col] += CT(w)
            end
        end
    end
    return nothing
end

function _bl_joint2d_varying!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType, dist_be, val_be, geom, ::Val{D}, blocks, brange,
    weights = NoWeights(), second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
) where {OT, CT, D}
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3) - 2
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            w = wi * _point_weight(weights, j)
            for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_be)
                (ok && 1 <= dbin <= n_dist) || continue
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                val = SFT.pair_value(sf_type, geom, frame, dist, du)
                col = _value_column(val_be, pair_axis_key(second_axis, val, Xi, Xj, dist), n_val)
                sums_bl[b - boff, dbin, col] += w * val
                counts_bl[b - boff, dbin, col] += CT(w)
            end
        end
    end
    return nothing
end

# Single-pass 1D (6 invariants): accumulator (B, 6, n_dist); for shared x the dist bin is constant over b.
# Only du_L and du_norm2 = ⟨du,du⟩ are needed.
@inline function _bl_sp1d_write!(sums_bl, counts_bl, b, bin, du_L, du_norm2, ::Type{CT},
                                 w = true) where {CT}
    vals = single_pass_invariants(du_L, du_norm2)
    @inbounds for t in 1:SINGLE_PASS_N
        sums_bl[b, t, bin] += w * vals[t]
        counts_bl[b, t, bin] += CT(w)
    end
    return nothing
end

function _bl_sp1d_shared!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xc::NTuple{D, AbstractVector}, ub::AbstractArray{<:Any, 3}, plan::AbstractSquaredDigitizePlan,
    geom::SFH.FlatGeometry, ::Val{D}, blocks, brange, weights, window::PairWindow, keybuf, idxbuf, rhbuf, sel,
) where {OT, CT, D}
    nb = size(sums_bl, 3)
    boff = first(brange) - 1
    vD = Val(D)
    chooses = _chooses_compaction(blocks)
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            Xi = _component_point(xc, i, vD)
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, Xi, jlo:last(jr), o, vD)
            ks = (jlo - o):(last(jr) - o)
            if chooses && _compacts(_sample_in_range(plan, keybuf, ks)...)
                for m in 1:_compact_in_range!(sel, plan, keybuf, ks)
                    k = Int(sel[m])
                    bin = squared_bin_select(plan, keybuf[k], idxbuf[k])
                    j = k + o
                    rh = _bl_direction(rhbuf, k, vD)
                    w = wi * _point_weight(weights, j)
                    @simd ivdep for b in brange
                        du = _bl_vel(ub, b, j, vD) - _bl_vel(ub, b, i, vD)
                        _bl_sp1d_write!(sums_bl, counts_bl, b - boff, bin, SFH.fma_dot(du, rh), SFH.fma_dot(du, du), CT, w)
                    end
                end
            else
                for k in ks
                    bin = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
                    1 <= bin <= nb || continue
                    j = k + o
                    rh = _bl_direction(rhbuf, k, vD)
                    w = wi * _point_weight(weights, j)
                    @simd ivdep for b in brange
                        du = _bl_vel(ub, b, j, vD) - _bl_vel(ub, b, i, vD)
                        _bl_sp1d_write!(sums_bl, counts_bl, b - boff, bin, SFH.fma_dot(du, rh), SFH.fma_dot(du, du), CT, w)
                    end
                end
            end
        end
    end
    return nothing
end

function _bl_sp1d_shared!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3}, dist_be, geom, ::Val{D}, blocks, brange,
    weights = NoWeights(),
) where {OT, CT, D}
    nb = size(sums_bl, 3)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        Xi = _bl_pt(x, i, vW)
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            bin = SFH.digitize(dist, dist_be)
            (ok && 1 <= bin <= nb) || continue
            rh = SFH.pair_direction(geom, frame, dist)
            w = wi * _point_weight(weights, j)
            @simd for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                _bl_sp1d_write!(sums_bl, counts_bl, b - boff, bin, SFH.fma_dot(du, rh), SFH.fma_dot(du, du), CT, w)
            end
        end
    end
    return nothing
end

function _bl_sp1d_varying!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3}, dist_be, geom, ::Val{D}, blocks, brange,
    weights = NoWeights(),
) where {OT, CT, D}
    nb = size(sums_bl, 3)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            w = wi * _point_weight(weights, j)
            @simd for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = SFH.digitize(dist, dist_be)
                if ok && 1 <= bin <= nb
                    du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                    _bl_sp1d_write!(sums_bl, counts_bl, b - boff, bin, SFH.increment_invariants(geom, frame, dist, du)...,
                                    CT, w)
                end
            end
        end
    end
    return nothing
end

# Single-pass 2D (6 invariants × per-invariant value bins): accumulator (B, 6, n_dist, n_val + 2), padded value columns.
@inline function _bl_sp2d_write!(sums_bl, counts_bl, b, dbin, vals, value_bins, n_val, ::Type{CT},
                                 w = true) where {CT}
    @sp2d_each_invariant value_bins t vb begin
        col = _value_column(vb, vals[t], n_val)
        @inbounds sums_bl[b, t, dbin, col] += w * vals[t]
        @inbounds counts_bl[b, t, dbin, col] += CT(w)
    end
    return nothing
end

function _bl_sp2d_shared!(
    sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3}, dist_be, value_bins, geom, ::Val{D}, blocks, brange,
    weights = NoWeights(),
) where {OT, CT, D}
    nb = size(sums_bl, 3); n_val = size(sums_bl, 4) - 2
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        Xi = _bl_pt(x, i, vW)
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            dbin = SFH.digitize(dist, dist_be)
            (ok && 1 <= dbin <= nb) || continue
            rh = SFH.pair_direction(geom, frame, dist)
            w = wi * _point_weight(weights, j)
            for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                vals = single_pass_invariants(SFH.fma_dot(du, rh), SFH.fma_dot(du, du))
                _bl_sp2d_write!(sums_bl, counts_bl, b - boff, dbin, vals, value_bins, n_val, CT, w)
            end
        end
    end
    return nothing
end

function _bl_sp2d_varying!(
    sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3}, dist_be, value_bins, geom, ::Val{D}, blocks, brange,
    weights = NoWeights(),
) where {OT, CT, D}
    nb = size(sums_bl, 3); n_val = size(sums_bl, 4) - 2
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for (ir, jr) in blocks, i in ir
        wi = _point_weight(weights, i)
        for j in max(i + 1, first(jr)):last(jr)
            w = wi * _point_weight(weights, j)
            for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_be)
                (ok && 1 <= dbin <= nb) || continue
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                vals = single_pass_invariants(SFH.increment_invariants(geom, frame, dist, du)...)
                _bl_sp2d_write!(sums_bl, counts_bl, b - boff, dbin, vals, value_bins, n_val, CT, w)
            end
        end
    end
    return nothing
end

# Drivers: stage to batch-leading, run a kernel through an executor, add the result into the caller's arrays.
# The outer pair index `i` is partitioned into chunks, and per-task accumulators are reduced at the end.
#
# executor(make_accum, make_scratch, run_chunk!, ifull, grid, B, accum_bytes, ws) → reduced (sums, counts), width B:
#   make_accum(bw)                          → fresh zeroed (sums_bl, counts_bl) of batch width bw
#   make_scratch()                          → a task's kernel scratch
#   run_chunk!(acc, scratch, isub, brange)  → kernel over outer i ∈ isub and batch b ∈ brange
#   grid                                    → the cull grid of the sweep, per-slice grids, or `nothing`
#   accum_bytes                             → bytes of one full-width accumulator, for the split model
#   ws                                      → CPUSFWorkspace to draw accumulators from, or `nothing`
#   serial  : one full-width accumulator over ifull
#   threaded: partitions (i, b); per-task accumulators are only as wide as their b-chunk

@inline function _bl_serial_exec(make_accum, make_scratch, run_chunk!, ifull, grid, B, accum_bytes, ws)
    acc = _bl_accum_pool(ws, make_accum, [B])[1]
    _bl_zero_accum!(acc)
    run_chunk!(acc, make_scratch(), ifull, 1:B)
    return acc
end

"""
    _bl_accum_pool(ws, make_accum, widths) -> Vector

One accumulator per task, of the given batch widths, drawn from the workspace when there is one and
allocated fresh otherwise. Always built outside the parallel region, so tasks only read their slot.
"""
_bl_accum_pool(::Nothing, make_accum::F, widths::AbstractVector{Int}) where {F} =
    [make_accum(w) for w in widths]

"""The full-width reduction accumulator, from the workspace when there is one."""
_bl_result_accum(::Nothing, make_accum::F, B::Int) where {F} = make_accum(B)

@inline function _bl_zero_accum!(acc)
    fill!(acc[1], zero(eltype(acc[1])))
    fill!(acc[2], zero(eltype(acc[2])))
    return acc
end

"""Bytes in one full-width `(sums, counts)` accumulator of shape `dims`."""
@inline _bl_accum_bytes(::Type{OT}, ::Type{CT}, dims::Vararg{Int}) where {OT, CT} =
    prod(dims) * (sizeof(OT) + sizeof(CT))

"""
    _bl_batch_chunk_count(accum_bytes, B, n_tasks) -> Int

How many chunks to split the batch axis into, given the full-width accumulator size in bytes.

Every task holds an accumulator, so the live footprint is `n_tasks * accum_bytes`. The count is 1 when
that fits `_BL_ACCUM_BUDGET`, else the smallest count that fits it, at most `min(n_tasks, B)`. Pair
geometry is recomputed once per batch chunk.
"""
@inline function _bl_batch_chunk_count(accum_bytes::Int, B::Int, n_tasks::Int)
    footprint = accum_bytes * n_tasks
    footprint <= _BL_ACCUM_BUDGET && return 1
    return clamp(cld(footprint, _BL_ACCUM_BUDGET), 1, min(n_tasks, B))
end

# Live accumulator footprint across all tasks, above which the batch axis is split.
const _BL_ACCUM_BUDGET = 64 * 1024 * 1024

"""Contiguous batch-axis chunks; `_bl_batch_chunk_count` picks `n`."""
@inline _bl_batch_chunks(B::Int, n::Int) =
    [(((k - 1) * B) ÷ n + 1):((k * B) ÷ n) for k in 1:n]

"""
    _bl_n_tasks(backend) -> Int

How many tasks a batch call on `backend` splits into, and therefore how many accumulators a
[`CPUSFWorkspace`](@ref) must hold: one for a serial or `AutoBackend()` call, and for a threaded
backend when OhMyThreads is not loaded.
"""
_bl_n_tasks(::CB.AbstractExecutionBackend) = 1
_bl_n_tasks(::CB.AbstractThreadedBackend) = _ohmythreads_loaded() ? Threads.nthreads() : 1
_bl_n_tasks(b::CB.AbstractMPIBackend) = _bl_n_tasks(CB.local_backend(b))
_bl_n_tasks(b::CB.AbstractDistributedBackend) = _bl_n_tasks(CB.local_backend(b))

"""
    _bl_executor(backend) -> executor

The batch-leading executor a local backend runs with, as a value. Backends that compose over a
local inner backend (MPI) look it up here; extensions add methods for the backends they provide.
"""
_bl_executor(::CB.AbstractSerialBackend) = _bl_serial_exec
_bl_executor(b::CB.AbstractExecutionBackend) = throw(ArgumentError(
    "no batch-leading executor for $(nameof(typeof(b))); use SerialBackend, or load the extension \
     providing it (ThreadedBackend needs OhMyThreads)."))

"""
    _bl_add_permuted!(dest, src_bl, perm)

Accumulate the batch-leading accumulator `src_bl` into `dest` through the permutation `perm`,
fusing the permute with the add via a lazy `PermutedDimsArray`.
"""
@inline function _bl_add_permuted!(dest, src_bl, perm)
    dest .+= PermutedDimsArray(src_bl, perm)
    return dest
end

"""
    _bl_cull(xb, ub, geom, distance_bins, culling, Val(fixed_x), weights) -> (grid, xs, us, ws)

The inputs sorted into cull grids, or unchanged with `grid === nothing` when `culling` declines.

Shared positions are sorted once for every slice: one grid, and the `(W, N)` positions, the
`(B, D, N)` fields and the weights in its order. Positions varying per slice are sorted per slice:
`grid[b]` is slice `b`'s grid (`nothing` where that slice declines), `xs[b]` and `us[b]` its
`(1, W, N)` positions and `(1, D, N)` fields in that order, `ws[b]` its weights.
"""
function _bl_cull(xb, ub, geom, distance_bins, culling::CullingPolicy, ::Val{true}, weights)
    _cull_enabled(culling) || return nothing, xb, ub, weights
    grid = cull_grid_for(ntuple(d -> view(xb, d, :), SFH.coordinate_width(geom)), geom, distance_bins, culling)
    grid === nothing && return nothing, xb, ub, weights
    p = grid.perm
    return grid, xb[:, p], ub[:, :, p], _permuted_point_weights(weights, p)
end

function _bl_cull(xb, ub, geom, distance_bins, culling::CullingPolicy, ::Val{false}, weights)
    _cull_enabled(culling) || return nothing, xb, ub, weights
    vW = SFH.coordinate_width(geom)
    grid_of(b) = cull_grid_for(ntuple(d -> view(xb, b, d, :), vW), geom, distance_bins, culling)
    grids = Base.promote_op(grid_of, Int)[grid_of(b) for b in axes(ub, 1)]
    all(isnothing, grids) && return nothing, xb, ub, weights
    perms = [g === nothing ? collect(axes(ub, 3)) : g.perm for g in grids]
    return grids, [xb[b:b, :, p] for (b, p) in pairs(perms)], [ub[b:b, :, p] for (b, p) in pairs(perms)],
           [_permuted_point_weights(weights, p) for p in perms]
end

"""
    _bl_chunk_runner(kernel!, grid, xs, us, ws, N) -> run_chunk!

The executor's `run_chunk!(acc, scratch, isub, brange)` for [`_bl_cull`](@ref)'s output: `kernel!(sums,
counts, x, u, blocks, brange, weights, scratch)` once over the slices of `brange` with one set of blocks, or,
for per-slice grids, once per slice with that slice's inputs and blocks into its row of `acc`.
"""
function _bl_chunk_runner(kernel!::K, grid, xs, us, ws, N::Int) where {K}
    grid isa AbstractVector ||
        return (acc, scratch, isub, br) -> kernel!(acc[1], acc[2], xs, us, pair_blocks(N, isub; grid), br, ws, scratch)
    return function (acc, scratch, isub, br)
        for (k, b) in enumerate(br)
            kernel!(selectdim(acc[1], 1, k:k), selectdim(acc[2], 1, k:k), xs[b], us[b],
                    pair_blocks(N, isub; grid = grid[b]), 1:1, ws[b], scratch)
        end
        return nothing
    end
end

"""
    _bl_sweep(executor, culled, kernel!, stage, make_accum, vD, N, B, accum_bytes, workspace) -> (sums_bl, counts_bl)

`executor` over [`_bl_cull`](@ref)'s output `culled`, the kernel loading the positions `stage(xs)` with the scratch
[`_bl_kernel_scratch`](@ref) builds for them.
"""
function _bl_sweep(executor::E, culled::Tuple, kernel!::K, stage::S, make_accum::A, vD::Val, N::Int, B::Int,
                   accum_bytes::Int, workspace) where {E, K, S, A}
    grid, xs, us, ws = culled
    xk = stage(xs)
    make_scratch() = _bl_kernel_scratch(xk, vD)
    return executor(make_accum, make_scratch, _bl_chunk_runner(kernel!, grid, xk, us, ws, N), 1:(N - 1), grid, B,
                    accum_bytes, workspace)
end

"""Slices from which a flat 1-D or single-pass batch over shared positions sums each pair across the slices in its
innermost loop, the pair's bin being the same in every slice; fewer slices loop the slices outside the pairs."""
const BL_SLICE_LANES_MIN = 4

"""
    _bl_mode(vS, Val(fixed_x), B, lanes_min, rows, lanes)

The kernel family of a batch: on a flat metric of SIMD width `vS`, `rows(vS)` — the slices looped outside each run
of pairs — for positions varying per slice or shared by fewer than `lanes_min` slices, else `lanes()`, the slices
innermost; `lanes()` on any other metric.
"""
@inline _bl_mode(::Nothing, vFX::Val, B::Int, lanes_min::Int, rows::R, lanes::L) where {R, L} = lanes()
@inline _bl_mode(vS::Val, ::Val{fixed_x}, B::Int, lanes_min::Int, rows::R, lanes::L) where {fixed_x, R, L} =
    (!fixed_x || B < lanes_min) ? rows(vS) : lanes()

"""
    _bl_slices(x, ub, geom, distance_bins, culling, Val(fixed_x), weights) -> (grid, xs, us, ws)

[`_bl_prepare`](@ref)'s staged inputs as the kernels that loop the slices outside the pairs load them, sorted into cull
grids where `culling` takes one. Shared `(W, N)` positions give one grid and one tuple of contiguous component
vectors, with `us[b]` slice `b`'s field components in that order; positions varying per slice give one grid,
positions, field and weights per slice ([`_cull_sorted`](@ref)).
"""
function _bl_slices(x, ub, geom, distance_bins, culling::CullingPolicy, ::Val{true}, weights)
    vW, vD = SFH.coordinate_width(geom), SFH.field_width(geom)
    xc = ntuple(d -> x[d, :], vW)
    grid = _cull_enabled(culling) ? cull_grid_for(xc, geom, distance_bins, culling) : nothing
    p = grid === nothing ? axes(ub, 3) : grid.perm
    xs = grid === nothing ? xc : apply_perm(xc, p)
    ws = grid === nothing ? weights : _permuted_point_weights(weights, p)
    return grid, xs, [ntuple(d -> ub[b, d, p], vD) for b in axes(ub, 1)], ws
end

function _bl_slices(xb, ub, geom, distance_bins, culling::CullingPolicy, ::Val{false}, weights)
    vW, vD = SFH.coordinate_width(geom), SFH.field_width(geom)
    slice(b) = _cull_sorted(ntuple(d -> xb[b, d, :], vW), ntuple(d -> ub[b, d, :], vD), weights, geom, distance_bins,
                            culling)
    sorted = Base.promote_op(slice, Int)[slice(b) for b in axes(ub, 1)]
    S = eltype(sorted)
    return Base.promote_op(s -> s[1], S)[s[1] for s in sorted], Base.promote_op(s -> s[2], S)[s[2] for s in sorted],
           Base.promote_op(s -> s[3], S)[s[3] for s in sorted], Base.promote_op(s -> s[4], S)[s[4] for s in sorted]
end

"""
    _bl_rows_sweep(executor, slices, kernel!, make_scratch, make_accum, N, B, accum_bytes, workspace)

`executor` over [`_bl_slices`](@ref)'s output: `kernel!(sums, counts, xc, us, blocks, brange, weights, scratch)` with
the shared positions and every slice of `brange`, or, for positions varying per slice, once per slice with that
slice's own inputs (`us` then one tuple of field components, into row 1).
"""
function _bl_rows_sweep(executor::E, slices::Tuple, kernel!::K, make_scratch::M, make_accum::A, N::Int, B::Int,
                        accum_bytes::Int, workspace) where {E, K, M, A}
    grid, xs, us, ws = slices
    return executor(make_accum, make_scratch, _bl_chunk_runner(kernel!, grid, xs, us, ws, N), 1:(N - 1), grid, B,
                    accum_bytes, workspace)
end

"""Row `k` of a batch-leading accumulator: one slice's histogram."""
@inline _bl_row(a::AbstractArray{<:Any, N}, k::Int) where {N} = view(a, k, ntuple(_ -> Colon(), Val(N - 1))...)

"""
    _bl_run_bins!(binbuf, sel, plan, keybuf, idxbuf, ks, compact) -> n

The distance bins of a run's pairs, which every slice shares: of its `n` in-range slots, compacted into `sel`, into
`binbuf[1:n]`; or, with `compact` false, of every slot `k ∈ ks` into `binbuf[k]` (`n = 0`).
"""
@inline function _bl_run_bins!(binbuf, sel, plan, keybuf, idxbuf, ks, compact::Bool, chooses::Bool)
    if compact
        n = _compact_in_range!(sel, plan, keybuf, ks)
        @inbounds for m in 1:n
            k = Int(sel[m])
            binbuf[m] = squared_bin_select(plan, keybuf[k], idxbuf[k])
        end
        return n
    end
    @inbounds for k in ks
        binbuf[k] = chooses ? squared_bin_select(plan, keybuf[k], idxbuf[k]) : squared_bin(plan, keybuf[k], idxbuf[k])
    end
    return 0
end

"""A task's scratch for the 1-D kernels over `N` points: the pair window, then the digitize keys, approximate bins,
directions, compacted slots, distance bins and values."""
function _bl_rows_scratch_1d(N::Int, ::Val{D}, ::Type{FT}, ::Type{OT}) where {D, FT, OT}
    window = _pair_window(N)
    L = _pair_scratch_length(window, N)
    return (window, Vector{FT}(undef, L), Vector{Int32}(undef, L), Matrix{FT}(undef, L, D), Vector{Int32}(undef, L),
            Vector{Int32}(undef, L), Vector{OT}(undef, L))
end

"""
    _bl_rows_1d!(sums_bl, counts_bl, xc, us, sf, plan, ::Val{D}, blocks, brange, weights, scratch)

The 1-D pairs `blocks` covers, for the shared positions `xc` and every slice `b ∈ brange` of the fields `us[b]`: each
run's geometry, in-range choice and bins once, then per slice a vectorized value pass and a scatter of the in-range
pairs into the slice's row. One slice's own inputs (`us` a tuple) take [`_pf_simd_pairs!`](@ref) into row 1.
"""
function _bl_rows_1d!(sums_bl::AbstractMatrix{OT}, counts_bl::AbstractMatrix{CT}, xc::NTuple{D}, us::AbstractVector,
                      sf, plan, ::Val{D}, blocks, brange, weights, scratch) where {OT, CT, D}
    window, keybuf, idxbuf, rhbuf, sel, binbuf, valbuf = scratch
    nb = n_histogram_bins(plan)
    boff = first(brange) - 1
    vD = Val(D)
    chooses = _chooses_compaction(blocks)
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, _component_point(xc, i, vD), jlo:last(jr), o, vD)
            ks = (jlo - o):(last(jr) - o)
            compact = chooses && _compacts(_sample_in_range(plan, keybuf, ks)...)
            n = _bl_run_bins!(binbuf, sel, plan, keybuf, idxbuf, ks, compact, chooses)
            for b in brange
                uc = us[b]
                Ui = _component_point(uc, i, vD)
                @simd for k in ks
                    valbuf[k] = sf(_component_point(uc, k + o, vD) - Ui, _bl_direction(rhbuf, k, vD))
                end
                s, c = _bl_row(sums_bl, b - boff), _bl_row(counts_bl, b - boff)
                if compact
                    for m in 1:n
                        _pf_accumulate!(s, c, valbuf, weights, wi, o, Int(sel[m]), Int(binbuf[m]))
                    end
                else
                    for k in ks
                        bin = Int(binbuf[k])
                        1 <= bin <= nb && _pf_accumulate!(s, c, valbuf, weights, wi, o, k, bin)
                    end
                end
            end
        end
    end
    return nothing
end

function _bl_rows_1d!(sums_bl::AbstractMatrix, counts_bl::AbstractMatrix, xc::NTuple{D}, uc::Tuple, sf, plan,
                      vD::Val{D}, blocks, brange, weights, scratch) where {D}
    window, keybuf, idxbuf, _, sel, _, valbuf = scratch
    _pf_simd_pairs!(_bl_row(sums_bl, 1), _bl_row(counts_bl, 1), sf, xc, uc, plan, vD, keybuf, valbuf, idxbuf, sel,
                    window, blocks, weights)
    return nothing
end

"""A task's scratch for the single-pass kernels: the pair window, then the digitize keys, approximate bins,
directions, compacted slots, distance bins, `δu_L` and `‖δu‖²`."""
function _bl_rows_scratch_sp1d(N::Int, ::Val{D}, ::Type{FT}, ::Type{OT}) where {D, FT, OT}
    window = _pair_window(N)
    L = _pair_scratch_length(window, N)
    return (window, Vector{FT}(undef, L), Vector{Int32}(undef, L), Matrix{FT}(undef, L, D), Vector{Int32}(undef, L),
            Vector{Int32}(undef, L), Vector{OT}(undef, L), Vector{OT}(undef, L))
end

"""
    _bl_rows_sp1d!(sums_bl, counts_bl, xc, us, plan, ::Val{D}, blocks, brange, weights, scratch)

The single-pass analogue of [`_bl_rows_1d!`](@ref): per slice a vectorized pass forms `δu_L` and `‖δu‖²`, and the
scatter adds the six invariants. One slice's own inputs take [`_pf_sp_simd_pairs!`](@ref) into row 1.
"""
function _bl_rows_sp1d!(sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3}, xc::NTuple{D},
                        us::AbstractVector, plan, ::Val{D}, blocks, brange, weights, scratch) where {OT, CT, D}
    window, keybuf, idxbuf, rhbuf, sel, binbuf, duLbuf, dn2buf = scratch
    nb = n_histogram_bins(plan)
    boff = first(brange) - 1
    vD = Val(D)
    chooses = _chooses_compaction(blocks)
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, _component_point(xc, i, vD), jlo:last(jr), o, vD)
            ks = (jlo - o):(last(jr) - o)
            compact = chooses && _compacts(_sample_in_range(plan, keybuf, ks)...)
            n = _bl_run_bins!(binbuf, sel, plan, keybuf, idxbuf, ks, compact, chooses)
            for b in brange
                uc = us[b]
                Ui = _component_point(uc, i, vD)
                @simd for k in ks
                    du = _component_point(uc, k + o, vD) - Ui
                    duLbuf[k] = SFH.fma_dot(du, _bl_direction(rhbuf, k, vD))
                    dn2buf[k] = SFH.fma_dot(du, du)
                end
                s, c = _bl_row(sums_bl, b - boff), _bl_row(counts_bl, b - boff)
                if compact
                    for m in 1:n
                        _sp1d_accumulate!(s, c, duLbuf, dn2buf, weights, wi, o, Int(sel[m]), Int(binbuf[m]))
                    end
                else
                    for k in ks
                        bin = Int(binbuf[k])
                        1 <= bin <= nb && _sp1d_accumulate!(s, c, duLbuf, dn2buf, weights, wi, o, k, bin)
                    end
                end
            end
        end
    end
    for b in brange
        _sp1d_derive_rows!(_bl_row(sums_bl, b - boff), _bl_row(counts_bl, b - boff))
    end
    return nothing
end

function _bl_rows_sp1d!(sums_bl::AbstractArray{<:Any, 3}, counts_bl::AbstractArray{<:Any, 3}, xc::NTuple{D}, uc::Tuple,
                        plan, vD::Val{D}, blocks, brange, weights, scratch) where {D}
    window, keybuf, idxbuf, _, sel, _, duLbuf, dn2buf = scratch
    _pf_sp_simd_pairs!(_bl_row(sums_bl, 1), _bl_row(counts_bl, 1), xc, uc, plan, vD, keybuf, duLbuf, dn2buf, idxbuf,
                       sel, window, blocks, weights)
    return nothing
end

"""A task's scratch for the joint kernels: the pair window, then the digitize keys, approximate bins, directions,
compacted slots, distance bins, values, value columns, the shared columns of a second axis that is not the value,
and the second-axis quantities [`_pf_2d_simd_pairs!`](@ref) reads."""
function _bl_rows_scratch_joint(N::Int, ::Val{D}, ::Type{FT}, ::Type{OT}, second_axis) where {D, FT, OT}
    window = _pair_window(N)
    L = _pair_scratch_length(window, N)
    valbuf = Vector{OT}(undef, L)
    return (window, Vector{FT}(undef, L), Vector{Int32}(undef, L), Matrix{FT}(undef, L, D), Vector{Int32}(undef, L),
            Vector{Int32}(undef, L), valbuf, Vector{Int32}(undef, L), Vector{Int32}(undef, L),
            needs_axis_buffer(second_axis) ? Vector{OT}(undef, L) : valbuf)
end

"""
    _bl_rows_joint!(sums_bl, counts_bl, xc, us, sf, plan, val_be, second_axis, ::Val{D}, blocks, brange, weights, scratch)

The joint analogue of [`_bl_rows_1d!`](@ref) into the padded `(n_dist, n_val + 2)` rows: a second axis other than the
value is binned once per pair for every slice; on the value axis each slice's value columns are formed in its value
pass where the value edges digitize in vector form, else in its scatter. One slice's own inputs take
[`_pf_2d_simd_pairs!`](@ref) into row 1.
"""
function _bl_rows_joint!(sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3}, xc::NTuple{D},
                         us::AbstractVector, sf, plan, val_be, second_axis, ::Val{D}, blocks, brange, weights,
                         scratch) where {OT, CT, D}
    window, keybuf, idxbuf, rhbuf, sel, binbuf, valbuf, colbuf, acolbuf, _ = scratch
    nb = n_histogram_bins(plan)
    n_val = n_histogram_bins(val_be)
    boff = first(brange) - 1
    vD = Val(D)
    vector_columns = has_vector_digitize(val_be, OT)
    chooses = _chooses_compaction(blocks)
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, _component_point(xc, i, vD), jlo:last(jr), o, vD)
            ks = (jlo - o):(last(jr) - o)
            s_in, s_n = _sample_in_range(plan, keybuf, ks)
            compact = chooses && _compacts(s_in, s_n)
            n = _bl_run_bins!(binbuf, sel, plan, keybuf, idxbuf, ks, compact, chooses)
            if needs_axis_buffer(second_axis)
                for k in ks
                    rh = _bl_direction(rhbuf, k, vD)
                    acolbuf[k] = _value_column(val_be, axis_quantity(second_axis, rh, one(eltype(rh))), n_val)
                end
            end
            columns = !needs_axis_buffer(second_axis) && vector_columns && (!chooses || !_sparse(s_in, s_n))
            for b in brange
                uc = us[b]
                Ui = _component_point(uc, i, vD)
                if columns
                    @simd for k in ks
                        v = OT(sf(_component_point(uc, k + o, vD) - Ui, _bl_direction(rhbuf, k, vD)))
                        valbuf[k] = v
                        colbuf[k] = _vector_value_column(val_be, v, n_val)
                    end
                else
                    @simd for k in ks
                        valbuf[k] = sf(_component_point(uc, k + o, vD) - Ui, _bl_direction(rhbuf, k, vD))
                    end
                end
                s, c = _bl_row(sums_bl, b - boff), _bl_row(counts_bl, b - boff)
                if compact
                    for m in 1:n
                        k = Int(sel[m])
                        _joint_accumulate!(s, c, valbuf, weights, wi, o, k, Int(binbuf[m]),
                                           _bl_joint_column(second_axis, columns, val_be, valbuf, colbuf, acolbuf, k, n_val))
                    end
                else
                    for k in ks
                        bin = Int(binbuf[k])
                        1 <= bin <= nb || continue
                        _joint_accumulate!(s, c, valbuf, weights, wi, o, k, bin,
                                           _bl_joint_column(second_axis, columns, val_be, valbuf, colbuf, acolbuf, k, n_val))
                    end
                end
            end
        end
    end
    return nothing
end

"""The padded value column of slot `k` in a joint row: the shared second-axis column, the value column the value pass
formed, or the value's column formed here."""
@inline function _bl_joint_column(second_axis, columns::Bool, val_be, valbuf, colbuf, acolbuf, k, n_val)
    needs_axis_buffer(second_axis) && return Int(@inbounds acolbuf[k])
    return columns ? Int(@inbounds colbuf[k]) : _value_column(val_be, @inbounds(valbuf[k]), n_val)
end

function _bl_rows_joint!(sums_bl::AbstractArray{<:Any, 3}, counts_bl::AbstractArray{<:Any, 3}, xc::NTuple{D},
                         uc::Tuple, sf, plan, val_be, second_axis, vD::Val{D}, blocks, brange, weights,
                         scratch) where {D}
    window, keybuf, idxbuf, _, sel, _, valbuf, colbuf, _, axbuf = scratch
    _pf_2d_simd_pairs!(_bl_row(sums_bl, 1), _bl_row(counts_bl, 1), sf, xc, uc, plan, val_be, vD, keybuf, valbuf,
                       idxbuf, colbuf, sel, window, blocks, second_axis, axbuf, weights)
    return nothing
end

"""A task's scratch for the single-pass 2D kernels: the pair window, then the digitize keys, approximate bins,
directions, compacted slots, distance bins, `δu_L`, `‖δu‖²`, the six value-column buffers, and the interleaved
histogram [`_sp2d_simd_pairs!`](@ref) fills for one slice's own inputs."""
function _bl_rows_scratch_sp2d(N::Int, ::Val{D}, ::Type{FT}, ::Type{OT}, ::Type{CT}, val_plan, n_bins::Int,
                               n_val::Int) where {D, FT, OT, CT}
    window = _pair_window(N)
    L = _pair_scratch_length(window, N)
    Lc = _sp2d_has_columns(val_plan, OT) ? L : 0
    return (window, Vector{FT}(undef, L), Vector{Int32}(undef, L), Matrix{FT}(undef, L, D), Vector{Int32}(undef, L),
            Vector{Int32}(undef, L), Vector{OT}(undef, L), Vector{OT}(undef, L),
            ntuple(_ -> Vector{Int32}(undef, Lc), Val(SINGLE_PASS_N)), _sp2d_histogram(OT, CT, n_bins, n_val))
end

"""
    _bl_rows_sp2d!(sums_bl, counts_bl, xc, us, plan, value_bins, ::Val{D}, blocks, brange, weights, scratch)

The single-pass 2D analogue of [`_bl_rows_1d!`](@ref) into the padded `(6, n_bins, n_val + 2)` rows: per slice a
vectorized pass forms `δu_L` and `‖δu‖²`, and the six invariants' value columns where
[`_sp2d_column_pass`](@ref) or [`_sp2d_invariant_column_pass`](@ref) holds for the run, as the point kernel does. One
slice's own inputs take [`_sp2d_simd_pairs!`](@ref).
"""
function _bl_rows_sp2d!(sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4}, xc::NTuple{D},
                        us::AbstractVector, plan, value_bins, ::Val{D}, blocks, brange, weights,
                        scratch) where {OT, CT, D}
    window, keybuf, idxbuf, rhbuf, sel, binbuf, duLbuf, dn2buf, C, _ = scratch
    nb = n_histogram_bins(plan)
    n_val = size(sums_bl, 4) - 2
    boff = first(brange) - 1
    vD = Val(D)
    vector_columns = has_vector_digitize(value_bins, OT)
    invariant_columns = _sp2d_invariant_linear(value_bins, OT)
    chooses = _chooses_compaction(blocks)
    C1, C2, C3, C4, C5, C6 = C
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, _component_point(xc, i, vD), jlo:last(jr), o, vD)
            ks = (jlo - o):(last(jr) - o)
            s_in, s_n = _sample_in_range(plan, keybuf, ks)
            compact = chooses && _compacts(s_in, s_n)
            n = _bl_run_bins!(binbuf, sel, plan, keybuf, idxbuf, ks, compact, chooses)
            fused = vector_columns && _sp2d_column_pass(OT, !chooses, s_in, s_n)
            columns = fused || (invariant_columns && _sp2d_invariant_column_pass(OT, s_in, s_n))
            for b in brange
                uc = us[b]
                Ui = _component_point(uc, i, vD)
                if fused
                    @simd ivdep for k in ks
                        du = _component_point(uc, k + o, vD) - Ui
                        duL, dn2 = OT(SFH.fma_dot(du, _bl_direction(rhbuf, k, vD))), OT(SFH.fma_dot(du, du))
                        duLbuf[k], dn2buf[k] = duL, dn2
                        v = single_pass_invariants(duL, dn2)
                        C1[k] = _vector_value_column(value_bins, v[1], n_val)
                        C2[k] = _vector_value_column(value_bins, v[2], n_val)
                        C3[k] = _vector_value_column(value_bins, v[3], n_val)
                        C4[k] = _vector_value_column(value_bins, v[4], n_val)
                        C5[k] = _vector_value_column(value_bins, v[5], n_val)
                        C6[k] = _vector_value_column(value_bins, v[6], n_val)
                    end
                else
                    @simd for k in ks
                        du = _component_point(uc, k + o, vD) - Ui
                        duLbuf[k] = SFH.fma_dot(du, _bl_direction(rhbuf, k, vD))
                        dn2buf[k] = SFH.fma_dot(du, du)
                    end
                    columns && _sp2d_invariant_columns!(C, value_bins, duLbuf, dn2buf, ks, n_val)
                end
                row = b - boff
                if compact
                    for m in 1:n
                        k = Int(sel[m])
                        _bl_sp2d_add!(sums_bl, counts_bl, row, Int(binbuf[m]), k, duLbuf, dn2buf, columns, C,
                                      value_bins, n_val, wi * _point_weight(weights, k + o))
                    end
                else
                    for k in ks
                        bin = Int(binbuf[k])
                        1 <= bin <= nb || continue
                        _bl_sp2d_add!(sums_bl, counts_bl, row, bin, k, duLbuf, dn2buf, columns, C, value_bins, n_val,
                                      wi * _point_weight(weights, k + o))
                    end
                end
            end
        end
    end
    return nothing
end

"""Add slot `k`'s six invariants, from its stored `δu_L` and `‖δu‖²`, into row `row` at distance bin `dbin`, each in
its padded value column: the stored one with `columns`, else formed here."""
@inline function _bl_sp2d_add!(sums_bl, counts_bl::AbstractArray{CT}, row, dbin, k, duLbuf, dn2buf, columns::Bool, C,
                               value_bins, n_val, w) where {CT}
    s = single_pass_invariants(@inbounds(duLbuf[k]), @inbounds(dn2buf[k]))
    @sp2d_each_invariant value_bins t vb begin
        col = columns ? Int(@inbounds(C[t][k])) : _value_column(vb, s[t], n_val)
        @inbounds sums_bl[row, t, dbin, col] += w * s[t]
        @inbounds counts_bl[row, t, dbin, col] += CT(w)
    end
    return nothing
end

function _bl_rows_sp2d!(sums_bl::AbstractArray{<:Any, 4}, counts_bl::AbstractArray{<:Any, 4}, xc::NTuple{D}, uc::Tuple,
                        plan, value_bins, vD::Val{D}, blocks, brange, weights, scratch) where {D}
    window, keybuf, idxbuf, _, sel, _, duLbuf, dn2buf, C, h = scratch
    n_bins, n_val = size(sums_bl, 3), size(sums_bl, 4) - 2
    fill!(h, zero(eltype(h)))
    _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, vD, keybuf, duLbuf, dn2buf, idxbuf, C, sel, window, n_val, blocks,
                      weights)
    _sp2d_unpack!(view(_bl_row(sums_bl, 1), :, :, 2:(n_val + 1)), view(_bl_row(counts_bl, 1), :, :, 2:(n_val + 1)), h,
                  n_bins, n_val)
    return nothing
end

function _bl_run_1d!(sums, counts, sf_type, x, u, distance_bins, geom, executor, workspace = nothing;
                     weights = NoWeights(), culling::CullingPolicy = AutoCulling())
    dist_be = digitize_plan(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, vFX = _bl_prepare(x, u, geom, workspace)
    vD = SFH.field_width(geom)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :sf1d, (n_bins,), OT, CT)
    make_accum(bw) = (zeros(OT, bw, n_bins), zeros(CT, bw, n_bins))
    accum_bytes = _bl_accum_bytes(OT, CT, B, n_bins)
    rows = vS -> begin
        plan = squared_digitize_plan(distance_bins)
        kernel! = (s, c, xc, us, blocks, br, w, scr) -> _bl_rows_1d!(s, c, xc, us, sf_type, plan, vS, blocks, br, w,
                                                                     scr)
        _bl_rows_sweep(executor, _bl_slices(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!,
                       () -> _bl_rows_scratch_1d(N, vS, eltype(x0), OT), make_accum, N, B, accum_bytes, workspace)
    end
    lanes = () -> begin
        sh_plan = _bl_shared_plan(geom, distance_bins)
        kernel! = _bl_by_layout(vFX,
            (s, c, xk, uk, blocks, br, w, scr) -> _bl_shared_1d!(s, c, xk, uk, sf_type, sh_plan, geom, vD, blocks, br,
                                                                 w, _bl_scratch_args(scr)...),
            (s, c, xk, uk, blocks, br, w, scr) -> _bl_varying_1d!(s, c, xk, uk, sf_type, dist_be, geom, vD, blocks, br,
                                                                  w))
        _bl_sweep(executor, _bl_cull(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!,
                  xs -> _bl_shared_positions(xs, geom), make_accum, vD, N, B, accum_bytes, workspace)
    end
    sums_bl, counts_bl = _bl_mode(_simd_width(geom), vFX, B, BL_SLICE_LANES_MIN, rows, lanes)
    _bl_add_permuted!(reshape(sums, n_bins, B), sums_bl, (2, 1))
    _bl_add_permuted!(reshape(counts, n_bins, B), counts_bl, (2, 1))
    return nothing
end

function _bl_run_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins, geom, executor,
                          workspace = nothing; weights = NoWeights(), culling::CullingPolicy = AutoCulling(),
                          second_axis::AbstractSecondAxisSource = InvariantValueAxis())
    dist_be = digitize_plan(distance_bins); val_be = digitize_plan(value_bins)
    n_dist = n_histogram_bins(dist_be); n_val = n_histogram_bins(val_be)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, vFX = _bl_prepare(x, u, geom, workspace)
    vD = SFH.field_width(geom)
    _validate_bl_geometry(geom, W, D)
    _require_value_axis(second_axis, geom)
    _validate_ws_layout(workspace, :joint2d, _bl_accum_tail(Val(:joint2d), n_dist, n_val), OT, CT)
    make_accum(bw) = (zeros(OT, bw, n_dist, n_val + 2), zeros(CT, bw, n_dist, n_val + 2))
    accum_bytes = _bl_accum_bytes(OT, CT, B, n_dist, n_val + 2)
    rows = vS -> begin
        plan = squared_digitize_plan(distance_bins)
        kernel! = (s, c, xc, us, blocks, br, w, scr) -> _bl_rows_joint!(s, c, xc, us, sf_type, plan, val_be,
                                                                        second_axis, vS, blocks, br, w, scr)
        _bl_rows_sweep(executor, _bl_slices(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!,
                       () -> _bl_rows_scratch_joint(N, vS, eltype(x0), OT, second_axis), make_accum, N, B,
                       accum_bytes, workspace)
    end
    lanes = () -> begin
        kernel! = _bl_by_layout(vFX,
            (s, c, xk, uk, blocks, br, w, _) -> _bl_joint2d_shared!(s, c, xk, uk, sf_type, dist_be, val_be, geom, vD,
                                                                    blocks, br, w, second_axis),
            (s, c, xk, uk, blocks, br, w, _) -> _bl_joint2d_varying!(s, c, xk, uk, sf_type, dist_be, val_be, geom, vD,
                                                                     blocks, br, w, second_axis))
        _bl_sweep(executor, _bl_cull(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!, identity,
                  make_accum, vD, N, B, accum_bytes, workspace)
    end
    sums_bl, counts_bl = _bl_mode(_simd_width(geom), vFX, B, typemax(Int), rows, lanes)
    _bl_add_permuted!(reshape(sums, n_dist, n_val, B), view(sums_bl, :, :, 2:(n_val + 1)), (2, 3, 1))
    _bl_add_permuted!(reshape(counts, n_dist, n_val, B), view(counts_bl, :, :, 2:(n_val + 1)), (2, 3, 1))
    return nothing
end

function _bl_run_sp1d!(sums, counts, x, u, distance_bins, geom, executor, workspace = nothing;
                       weights = NoWeights(), culling::CullingPolicy = AutoCulling())
    dist_be = digitize_plan(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, vFX = _bl_prepare(x, u, geom, workspace)
    vD = SFH.field_width(geom)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :single_pass, (SINGLE_PASS_N, n_bins), OT, CT)
    make_accum(bw) = (zeros(OT, bw, SINGLE_PASS_N, n_bins), zeros(CT, bw, SINGLE_PASS_N, n_bins))
    accum_bytes = _bl_accum_bytes(OT, CT, B, SINGLE_PASS_N, n_bins)
    rows = vS -> begin
        plan = squared_digitize_plan(distance_bins)
        kernel! = (s, c, xc, us, blocks, br, w, scr) -> _bl_rows_sp1d!(s, c, xc, us, plan, vS, blocks, br, w, scr)
        _bl_rows_sweep(executor, _bl_slices(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!,
                       () -> _bl_rows_scratch_sp1d(N, vS, eltype(x0), OT), make_accum, N, B, accum_bytes, workspace)
    end
    lanes = () -> begin
        sh_plan = _bl_shared_plan(geom, distance_bins)
        kernel! = _bl_by_layout(vFX,
            (s, c, xk, uk, blocks, br, w, scr) -> _bl_sp1d_shared!(s, c, xk, uk, sh_plan, geom, vD, blocks, br, w,
                                                                   _bl_scratch_args(scr)...),
            (s, c, xk, uk, blocks, br, w, scr) -> _bl_sp1d_varying!(s, c, xk, uk, dist_be, geom, vD, blocks, br, w))
        _bl_sweep(executor, _bl_cull(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!,
                  xs -> _bl_shared_positions(xs, geom), make_accum, vD, N, B, accum_bytes, workspace)
    end
    sums_bl, counts_bl = _bl_mode(_simd_width(geom), vFX, B, BL_SLICE_LANES_MIN, rows, lanes)
    _bl_add_permuted!(reshape(sums, SINGLE_PASS_N, n_bins, B), sums_bl, (2, 3, 1))
    _bl_add_permuted!(reshape(counts, SINGLE_PASS_N, n_bins, B), counts_bl, (2, 3, 1))
    return nothing
end

function _bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins, geom, executor, workspace = nothing;
                       weights = NoWeights(), culling::CullingPolicy = AutoCulling())
    dist_be = digitize_plan(distance_bins); val_plan = digitize_plan(value_bins)
    n_bins = n_histogram_bins(dist_be)
    n_val = size(sums, 3)
    _validate_value_bins!(val_plan, n_val)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, vFX = _bl_prepare(x, u, geom, workspace)
    vD = SFH.field_width(geom)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :single_pass_2d, _bl_accum_tail(Val(:single_pass_2d), n_bins, n_val), OT, CT)
    make_accum(bw) = (zeros(OT, bw, SINGLE_PASS_N, n_bins, n_val + 2), zeros(CT, bw, SINGLE_PASS_N, n_bins, n_val + 2))
    accum_bytes = _bl_accum_bytes(OT, CT, B, SINGLE_PASS_N, n_bins, n_val + 2)
    rows = vS -> begin
        plan = squared_digitize_plan(distance_bins)
        kernel! = (s, c, xc, us, blocks, br, w, scr) -> _bl_rows_sp2d!(s, c, xc, us, plan, val_plan, vS, blocks, br, w,
                                                                       scr)
        _bl_rows_sweep(executor, _bl_slices(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!,
                       () -> _bl_rows_scratch_sp2d(N, vS, eltype(x0), OT, CT, val_plan, n_bins, n_val), make_accum,
                       N, B, accum_bytes, workspace)
    end
    lanes = () -> begin
        kernel! = _bl_by_layout(vFX,
            (s, c, xk, uk, blocks, br, w, _) -> _bl_sp2d_shared!(s, c, xk, uk, dist_be, val_plan, geom, vD, blocks, br,
                                                                 w),
            (s, c, xk, uk, blocks, br, w, _) -> _bl_sp2d_varying!(s, c, xk, uk, dist_be, val_plan, geom, vD, blocks,
                                                                  br, w))
        _bl_sweep(executor, _bl_cull(x0, u0, geom, distance_bins, culling, vFX, weights), kernel!, identity,
                  make_accum, vD, N, B, accum_bytes, workspace)
    end
    sums_bl, counts_bl = _bl_mode(_simd_width(geom), vFX, B, typemax(Int), rows, lanes)
    interior = 2:(n_val + 1)
    _bl_add_permuted!(reshape(sums, SINGLE_PASS_N, n_bins, n_val, B), view(sums_bl, :, :, :, interior), (2, 3, 4, 1))
    _bl_add_permuted!(reshape(counts, SINGLE_PASS_N, n_bins, n_val, B), view(counts_bl, :, :, :, interior),
                      (2, 3, 4, 1))
    return nothing
end
