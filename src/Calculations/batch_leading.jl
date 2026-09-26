# CPU batch kernels over a batch-leading `(B, D, N)` working buffer, specialized on `Val(D)`.
#
# `D` reaches every kernel as a type parameter, so each `SVector{D}` has a compile-time type and the
# body allocates nothing; the one dynamic dispatch on `Val(D)` is amortized over the whole `O(N²B)`
# sweep. The batch axis is innermost, so `@simd for b` reads unit stride, and with shared positions
# the bin is constant across `b`, making the accumulation contiguous.

using Distances: Distances as DI

@inline _bl_unwrap(u) = (u, false)              # (array, already_batch_leading)
@inline _bl_unwrap(u::BatchLeading) = (u.data, true)

# Prepare batch-leading (B,D,N) buffers. Handles the default plain `(D,N,B...)` (transposed
# once) and `BatchLeading` `(B,D,N)` (zero-copy). `x` may be fixed (D,N) or varying. Called
# once per top-level call (not in the hot loop), so the type-instability of the branch is a
# harmless function barrier. Returns (xb, ub, B, D, W, N, fixed_x, geom): `D` and `W` are the
# field and coordinate widths the kernels load, and `geom` carries the velocity dimension.
"""
Component-first view of an input, so `prepare_pair_inputs` — which reads components from axis 1 —
can convert it. Only a batch-leading `(B, W, N)` array needs permuting; the default `(W, N, B…)`
layout is already component-first. `permutedims` costs one `O(N·B)` pass per call.
"""
@inline _bl_component_first(a, is_batch_leading::Bool) =
    is_batch_leading ? permutedims(a, (2, 3, 1)) : a

function _bl_prepare(x, u, distance_metric = DI.Euclidean(), workspace = nothing)
    u_raw, u_bl = _bl_unwrap(u)
    x_raw, x_bl = _bl_unwrap(x)
    fixed_x = ndims(x_raw) == 2
    # The velocity dimension, read before any conversion — this is what fixes the geometry, and it
    # is not recoverable from the converted arrays.
    D_in = u_bl ? size(u_raw, 2) : size(u_raw, 1)
    geom = SFH.pair_geometry_for(distance_metric, Val(D_in))
    if !(geom isa SFH.FlatGeometry)
        # Convert once per call, component-first, then let the layout code below run unchanged.
        x_raw, u_raw = SFH.prepare_pair_inputs(
            geom, _bl_component_first(x_raw, x_bl), _bl_component_first(u_raw, u_bl),
        )
        x_bl = false
        u_bl = false
    end
    # `W` is the coordinate width and `D` the field width the kernels load; they differ from each
    # other and from the velocity dimension on a sphere.
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
    return xb, ub, B, D, W, N, fixed_x, geom
end

# Statically-sized, unchecked loads for the two layouts the batch drivers hold: `(W, N)` shared
# positions and `(B, W, N)` batch-leading, indexed within the shapes `_bl_prepare` validated. The width
# comes from the geometry, never from the velocity rank.
@inline _bl_pt(x::AbstractMatrix, i, ::Val{W}) where {W} =
    SA.SVector{W}(ntuple(d -> @inbounds(x[d, i]), Val(W)))
@inline _bl_pt(xb::AbstractArray{<:Any, 3}, b, i, ::Val{W}) where {W} =
    SA.SVector{W}(ntuple(d -> @inbounds(xb[b, d, i]), Val(W)))
@inline _bl_vel(ub, b, i, ::Val{D}) where {D} =
    SA.SVector{D}(ntuple(d -> @inbounds(ub[b, d, i]), Val(D)))

"""
Throw unless the staged coordinate width matches what `geom` needs.

The batch entry points take raw `(D, N, B…)` arrays, so this is where a mismatched `x` is caught,
before the kernels load it under `@inbounds`.
"""
@inline function _validate_bl_geometry(geom, W::Int, D::Int)
    want = _val_int(SFH.coordinate_width(geom))
    W == want || throw(
        DimensionMismatch(
            "$(nameof(typeof(geom))) locates a point with $want coordinate(s) on axis 1 of x, but " *
            "got $W (velocity dimension D=$D)",
        ),
    )
    return nothing
end

# The workspace's transpose buffers, or `nothing` for the allocate-fresh path.
@inline _ws_ub(::Nothing) = nothing
@inline _ws_xb(::Nothing) = nothing

# (D,N,B) -> (B,D,N) materialized batch-leading buffer (lazy PermutedDimsArray would put the
# strided read back in the hot loop, so we materialize — cheap, O(D·N·B) ≪ O(N²·B)).
@inline _to_batch_leading(u_DNB, ::Nothing) = permutedims(u_DNB, (3, 1, 2))
@inline _to_batch_leading(u_DNB, dest::AbstractArray) = permutedims!(dest, u_DNB, (3, 1, 2))

# ----------------------------------------------------------------------------------------
# Shared positions ("same surface"): x is (D,N) fixed; geometry computed ONCE per pair.
# ub :: (B, D, N) ; sums_bl, counts_bl :: (B, n_bins)
# ----------------------------------------------------------------------------------------
"""The distance plan of a shared-position kernel: squared on a flat metric, whose kernels digitize
`r²` in a vectorized geometry pass; the bins' own plan otherwise."""
@inline _bl_shared_plan(::SFH.FlatGeometry, bins) = squared_digitize_plan(bins)
@inline _bl_shared_plan(geom, bins) = digitize_plan(bins)

"""Shared positions as a kernel loads them: contiguous component vectors on a flat metric, the
`(W, N)` matrix otherwise."""
@inline _bl_shared_positions(x::AbstractMatrix, ::SFH.FlatGeometry{D}) where {D} = ntuple(d -> x[d, :], Val(D))
@inline _bl_shared_positions(x, geom) = x

"""Per-call buffers of a flat shared-position kernel under `window`: digitize key, approximate bin,
direction."""
@inline function _bl_geometry_scratch(window::PairWindow, xc::NTuple{W, AbstractVector{T}}, ::Val{W}) where {T, W}
    L = _pair_scratch_length(window, length(xc[1]))
    return Vector{T}(undef, L), Vector{Int32}(undef, L), Matrix{T}(undef, L, W)
end

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
    sums_bl::AbstractMatrix, counts_bl::AbstractMatrix, xc::NTuple{D, AbstractVector}, ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType, plan::AbstractSquaredDigitizePlan, geom::SFH.FlatGeometry,
    vD::Val{D}, blocks, brange, weights = NoWeights(),
) where {D}
    window = _pair_window(length(xc[1]))
    return _bl_shared_1d!(sums_bl, counts_bl, xc, ub, sf_type, plan, geom, vD, blocks, brange, weights, window,
                          _bl_geometry_scratch(window, xc, vD)...)
end

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
    keybuf, idxbuf, rhbuf,
) where {OT, CT, D}
    nb = size(sums_bl, 2)
    boff = first(brange) - 1
    vD = Val(D)
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            Xi = _component_point(xc, i, vD)
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, Xi, jlo:last(jr), o, vD)
            for k in (jlo - o):(last(jr) - o)
                bin = squared_bin(plan, keybuf[k], idxbuf[k])
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
            # Loop-invariant across the b strip: one frame, one direction and one pair weight serve
            # every field, since a weight belongs to the point and not to the slice.
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

# ----------------------------------------------------------------------------------------
# Varying positions: x is (B,D,N) too; geometry depends on b ⇒ computed inside the b loop.
# ----------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------
# Joint 2D (single SF): output accumulator (B, n_dist, n_val). vbin varies per b (scatter on
# the value axis) ⇒ plain b-loop (still 0-alloc, type-stable). dbin constant for shared x.
# ----------------------------------------------------------------------------------------
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
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3)
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
                vbin = _slice_axis_bin(abin, val, val_be)
                if 1 <= vbin <= n_val
                    sums_bl[b - boff, dbin, vbin] += w * val
                    counts_bl[b - boff, dbin, vbin] += CT(w)
                end
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
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3)
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
                vbin = SFH.digitize(pair_axis_key(second_axis, val, Xi, Xj, dist), val_be)
                if 1 <= vbin <= n_val
                    sums_bl[b - boff, dbin, vbin] += w * val
                    counts_bl[b - boff, dbin, vbin] += CT(w)
                end
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------------------
# Single-pass 1D (6 invariants). Accumulator (B, 6, n_dist): for shared x the dist bin is
# constant over b, so each of the 6 writes is contiguous in b (vectorizes). Note: only du_L
# and du_norm2=⟨du,du⟩ are needed — the old `du_T = mδu_t(...)` was dead work (now removed).
# ----------------------------------------------------------------------------------------
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
    sums_bl::AbstractArray{<:Any, 3}, counts_bl::AbstractArray{<:Any, 3}, xc::NTuple{D, AbstractVector},
    ub::AbstractArray{<:Any, 3}, plan::AbstractSquaredDigitizePlan, geom::SFH.FlatGeometry, vD::Val{D}, blocks,
    brange, weights = NoWeights(),
) where {D}
    window = _pair_window(length(xc[1]))
    return _bl_sp1d_shared!(sums_bl, counts_bl, xc, ub, plan, geom, vD, blocks, brange, weights, window,
                            _bl_geometry_scratch(window, xc, vD)...)
end

function _bl_sp1d_shared!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xc::NTuple{D, AbstractVector}, ub::AbstractArray{<:Any, 3}, plan::AbstractSquaredDigitizePlan,
    geom::SFH.FlatGeometry, ::Val{D}, blocks, brange, weights, window::PairWindow, keybuf, idxbuf, rhbuf,
) where {OT, CT, D}
    nb = size(sums_bl, 3)
    boff = first(brange) - 1
    vD = Val(D)
    @inbounds for (ir, jr) in blocks
        _check_run_fits(window, keybuf, jr)
        o = _slot_offset(window, jr)
        for i in ir
            jlo = max(i + 1, first(jr))
            jlo > last(jr) && continue
            Xi = _component_point(xc, i, vD)
            wi = _point_weight(weights, i)
            _bl_block_geometry!(keybuf, idxbuf, rhbuf, plan, xc, Xi, jlo:last(jr), o, vD)
            for k in (jlo - o):(last(jr) - o)
                bin = squared_bin(plan, keybuf[k], idxbuf[k])
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

# ----------------------------------------------------------------------------------------
# Single-pass 2D (6 invariants × per-invariant value bins). Accumulator (B, 6, n_dist, n_val);
# per-invariant value bin ⇒ scatter ⇒ plain b-loop.
# ----------------------------------------------------------------------------------------
@inline function _bl_sp2d_write!(sums_bl, counts_bl, b, dbin, vals, value_bins, n_val, ::Type{CT},
                                 w = true) where {CT}
    @sp2d_each_invariant value_bins t vb begin
        vbin = SFH.digitize(vals[t], vb)
        if 1 <= vbin <= (length(vb) - 1) && vbin <= n_val
            @inbounds sums_bl[b, t, dbin, vbin] += w * vals[t]
            @inbounds counts_bl[b, t, dbin, vbin] += CT(w)
        end
    end
    return nothing
end

function _bl_sp2d_shared!(
    sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3}, dist_be, value_bins, geom, ::Val{D}, blocks, brange,
    weights = NoWeights(),
) where {OT, CT, D}
    nb = size(sums_bl, 3); n_val = size(sums_bl, 4)
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
    nb = size(sums_bl, 3); n_val = size(sums_bl, 4)
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

# ========================================================================================
# Drivers: prep (transpose to batch-leading) + run kernel via an EXECUTOR + transpose back.
#
# Parallelism is over the outer pair index `i`, so each pair's geometry is computed once and the
# inner batch loop over `b` stays full and SIMD-vectorized. `i` is partitioned into round-robin
# chunks, which balances the triangle, and thread-local accumulators are reduced at the end.
#
# executor(make_accum, run_chunk!, ifull, B, accum_bytes, ws) → reduced (sums, counts), width B:
#   make_accum(bw)                     → fresh zeroed (sums_bl, counts_bl) of batch width bw
#   run_chunk!(acc, isub, brange)      → kernel over outer i ∈ isub and batch b ∈ brange
#   accum_bytes                        → bytes of one full-width accumulator, for the split model
#   ws                                 → CPUSFWorkspace to draw accumulators from, or `nothing`
#   serial  : one full-width accumulator over ifull
#   threaded: partitions (i, b); per-task accumulators are only as wide as their b-chunk
# ========================================================================================

@inline function _bl_serial_exec(make_accum, run_chunk!, ifull, B, accum_bytes, ws)
    acc = _bl_accum_pool(ws, make_accum, [B])[1]
    _bl_zero_accum!(acc)
    run_chunk!(acc, ifull, 1:B)
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

Every task holds an accumulator, so the live footprint is `n_tasks * accum_bytes` — that is what
turns into GC and page-fault time, and it is what a batch chunk divides. Splitting `b` costs pair
geometry, recomputed once per batch chunk, so the right answer is the SMALLEST split that brings
the footprint under budget. Kernels whose footprint already fits (the 1D batch accumulator is tens
of KiB) keep `Bc == 1` and never pay geometry twice; the single-pass 2D accumulator at large `B`
and many threads reaches hundreds of MiB and is split until it fits.
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
[`CPUSFWorkspace`](@ref) must hold. A threaded backend is refused without the OhMyThreads
extension, so with the extension absent the count is the one task a serial or `AutoBackend()` call
will use.
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

Add the batch-leading accumulator `src_bl` into `dest` through the permutation `perm`.

`!` entry points across the package **accumulate** into the caller's buffers; zeroing belongs to the
non-mutating wrappers. `PermutedDimsArray` is a lazy view, so this fuses the permute with the add
and allocates nothing.
"""
@inline function _bl_add_permuted!(dest, src_bl, perm)
    dest .+= PermutedDimsArray(src_bl, perm)
    return dest
end

"""
    _bl_cull(xb, ub, geom, distance_bins, culling, fixed_x, weights) -> (grid, xs, us, ws)

The inputs sorted into cull grids, or unchanged with `grid === nothing` when `culling` declines.

Shared positions are sorted once for every slice: one grid, and the `(W, N)` positions, the
`(B, D, N)` fields and the weights in its order. Positions varying per slice are sorted per slice:
`grid[b]` is slice `b`'s grid (`nothing` where that slice declines), `xs[b]` and `us[b]` its
`(1, W, N)` positions and `(1, D, N)` fields in that order, `ws[b]` its weights.
"""
function _bl_cull(xb, ub, geom, distance_bins, culling::CullingPolicy, fixed_x::Bool, weights)
    _cull_enabled(culling) || return nothing, xb, ub, weights
    W = _val_int(SFH.coordinate_width(geom))
    if fixed_x
        grid = cull_grid_for(ntuple(d -> view(xb, d, :), W), geom, distance_bins, culling)
        grid === nothing && return nothing, xb, ub, weights
        p = grid.perm
        return grid, xb[:, p], ub[:, :, p], _permuted_point_weights(weights, p)
    end
    grids = [cull_grid_for(ntuple(d -> view(xb, b, d, :), W), geom, distance_bins, culling) for b in axes(ub, 1)]
    all(isnothing, grids) && return nothing, xb, ub, weights
    perms = [g === nothing ? collect(axes(ub, 3)) : g.perm for g in grids]
    return grids, [xb[b:b, :, p] for (b, p) in pairs(perms)], [ub[b:b, :, p] for (b, p) in pairs(perms)],
           [_permuted_point_weights(weights, p) for p in perms]
end

"""
    _bl_chunk_runner(kernel!, grid, xs, us, ws, N) -> run_chunk!

The executor's `run_chunk!(acc, isub, brange)` for [`_bl_cull`](@ref)'s output: `kernel!(sums,
counts, x, u, blocks, brange, weights)` once over the slices of `brange` with one set of blocks, or,
for per-slice grids, once per slice with that slice's inputs and blocks into its row of `acc`.
"""
function _bl_chunk_runner(kernel!::K, grid, xs, us, ws, N::Int) where {K}
    grid isa AbstractVector ||
        return (acc, isub, br) -> kernel!(acc[1], acc[2], xs, us, pair_blocks(N, isub; grid), br, ws)
    return function (acc, isub, br)
        for (k, b) in enumerate(br)
            kernel!(selectdim(acc[1], 1, k:k), selectdim(acc[2], 1, k:k), xs[b], us[b],
                    pair_blocks(N, isub; grid = grid[b]), 1:1, ws[b])
        end
        return nothing
    end
end

function _bl_run_1d!(sums, counts, sf_type, x, u, distance_bins, distance_metric, executor, workspace = nothing;
                     weights = NoWeights(), culling::CullingPolicy = AutoCulling())
    dist_be = digitize_plan(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :sf1d, (n_bins,), OT, CT)
    grid, xs, us, ws = _bl_cull(x0, u0, geom, distance_bins, culling, fixed_x, weights)
    sh_plan = _bl_shared_plan(geom, distance_bins)
    make_accum(bw) = (zeros(OT, bw, n_bins), zeros(CT, bw, n_bins))
    kernel! = fixed_x ?
        ((s, c, xk, uk, blocks, br, w) -> _bl_shared_1d!(s, c, xk, uk, sf_type, sh_plan, geom, vD, blocks, br, w)) :
        ((s, c, xk, uk, blocks, br, w) -> _bl_varying_1d!(s, c, xk, uk, sf_type, dist_be, geom, vD, blocks, br, w))
    run_chunk! = _bl_chunk_runner(kernel!, grid, fixed_x ? _bl_shared_positions(xs, geom) : xs, us, ws, N)
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, n_bins), workspace)
    _bl_add_permuted!(reshape(sums, n_bins, B), sums_bl, (2, 1))
    _bl_add_permuted!(reshape(counts, n_bins, B), counts_bl, (2, 1))
    return nothing
end

function _bl_run_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins, distance_metric, executor,
                          workspace = nothing; weights = NoWeights(), culling::CullingPolicy = AutoCulling(),
                          second_axis::AbstractSecondAxisSource = InvariantValueAxis())
    dist_be = digitize_plan(distance_bins); val_be = digitize_plan(value_bins)
    n_dist = n_histogram_bins(dist_be); n_val = n_histogram_bins(val_be)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _require_value_axis(second_axis, geom)
    _validate_ws_layout(workspace, :joint2d, (n_dist, n_val), OT, CT)
    grid, xs, us, ws = _bl_cull(x0, u0, geom, distance_bins, culling, fixed_x, weights)
    make_accum(bw) = (zeros(OT, bw, n_dist, n_val), zeros(CT, bw, n_dist, n_val))
    kernel! = fixed_x ?
        ((s, c, xk, uk, blocks, br, w) -> _bl_joint2d_shared!(s, c, xk, uk, sf_type, dist_be, val_be, geom, vD,
                                                              blocks, br, w, second_axis)) :
        ((s, c, xk, uk, blocks, br, w) -> _bl_joint2d_varying!(s, c, xk, uk, sf_type, dist_be, val_be, geom, vD,
                                                               blocks, br, w, second_axis))
    run_chunk! = _bl_chunk_runner(kernel!, grid, xs, us, ws, N)
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, n_dist, n_val), workspace)
    _bl_add_permuted!(reshape(sums, n_dist, n_val, B), sums_bl, (2, 3, 1))
    _bl_add_permuted!(reshape(counts, n_dist, n_val, B), counts_bl, (2, 3, 1))
    return nothing
end

function _bl_run_sp1d!(sums, counts, x, u, distance_bins, distance_metric, executor, workspace = nothing;
                       weights = NoWeights(), culling::CullingPolicy = AutoCulling())
    dist_be = digitize_plan(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :single_pass, (SINGLE_PASS_N, n_bins), OT, CT)
    grid, xs, us, ws = _bl_cull(x0, u0, geom, distance_bins, culling, fixed_x, weights)
    sh_plan = _bl_shared_plan(geom, distance_bins)
    make_accum(bw) = (zeros(OT, bw, SINGLE_PASS_N, n_bins), zeros(CT, bw, SINGLE_PASS_N, n_bins))
    kernel! = fixed_x ?
        ((s, c, xk, uk, blocks, br, w) -> _bl_sp1d_shared!(s, c, xk, uk, sh_plan, geom, vD, blocks, br, w)) :
        ((s, c, xk, uk, blocks, br, w) -> _bl_sp1d_varying!(s, c, xk, uk, dist_be, geom, vD, blocks, br, w))
    run_chunk! = _bl_chunk_runner(kernel!, grid, fixed_x ? _bl_shared_positions(xs, geom) : xs, us, ws, N)
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, SINGLE_PASS_N, n_bins), workspace)
    _bl_add_permuted!(reshape(sums, SINGLE_PASS_N, n_bins, B), sums_bl, (2, 3, 1))
    _bl_add_permuted!(reshape(counts, SINGLE_PASS_N, n_bins, B), counts_bl, (2, 3, 1))
    return nothing
end

function _bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins, distance_metric, executor, workspace = nothing;
                       weights = NoWeights(), culling::CullingPolicy = AutoCulling())
    dist_be = digitize_plan(distance_bins); val_plan = digitize_plan(value_bins)
    n_bins = n_histogram_bins(dist_be)
    n_val = size(sums, 3)
    _validate_value_bins!(val_plan, n_val)
    OT = eltype(sums); CT = eltype(counts)
    x0, u0, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :single_pass_2d, (SINGLE_PASS_N, n_bins, n_val), OT, CT)
    grid, xs, us, ws = _bl_cull(x0, u0, geom, distance_bins, culling, fixed_x, weights)
    make_accum(bw) = (zeros(OT, bw, SINGLE_PASS_N, n_bins, n_val), zeros(CT, bw, SINGLE_PASS_N, n_bins, n_val))
    kernel! = fixed_x ?
        ((s, c, xk, uk, blocks, br, w) -> _bl_sp2d_shared!(s, c, xk, uk, dist_be, val_plan, geom, vD, blocks, br, w)) :
        ((s, c, xk, uk, blocks, br, w) -> _bl_sp2d_varying!(s, c, xk, uk, dist_be, val_plan, geom, vD, blocks, br, w))
    run_chunk! = _bl_chunk_runner(kernel!, grid, xs, us, ws, N)
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, SINGLE_PASS_N, n_bins, n_val), workspace)
    _bl_add_permuted!(reshape(sums, SINGLE_PASS_N, n_bins, n_val, B), sums_bl, (2, 3, 4, 1))
    _bl_add_permuted!(reshape(counts, SINGLE_PASS_N, n_bins, n_val, B), counts_bl, (2, 3, 4, 1))
    return nothing
end
