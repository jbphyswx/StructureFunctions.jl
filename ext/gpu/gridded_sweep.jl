# The direct lag sweep on a device: the route a non-polynomial operator and a value histogram take on a
# grid, which the transform cannot express because it computes polynomial moments of each lag.
#
# A unit of `L` lanes owns one slice of one (slab pair, lag), or one of `n_chunks` parts of it when the lags
# alone would not fill the device; its lanes sweep the lag's cells at the stride of the chunked unit
# (`SFC._lag_fold`), so consecutive lanes read consecutive cells, and a workgroup of `WG` lanes holds `WG ÷ L`
# units ([`LagPlan`](@ref)).
# The lanes an item takes follow the launch's total work and the lag's cells ([`_lag_plan`](@ref)). A schedule
# whose pairs have unequal lag boxes indexes the (slab pair, lag) items through the prefix sum of the boxes'
# volumes (`SFC._lag_boxes`).

"""`A[I...] += x` as a device atomic."""
@inline _atomic_add!(A, x, I::Vararg{Int}) = (@atomic A[I...] += x; nothing)

"""Lanes of a lag-sweep workgroup."""
const LAG_WG = 128

"""Lanes per multiprocessor a lag-sweep launch spreads over its items, at least."""
const LAG_LANES_PER_SM = 2048

"""Lanes per multiprocessor a lag-sweep launch spreads over its items, at most."""
const LAG_MAX_LANES_PER_SM = 131072

"""Fewest lanes an item takes, cells allowing."""
const LAG_MIN_LANES = 8

"""Fewest cells a lane sweeps."""
const LAG_MIN_CELLS_PER_LANE = 16

"""Most cells a lane sweeps before its item takes more lanes, the launch's lanes allowing."""
const LAG_MAX_CELLS_PER_LANE = 256

"""Most lanes of a unit."""
const LAG_UNIT_LANES = 64

"""Most units an item is split into."""
const LAG_MAX_CHUNKS = 16

"""Launch plan of the lag-sweep kernels: workgroups of `WG` lanes holding units of `L` lanes, each (slab pair,
lag) of a slice split into `n_chunks` units."""
struct LagPlan{WG, L}
    n_chunks::Int
end

"""The plan for `n_items` (slab pair, lag) items of `nt` slices over slabs of `cells` cells on the device `caps`
describes: each item takes the lanes `LAG_LANES_PER_SM` per multiprocessor spread over the items and slices, at
least `LAG_MIN_LANES` and enough that no lane sweeps more than `LAG_MAX_CELLS_PER_LANE` cells, at most
`LAG_MAX_LANES_PER_SM` per multiprocessor spread the same way and one per `LAG_MIN_CELLS_PER_LANE` cells, rounded
down to a power of two; they form units of at most `LAG_UNIT_LANES` lanes, at most `LAG_MAX_CHUNKS` of them."""
function _lag_plan(caps::SFC.GPUDeviceCaps, n_items::Int, nt::Int, cells::Int)
    n_work = n_items * nt
    lanes = max(LAG_LANES_PER_SM * caps.n_sms ÷ n_work, cells ÷ LAG_MAX_CELLS_PER_LANE, LAG_MIN_LANES)
    lanes = min(lanes, max(1, LAG_MAX_LANES_PER_SM * caps.n_sms ÷ n_work), cells ÷ LAG_MIN_CELLS_PER_LANE)
    lanes = prevpow(2, max(1, lanes))
    L = min(lanes, LAG_UNIT_LANES, LAG_WG)
    return LagPlan{LAG_WG, L}(min(max(1, lanes ÷ L), LAG_MAX_CHUNKS))
end

"""The slice, (slab pair, lag) item and chunk of work unit `unit`."""
@inline function _lag_unit_parts(unit::Int, n_work::Int, n_chunks::Int)
    t = (unit - 1) ÷ n_work + 1
    w = unit - (t - 1) * n_work
    item = (w - 1) ÷ n_chunks + 1
    return t, item, w - (item - 1) * n_chunks
end

"""
    _lag_unit(group, lid, Val(WG), Val(L), n_work, n_chunks, n_units) -> (slot, t, item, lane, n_lanes, ok)

Lane `lid` of workgroup `group` (a device's index type, `Int32` on some) of `WG` lanes: its unit's accumulator
slot in the workgroup, the unit's slice and item, its lane and lane count within the item's chunked unit, and
whether the unit exists.
"""
@inline function _lag_unit(group::Integer, lid::Integer, ::Val{WG}, ::Val{L}, n_work::Int, n_chunks::Int,
                           n_units::Int) where {WG, L}
    l = Int(lid) - 1
    slot = l ÷ L + 1
    unit = (Int(group) - 1) * (WG ÷ L) + slot
    t, item, chunk = _lag_unit_parts(unit, n_work, n_chunks)
    return slot, t, item, (chunk - 1) * L + l % L + 1, n_chunks * L, unit <= n_units
end

"""The lag of an item and what `_lag_visit` knows of it, with the first data columns of its two slabs."""
@inline function _lag_item(sf, s, su, plan, nb, boxes, pair_i, pair_j, item, n_pairs, Nu, ::Val{D}, ::Val{V},
                           ::Val{K}, ::Val{Dg}, ::Val{UB}) where {D, V, K, Dg, UB}
    p, h = SFC._item_pair_lag(Val(UB), Val(Dg), item, 1, n_pairs, boxes)
    I = Int(@inbounds pair_i[p])
    J = Int(@inbounds pair_j[p])
    return h, (I - 1) * Nu, (J - 1) * Nu, SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
end

"""
    _lag_share(second_axis, transport, sf, geometry, data, valid, weights, su, strides, h, r2, half_dim, baseI,
               baseJ, toff, factor, lane, n_lanes, Val(D), Val(V), Val(K), Val(NS), AT) -> (part, n)

Lane `lane` of `n_lanes`: its `NS` contributions to the workgroup's accumulators ([`_lag_part`](@ref)) and
its pair count, over the lag's pairs it sweeps. Data columns are the cells shifted by `toff`; weights are
indexed by cell.
"""
@inline function _lag_share(second_axis, transport, sf, geometry, data, valid, weights, su, strides, h, r2, half_dim,
                            baseI, baseJ, toff, factor, lane, n_lanes, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{NS},
                            ::Type{AT}) where {D, V, K, NS, AT}
    return SFC.with_frames(transport, geometry) do frames
        M = length(frames)
        T = float(promote_type(eltype(data), eltype(frames[1].dir)))
        op = @inline (acc, k, kp) -> begin
            ok = @inbounds(valid[k + toff]) & @inbounds(valid[kp + toff])
            w = SFC._pair_weight(weights, k, kp)
            vals = SFC._lag_values(transport, sf, frames, data, k + toff, kp + toff, r2, Val(D), Val(V), Val(K), T)
            (SFC._accumulate(acc[1], vals, ok, w), SFC._count_pair(acc[2], ok, w))
        end
        init = (ntuple(@inline(_ -> SFC._zero_value(sf, T)), Val(M)), SFC._zero_count(weights))
        totals, n = SFC._lag_fold(op, init, su, strides, h, half_dim, baseI, baseJ, lane, n_lanes)
        (_lag_part(second_axis, totals, factor, Val(NS), AT), n)
    end
end

"""One lane's contribution to the workgroup's `NS` accumulators and its pair count: the frame average of
the lag's value, a row per single-pass invariant (distance histogram), or each frame's share of it in its
own slot (histogram over the angle)."""
@inline function _lag_part(::Nothing, totals::NTuple{M}, factor, ::Val{NS}, ::Type{AT}) where {M, NS, AT}
    v = factor * sum(totals) / M
    return ntuple(@inline(q -> AT(@inbounds v[q])), Val(NS))
end

@inline _lag_part(::SFC.SeparationAngleAxis, totals::NTuple{M}, factor, ::Val{NS}, ::Type{AT}) where {M, NS, AT} =
    ntuple(@inline(m -> m <= M ? AT(factor * @inbounds(totals[m]) / M) : zero(AT)), Val(NS))

"""Accumulators of a workgroup: a row per value of `sf`, or one slot per image of a lag."""
_lag_slots(::Nothing, sf, ::Val{Dg}) where {Dg} = Val(prod(SFC._value_dims(sf); init = 1))
_lag_slots(::SFC.SeparationAngleAxis, sf, ::Val{Dg}) where {Dg} = Val(1 << Dg)

"""Flush accumulator `q` of a workgroup, the lag's bin `b` and slice `t`, into the histogram."""
@inline function _lag_flush!(sums, counts, ::Nothing, sf, q, s, n, b, t, transport, geometry, r2, axis_plan, n_axis)
    _atomic_add!(sums, s, SFC._value_row(sf, q)..., b, t)
    _atomic_add!(counts, n, SFC._value_row(sf, q)..., b, t)
    return nothing
end

@inline function _lag_flush!(sums, counts, second_axis::SFC.SeparationAngleAxis, sf, q, s, n, b, t, transport,
                             geometry, r2, axis_plan, n_axis)
    M, bθ = SFC.with_frames(transport, geometry) do frames
        length(frames), q <= length(frames) ?
            SFH.digitize(SFC.axis_quantity(second_axis, @inbounds(frames[q]).dir, r2), axis_plan) : 0
    end
    if 1 <= bθ <= n_axis
        _atomic_add!(sums, s, b, bθ, t)
        _atomic_add!(counts, convert(eltype(counts), n / M), b, bθ, t)
    end
    return nothing
end

KA.@kernel unsafe_indices = true function _gridded_lag_kernel!(
    sums, counts, sf, s, su, @Const(data), valid, weights, plan, nb::Int, transport, axis_plan, n_axis::Int,
    second_axis, boxes, @Const(pair_i), @Const(pair_j), n_pairs::Int, n_work::Int, n_chunks::Int, n_units::Int,
    strides, Nu::Int, n_grid::Int, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{Dg}, ::Val{UB}, ::Val{WG}, ::Val{L},
    ::Val{NS},
) where {D, V, K, Dg, UB, WG, L, NS}
    acc_s = @localmem eltype(sums) (NS * (WG ÷ L),)
    acc_c = @localmem eltype(counts) (WG ÷ L,)
    lid = @index(Local, Linear)
    k = Int(lid)
    while k <= NS * (WG ÷ L)
        @inbounds acc_s[k] = zero(eltype(sums))
        k += WG
    end
    if lid <= WG ÷ L
        @inbounds acc_c[lid] = zero(eltype(counts))
    end
    @synchronize

    lid = @index(Local, Linear)
    grp = @index(Group, Linear)
    slot, t, item, lane, n_lanes, ok = _lag_unit(grp, lid, Val(WG), Val(L), n_work, n_chunks, n_units)
    if ok
        h, baseI, baseJ, v = _lag_item(sf, s, su, plan, nb, boxes, pair_i, pair_j, item, n_pairs, Nu, Val(D), Val(V),
                                       Val(K), Val(Dg), Val(UB))
        if v !== nothing
            b, r2, factor, geometry, self_reverse = v
            half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
            part, n = _lag_share(second_axis, transport, sf, geometry, data, valid, weights, su, strides, h, r2,
                                 half_dim, baseI, baseJ, (t - 1) * n_grid, factor, lane, n_lanes, Val(D), Val(V),
                                 Val(K), Val(NS), eltype(sums))
            for q in 1:NS
                @atomic acc_s[(slot - 1) * NS + q] += @inbounds part[q]
            end
            @atomic acc_c[slot] += convert(eltype(counts), n)
        end
    end
    @synchronize

    lid = @index(Local, Linear)
    grp = @index(Group, Linear)
    k = Int(lid)
    while k <= NS * (WG ÷ L)
        fslot = (k - 1) ÷ NS + 1
        unit = (Int(grp) - 1) * (WG ÷ L) + fslot
        if unit <= n_units
            ft, fitem, _ = _lag_unit_parts(unit, n_work, n_chunks)
            _, _, _, fv = _lag_item(sf, s, su, plan, nb, boxes, pair_i, pair_j, fitem, n_pairs, Nu, Val(D), Val(V),
                                    Val(K), Val(Dg), Val(UB))
            if fv !== nothing
                fb, fr2, _, fgeometry, _ = fv
                _lag_flush!(sums, counts, second_axis, sf, k - (fslot - 1) * NS, @inbounds(acc_s[k]),
                            @inbounds(acc_c[fslot]), fb, ft, transport, fgeometry, fr2, axis_plan, n_axis)
            end
        end
        k += WG
    end
end

KA.@kernel unsafe_indices = true function _gridded_scatter_kernel!(
    sums, counts, sf, s, su, @Const(data), valid, weights, plan, nb::Int, transport, axis_plan, n_axis::Int,
    boxes, @Const(pair_i), @Const(pair_j), n_pairs::Int, n_work::Int, n_chunks::Int, n_units::Int, strides, Nu::Int,
    n_grid::Int, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{Dg}, ::Val{UB}, ::Val{WG}, ::Val{L},
) where {D, V, K, Dg, UB, WG, L}
    lid = @index(Local, Linear)
    grp = @index(Group, Linear)
    _, t, item, lane, n_lanes, ok = _lag_unit(grp, lid, Val(WG), Val(L), n_work, n_chunks, n_units)
    if ok
        h, baseI, baseJ, v = _lag_item(sf, s, su, plan, nb, boxes, pair_i, pair_j, item, n_pairs, Nu, Val(D), Val(V),
                                       Val(K), Val(Dg), Val(UB))
        if v !== nothing
            b, r2, factor, geometry, self_reverse = v
            half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
            toff = (t - 1) * n_grid
            SFC.with_frames(transport, geometry) do frames
                T = float(promote_type(eltype(data), eltype(frames[1].dir)))
                op = @inline (acc, k, kp) -> SFC._scatter_pair!(_atomic_add!, sums, counts, (t,), transport, sf, frames,
                                                                 data, valid, weights, k + toff, kp + toff, k, r2, b,
                                                                 factor, axis_plan, n_axis, Val(D), Val(V), Val(K), T)
                SFC._lag_fold(op, nothing, su, strides, h, half_dim, baseI, baseJ, lane, n_lanes)
            end
        end
    end
end

SFC.device_lag_sweep!(sums::AbstractArray, counts::AbstractArray, backend::CB.AbstractGPUBackend, sf, s,
                      su::SFC.UniformLagSchedule, data::AbstractArray{<:Any, 3}, valid, weights, plan, nb::Int, r_max,
                      transport, axis, vD::Val, vV::Val, vK::Val) =
    _device_lag_sweep!(_lag_plan, sums, counts, backend, sf, s, su, data, valid, weights, plan, nb, r_max, transport,
                       axis, vD, vV, vK)

"""[`SFC.device_lag_sweep!`](@ref) launched with the plan `choose(caps, n_items, nt, cells)` returns
(see [`_lag_plan`](@ref))."""
function _device_lag_sweep!(
    choose::C, sums, counts, backend::CB.AbstractGPUBackend, sf, s, su::SFC.UniformLagSchedule{Dg},
    data::AbstractArray{<:Any, 3}, valid, weights, plan, nb::Int, r_max, transport, axis, ::Val{D}, ::Val{V}, ::Val{K},
) where {C, Dg, D, V, K}
    ka = backend.backend
    _check_gpu_residency(sums, counts, ka)
    to = x -> KA.adapt(ka, x)
    items = SFC.sweep_items(s, r_max, 1, false)
    isempty(items) && return sums, counts
    UB, boxes, host_off = SFC._lag_boxes(s, su, items, r_max, to)
    n_pairs = length(items)
    n_items = SFC._lag_items(boxes.n_box, host_off, 1, n_pairs)
    n_items == 0 && return sums, counts
    nt = size(data, 3)
    cells = SFC.n_cells(su)
    lp = choose(SFC.gpu_device_caps(ka), n_items, nt, cells)
    axis_plan, n_axis, second_axis = axis === nothing ? (nothing, 0, nothing) : axis
    dvalid = valid isa SFC.AllValid ? valid : to(vec(Matrix{Bool}(valid)))
    args = (sums, counts, sf, to(s), su, reshape(to(data), size(data, 1), :), dvalid, _sf_weights_to_device(ka, weights),
            to(plan), nb, transport, to(axis_plan), n_axis)
    _lag_launch!(lp, ka, args, second_axis, sf, boxes, to(Int32[it[1] for it in items]),
                 to(Int32[it[2] for it in items]), n_pairs, n_items, nt, SFC.grid_strides(su), cells, size(data, 2),
                 Val(D), Val(V), Val(K), Val(Dg), Val(UB), to)
    return sums, counts
end

"""Launch the lag-sweep kernel of `second_axis` with the plan `lp` over `n_items` items of `nt` slices."""
function _lag_launch!(lp::LagPlan{WG, L}, ka, args, second_axis, sf, boxes, pair_i, pair_j, n_pairs::Int, n_items::Int,
                      nt::Int, strides, cells::Int, n_grid::Int, vD, vV, vK, ::Val{Dg}, vUB, to) where {WG, L, Dg}
    n_work = n_items * lp.n_chunks
    n_units = n_work * nt
    tail = (boxes, pair_i, pair_j, n_pairs, n_work, lp.n_chunks, n_units, strides, cells, n_grid, vD, vV, vK, Val(Dg),
            vUB, Val(WG), Val(L))
    ndrange = cld(n_units, WG ÷ L) * WG
    if second_axis isa SFC.InvariantValueAxis
        _gridded_scatter_kernel!(ka, WG)(args..., tail...; ndrange)
    else
        _gridded_lag_kernel!(ka, WG)(args..., to(second_axis), tail..., _lag_slots(second_axis, sf, Val(Dg)); ndrange)
    end
    return nothing
end
