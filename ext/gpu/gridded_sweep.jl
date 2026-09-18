# The direct lag sweep on a device: the route a non-polynomial operator takes on a grid, which the
# transform cannot express because it computes polynomial moments.
#
# One work item owns one (slab pair, lag) and reduces over the cells of the uniform directions,
# which is the unit the host loop walks. Every pair is offered the box over all pairs and
# `_lag_visit` rejects the lags a pair cannot reach, so the item index needs no prefix sum of
# per-pair volumes; on a schedule whose pairs have unequal boxes that launches dead items, and
# `uniform_lag_box` is the trait that says which schedules those are.

KA.@kernel unsafe_indices = true function _gridded_lag_kernel!(
    sums, counts, sf, s, su, @Const(data), valid, weights, plan, nb::Int,
    transport, n_box::Int, box_lo, box_len, box_strides, strides, Nu::Int,
    @Const(pair_i), @Const(pair_j), n_items::Int,
    ::Val{D}, ::Val{V}, ::Val{K}, ::Val{Dg},
) where {D, V, K, Dg}
    gid = @index(Global)
    if gid <= n_items
        p = (gid - 1) ÷ n_box + 1
        h = SFC._decode_lag(gid - (p - 1) * n_box, box_lo, box_len, box_strides)
        I = Int(@inbounds pair_i[p])
        J = Int(@inbounds pair_j[p])
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        if v !== nothing
            b, r2, factor, geometry, self_reverse = v
            half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
            baseI = (I - 1) * Nu
            baseJ = (J - 1) * Nu
            totals, n_pairs = SFC.with_frames(transport, geometry) do frames
                SFC._lag_reduce(transport, sf, data, valid, weights, Val(D), Val(V), Val(K),
                                Val(Dg), su, strides, h, frames, r2, half_dim, baseI, baseJ)
            end
            @atomic sums[b] += convert(eltype(sums), factor * sum(totals) / length(totals))
            @atomic counts[b] += convert(eltype(counts), n_pairs)
        end
    end
end

function SFC.device_lag_sweep!(
    sums::AbstractVector, counts::AbstractVector, backend::CB.AbstractGPUBackend,
    sf, s, su::SFC.UniformLagSchedule{Dg}, data, valid, weights, plan, nb::Int, r_max, transport,
    ::Val{D}, ::Val{V}, ::Val{K},
) where {Dg, D, V, K}
    ka = backend.backend
    items = SFC.sweep_items(s, r_max, 1, false)
    isempty(items) && return sums, counts
    pair_i = Int32[it[1] for it in items]
    pair_j = Int32[it[2] for it in items]

    lims = SFC.lag_limits(s, r_max)
    box_lo = ntuple(d -> first(SFC.lag_range(su, d, lims[d])), Val(Dg))
    box_len = ntuple(d -> length(SFC.lag_range(su, d, lims[d])), Val(Dg))
    n_box = prod(box_len)
    box_strides = SFC._lag_strides(box_len)
    n_items = length(items) * n_box

    dsums = KA.adapt(ka, zeros(eltype(sums), length(sums)))
    dcounts = KA.adapt(ka, zeros(eltype(counts), length(counts)))
    kernel = _gridded_lag_kernel!(ka, 64)
    kernel(dsums, dcounts, sf, KA.adapt(ka, s), su, KA.adapt(ka, data),
           KA.adapt(ka, valid), _sf_weights_to_device(ka, weights), KA.adapt(ka, plan), nb,
           transport, n_box, box_lo, box_len, box_strides, SFC.grid_strides(su), SFC.n_cells(su),
           KA.adapt(ka, pair_i), KA.adapt(ka, pair_j), n_items,
           Val(D), Val(V), Val(K), Val(Dg); ndrange = n_items)
    KA.synchronize(ka)

    sums .+= Array(dsums)
    counts .+= Array(dcounts)
    return sums, counts
end
