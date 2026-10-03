# Moment sets: what one pair adds to a device histogram. A device kernel's `sf_type` argument is a moment set: a
# pairwise operator (its value), `SFT.SinglePassInvariants()`, `TensorComponents{P, D}()` or a `FieldValue`.

"""Value bin of moment `m`. A tuple plan is indexed by an unrolled chain, so its columns may differ in type."""
@inline _sf_value_bin(plan, x, m) = SFH.digitize(x, plan)
@inline _sf_value_bin(plan::Tuple, x, m) = _sf_tuple_bin(plan, x, m)
@inline _sf_tuple_bin(plan::Tuple{Any}, x, m) = SFH.digitize(x, first(plan))
@inline _sf_tuple_bin(plan::Tuple, x, m) =
    m == 1 ? SFH.digitize(x, first(plan)) : _sf_tuple_bin(Base.tail(plan), x, m - 1)

"""Element `m` of the tuple `t` by a chain of comparisons, so a run-time `m` selects among registers."""
@inline _sf_tuple_at(t::Tuple{Any}, m) = first(t)
@inline _sf_tuple_at(t::Tuple, m) = m == 1 ? first(t) : _sf_tuple_at(Base.tail(t), m - 1)

"""The packed components of a pair's rank-`P` increment tensor over `D` components, in
`SFT.symmetric_indices(Val(D), Val(P))` order."""
struct TensorComponents{P, D} end

"""The value of the operator `sf` for a packed multi-field column of `V` vector fields of width `F`
followed by `K` scalars."""
struct FieldValue{S, F, V, K}
    sf::S
end
FieldValue(sf::S, ::Val{F}, ::Val{V}, ::Val{K}) where {S, F, V, K} = FieldValue{S, F, V, K}(sf)

"""Width, as a `Val`, of the field column a kernel stages per point for the moment set `M`: the field
width of `geom`, or a packed multi-field column."""
@inline _sf_field_width(M, geom) = SFH.field_width(geom)
@inline _sf_field_width(::FieldValue{S, F, V, K}, geom) where {S, F, V, K} = Val(V * F + K)

"""Number of moments one pair adds under the moment set `M`."""
@inline _sf_nmom(::SFT.AbstractPairwiseStructureFunctionType) = 1
@inline _sf_nmom(::SFT.SinglePassInvariants) = SINGLE_PASS_N
@inline _sf_nmom(::TensorComponents{P, D}) where {P, D} = length(SFT.symmetric_indices(Val(D), Val(P)))
@inline _sf_nmom(::FieldValue) = 1

"""
    _sf_pair_moments(M, geom, frame, r, Xi, Xj, Ui, Uj) -> NTuple{_sf_nmom(M)}

The moments the pair of points `Xi`, `Xj` carrying `Ui`, `Uj` adds under the moment set `M`, from its
`frame` and separation `r`.
"""
@inline _sf_pair_moments(sf::SFT.AbstractPairwiseStructureFunctionType, geom, frame, r, Xi, Xj, Ui, Uj) =
    (SFT.pair_value(sf, geom, frame, r, SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)),)
@inline _sf_pair_moments(::SFT.SinglePassInvariants, geom, frame, r, Xi, Xj, Ui, Uj) =
    single_pass_invariants(SFH.increment_invariants(geom, frame, r, SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj))...)
@inline function _sf_pair_moments(::TensorComponents{P, D}, geom, frame, r, Xi, Xj, Ui, Uj) where {P, D}
    du = _tensor_reading(Val(P), geom, frame) * SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
    idx = SFT.symmetric_indices(Val(D), Val(P))
    return ntuple(n -> prod(k -> @inbounds(du[k]), idx[n]), Val(length(idx)))
end
@inline function _sf_pair_moments(fv::FieldValue{S, F, V, K}, geom, frame, r, Xi, Xj, Ui, Uj) where {S, F, V, K}
    inc = field_increment(Val(F), Val(V), Val(K), geom, frame, Ui, Uj)
    v = SFT.pair_value(fv.sf, geom, frame, r, inc)
    return (SFT.is_odd_in_scalars(fv.sf) ? SFH.pair_orientation(geom, frame) * v : v,)
end

"""The same moments for kernels that form the unit separation `rhat` once per pair and sweep a strip of
fields over it."""
@inline _sf_pair_moments_along(M, geom, frame, r, rhat, Xi, Xj, Ui, Uj) =
    _sf_pair_moments(M, geom, frame, r, Xi, Xj, Ui, Uj)
@inline _sf_pair_moments_along(sf::SFT.AbstractPairwiseStructureFunctionType, geom, frame, r, rhat,
                               Xi, Xj, Ui, Uj) =
    (sf(SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj), rhat),)
@inline function _sf_pair_moments_along(::SFT.SinglePassInvariants, geom, frame, r, rhat, Xi, Xj, Ui, Uj)
    dU = SFH.pair_delta(geom, frame, Xi, Xj, Ui, Uj)
    return single_pass_invariants(SFH.fma_dot(dU, rhat), SFH.fma_dot(dU, dU))
end

"""
Moments of `M` a 1-D histogram accumulates per pair. `T2 = S2 - L2` and `L1T2 = S3 - L3` hold for every
pair, and a histogram bin is a sum, so both single-pass differences are recovered exactly at flush by
[`_sf_flush_moment`](@ref) and cost no atomic on any pair.
"""
@inline _sf_accum_moments(M) = ntuple(identity, Val(_sf_nmom(M)))
@inline _sf_accum_moments(::SFT.SinglePassInvariants) = (1, 2, 4, 5)

"""Value of moment `m` of `M` in bin `bin` of a 1-D histogram `ssum` of `NB` bins per moment."""
@inline _sf_flush_moment(M, ssum, NB::Int, m::Int, bin::Int) = @inbounds ssum[(m - 1) * NB + bin]
@inline function _sf_flush_moment(::SFT.SinglePassInvariants, ssum, NB::Int, m::Int, bin::Int)
    @inbounds begin
        m == 3 && return ssum[bin] - ssum[NB + bin]
        m == 6 && return ssum[3 * NB + bin] - ssum[4 * NB + bin]
        return ssum[(m - 1) * NB + bin]
    end
end

"""Workspace kind a 1-D (`Val(1)`) or distance × value (`Val(2)`) sweep of the moment set `M` runs under."""
@inline _sf_workspace_kind(M, ::Val{1}) = :sf1d
@inline _sf_workspace_kind(M, ::Val{2}) = :joint2d
@inline _sf_workspace_kind(::SFT.SinglePassInvariants, ::Val{1}) = :single_pass
@inline _sf_workspace_kind(::SFT.SinglePassInvariants, ::Val{2}) = :single_pass_2d
