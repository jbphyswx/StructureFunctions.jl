module StructureFunctionsFFTExt

using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, HelperFunctions as SFH
using SpectralBackends: SpectralBackends as SB

const CB = SFC.CB

"""
    _pad_dims(schedule, r_max) -> NTuple{Du, Int}

Transform length per uniform direction. A periodic direction needs none — the circular correlation
is the sum wanted. A bounded one is padded to at least `n + h_max`, with `h_max` the largest lag any
slab pair reads within `r_max`, so the circular correlation equals the linear one at every lag that
is read; rounded up to a length the transform factors well.
"""
_pad_dims(s::SFC.AbstractSeparableSchedule, r_max) = _pad_dims(SFC.uniform_axes(s), SFC.lag_limits(s, r_max))

@inline function _pad_dims(su::SFC.UniformLagSchedule{Dg}, lims::NTuple{Dg, Int}) where {Dg}
    return ntuple(Val(Dg)) do d
        n = @inbounds su.dims[d]
        @inbounds(su.periodic[d]) && return n
        nextprod((2, 3, 5), n + min(n - 1, lims[d]))
    end
end

"""Embed `src` at the origin of a zero array of size `P`, in `src`'s array family."""
function _embed(src::AbstractArray{FT, Dg}, P::NTuple{Dg, Int}) where {FT, Dg}
    out = fill!(similar(src, P), zero(FT))
    view(out, map(n -> 1:n, size(src))...) .= src
    return out
end

"""
    MonomialTransforms(data, valid, weights, uniform, P, plan, ::Val{Pm})
    MonomialTransforms(data, uniform, P, valid, weights = NoWeights())

Forward transforms of the masked, weighted monomials of one slab of a packed field, built on first use
and kept, in the array family of `data`. A monomial is named by the sorted components it multiplies,
zero-padded to `Pm` entries; the all-zero key is the mask (times the weights), transformed at
construction. `hits` and `misses` count the lookups that found and built a transform.
"""
mutable struct MonomialTransforms{FT, Dg, T, Pm, DT <: AbstractMatrix, VT, WT, PL, AT <: AbstractArray{Complex{FT}, Dg}}
    data::DT
    valid::VT
    weights::WT
    schedule::SFC.UniformLagSchedule{Dg, T}
    P::NTuple{Dg, Int}
    plan::PL
    cache::Dict{NTuple{Pm, Int}, AT}
    hits::Int
    misses::Int
end

function MonomialTransforms(
    data::AbstractMatrix, valid, weights, s::SFC.UniformLagSchedule{Dg, T}, P::NTuple{Dg, Int}, plan, ::Val{Pm},
) where {Dg, T, Pm}
    FT = float(eltype(data))
    mask = ntuple(_ -> 0, Val(Pm))
    F0 = plan * _embed(_held_monomial(data, valid, weights, s.dims, mask, FT), P)
    cache = Dict{NTuple{Pm, Int}, typeof(F0)}(mask => F0)
    return MonomialTransforms{FT, Dg, T, Pm, typeof(data), typeof(valid), typeof(weights), typeof(plan), typeof(F0)}(
        data, valid, weights, s, P, plan, cache, 0, 1,
    )
end

MonomialTransforms(data::AbstractMatrix, s::SFC.UniformLagSchedule, P::Tuple, valid, weights = SFC.NoWeights()) =
    MonomialTransforms(data, valid, weights, s, P, AbstractFFTs.plan_rfft(_zeros_like(data, P)), Val(2))

"""A zero array of shape `P` in `data`'s array family, in its float type."""
_zeros_like(data::AbstractArray, P::Tuple) =
    fill!(similar(parent(data), float(eltype(data)), P), zero(float(eltype(data))))

# The masked, weighted monomial of one slab, shaped as its cells.
_held_monomial(data::AbstractMatrix, valid, weights, dims::NTuple{Dg, Int}, key::NTuple{Pm, Int},
               ::Type{FT}) where {Dg, Pm, FT} =
    reshape(SFC._held_monomial_vector(data, valid, weights, key, FT), dims)

"""The forward transform of the masked, weighted monomial `key`, from the cache or built once."""
function monomial_transform!(mt::MonomialTransforms{FT, Dg, T, Pm}, key::Tuple) where {FT, Dg, T, Pm}
    k = SFC._padded_key(Val(Pm), key)
    cached = get(mt.cache, k, nothing)
    if cached !== nothing
        mt.hits += 1
        return cached
    end
    mt.misses += 1
    F = mt.plan * _embed(_held_monomial(mt.data, mt.valid, mt.weights, mt.schedule.dims, k, FT), mt.P)
    mt.cache[k] = F
    return F
end

"""Pairs with both ends held, at every lag of the padded grid: the two masks' cross-correlation."""
pair_counts(mtI::MonomialTransforms, mtJ::MonomialTransforms) =
    AbstractFFTs.irfft(conj.(monomial_transform!(mtI, ())) .* monomial_transform!(mtJ, ()), mtI.P[1])

pair_counts(mt::MonomialTransforms) = pair_counts(mt, mt)

"""
    moment_component(mtI, mtJ, j::NTuple{P, Int}) -> Array

`Σ_x m(x) m(x+h) Π_k δu[j_k]` at every lag `h` of the padded grid, for the sorted multi-index `j`,
with `x` in the first slab and `x + h` in the second.

Expanding each factor `δu = u(x+h) − u(x)` gives `2^P` cross-correlations of masked monomials, one per
choice of which factors sit at `x + h`; each is a product in the transform domain, and the whole sum
is inverted once.
"""
function moment_component(
    mtI::MonomialTransforms{FT, Dg}, mtJ::MonomialTransforms{FT, Dg}, j::NTuple{P, Int},
) where {FT, Dg, P}
    spec = fill!(similar(monomial_transform!(mtI, ())), zero(Complex{FT}))
    for mask in 0:((1 << P) - 1)
        primed = Tuple(j[k] for k in 1:P if (mask >> (k - 1)) & 1 == 1)
        unprimed = Tuple(j[k] for k in 1:P if (mask >> (k - 1)) & 1 == 0)
        sign = isodd(P - length(primed)) ? -one(FT) : one(FT)
        Fu = monomial_transform!(mtI, unprimed)
        Fp = monomial_transform!(mtJ, primed)
        @. spec += sign * conj(Fu) * Fp
    end
    return AbstractFFTs.irfft(spec, mtI.P[1])
end

moment_component(mt::MonomialTransforms, j::NTuple) = moment_component(mt, mt, j)

# ---------------------------------------------------------------------------------------------------
# The moments one slab pair needs, as columns of one batched inverse transform
# ---------------------------------------------------------------------------------------------------

# Fill the columns for slab pair (I, J) from the two slabs' forward transforms and invert them all at
# once; returns the `(lags, columns)` matrix of raw moments.
function _pair_inverse!(scratch, fwdI::AbstractVector, fwdJ::AbstractVector, columns::AbstractVector, iplan)
    specf = scratch.specf
    n = size(specf, 1)
    @inbounds for (c, terms) in enumerate(columns)
        for lin in 1:n
            specf[lin, c] = 0
        end
        for (sign, ki, kj) in terms
            FI = fwdI[ki]
            FJ = fwdJ[kj]
            @simd for lin in 1:n
                specf[lin, c] += sign * conj(FI[lin]) * FJ[lin]
            end
        end
    end
    LA.mul!(scratch.out, iplan, scratch.spec)
    return scratch.outf
end

# ---------------------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------------------

SFC.transform_engine(sf, data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, vD::Val, vV::Val, vK::Val,
                     valid, weights, tag; to = identity) =
    _transform_prepare(sf, data, s, dist_be, vD, vV, vK, valid, weights, tag; to)

function _transform_prepare(
    sf, data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    valid, weights, tag; to = identity,
) where {D, V, K}
    SFT.is_polynomial_operator(sf) || throw(ArgumentError(
        "$(typeof(sf)) is not a polynomial in the increment, so the transform cannot produce it. " *
        "Omit the spectral backend for the lag sweep, which evaluates any pairwise operator exactly.",
    ))
    SFC._check_grid_field(sf, data, s, Val(D), Val(V), Val(K))
    su = SFC.uniform_axes(s)
    T = eltype(su.spacing)
    r_max = SFC._cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    P = _pad_dims(s, r_max)
    W = V * D + K
    p = SFT.order(sf)
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    dp0, vp0, wp0 = SFC.separable_layout(s, data, valid, w)
    dp = to(dp0)
    vp = vp0 isa SFC.AllValid ? vp0 : to(Vector{Bool}(vp0))
    wp = wp0 isa SFC.NoWeights ? wp0 : to(wp0)
    fwd, keys = _slab_transforms(tag, s, dp, vp, wp, su, P, Val(W), Val(p); to)
    key_index = Dict(k => i for (i, k) in enumerate(keys))
    transport = SFC.lag_transport(s)
    columns = SFC._columns(transport, Val(W), Val(p), key_index)
    # the count column: the two masks' correlation, the weighted pair mass, or a soft-binned kernel mass
    soft = SFC._soft_binned(s)
    masked = !(valid isa SFC.AllValid) || !(wp isa SFC.NoWeights) || soft
    masked && push!(columns, [(1, key_index[keys[1]], key_index[keys[1]])])
    N = SFC._inverse_count(transport, W, p)
    weighted = (wp isa SFC.NoWeights && !soft) ? Val(false) : Val(true)
    return (; s, su, P, r_max, fwd, columns, masked, weighted, transport, vW = Val(W), vP = Val(p), vN = Val(N))
end

# One transform set per slab, every monomial of degree ≤ Pm built up front so tasks only read; the
# first key is the mask. A grid's slabs are transformed by FFT; a scattered schedule's single slab by
# the non-uniform FFT provider.
function _slab_transforms(
    ::SB.AbstractFastFourierTransformSpectralBackend, s::SFC.AbstractSeparableSchedule, dp, vp, wp, su, P,
    ::Val{W}, ::Val{Pm}; to,
) where {W, Pm}
    keys = SFC._monomial_keys(Val(W), Val(Pm))
    fplan = plan_rfft(_zeros_like(dp, P))
    Nu = SFC.n_cells(su)
    fwd = map(1:SFC.n_slabs(s)) do I
        cols = ((I - 1) * Nu + 1):(I * Nu)
        mt = MonomialTransforms(view(dp, :, cols), vp isa SFC.AllValid ? vp : view(vp, cols),
                                wp isa SFC.NoWeights ? wp : view(wp, cols), su, P, fplan, Val(Pm))
        [monomial_transform!(mt, k) for k in keys]
    end
    return fwd, keys
end

function _slab_transforms(
    tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend, s::SFC.ScatteredModesSchedule, dp, vp, wp, su, P,
    ::Val{W}, ::Val{Pm}; to,
) where {W, Pm}
    keys = SFC._monomial_keys(Val(W), Val(Pm))
    return [SFC.nufft_monomial_transforms(tag, s, dp, vp, wp, keys, Val(Pm); to)], keys
end

# The batched inverse plan over `ncols` columns and the per-task scratch it fills, in the transforms'
# array family.
function _inverse_plan(eng, ncols::Int)
    F1 = eng.fwd[1][1]
    CT = eltype(F1)
    FT = real(CT)
    P = eng.P
    Ph = size(F1)
    proto = fill!(similar(F1, Ph..., ncols), zero(CT))
    iplan = AbstractFFTs.plan_irfft(proto, P[1], 1:length(P))
    make_scratch = () -> begin
        spec = fill!(similar(F1, Ph..., ncols), zero(CT))
        out = fill!(similar(F1, FT, P..., ncols), zero(FT))
        (spec = spec, specf = reshape(spec, :, ncols), out = out, outf = reshape(out, :, ncols))
    end
    return iplan, make_scratch
end

function _transform_item!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT}, sf, eng, iplan, item::NTuple{4, Int}, scratch, plan,
    nb, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {OT, CT, D, V, K, W, Po, N}
    I, J = item[1], item[2]
    out = _pair_inverse!(scratch, eng.fwd[I], eng.fwd[J], eng.columns, iplan)
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    @inbounds for H in SFC._pair_lags(s, su, I, J, eng.r_max)
        h = Tuple(H)
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        inv_r = inv(sqrt(r2))
        val = SFC.with_frames(tr, geometry) do frames
            SFC._lag_value(tr, sf, out, idx, scale, frames, inv_r, Val(W), Val(Po), Val(N), Val(V), Val(K))
        end
        sums[b] += OT(factor * val)
        counts[b] += CT(n_pairs)
    end
    return nothing
end

function _transform_item!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, sf, eng, iplan, item::NTuple{4, Int}, scratch, plan,
    nb, axis_edges, na, second_axis, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{W}, ::Val{Po}, ::Val{N},
) where {OT, CT, D, V, K, W, Po, N}
    I, J = item[1], item[2]
    out = _pair_inverse!(scratch, eng.fwd[I], eng.fwd[J], eng.columns, iplan)
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    @inbounds for H in SFC._pair_lags(s, su, I, J, eng.r_max)
        h = Tuple(H)
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        inv_r = inv(sqrt(r2))
        Mo = SFC._lag_moments(tr, out, idx, scale, nothing, nothing, Val(W), Val(Po), Val(N))
        SFC.with_frames(tr, geometry) do frames
            Mi = length(frames)
            Mi > 1 && CT <: Integer && throw(ArgumentError(
                "a lag that half-turns a periodic direction splits each pair between its two " *
                "directions, so a joint histogram over angle needs a floating-point count type; got $CT",
            ))
            for f in frames
                bθ = SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges)
                1 <= bθ <= na || continue
                sums[b, bθ] += OT(factor * SFT.moment_contract(sf, Mo, f.dir * inv_r, Val(V), Val(K)) / Mi)
                counts[b, bθ] += CT(n_pairs / Mi)
            end
        end
    end
    return nothing
end

function SFC.gridded_sweep!(
    sums::AbstractVector, counts::AbstractVector, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {D, V, K}
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    length(sums) == nb && length(counts) == nb || throw(DimensionMismatch(
        "sums and counts must have length $nb; got $(length(sums)) and $(length(counts))",
    ))
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._check_weighted_counts(w, eltype(counts))
    _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                      nothing)
    return sums, counts
end

function SFC.gridded_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    second_axis::SFC.SeparationAngleAxis,
) where {D, V, K}
    SFC._require_directional(s)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    axis_edges = SFC.BinEdges(axis_be)
    na = SFC.n_histogram_bins(axis_edges)
    size(sums) == (nb, na) && size(counts) == (nb, na) || throw(DimensionMismatch(
        "sums and counts must be ($nb, $na); got $(size(sums)) and $(size(counts))",
    ))
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._check_weighted_counts(w, eltype(counts))
    _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                      (axis_edges, na, second_axis))
    return sums, counts
end

# ---------------------------------------------------------------------------------------------------
# The non-uniform FFT route: the same engine on a ScatteredModesSchedule, its forward transforms from
# a NUFFT provider. Counts are a kernel-weighted pair mass and need a floating-point type.
# ---------------------------------------------------------------------------------------------------

_soft_counts(::Type{CT}) where {CT} = CT <: AbstractFloat ? nothing : throw(ArgumentError(
    "the non-uniform FFT route's counts are a kernel-weighted pair mass; pass Float64 counts",
))

function SFC.gridded_sweep!(
    sums::AbstractVector, counts::AbstractVector{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.ScatteredModesSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {CT, D, V, K}
    _soft_counts(CT)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    length(sums) == nb && length(counts) == nb || throw(DimensionMismatch(
        "sums and counts must have length $nb; got $(length(sums)) and $(length(counts))",
    ))
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                      nothing)
    return sums, counts
end

function SFC.gridded_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.ScatteredModesSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
    tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    second_axis::SFC.SeparationAngleAxis,
) where {CT, D, V, K}
    _soft_counts(CT)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    axis_edges = SFC.BinEdges(axis_be)
    na = SFC.n_histogram_bins(axis_edges)
    size(sums) == (nb, na) && size(counts) == (nb, na) || throw(DimensionMismatch(
        "sums and counts must be ($nb, $na); got $(size(sums)) and $(size(counts))",
    ))
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    _transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, Val(D), Val(V), Val(K), valid, w, tag,
                      (axis_edges, na, second_axis))
    return sums, counts
end

_scattered_needs_nufft(tag) = throw(ArgumentError(
    "a ScatteredModesSchedule sums its pairs by non-uniform FFT, which $(nameof(typeof(tag))) does not name; pass " *
    "NonuniformFFTsSpectralBackend() or FINUFFTSpectralBackend() " *
    "with one provider loaded. Auto never selects the soft-binned route.",
))
_nufft_needs_scattered(s) = throw(ArgumentError(
    "a non-uniform FFT tag is for a ScatteredModesSchedule; a $(nameof(typeof(s))) transforms with " *
    "FastFourierTransformSpectralBackend().",
))

for tagT in (:(SB.AbstractFastFourierTransformSpectralBackend), :(SB.AutoSpectralBackend))
    @eval begin
        SFC.gridded_sweep!(::AbstractVector, ::AbstractVector, ::SFT.AbstractPairwiseStructureFunctionType,
                           ::AbstractMatrix, ::SFC.ScatteredModesSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
                           tag::$tagT; kwargs...) where {D, V, K} = _scattered_needs_nufft(tag)
        SFC.gridded_sweep!(::AbstractMatrix, ::AbstractMatrix, ::SFT.AbstractPairwiseStructureFunctionType,
                           ::AbstractMatrix, ::SFC.ScatteredModesSchedule, dist_be, axis_be, ::Val{D}, ::Val{V},
                           ::Val{K}, tag::$tagT; kwargs...) where {D, V, K} = _scattered_needs_nufft(tag)
    end
end

SFC.gridded_sweep!(::AbstractVector, ::AbstractVector, ::SFT.AbstractPairwiseStructureFunctionType, ::AbstractMatrix,
                   s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
                   ::SB.AbstractNonUniformFastFourierTransformSpectralBackend; kwargs...) where {D, V, K} =
    _nufft_needs_scattered(s)
SFC.gridded_sweep!(::AbstractMatrix, ::AbstractMatrix, ::SFT.AbstractPairwiseStructureFunctionType, ::AbstractMatrix,
                   s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
                   ::SB.AbstractNonUniformFastFourierTransformSpectralBackend; kwargs...) where {D, V, K} =
    _nufft_needs_scattered(s)

# ---------------------------------------------------------------------------------------------------
# Tensors on grids: the same engine, each lag's symmetric moment store binned instead of contracted,
# and the dense tensor assembled once at the end.
# ---------------------------------------------------------------------------------------------------

# The transform a tensor sweep runs with: `Auto` is the FFT on a grid, and a tag must match its schedule.
_tensor_tag(tag::SB.AbstractFastFourierTransformSpectralBackend, ::SFC.AbstractSeparableSchedule) = tag
_tensor_tag(tag::SB.AbstractFastFourierTransformSpectralBackend, ::SFC.ScatteredModesSchedule) = _scattered_needs_nufft(tag)
_tensor_tag(tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend, ::SFC.ScatteredModesSchedule) = tag
_tensor_tag(::SB.AbstractNonUniformFastFourierTransformSpectralBackend, s::SFC.AbstractSeparableSchedule) = _nufft_needs_scattered(s)
_tensor_tag(::SB.AbstractAutoSpectralBackend, ::SFC.AbstractSeparableSchedule) = SB.FastFourierTransformSpectralBackend()
_tensor_tag(tag::SB.AbstractAutoSpectralBackend, ::SFC.ScatteredModesSchedule) = _scattered_needs_nufft(tag)

const TensorTag = Union{SB.AbstractFastFourierTransformSpectralBackend, SB.AbstractNonUniformFastFourierTransformSpectralBackend,
                        SB.AbstractAutoSpectralBackend}

function _tensor_engine(order::Val{P}, data, s, dist_be, ::Val{D}, valid, weights, counts, tag) where {P, D}
    tag = _tensor_tag(tag, s)
    w = SFC._pair_weights(weights, size(data, 2), float(eltype(data)))
    SFC._check_weighted_counts(w, eltype(counts))
    SFC._soft_binned(s) && _soft_counts(eltype(counts))
    eng = _transform_prepare(SFT.MomentTensorOperator{P}(), data, s, dist_be, Val(D), Val(1), Val(0), valid, w, tag)
    return eng, binomial(D + P - 1, P)
end

function SFC.gridded_tensor_sweep!(
    sums::AbstractArray{OT}, counts::AbstractVector{CT}, order::Val{P}, data::AbstractMatrix,
    s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, tag::TensorTag;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {OT, CT, P, D}
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    size(sums) == (ntuple(_ -> D, P)..., nb) && length(counts) == nb || throw(DimensionMismatch(
        "sums must be $((ntuple(_ -> D, P)..., nb)) and counts of length $nb; got $(size(sums)) and $(length(counts))",
    ))
    eng, Ns = _tensor_engine(order, data, s, dist_be, Val(D), valid, weights, counts, tag)
    iplan, make_scratch = _inverse_plan(eng, length(eng.columns))
    items = SFC.sweep_items(s, eng.r_max, SFC.sweep_tasks(backend), false)
    sf = SFT.MomentTensorOperator{P}()
    sym = zeros(OT, Ns, nb)
    body! = (ls, lc, it, scratch) -> _transform_tensor_item!(ls, lc, sf, eng, iplan, it, scratch, plan, nb, nothing,
                                                             Val(D), eng.vW, eng.vP, eng.vN, Val(Ns))
    SFC.sweep_reduce!(sym, counts, backend, items, make_scratch, body!)
    SFC._expand_symmetric!(sums, sym, Val(D), Val(P))
    return sums, counts
end

function SFC.gridded_tensor_sweep!(
    sums::AbstractArray{OT}, counts::AbstractMatrix{CT}, order::Val{P}, data::AbstractMatrix,
    s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, tag::TensorTag;
    valid = SFC.AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    second_axis::SFC.SeparationAngleAxis,
) where {OT, CT, P, D}
    SFC._require_directional(s)
    plan = SFC.squared_digitize_plan(dist_be)
    nb = SFC.n_histogram_bins(plan)
    axis_edges = SFC.BinEdges(axis_be)
    na = SFC.n_histogram_bins(axis_edges)
    size(sums) == (ntuple(_ -> D, P)..., nb, na) && size(counts) == (nb, na) || throw(DimensionMismatch(
        "sums must be $((ntuple(_ -> D, P)..., nb, na)) and counts ($nb, $na); got $(size(sums)) and $(size(counts))",
    ))
    eng, Ns = _tensor_engine(order, data, s, dist_be, Val(D), valid, weights, counts, tag)
    iplan, make_scratch = _inverse_plan(eng, length(eng.columns))
    items = SFC.sweep_items(s, eng.r_max, SFC.sweep_tasks(backend), false)
    sf = SFT.MomentTensorOperator{P}()
    sym = zeros(OT, Ns, nb, na)
    body! = (ls, lc, it, scratch) -> _transform_tensor_item!(ls, lc, sf, eng, iplan, it, scratch, plan, nb,
                                                             (axis_edges, na, second_axis), Val(D), eng.vW, eng.vP,
                                                             eng.vN, Val(Ns))
    SFC.sweep_reduce!(sym, counts, backend, items, make_scratch, body!)
    SFC._expand_symmetric!(sums, sym, Val(D), Val(P))
    return sums, counts
end

function _transform_tensor_item!(
    sym::AbstractMatrix{OT}, counts::AbstractVector{CT}, sf, eng, iplan, item::NTuple{4, Int}, scratch, plan, nb,
    ::Nothing, ::Val{D}, ::Val{W}, ::Val{Po}, ::Val{N}, ::Val{Ns},
) where {OT, CT, D, W, Po, N, Ns}
    I, J = item[1], item[2]
    out = _pair_inverse!(scratch, eng.fwd[I], eng.fwd[J], eng.columns, iplan)
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    @inbounds for H in SFC._pair_lags(s, su, I, J, eng.r_max)
        h = Tuple(H)
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(1), Val(0))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        m = SFC._lag_tensor(tr, out, idx, scale, geometry, Val(W), Val(Po), Val(N))
        f = SFC._tensor_factor(tr, factor)
        for n in 1:Ns
            sym[n, b] += OT(f * m[n])
        end
        counts[b] += CT(n_pairs)
    end
    return nothing
end

function _transform_tensor_item!(
    sym::AbstractArray{OT, 3}, counts::AbstractMatrix{CT}, sf, eng, iplan, item::NTuple{4, Int}, scratch, plan, nb,
    axis::Tuple, ::Val{D}, ::Val{W}, ::Val{Po}, ::Val{N}, ::Val{Ns},
) where {OT, CT, D, W, Po, N, Ns}
    axis_edges, na, second_axis = axis
    I, J = item[1], item[2]
    out = _pair_inverse!(scratch, eng.fwd[I], eng.fwd[J], eng.columns, iplan)
    s, su, P = eng.s, eng.su, eng.P
    T = eltype(su.spacing)
    strides = SFC._lag_strides(P)
    tr = eng.transport
    @inbounds for H in SFC._pair_lags(s, su, I, J, eng.r_max)
        h = Tuple(H)
        v = SFC._lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(1), Val(0))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        idx = SFC._lag_index(h, P, strides)
        n_pairs = SFC._named_pairs(eng.weighted, eng.masked, out, idx, size(out, 2), su, h, self_reverse)
        scale = self_reverse ? T(0.5) : one(T)
        m = SFC._lag_tensor(tr, out, idx, scale, geometry, Val(W), Val(Po), Val(N))
        fac = SFC._tensor_factor(tr, factor)
        SFC.with_frames(tr, geometry) do frames
            Mi = length(frames)
            Mi > 1 && CT <: Integer && throw(ArgumentError(
                "a lag that half-turns a periodic direction splits each pair between its two " *
                "directions, so a joint histogram over angle needs a floating-point count type; got $CT",
            ))
            for f in frames
                bθ = SFH.digitize(SFC.axis_quantity(second_axis, f.dir, r2), axis_edges)
                1 <= bθ <= na || continue
                for n in 1:Ns
                    sym[n, b, bθ] += OT(fac * m[n] / Mi)
                end
                counts[b, bθ] += CT(n_pairs / Mi)
            end
        end
    end
    return nothing
end

# The CPU engine: one inverse per slab pair, the lags of each read on the host.
function _transform_sweep!(
    sums, counts, backend::CB.AbstractExecutionBackend, sf, data, s, dist_be, plan, nb, ::Val{D}, ::Val{V},
    ::Val{K}, valid, weights, tag, axis,
) where {D, V, K}
    eng = _transform_prepare(sf, data, s, dist_be, Val(D), Val(V), Val(K), valid, weights, tag)
    iplan, make_scratch = _inverse_plan(eng, length(eng.columns))
    items = SFC.sweep_items(s, eng.r_max, SFC.sweep_tasks(backend), false)
    body! = _item_body(sf, eng, iplan, plan, nb, axis, Val(D), Val(V), Val(K), eng.vW, eng.vP, eng.vN)
    SFC.sweep_reduce!(sums, counts, backend, items, make_scratch, body!)
    return nothing
end

_item_body(sf, eng, iplan, plan, nb, ::Nothing, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, scratch) -> _transform_item!(ls, lc, sf, eng, iplan, it, scratch, plan, nb, vD, vV, vK, vW, vP, vN)

_item_body(sf, eng, iplan, plan, nb, axis::Tuple, vD, vV, vK, vW, vP, vN) =
    (ls, lc, it, scratch) -> _transform_item!(ls, lc, sf, eng, iplan, it, scratch, plan, nb, axis[1], axis[2],
                                              axis[3], vD, vV, vK, vW, vP, vN)

# A device runs the engine through the KernelAbstractions extension.
_transform_sweep!(
    sums, counts, backend::CB.AbstractGPUBackend, sf, data, s, dist_be, plan, nb, vD::Val, vV::Val, vK::Val,
    valid, weights, tag, axis,
) = SFC.device_transform_sweep!(sums, counts, backend, sf, data, s, dist_be, plan, nb, vD, vV, vK, valid, weights,
                                tag, axis)

# `Auto` is answered here because only this extension knows what a transform would cost: the sweep
# visits every lag of every slab pair over the slab's cells, the transform pays one forward transform
# per monomial per slab and one inverse per raw moment per slab pair, however few lags are wanted.
# Dispatching on the concrete tag out-specialises the core method.
_auto_transform(sf, s, dist_be, W::Int, valid, weights, ::CB.AbstractExecutionBackend) =
    _prefers_transform(sf, s, dist_be, W, valid, weights)

# A device runs only the transform.
_auto_transform(sf, s, dist_be, W::Int, valid, weights, ::CB.AbstractGPUBackend) = true

function _prefers_transform(sf, s::SFC.AbstractSeparableSchedule, dist_be, W::Int, valid, weights)
    SFT.is_polynomial_operator(sf) || return false
    su = SFC.uniform_axes(s)
    Dg = SFC.grid_dimension(su)
    T = eltype(su.spacing)
    r_max = SFC._cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    lims = SFC.lag_limits(s, r_max)
    n_lags = prod(ntuple(d -> length(SFC.lag_range(su, d, lims[d])), Dg))
    n = prod(_pad_dims(s, r_max))
    p = SFT.order(sf)
    n_pairs = length(SFC.sweep_items(s, r_max, 1, false))
    count_column = (valid isa SFC.AllValid && weights === nothing) ? 0 : 1
    per_pair = SFC._inverse_count(SFC.lag_transport(s), W, p) + count_column
    transforms = SFC.n_slabs(s) * binomial(W + p, p) + n_pairs * per_pair
    return transforms * n * log2(max(2, n)) < n_pairs * n_lags * SFC.n_cells(su)
end

function SFC.gridded_sweep!(
    sums::AbstractVector, counts::AbstractVector, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K},
    ::SB.AutoSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {D, V, K}
    return _auto_transform(sf, s, dist_be, V * D + K, valid, weights, backend) ?
        SFC.gridded_sweep!(sums, counts, sf, data, s, dist_be, Val(D), Val(V), Val(K),
                           SB.FastFourierTransformSpectralBackend(); valid, weights, backend) :
        SFC.gridded_lag_sweep!(sums, counts, sf, data, s, dist_be, Val(D), Val(V), Val(K); valid, weights, backend)
end

function SFC.gridded_sweep!(
    sums::AbstractMatrix, counts::AbstractMatrix, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::SFC.AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K},
    ::SB.AutoSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    second_axis::SFC.SeparationAngleAxis,
) where {D, V, K}
    return _auto_transform(sf, s, dist_be, V * D + K, valid, weights, backend) ?
        SFC.gridded_sweep!(sums, counts, sf, data, s, dist_be, axis_be, Val(D), Val(V), Val(K),
                           SB.FastFourierTransformSpectralBackend(); valid, weights, backend, second_axis) :
        SFC.gridded_lag_sweep!(sums, counts, sf, data, s, dist_be, axis_be, Val(D), Val(V), Val(K);
                               valid, weights, backend, second_axis)
end

# ---------------------------------------------------------------------------------------------------
# Spectra from the lag-space structure function
# ---------------------------------------------------------------------------------------------------

"""
    _trace_lags(mt, ::Val{D}) -> (trace, counts)

`Σ_pairs ‖δu‖²` and the pair count at every lag of the padded grid, both weighted when the transforms
are. Only the diagonal moment components are formed, so the monomials transformed are the mask, each
component and each square; with every cell held and no weights the count is the box overlap and no
mask transform is taken.
"""
function _trace_lags(mt::MonomialTransforms{FT, Dg}, ::Val{D}) where {FT, Dg, D}
    trace = moment_component(mt, (1, 1))
    for d in 2:D
        trace .+= moment_component(mt, (d, d))
    end
    complete = mt.valid isa SFC.AllValid && mt.weights isa SFC.NoWeights
    pc = complete ? nothing : pair_counts(mt)
    counts = _count_array(trace, mt.weights)
    P = mt.P
    @inbounds for I in CartesianIndices(trace)
        h = _lag_of_index(Tuple(I), P)
        counts[I] = pc === nothing ? _padded_pair_count(mt.schedule, h) : _count_value(pc[I], mt.weights)
    end
    return trace, counts
end

_count_array(trace, ::SFC.NoWeights) = similar(trace, Int)
_count_array(trace, ::AbstractVector) = similar(trace)
_count_value(c, ::SFC.NoWeights) = round(Int, c)
_count_value(c, ::AbstractVector) = c

# The signed lag a position of the padded transform holds.
@inline _lag_of_index(I::NTuple{Dg, Int}, P::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> (m = I[d] - 1; m > P[d] ÷ 2 ? m - P[d] : m), Val(Dg))

# Pairs a lag of the padded grid names on a complete field: zero beyond the cells of a bounded direction.
@inline function _padded_pair_count(s::SFC.UniformLagSchedule{Dg}, h::NTuple{Dg, Int}) where {Dg}
    n = 1
    @inbounds for d in 1:Dg
        n *= s.periodic[d] ? s.dims[d] : max(0, s.dims[d] - abs(h[d]))
    end
    return n
end

"""Weighted variance of the held cells about their weighted mean, summed over the field's components."""
function _held_variance(data::AbstractMatrix, valid, weights)
    W, n = size(data)
    T = float(eltype(data))
    acc = zero(T)
    for c in 1:W
        m = zero(T)
        k = zero(T)
        @inbounds for i in 1:n
            valid[i] || continue
            w = _cell_weight(weights, i)
            m += w * data[c, i]
            k += w
        end
        k > 0 || throw(ArgumentError("no held cell carries weight"))
        m /= k
        @inbounds for i in 1:n
            valid[i] && (acc += _cell_weight(weights, i) * (data[c, i] - m)^2 / k)
        end
    end
    return acc
end

@inline _cell_weight(::SFC.NoWeights, i) = true
@inline _cell_weight(w::AbstractVector, i) = @inbounds w[i]

function SFC.gridded_spectrum(
    u::AbstractArray, s::SFC.UniformLagSchedule{Dg, T}, ::Val{D},
    ::SB.AbstractFastFourierTransformSpectralBackend; valid = SFC.AllValid(), weights = nothing,
    taper::SFC.AbstractTaper = SFC.NoTaper(),
    missing_lags::SFC.AbstractMissingLagPolicy = SFC.RefuseMissingLags(),
) where {Dg, T, D}
    size(u, 1) == D || throw(DimensionMismatch("field has $(size(u, 1)) components, declared $D"))
    data = reshape(u, D, :)
    SFC._check_grid_field(SFT.S2SFType(), data, s, Val(D), Val(1), Val(0))
    w = SFC._pair_weights(weights, size(data, 2), T)
    # every lag of a bounded direction, |h| < n, transformed linearly: P ≥ 2n − 1; a periodic one needs
    # no padding
    P = _pad_dims(s, T(Inf))
    mt = MonomialTransforms(data, s, P, valid, w)
    trace, counts = _trace_lags(mt, Val(D))
    σ2 = _held_variance(data, valid, w)
    r_max = sqrt(sum(d -> (s.periodic[d] ? (s.dims[d] ÷ 2) * s.spacing[d] : (s.dims[d] - 1) * s.spacing[d])^2,
                     1:Dg))

    # C(h) = C(0) − D(h)/2 with C(0) the held cells' variance, weighted by the taper; a lag no held pair
    # names is structurally absent beyond a bounded direction's cells and a missing datum within them
    cov = similar(trace)
    @inbounds for I in CartesianIndices(trace)
        h = _lag_of_index(Tuple(I), P)
        n = counts[I]
        if n > 0
            r = sqrt(sum(d -> (h[d] * s.spacing[d])^2, 1:Dg))
            cov[I] = SFC.taper_weight(taper, r, r_max) * (σ2 - trace[I] / (2n))
        elseif all(d -> s.periodic[d] || abs(h[d]) < s.dims[d], 1:Dg)
            cov[I] = _missing_lag_value(missing_lags, h, T)
        else
            cov[I] = zero(T)
        end
    end

    density = real.(AbstractFFTs.fft(cov)) .* (prod(abs, s.spacing) / T(2π)^Dg)
    wavenumbers = ntuple(d -> T(2π) .* AbstractFFTs.fftfreq(P[d], 1 / abs(s.spacing[d])), Val(Dg))
    return wavenumbers, density
end

_missing_lag_value(::SFC.RefuseMissingLags, h, ::Type{T}) where {T} = throw(ArgumentError(
    "lag $h is named by no pair of held cells, so its structure function is undefined; pass " *
    "`missing_lags = ZeroDeviationAtMissingLags()` to set its covariance to zero, or fill the field.",
))
_missing_lag_value(::SFC.ZeroDeviationAtMissingLags, h, ::Type{T}) where {T} = zero(T)

end # module
