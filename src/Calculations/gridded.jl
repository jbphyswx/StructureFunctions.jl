# Gridded enumeration. On a uniform grid the geometry is shift-invariant, so every pair sharing a lag
# vector shares its separation, its direction and its distance bin; those are computed once per lag
# and the field difference becomes a shifted-array reduction with one histogram write per lag.

"""
    AllValid()

Every cell holds a usable datum, as an indexable value rather than an absent one.

Distinct from a grid's own mask, which says which cells *exist*: a cell can exist and still hold
nothing, and it is the field that decides. [`field_validity`](@ref) combines the two. Spelling the
complete-field case as a value that answers `true` keeps one kernel — the test folds away, so a
complete field pays nothing, and no separate masked kernel can drift from the unmasked one.
"""
struct AllValid end

@inline Base.getindex(::AllValid, ::Integer) = true

"""
    field_validity(u, ::Val{D}[, cell_mask]) -> BitVector or AllValid

Which cells of `u` hold a usable datum: every component finite, and — where a `cell_mask` is given,
as a grid carries one — the cell marked as existing.

Returns [`AllValid`](@ref) when nothing is excluded, so a complete field costs nothing downstream.
"""
function field_validity(u::AbstractArray, ::Val{D}, cell_mask = nothing) where {D}
    n = length(u) ÷ D
    uf = reshape(u, D, n)
    v = trues(n)
    any_invalid = false
    @inbounds for k in 1:n
        ok = true
        for c in 1:D
            ok &= isfinite(uf[c, k])
        end
        cell_mask === nothing || (ok &= cell_mask[k])
        v[k] = ok
        any_invalid |= !ok
    end
    return any_invalid ? v : AllValid()
end

"""
    increment_tensor_value(sf, T, r̂) -> value

The operator's summed value over a set of pairs, from that set's second-order increment tensor
`T[a,b] = Σ δu_a δu_b` and their shared separation direction.

Every operator that is a quadratic form in `δu` is a contraction of `T`, so a path that produces `T`
for many pairs at once — a correlation-theorem sweep, where `T` for every lag costs `O(M log M)` —
evaluates the operator without ever forming a pair. Anything not a quadratic form has no such
contraction and says so.
"""
increment_tensor_value(sf::SFT.AbstractStructureFunctionType, T, r_hat) = throw(ArgumentError(
    "$(typeof(sf)) is not a quadratic form in δu, so it is not a contraction of the second-order " *
    "increment tensor. Use the lag sweep, which evaluates any pairwise operator exactly.",
))

@inline increment_tensor_value(::SFT.SecondOrderStructureFunctionType, T, r_hat) = LA.tr(T)

@inline increment_tensor_value(::SFT.ProjectedStructureFunctionType{2, 0}, T, r_hat) =
    LA.dot(r_hat, T, r_hat)

@inline increment_tensor_value(::SFT.ProjectedStructureFunctionType{0, 2}, T, r_hat) =
    LA.tr(T) - LA.dot(r_hat, T, r_hat)

@inline function increment_tensor_value(
    ::SFT.TransverseComponentSecondOrderStructureFunctionType, T, r_hat,
)
    D = length(r_hat)
    D > 1 || throw(ArgumentError(
        "T2ComponentSF averages over the transverse directions, of which there are none at D = 1",
    ))
    return (LA.tr(T) - LA.dot(r_hat, T, r_hat)) / (D - 1)
end

"""Whether [`increment_tensor_value`](@ref) has a contraction for this operator."""
@inline is_quadratic_operator(sf::SFT.AbstractStructureFunctionType) = false
@inline is_quadratic_operator(::SFT.SecondOrderStructureFunctionType) = true
@inline is_quadratic_operator(::SFT.ProjectedStructureFunctionType{2, 0}) = true
@inline is_quadratic_operator(::SFT.ProjectedStructureFunctionType{0, 2}) = true
@inline is_quadratic_operator(::SFT.TransverseComponentSecondOrderStructureFunctionType) = true

"""
    gridded_sweep!(sums, counts, sf, u, schedule, distance_bins, ::Val{D}, spectral_backend)

Accumulate the histogram of every pair `schedule` names, by the algorithm `spectral_backend` selects.

Which algorithm sums the pairs is an axis of its own, orthogonal to which hardware runs it: the lag
sweep visits each lag and is exact for any pairwise operator, while a transform produces the
second-order increment tensor for every lag at once and so serves the operators that are quadratic
forms in `δu`. `AutoSpectralBackend` picks whichever does less work, and resolves to the lag sweep
unless a transform is loaded — an extension supplies the transform, and answers `Auto` with a cost
comparison because only it knows what a transform would cost.
"""
function gridded_sweep!(sums, counts, sf, u, schedule, distance_bins, ::Val{D},
                        spectral_backend; valid = AllValid()) where {D}
    throw(ArgumentError(
        "no method sums a gridded calculation with $(typeof(spectral_backend)). Load the package " *
        "that supplies it — `using SpectralBackends` for the direct sum, and additionally an " *
        "AbstractFFTs implementation (`using FFTW` on CPU) for a transform. Omitting the argument " *
        "sweeps the lags, which needs neither.",
    ))
end

"""
    UniformLagSchedule(dims, spacing, periodic)

Lag enumeration for a uniform rectilinear grid of `dims` cells, with constant `spacing` along each
direction, each direction wrapping or not per `periodic`.

A lag is an integer offset vector naming every pair `(cell, cell + lag)`, so the separation, its
direction and its distance bin are properties of the lag alone. Directions are positional and match
the trailing axes of the field the schedule is swept over.

`spacing` is signed, as a descending axis reports it; only its magnitude enters a separation.
"""
struct UniformLagSchedule{Dg, T}
    dims::NTuple{Dg, Int}
    spacing::NTuple{Dg, T}
    periodic::NTuple{Dg, Bool}
end

"""Cells the schedule covers."""
@inline n_grid_cells(s::UniformLagSchedule) = prod(s.dims)

"""Flat-index stride of each direction in a field stored `(component, cells...)`."""
@inline function grid_strides(s::UniformLagSchedule{Dg}) where {Dg}
    return ntuple(Val(Dg)) do d
        p = 1
        for k in 1:(d - 1)
            p *= @inbounds s.dims[k]
        end
        p
    end
end

"""
    lag_range(schedule, d, r_max)

Representative lags along direction `d`, one per distinct separation, bounded by `r_max`.

A periodic direction of `n` cells has exactly `n` distinct offsets, and its representatives are
taken in `-((n-1)÷2):(n÷2)`, so each is the minimum image and its magnitude is the true separation.
A bounded direction reaches `±(n-1)`.
"""
@inline function lag_range(s::UniformLagSchedule, d::Integer, r_max)
    n = @inbounds s.dims[d]
    step = abs(@inbounds s.spacing[d])
    lim = (isfinite(r_max) && step > 0) ? floor(Int, r_max / step) : typemax(Int)
    @inbounds if s.periodic[d]
        return (-min((n - 1) ÷ 2, lim)):min(n ÷ 2, lim)
    end
    m = min(n - 1, lim)
    return (-m):m
end

# The representative of `-h`. A periodic direction of even length has one half-turn offset that is
# its own reverse, and a lag built only from those names each pair twice.
@inline _lag_negate(s::UniformLagSchedule, d::Integer, h::Integer) =
    @inbounds (s.periodic[d] && iseven(s.dims[d]) && h == s.dims[d] ÷ 2) ? h : -h

"""
    _lag_segments(schedule, d, h, half) -> ((range, offset), (range, offset))

Where direction `d` may sit for lag `h`, split so that each part has a **constant** flat offset: the
part whose partner stays in range, and the part that wraps. The second range is empty unless the
direction is periodic and the lag is nonzero.

`half` restricts the direction to its first half, which is what makes a half-turn lag name each pair
once: that lag is its own reverse, so `x ↦ x + h` pairs the halves and one of them is a complete set
of representatives. Nothing wraps out of the first half, so it is a single segment.
"""
@inline function _lag_segments(s::UniformLagSchedule, d::Integer, h::Integer, half::Bool)
    n = @inbounds s.dims[d]
    per = @inbounds s.periodic[d]
    half && return ((1:(n ÷ 2), h), (1:0, 0))
    if h >= 0
        return (per && h > 0) ?
            ((1:(n - h), h), (((n - h + 1):n), h - n)) :
            ((1:(n - h), h), (1:0, 0))
    end
    return per ?
        (((1 - h):n, h), (1:(-h), h + n)) :
        (((1 - h):n, h), (1:0, 0))
end

"""
    _ambiguous_dirs(schedule, h) -> NTuple{Dg, Bool}

Directions along which lag `h` reaches exactly half the period, so `+h_d` and `-h_d` are the same
offset and name two minimal paths of equal length.

Only a periodic direction with an even cell count has one, and only at that single offset.
"""
@inline _ambiguous_dirs(s::UniformLagSchedule{Dg}, h::NTuple{Dg, Int}) where {Dg} =
    ntuple(d -> @inbounds(s.periodic[d] && iseven(s.dims[d]) && abs(h[d]) == s.dims[d] ÷ 2), Val(Dg))

"""
    _lag_images(dx, ambiguous, ::Val{K}) -> NTuple{2^K, typeof(dx)}

The `2^K` displacements of equal length that join the two ends of a lag: every choice of sign along
the `K` directions the lag half-turns. `K = 0` gives `dx` alone.
"""
@inline function _lag_images(dx::SA.SVector{D, T}, amb::NTuple{Dg, Bool}, ::Val{K}) where {D, T, Dg, K}
    # Which sign bit of the image index belongs to each direction; 0 for the unambiguous ones.
    bit_of = ntuple(Val(Dg)) do d
        b = 0
        for k in 1:d
            @inbounds amb[k] && (b += 1)
        end
        @inbounds amb[d] ? b : 0
    end
    return ntuple(Val(1 << K)) do m
        SA.SVector{D, T}(ntuple(Val(D)) do d
            flip = d <= Dg && @inbounds(bit_of[d]) != 0 &&
                   ((m - 1) >> (@inbounds(bit_of[d]) - 1)) & 1 == 1
            flip ? -(@inbounds dx[d]) : @inbounds dx[d]
        end)
    end
end

"""
    _lag_reduce_images(sf, u_flat, ::Val{D}, ::Val{Dg}, schedule, strides, h, dx, r2, half_dim, amb)

[`_lag_reduce`](@ref) with the lag's equal-length displacements built first. The half-turn count is a
value, so it is resolved to a type parameter here — the counts a grid can produce are spelled out,
and any wider one still resolves through the same expression.
"""
@inline function _lag_reduce_images(
    sf, u_flat, valid, ::Val{D}, ::Val{Dg}, s, strides, h, dx, r2, half_dim::Int,
    amb::NTuple{Dg, Bool},
) where {D, Dg}
    go(vk) = _lag_reduce(sf, u_flat, valid, Val(D), Val(Dg), s, strides, h,
                         _lag_images(dx, amb, vk), r2, half_dim)
    K = count(amb)
    K == 0 && return go(Val(0))
    K == 1 && return go(Val(1))
    K == 2 && return go(Val(2))
    K == 3 && return go(Val(3))
    return go(Val(K))
end

"""
    _lag_reduce(sf, u_flat, ::Val{D}, ::Val{Dg}, schedule, strides, h, images, r2, half_dim)
        -> (sum, n_pairs)

Sum the operator over every pair one lag names, with the lag's equal-length displacements `images`
and squared separation `r2` already fixed.

The lag's cells are swept as boxes of constant flat offset, so the innermost loop is unit-stride and
the whole lag contributes one number. A wrapped direction contributes a second box rather than a
per-cell branch. Each pair's value is the mean over `images`, which is one term — the displacement
itself — unless the lag half-turns a periodic direction.
"""
@inline function _lag_reduce(
    sf::SFT.AbstractPairwiseStructureFunctionType, u_flat::AbstractMatrix{UT}, valid,
    ::Val{D}, ::Val{Dg}, s::UniformLagSchedule, strides::NTuple{Dg, Int},
    h::NTuple{Dg, Int}, images::NTuple{M, SA.SVector{D, T}}, r2, half_dim::Int,
) where {UT, D, Dg, M, T}
    total = zero(SFT._sf_raw(sf, SA.SVector{D, UT}(ntuple(_ -> zero(UT), Val(D))), images[1], r2))
    n_pairs = 0
    inv_m = inv(M)
    @inbounds for combo in 0:((1 << Dg) - 1)
        segs = ntuple(Val(Dg)) do d
            _lag_segments(s, d, h[d], d == half_dim)[1 + ((combo >> (d - 1)) & 1)]
        end
        ranges = map(first, segs)
        any(isempty, ranges) && continue
        off = 0
        for d in 1:Dg
            off += segs[d][2] * strides[d]
        end
        r1 = ranges[1]
        outer = CartesianIndices(Base.tail(ranges))
        for J in outer
            base = 0
            for d in 2:Dg
                base += (J[d - 1] - 1) * strides[d]
            end
            @simd for i1 in r1
                k = base + i1
                kp = k + off
                ok = valid[k] & valid[kp]
                δu = SA.SVector{D, UT}(ntuple(c -> u_flat[c, kp] - u_flat[c, k], Val(D)))
                v = zero(total)
                for m in 1:M                          # `M` is a type parameter, so this unrolls
                    v += SFT._sf_raw(sf, δu, images[m], r2)
                end
                # selected, never multiplied by `ok`: an empty cell may hold NaN, and NaN * 0 is NaN
                total += ok ? v * inv_m : zero(total)
                n_pairs += ok
            end
        end
    end
    return total, n_pairs
end

"""
    gridded_lag_sweep!(sums, counts, sf, u, schedule, distance_bins, ::Val{D})

Accumulate every pair `schedule` names into the 1-D distance histogram `sums`/`counts`.

`u` is stored `(component, cells...)` with its trailing axes matching `schedule.dims`, and `D` is its
component count, which may exceed the grid's dimension — a lag then lies in the grid's directions
and is zero along the rest. `valid` says which cells hold a datum; a pair counts only when both of
its ends do (see [`field_validity`](@ref)).

Exact, not approximate: each unordered pair is counted once, because lags are enumerated one per
distinct separation and only one of `±lag` is kept. A lag equal to its own reverse is swept over
half of one direction, which is one representative per pair.

Where a periodic direction has an even cell count, its half-period offset joins two cells by two
minimal paths of equal length and opposite sign, so the separation direction is not unique there.
The operator is averaged over those equal-length displacements, which is the only choice that does
not favour one of them; an operator odd in the separation direction therefore vanishes on the pairs
whose every offset half-turns, as it must when no direction is preferred.
"""
function gridded_lag_sweep!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    u::AbstractArray, s::UniformLagSchedule{Dg, T}, dist_be, ::Val{D};
    valid = AllValid(),
) where {OT, CT, Dg, T, D}
    D >= Dg || throw(ArgumentError(
        "the field has $D components but the grid has $Dg directions; a lag needs a component per " *
        "direction",
    ))
    size(u)[2:end] == s.dims || throw(DimensionMismatch(
        "field cells $(size(u)[2:end]) do not match the schedule's dims $(s.dims)",
    ))
    size(u, 1) == D || throw(DimensionMismatch(
        "field has $(size(u, 1)) components, declared $D",
    ))

    plan = squared_digitize_plan(dist_be)
    nb = n_histogram_bins(plan)
    length(sums) == nb && length(counts) == nb || throw(DimensionMismatch(
        "sums and counts must have length $nb; got $(length(sums)) and $(length(counts))",
    ))
    u_flat = reshape(u, D, n_grid_cells(s))
    strides = grid_strides(s)
    r_max = _cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    lags = ntuple(d -> lag_range(s, d, r_max), Val(Dg))

    @inbounds for H in CartesianIndices(lags)
        h = Tuple(H)
        all(iszero, h) && continue                       # a cell paired with itself
        hn = ntuple(d -> _lag_negate(s, d, h[d]), Val(Dg))
        h < hn && continue                               # keep one of `±lag`
        dx = SA.SVector{D, T}(ntuple(d -> d <= Dg ? T(h[d]) * s.spacing[d] : zero(T), Val(D)))
        r2 = LA.dot(dx, dx)
        b = squared_digitize(plan, r2)
        1 <= b <= nb || continue
        # A self-reverse lag pairs the two halves of any direction it half-turns, so sweeping one of
        # them visits each pair once.
        half_dim = (h == hn) ? findfirst(!iszero, h)::Int : 0
        total, n_pairs = _lag_reduce_images(sf, u_flat, valid, Val(D), Val(Dg), s, strides, h, dx,
                                            r2, half_dim, _ambiguous_dirs(s, h))
        sums[b] += OT(total)
        counts[b] += CT(n_pairs)
    end
    return sums, counts
end
