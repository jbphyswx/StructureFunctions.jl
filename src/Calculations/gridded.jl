# Gridded enumeration. On a uniform grid the geometry is shift-invariant, so every pair sharing a lag
# vector shares its separation, its direction and its distance bin; those are computed once per lag
# and the field difference becomes a shifted-array reduction with one histogram write per lag.

"""
    AllValid()

Every cell holds a usable datum, as an indexable value.

Distinct from a grid's own mask, which says which cells *exist*: a cell can exist and still hold
nothing, and it is the field that decides. [`field_validity`](@ref) combines the two. Spelling the
complete-field case as a value that answers `true` keeps one kernel — the test folds away, so a
complete field pays nothing, and no separate masked kernel can drift from the unmasked one.
"""
struct AllValid end

@inline Base.getindex(::AllValid, ::Integer) = true

"""
    field_validity(u[, cell_mask]) -> Bool vector or AllValid

Which cells of `u` hold a usable datum: every component finite, and — where a `cell_mask` is given,
as a grid carries one — the cell marked as existing. `u` is `(components, cells...)` or a `Fields`; the
vector is in the array family of `u`.

Returns [`AllValid`](@ref) when nothing is excluded, so a complete field costs nothing downstream.
"""
function field_validity(u::AbstractArray, cell_mask = nothing)
    v = vec(mapreduce(isfinite, &, reshape(u, size(u, 1), :); dims = 1))
    cell_mask === nothing || (v .&= cell_mask)
    return all(v) ? AllValid() : v
end

field_validity(f::MF.Fields, cell_mask = nothing) = field_validity(MF.packed(f), cell_mask)

"""
    NoWeights()

Every pair counts once, as an indexable value: the unweighted sweep is the same kernel with the
weight folded away, as [`AllValid`](@ref) folds the mask away.
"""
struct NoWeights end

"""
    _pair_weights(weights, n, FT) -> NoWeights or AbstractVector{FT}

The per-cell pair weights a sweep multiplies by: `nothing` means none; a vector must hold one finite
weight per cell and is copied to the requested float type on its existing backend.
"""
_pair_weights(::Nothing, ::Int, ::Type) = NoWeights()
_pair_weights(w::NoWeights, ::Int, ::Type) = w
function _pair_weights(w::AbstractVector, n::Int, ::Type{FT}) where {FT}
    length(w) == n || throw(DimensionMismatch("$(length(w)) weights for $n cells"))
    all(isfinite, w) || throw(ArgumentError("every pair weight must be finite"))
    return FT.(w)
end

"""Weighted pairs make the counts a weighted pair mass, which needs a floating-point count type.
Takes the weights [`_pair_weights`](@ref) returns."""
_check_weighted_counts(::NoWeights, ::Type) = nothing
_check_weighted_counts(::AbstractVector, ::Type{CT}) where {CT} = CT <: AbstractFloat ? nothing : throw(ArgumentError(
    "pair weights make the counts a weighted pair mass, which the count type $CT cannot hold; pass Float64 as the count type, or Float64 counts",
))

"""
    cell_measure(grid) -> Vector

The measure — length, area or volume — of every cell of `grid`, in the cell order the gridded entries
use. As `weights = cell_measure(grid)` it makes a structure function an area average. Supplied by the
FlowGeometries extension.
"""
function cell_measure end

"""
    _packed(u) -> (data, Val(D), Val(V), Val(K))

The one form every gridded kernel indexes: the `(V·D + K, cells)` matrix of a field with its field
layout as type parameters. A bare `(D, cells...)` array is one vector field of width `D`.
"""
@inline _packed(f::MF.Fields{D, V, K}) where {D, V, K} = (MF.packed(f), Val(D), Val(V), Val(K))

function _packed(u::AbstractArray)
    D = size(u, 1)
    return (reshape(u, D, :), Val(D), Val(1), Val(0))
end

"""
    _check_grid_field(sf, data, schedule, ::Val{D}, ::Val{V}, ::Val{K})

The operator reads only fields the field carries; the packed field is `(V·D + K, n_cells)`; and a
field carrying vector fields has at least one component per grid direction, so that a lag has a
component to lie along.
"""
function _check_grid_field(sf, data::AbstractMatrix, s, ::Val{D}, ::Val{V}, ::Val{K}) where {D, V, K}
    validate_fields(sf, Val(V), Val(K))
    Dg = grid_dimension(s)
    (V == 0 || D >= Dg) || throw(ArgumentError(
        "the field has $D components but the grid has $Dg directions; a lag needs a component per " *
        "direction",
    ))
    size(data, 1) == V * D + K || throw(DimensionMismatch(
        "field has $(size(data, 1)) components, declared $V vector field(s) of width $D and $K " *
        "scalar field(s), $(V * D + K) components",
    ))
    size(data, 2) == n_cells(s) || throw(DimensionMismatch(
        "field holds $(size(data, 2)) cells, the grid $(n_cells(s))",
    ))
    return nothing
end

"""The width a lag direction is expressed in: the vector fields' width, or the grid's without any."""
@inline _direction_width(::Val{D}, ::Val{V}, ::Val{Dg}) where {D, V, Dg} = Val(V == 0 ? Dg : D)

"""One pair's increment from the packed field: the plain vector for one vector field, else a multi-field."""
@inline _lag_increment(::Val{D}, ::Val{1}, ::Val{0}, data::AbstractMatrix{T}, k, kp) where {D, T} =
    SA.SVector{D, T}(ntuple(@inline(c -> @inbounds(data[c, kp] - data[c, k])), Val(D)))

@inline function _lag_increment(
    ::Val{D}, ::Val{V}, ::Val{K}, data::AbstractMatrix{T}, k, kp,
) where {D, V, K, T}
    vectors = ntuple(@inline(a -> SA.SVector{D, T}(ntuple(@inline(d -> @inbounds(data[(a - 1) * D + d, kp] -
                                                                                 data[(a - 1) * D + d, k])), Val(D)))),
                     Val(V))
    scalars = ntuple(@inline(c -> @inbounds(data[V * D + c, kp] - data[V * D + c, k])), Val(K))
    return MF.FieldIncrement{D, V, K, T}(vectors, scalars)
end

"""Split a transported packed increment into what the operator reads."""
@inline _split_increment(::Val{D}, ::Val{1}, ::Val{0}, δ::SA.SVector{D}) where {D} = δ

@inline function _split_increment(::Val{D}, ::Val{V}, ::Val{K}, δ::SA.SVector{W, T}) where {D, V, K, W, T}
    vectors = ntuple(@inline(a -> SA.SVector{D, T}(ntuple(@inline(d -> @inbounds(δ[(a - 1) * D + d])), Val(D)))), Val(V))
    scalars = ntuple(@inline(c -> @inbounds(δ[V * D + c])), Val(K))
    return MF.FieldIncrement{D, V, K, T}(vectors, scalars)
end

# Which end of a lag's pairs is read first: the sign of the displacement along the first direction
# that separates the ends — unless that direction half-turns, when neither end comes first (see
# `pair_orientation`). An operator odd in a scalar increment takes this as a factor; the rest see 1.
@inline function _lag_orientation(dx, amb::NTuple{Dg, Bool}) where {Dg}
    @inbounds for d in eachindex(dx)
        iszero(dx[d]) && continue
        return (d <= Dg && amb[d]) ? 0 : (dx[d] > 0 ? 1 : -1)
    end
    return 0
end

# ---------------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------------


"""
    gridded_sweep!(sums, counts, sf, u, schedule, distance_bins[, axis_bins], ::Val{D}, spectral_backend; valid, weights, backend)
    gridded_sweep!(sums, counts, sf, fields, schedule, distance_bins[, axis_bins], spectral_backend; valid, weights, backend)

Accumulate the histogram of every pair `schedule` names, by the algorithm `spectral_backend` selects.
`valid` and `weights` are as on [`gridded_lag_sweep!`](@ref).

Which algorithm sums the pairs is an axis of its own, orthogonal to which hardware runs it: the lag
sweep visits each lag and is exact for any pairwise operator, while a transform produces the increment
moment tensor of every lag at once and so serves the operators that are polynomials in `δu`.
`AutoSpectralBackend` picks whichever does less work, and resolves to the lag sweep unless a transform
is loaded — an extension supplies the transform, and answers `Auto` with a cost comparison because
only it knows what a transform would cost. `backend` names the hardware, as on the unstructured entry.

With `axis_bins` the histogram is joint in separation and `second_axis`: each pair's value
([`InvariantValueAxis`](@ref)), which only the lag sweep bins, or the angle a
[`SeparationAngleAxis`](@ref) reads from each lag; `sums` and `counts` are then `(n_distance, n_axis)`.
With `sf = SinglePassInvariants()` the distance histogram holds the six single-pass invariants,
`sums` and `counts` `(6, n_distance)`.
"""
gridded_sweep!(sums::AbstractArray, counts::AbstractArray, sf, data::AbstractMatrix, schedule, distance_bins,
               ::Val{D}, ::Val{V}, ::Val{K}, ::Union{SB.AbstractDirectSumSpectralBackend, SB.AbstractAutoSpectralBackend}; kwargs...) where {D, V, K} =
    gridded_lag_sweep!(sums, counts, sf, data, schedule, distance_bins, Val(D), Val(V), Val(K); kwargs...)

gridded_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, data::AbstractMatrix, schedule, distance_bins,
               axis_bins, ::Val{D}, ::Val{V}, ::Val{K}, ::Union{SB.AbstractDirectSumSpectralBackend, SB.AbstractAutoSpectralBackend}; kwargs...) where {D, V, K} =
    gridded_lag_sweep!(sums, counts, sf, data, schedule, distance_bins, axis_bins, Val(D), Val(V), Val(K); kwargs...)

gridded_sweep!(::AbstractArray, ::AbstractArray, sf, ::AbstractMatrix, schedule, distance_bins, ::Val{D}, ::Val{V},
               ::Val{K}, tag::SB.AbstractSpectralBackend; kwargs...) where {D, V, K} = _no_transform_loaded(tag, schedule)

gridded_sweep!(::AbstractMatrix, ::AbstractMatrix, sf, ::AbstractMatrix, schedule, distance_bins, axis_bins, ::Val{D},
               ::Val{V}, ::Val{K}, tag::SB.AbstractSpectralBackend; kwargs...) where {D, V, K} =
    _no_transform_loaded(tag, schedule)

gridded_sweep!(::AbstractArray, ::AbstractArray, sf, ::AbstractMatrix, schedule, distance_bins, ::Val{D}, ::Val{V},
               ::Val{K}, spectral_backend; kwargs...) where {D, V, K} = _not_a_spectral_tag(spectral_backend)

gridded_sweep!(::AbstractMatrix, ::AbstractMatrix, sf, ::AbstractMatrix, schedule, distance_bins, axis_bins, ::Val{D},
               ::Val{V}, ::Val{K}, spectral_backend; kwargs...) where {D, V, K} = _not_a_spectral_tag(spectral_backend)

_not_a_spectral_tag(x) = throw(ArgumentError(
    "spectral_backend must be a SpectralBackends tag — AutoSpectralBackend(), DirectSumSpectralBackend(), " *
    "FastFourierTransformSpectralBackend() or a non-uniform FFT tag; got $(typeof(x)). Omitting it sweeps the lags.",
))

# The array and `Fields` forms of both sweeps route through `_packed`, so every kernel sees one layout.
function gridded_sweep!(sums::AbstractArray, counts::AbstractArray, sf, u::AbstractArray, schedule,
                        distance_bins, ::Val{D}, spectral_backend; kwargs...) where {D}
    size(u, 1) == D || throw(DimensionMismatch("field has $(size(u, 1)) components, declared $D"))
    return gridded_sweep!(sums, counts, sf, reshape(u, D, :), schedule, distance_bins, Val(D), Val(1),
                          Val(0), spectral_backend; kwargs...)
end

gridded_sweep!(sums::AbstractVector, counts::AbstractVector, sf, f::MF.Fields{D, V, K}, schedule,
               distance_bins, spectral_backend; kwargs...) where {D, V, K} =
    gridded_sweep!(sums, counts, sf, MF.packed(f), schedule, distance_bins, Val(D), Val(V), Val(K),
                   spectral_backend; kwargs...)

function gridded_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, u::AbstractArray, schedule,
                        distance_bins, axis_bins, ::Val{D}, spectral_backend; kwargs...) where {D}
    size(u, 1) == D || throw(DimensionMismatch("field has $(size(u, 1)) components, declared $D"))
    return gridded_sweep!(sums, counts, sf, reshape(u, D, :), schedule, distance_bins, axis_bins,
                          Val(D), Val(1), Val(0), spectral_backend; kwargs...)
end

gridded_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, f::MF.Fields{D, V, K}, schedule,
               distance_bins, axis_bins, spectral_backend; kwargs...) where {D, V, K} =
    gridded_sweep!(sums, counts, sf, MF.packed(f), schedule, distance_bins, axis_bins, Val(D), Val(V),
                   Val(K), spectral_backend; kwargs...)

function gridded_lag_sweep!(sums::AbstractArray, counts::AbstractArray, sf, u::AbstractArray,
                            schedule, distance_bins, ::Val{D}; kwargs...) where {D}
    size(u, 1) == D || throw(DimensionMismatch("field has $(size(u, 1)) components, declared $D"))
    return gridded_lag_sweep!(sums, counts, sf, reshape(u, D, :), schedule, distance_bins, Val(D),
                              Val(1), Val(0); kwargs...)
end

gridded_lag_sweep!(sums::AbstractVector, counts::AbstractVector, sf, f::MF.Fields{D, V, K}, schedule,
                   distance_bins; kwargs...) where {D, V, K} =
    gridded_lag_sweep!(sums, counts, sf, MF.packed(f), schedule, distance_bins, Val(D), Val(V), Val(K);
                       kwargs...)

function gridded_lag_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, u::AbstractArray,
                            schedule, distance_bins, axis_bins, ::Val{D}; kwargs...) where {D}
    size(u, 1) == D || throw(DimensionMismatch("field has $(size(u, 1)) components, declared $D"))
    return gridded_lag_sweep!(sums, counts, sf, reshape(u, D, :), schedule, distance_bins, axis_bins,
                              Val(D), Val(1), Val(0); kwargs...)
end

gridded_lag_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, f::MF.Fields{D, V, K}, schedule,
                   distance_bins, axis_bins; kwargs...) where {D, V, K} =
    gridded_lag_sweep!(sums, counts, sf, MF.packed(f), schedule, distance_bins, axis_bins, Val(D),
                       Val(V), Val(K); kwargs...)

# ---------------------------------------------------------------------------------------------------
# Schedules with a uniform direction
# ---------------------------------------------------------------------------------------------------

"""
    AbstractSeparableSchedule

A pair enumeration with at least one uniform direction. The cells factor into *slabs*, one per index
on the remaining directions, and within a pair of slabs every pair of cells is named by an integer
lag along the uniform directions. Both sweeps run over `(slab, slab, lag)`: the direct one reduces
the field over each lag, the transform produces every lag of a slab pair at once.

A schedule provides [`uniform_axes`](@ref), [`n_slabs`](@ref), `enumerated_pairs`,
[`separable_layout`](@ref), `lag_limits`, [`lag_transport`](@ref) and `lag_frame`.
"""
abstract type AbstractSeparableSchedule end

# A transform's tag no loaded method takes: a separable schedule wants the AbstractFFTs extension; any
# other schedule has no lag to transform along.
_no_transform_loaded(tag, ::AbstractSeparableSchedule) = throw(ArgumentError(_ABSTRACTFFTS_LOADED[] ?
    "no loaded extension transforms a grid with $(typeof(tag)); FastFourierTransformSpectralBackend() is the " *
    "transform, and DirectSumSpectralBackend() or AutoSpectralBackend() enumerate the lags." :
    "$(typeof(tag)) needs a transform this session has not loaded: `using FFTW: FFTW` on CPU, or another AbstractFFTs " *
    "implementation, supplies it for every grid with a uniform direction. DirectSumSpectralBackend and " *
    "AutoSpectralBackend need none.",
))

_no_transform_loaded(tag, schedule) = throw(ArgumentError(
    "$(typeof(tag)) transforms along a uniform direction, and a $(nameof(typeof(schedule))) has none: its pairs " *
    "share no lag. Omit the tag, or pass DirectSumSpectralBackend or AutoSpectralBackend, to enumerate the pairs.",
))

"""
    IdentityTransport()
    FrameTransport()

How the two ends of a pair are compared. Under `IdentityTransport` the increment is the plain
difference `u(J, i+h) - u(I, i)` and the lag's displacement is its direction. Under `FrameTransport`
each end is first expressed in the pair's own frame, `B·u(J, i+h) - A·u(I, i)` with `W×W` matrices
fixed by the lag, and the longitudinal direction is `ê₁`.
"""
abstract type AbstractLagTransport end

"""The transport of a flat schedule: the increment is the plain difference and the lag's displacement its direction; see [`AbstractLagTransport`](@ref)."""
struct IdentityTransport <: AbstractLagTransport end

"""The transport of a curved schedule: each end is expressed in the pair's own frame before differencing; see [`AbstractLagTransport`](@ref)."""
struct FrameTransport <: AbstractLagTransport end

"""
    uniform_axes(schedule) -> UniformLagSchedule

The uniform directions, in the order they run within a slab.
"""
function uniform_axes end

"""
    n_slabs(schedule) -> Int

How many slabs the cells factor into; one for a schedule whose every direction is uniform.
"""
function n_slabs end

"""
    enumerated_pairs(schedule, r_max)

The slab pairs `(I, J)`, `I ≤ J`, whose cells can come within `r_max` of each other.
"""
function enumerated_pairs end

"""
    n_enumerated_pairs(schedule, r_max) -> Int

How many pairs [`enumerated_pairs`](@ref) yields, without building them. The cost model asks this of
a schedule it has not decided to sweep yet, so materialising the pairs there would allocate the
whole `O(n_slabs²)` list to read one number off it.
"""
n_enumerated_pairs(s, r_max) = count(_ -> true, enumerated_pairs(s, r_max))

"""
    separable_layout(schedule, data, valid, weights) -> (data, valid, weights)

The packed field, its validity and its pair weights with slab `I` occupying columns
`(I-1)·N_u + 1 : I·N_u`, `N_u` the cells of a slab, and the uniform directions running column-major
within it.
"""
function separable_layout end

"""
    lag_limits(schedule, I, J, r_max) -> NTuple{Du, Int}
    lag_limits(schedule, r_max) -> NTuple{Du, Int}

The largest `|h_d|` along each uniform direction at which a pair of the two slabs can still lie
within `r_max`; the two-argument form bounds it over every slab pair.
"""
function lag_limits end

"""
    uniform_lag_box(schedule) -> Bool

Whether [`lag_limits`](@ref)`(schedule, I, J, r_max)` is the same box for every slab pair, so a device
launch indexes its work items by division on one box. A schedule that answers `false` carries one box
per pair, and the launch is indexed through the prefix sum of their volumes.
"""
uniform_lag_box(::AbstractSeparableSchedule) = true

"""
    lag_transport(schedule) -> AbstractLagTransport

See [`IdentityTransport`](@ref).
"""
function lag_transport end

"""
    lag_frame(schedule, I, J, h, ::Val{D}, ::Val{V}, ::Val{K}) -> (ok, r2, factor, geometry)

The pairs lag `h` names between slabs `I` and `J`: whether their separation direction is defined,
their squared separation, the reading factor for operators odd in a scalar increment, and the
geometry [`with_frames`](@ref) turns into one `(dir, A, B)` frame per equal-length displacement.
"""
function lag_frame end

"""
    with_frames(f, transport, geometry)

Call `f(frames)` with the lag's frames as a tuple whose length is a type parameter. `dir` is the
displacement under `IdentityTransport` and the unit longitudinal direction `ê₁` under
`FrameTransport`; `A`, `B` are the transport matrices, `nothing` where there are none.
"""
@inline function with_frames(f, ::IdentityTransport, g)
    return _with_images(g.dx, g.amb) do images
        f(map(_identity_frame, images))
    end
end

@inline _identity_frame(dx) = (dir = dx, A = nothing, B = nothing)

@inline function with_frames(f, ::FrameTransport, g)
    g.two || return f(((dir = g.dir, A = g.A, B = g.B),))
    return f(((dir = g.dir, A = g.A, B = g.B), (dir = g.dir, A = g.A2, B = g.B2)))
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
struct UniformLagSchedule{Dg, T} <: AbstractSeparableSchedule
    dims::NTuple{Dg, Int}
    spacing::NTuple{Dg, T}
    periodic::NTuple{Dg, Bool}
end

"""Cells the schedule covers."""
@inline n_grid_cells(s::UniformLagSchedule) = prod(s.dims)
@inline n_cells(s::UniformLagSchedule) = prod(s.dims)

"""Directions the schedule's lags span."""
@inline grid_dimension(::UniformLagSchedule{Dg}) where {Dg} = Dg

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

"""The largest `|h|` along direction `d` whose displacement stays within `r_max`."""
@inline function _lag_limit(s::UniformLagSchedule, d::Integer, r_max)
    step = abs(@inbounds s.spacing[d])
    return (isfinite(r_max) && step > 0) ? floor(Int, r_max / step) : typemax(Int)
end

"""
    lag_range(schedule, d, lim::Integer)
    lag_range(schedule, d, r_max::AbstractFloat)

Representative lags along direction `d`, one per distinct separation, with `|h| ≤ lim`.

A periodic direction of `n` cells has exactly `n` distinct offsets, and its representatives are
taken in `-((n-1)÷2):(n÷2)`, so each is the minimum image and its magnitude is the true separation.
A bounded direction reaches `±(n-1)`.
"""
@inline function lag_range(s::UniformLagSchedule, d::Integer, lim::Integer)
    n = @inbounds s.dims[d]
    @inbounds if s.periodic[d]
        return (-min((n - 1) ÷ 2, lim)):min(n ÷ 2, lim)
    end
    m = min(n - 1, lim)
    return (-m):m
end

@inline lag_range(s::UniformLagSchedule, d::Integer, r_max::AbstractFloat) =
    lag_range(s, d, _lag_limit(s, d, r_max))

@inline uniform_axes(s::UniformLagSchedule) = s
@inline n_slabs(::UniformLagSchedule) = 1
@inline enumerated_pairs(::UniformLagSchedule, r_max) = ((1, 1),)
@inline n_enumerated_pairs(::UniformLagSchedule, r_max) = 1
@inline separable_layout(::UniformLagSchedule, data, valid, weights) = (data, valid, weights)
@inline lag_limits(s::UniformLagSchedule{Dg}, r_max) where {Dg} =
    ntuple(d -> _lag_limit(s, d, r_max), Val(Dg))
@inline lag_limits(s::UniformLagSchedule, I, J, r_max) = lag_limits(s, r_max)
@inline lag_transport(::UniformLagSchedule) = IdentityTransport()

@inline function lag_frame(
    s::UniformLagSchedule{Dg}, I, J, h::NTuple{Dg, Int}, ::Val{D}, ::Val{V}, ::Val{K},
) where {Dg, D, V, K}
    dx = _lag_displacement(s, h, _direction_width(Val(D), Val(V), Val(Dg)))
    amb = _ambiguous_dirs(s, h)
    return true, SFH.norm2(dx), _lag_orientation(dx, amb), (dx = dx, amb = amb)
end

# The representative of `-h`. A periodic direction of even length has one half-turn offset that is
# its own reverse, and a lag built only from those names each pair twice.
@inline _lag_negate(s::UniformLagSchedule, d::Integer, h::Integer) =
    @inbounds (s.periodic[d] && iseven(s.dims[d]) && h == s.dims[d] ÷ 2) ? h : -h

"""The displacement a lag names, in the direction width `Dr`, zero along directions the grid lacks."""
@inline _lag_displacement(s::UniformLagSchedule{Dg, T}, h::NTuple{Dg, Int}, ::Val{Dr}) where {Dg, T, Dr} =
    SA.SVector{Dr, T}(ntuple(d -> d <= Dg ? T(h[d]) * @inbounds(s.spacing[d]) : zero(T), Val(Dr)))

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
    _with_images(f, dx, ambiguous)

Call `f(images)` with the lag's equal-length displacements as a tuple whose length is a type
parameter: the half-turn count is a value, and the counts a grid can produce are spelled out so each
resolves to its own `Val`.
"""
@generated function _with_images(f, dx, amb::NTuple{Dg, Bool}) where {Dg}
    branches = [:(K == $k && return f(_lag_images(dx, amb, Val($k)))) for k in 0:(Dg - 1)]
    return quote
        $(Expr(:meta, :inline))
        K = count(amb)
        $(branches...)
        return f(_lag_images(dx, amb, Val($Dg)))
    end
end

"""
    RectilinearLagSchedule(uniform, enumerated, axis_order)

Lag enumeration for a rectilinear grid with some uniform directions and some given as coordinate
lists. `uniform` describes the uniform directions, `enumerated` holds one coordinate vector per
remaining direction, and `axis_order[k]` is the field direction at position `k` of
`(uniform..., enumerated...)`.

Every pair of enumerated indices is a pair of slabs, within which the pairs are named by lags along
the uniform directions exactly as on a uniform grid; an enumerated direction never wraps.
"""
struct RectilinearLagSchedule{Du, De, N, T, C <: NTuple{De, AbstractVector{T}}} <: AbstractSeparableSchedule
    uniform::UniformLagSchedule{Du, T}
    enumerated::C
    axis_order::NTuple{N, Int}
    positions::NTuple{N, Int}
    function RectilinearLagSchedule(
        uniform::UniformLagSchedule{Du, T}, enumerated::C, axis_order::NTuple{N, Int},
    ) where {Du, T, De, C <: NTuple{De, AbstractVector{T}}, N}
        N == Du + De || throw(DimensionMismatch(
            "$Du uniform and $De enumerated directions, but axis_order names $N",
        ))
        sort(collect(axis_order)) == 1:N || throw(ArgumentError(
            "axis_order must be a permutation of 1:$N; got $axis_order",
        ))
        positions = ntuple(d -> findfirst(==(d), axis_order)::Int, Val(N))
        return new{Du, De, N, T, C}(uniform, enumerated, axis_order, positions)
    end
end

@inline _enumerated_dims(s::RectilinearLagSchedule) = map(length, s.enumerated)
@inline n_slabs(s::RectilinearLagSchedule) = prod(_enumerated_dims(s))
@inline n_cells(s::RectilinearLagSchedule) = n_cells(s.uniform) * n_slabs(s)
@inline grid_dimension(::RectilinearLagSchedule{Du, De, N}) where {Du, De, N} = N
@inline uniform_axes(s::RectilinearLagSchedule) = s.uniform
@inline lag_limits(s::RectilinearLagSchedule, r_max) = lag_limits(s.uniform, r_max)
@inline lag_limits(s::RectilinearLagSchedule, I, J, r_max) = lag_limits(s.uniform, r_max)
@inline lag_transport(::RectilinearLagSchedule) = IdentityTransport()

"""The grid's cell counts in field order."""
@inline function _field_dims(s::RectilinearLagSchedule{Du, De, N}) where {Du, De, N}
    return ntuple(Val(N)) do d
        k = @inbounds s.positions[d]
        k <= Du ? (@inbounds s.uniform.dims[k]) : length(@inbounds s.enumerated[k - Du])
    end
end

function separable_layout(s::RectilinearLagSchedule{Du, De, N}, data::AbstractMatrix, valid, weights) where {Du, De, N}
    s.axis_order == ntuple(identity, Val(N)) && return data, valid, weights
    dims = _field_dims(s)
    W = size(data, 1)
    perm = (1, (s.axis_order .+ 1)...)
    dp = reshape(permutedims(reshape(data, W, dims...), perm), W, :)
    vp = valid isa AllValid ? valid : vec(permutedims(reshape(valid, dims), s.axis_order))
    wp = weights isa NoWeights ? weights : vec(permutedims(reshape(weights, dims), s.axis_order))
    return dp, vp, wp
end

# Squared distance between two slabs' enumerated coordinates; no pair of theirs is closer.
@inline function _slab_gap2(s::RectilinearLagSchedule{Du, De}, I::Int, J::Int) where {Du, De}
    ce = CartesianIndices(_enumerated_dims(s))
    cI, cJ = ce[I], ce[J]
    acc = zero(eltype(s.enumerated[1]))
    @inbounds for e in 1:De
        δ = s.enumerated[e][cJ[e]] - s.enumerated[e][cI[e]]
        acc += δ * δ
    end
    return acc
end

function enumerated_pairs(s::RectilinearLagSchedule, r_max)
    n = n_slabs(s)
    bound = r_max * r_max
    return ((I, J) for I in 1:n for J in I:n if _slab_gap2(s, I, J) <= bound)
end

@inline function lag_frame(
    s::RectilinearLagSchedule{Du, De, N, T}, I, J, h::NTuple{Du, Int}, ::Val{D}, ::Val{V}, ::Val{K},
) where {Du, De, N, T, D, V, K}
    su = s.uniform
    ce = CartesianIndices(_enumerated_dims(s))
    cI, cJ = ce[I], ce[J]
    # displacement and half-turn flags in slab order, then in the field's order
    dxp = ntuple(Val(N)) do k
        if k <= Du
            T(h[k]) * @inbounds(su.spacing[k])
        else
            e = k - Du
            @inbounds s.enumerated[e][cJ[e]] - s.enumerated[e][cI[e]]
        end
    end
    ambp = _ambiguous_dirs(su, h)
    dx = _ordered_displacement(dxp, s.positions, _direction_width(Val(D), Val(V), Val(N)))
    amb = ntuple(d -> (k = @inbounds(s.positions[d]); k <= Du && @inbounds(ambp[k])), Val(N))
    return true, SFH.norm2(dx), _lag_orientation(dx, amb), (dx = dx, amb = amb)
end

@inline function _ordered_displacement(dxp::Tuple, positions::NTuple{N, Int}, ::Val{Dr}) where {N, Dr}
    T = eltype(dxp)
    return SA.SVector{Dr, T}(ntuple(d -> d <= N ? (@inbounds dxp[positions[d]]) : zero(T), Val(Dr)))
end

# ---------------------------------------------------------------------------------------------------
# Work partition and execution
# ---------------------------------------------------------------------------------------------------

"""
    sweep_items(schedule, r_max, n_tasks, split_lags) -> Vector{NTuple{4, Int}}

The independent units a sweep is split into: `(I, J, part, n_parts)` names slab pair `(I, J)` and,
with `split_lags`, the `part`-th of `n_parts` round-robin shares of its lags. A schedule with fewer
slab pairs than `n_tasks` splits each pair's lags so every task has work.
"""
function sweep_items(s::AbstractSeparableSchedule, r_max, n_tasks::Int, split_lags::Bool)
    pairs = collect(enumerated_pairs(s, r_max))
    n_parts = (split_lags && length(pairs) < n_tasks && !isempty(pairs)) ? cld(n_tasks, length(pairs)) : 1
    items = Vector{NTuple{4, Int}}(undef, length(pairs) * n_parts)
    k = 0
    for (I, J) in pairs, part in 1:n_parts
        items[k += 1] = (I, J, part, n_parts)
    end
    return items
end

"""
    sweep_reduce!(sums, counts, backend, items, make_scratch, body!)

Run `body!(sums, counts, item, scratch)` over every item on `backend`; a threaded backend gives each
task private histograms and its own `make_scratch()`, and adds the partials into `sums`/`counts`.
"""
function sweep_reduce!(sums, counts, ::CB.AbstractSerialBackend, items, make_scratch, body!)
    scratch = make_scratch()
    for it in items
        body!(sums, counts, it, scratch)
    end
    return nothing
end

sweep_reduce!(sums, counts, ::CB.AbstractThreadedBackend, items, make_scratch, body!) =
    threaded_sweep_reduce!(sums, counts, items, make_scratch, body!)

function sweep_reduce!(sums, counts, ::CB.AbstractAutoBackend, items, make_scratch, body!)
    _gridded_threads() > 1 && return threaded_sweep_reduce!(sums, counts, items, make_scratch, body!)
    return sweep_reduce!(sums, counts, CB.SerialBackend(), items, make_scratch, body!)
end

sweep_reduce!(sums, counts, backend::CB.AbstractExecutionBackend, items, make_scratch, body!) =
    throw(ArgumentError(
        "$(typeof(backend)) supplies no `sweep_reduce!`. The serial and threaded backends define " *
        "one here, the Distributed and MPI extensions define theirs, and a device backend does not " *
        "take this path at all: a gridded route reaches a device through its own hook " *
        "(`device_transform_sweep!`, `device_lag_sweep!`), so reaching here with one means this " *
        "route has no device kernel yet",
    ))

"""Threaded gridded sweep over work items; the OhMyThreads extension supplies the method."""
function threaded_sweep_reduce! end

"""
    transform_engine(sf, data, schedule, distance_bins, ::Val{D}, ::Val{V}, ::Val{K}, valid, weights, tag; to = identity, workspace = nothing, slice = (1, 1))

The transform engine's prepared state for a field: the forward transforms of every masked, weighted
monomial of every slab — by FFT for a grid's slabs, by non-uniform FFT for a
[`ScatteredModesSchedule`](@ref), as the spectral `tag` names — the inverse columns and the lag
bookkeeping, every array moved by `to`. The field is slice `t` of `slice = (t, nt)`: its FFT spectra are
slice `t` of one `(blk, nchunks, n_monomials, nt)` array kept in `workspace` (the engine's `spectra`),
each chunk of slabs contiguous at 64-byte alignment. Supplied by the AbstractFFTs extension.
"""
function transform_engine end

"""
    device_transform_sweep!(sums, counts, backend, sf, data, schedule, distance_bins, plan, nb, ::Val{D}, ::Val{V}, ::Val{K}, valid, weights, tag, axis, workspace)

The transform engine on a device backend, supplied by the KernelAbstractions extension together with
an AbstractFFTs implementation for the device's arrays. `axis` is `nothing` for the distance
histogram or `(axis_edges, n_angle, second_axis)` for the joint one; `workspace` is a
[`TransformWorkspace`](@ref) or `nothing`.
"""
function device_transform_sweep! end

"""
    TransformWorkspace()

The buffers and plans a transform on a grid keeps from one call to the next: each slice's forward
spectra, the forward stage's monomial scratch and plans, the inverse scratch of the executors on this
process, and the non-uniform FFT plans set to a scattered schedule's points. Pass it as `workspace` to
repeated transform calls on one grid, as a time series makes; a call whose sizes or array types differ
rebuilds what it keeps and releases what it replaces. A workspace serves one call at a time.
"""
mutable struct TransformWorkspace
    kept::Dict{Any, Pair{Any, Any}}
    pool::Vector{Pair{Any, Any}}
    lock::ReentrantLock
end

TransformWorkspace() = TransformWorkspace(Dict{Any, Pair{Any, Any}}(), Pair{Any, Any}[], ReentrantLock())

"""The buffers `workspace` keeps under `slot`, rebuilt by `build()` unless they were built for `sizes`; the
buffers they replace are released."""
_kept!(::Nothing, slot, sizes, build) = build()
function _kept!(ws::TransformWorkspace, slot, sizes, build)
    held = lock(() -> get(ws.kept, slot, nothing), ws.lock)
    held !== nothing && first(held) == sizes && return last(held)
    fresh = build()
    lock(() -> (ws.kept[slot] = sizes => fresh), ws.lock)
    held === nothing || _release_plan!(last(held))
    return fresh
end

"""One executor's scratch of `sizes` from `workspace`'s pool, or `build()` when none is free. The first element
of `sizes` names the kind of scratch; free sets of that kind and other sizes are released."""
_borrow!(::Nothing, sizes, build) = build()
function _borrow!(ws::TransformWorkspace, sizes, build)
    stale, free = lock(ws.lock) do
        stale = filter(e -> first(first(e)) === first(sizes) && first(e) != sizes, ws.pool)
        filter!(e -> first(first(e)) !== first(sizes) || first(e) == sizes, ws.pool)
        i = findlast(e -> first(e) == sizes, ws.pool)
        stale, i === nothing ? nothing : last(popat!(ws.pool, i))
    end
    foreach(e -> _release_plan!(last(e)), stale)
    return free === nothing ? build() : free
end

_give_back!(::Nothing, sizes, scratch) = nothing
_give_back!(ws::TransformWorkspace, sizes, scratch) = (lock(() -> push!(ws.pool, sizes => scratch), ws.lock); nothing)

"""
    _release_plan!(x)

Free what a transform plan kept in a [`TransformWorkspace`](@ref) holds outside Julia's memory; a tuple or named
tuple of buffers releases each member. The transform extensions add methods for their plans.
"""
_release_plan!(x::Union{Tuple, NamedTuple}) = foreach(_release_plan!, x)
_release_plan!(_) = nothing

"""Release every plan `ws` holds and drop what it keeps; `ws` is not used again."""
function _release_plans!(ws::TransformWorkspace)
    foreach(held -> _release_plan!(last(held)), values(ws.kept))
    foreach(held -> _release_plan!(last(held)), ws.pool)
    empty!(ws.kept)
    empty!(ws.pool)
    return nothing
end

"""The workspace the executors of `backend` share: `workspace` on this process, none across worker processes."""
_local_workspace(workspace, ::CB.AbstractExecutionBackend) = workspace
_local_workspace(workspace, ::Union{CB.AbstractDistributedBackend, CB.AbstractMPIBackend}) = nothing

"""
    _executor_scratch(workspace, backend, sizes, build) -> (make, done)

The scratch source of one sweep: `make()` gives an executor a set borrowed from `workspace` or built by
`build()`, and `done()` returns every set borrowed. Without a workspace, and on worker processes, each
executor builds its own.
"""
_executor_scratch(workspace, backend, sizes, build) = _executor_scratch(_local_workspace(workspace, backend), sizes, build)
_executor_scratch(::Nothing, sizes, build) = (build, Returns(nothing))
function _executor_scratch(ws::TransformWorkspace, sizes, build)
    lent = Any[]
    make = () -> (scratch = _borrow!(ws, sizes, build); lock(() -> push!(lent, scratch), ws.lock); scratch)
    done = () -> foreach(scratch -> _give_back!(ws, sizes, scratch), lent)
    return make, done
end

"""Tasks a backend sweeps with, which is how far the work is split."""
@inline sweep_tasks(::CB.AbstractSerialBackend) = 1
@inline sweep_tasks(::CB.AbstractThreadedBackend) = _gridded_threads()
@inline sweep_tasks(::CB.AbstractAutoBackend) = _gridded_threads()
@inline sweep_tasks(::CB.AbstractExecutionBackend) = 1

@inline _gridded_threads() = _ohmythreads_loaded() ? Threads.nthreads() : 1

# ---------------------------------------------------------------------------------------------------
# The direct sweep
# ---------------------------------------------------------------------------------------------------

"""
    _lag_visit(sf, schedule, uniform, I, J, h, plan, nb, ::Val{D}, ::Val{V}, ::Val{K})
        -> (bin, r2, factor, geometry, self_reverse) or nothing

What both sweeps need to know about one lag of one slab pair before touching the field: `nothing`
when the lag names no pair to bin. Within one slab only one of `±h` is visited and the lag equal to
its own reverse is flagged, since it names every pair twice. `factor` is the lag's reading for an
operator odd in a scalar increment and `1` for every other.
"""
@inline function _lag_visit(
    sf, s::AbstractSeparableSchedule, su::UniformLagSchedule{Dg}, I::Int, J::Int, h::NTuple{Dg, Int},
    plan, nb::Int, ::Val{D}, ::Val{V}, ::Val{K},
) where {Dg, D, V, K}
    same = I == J
    same && all(iszero, h) && return nothing
    hn = ntuple(d -> _lag_negate(su, d, h[d]), Val(Dg))
    same && h < hn && return nothing
    ok, r2, orientation, geometry = lag_frame(s, I, J, h, Val(D), Val(V), Val(K))
    ok || return nothing
    b = squared_digitize(plan, r2)
    1 <= b <= nb || return nothing
    factor = SFT.is_odd_in_scalars(sf) ? orientation : 1
    return b, r2, factor, geometry, same && h == hn
end

"""The lags of slab pair `(I, J)` that can reach `r_max`, as a Cartesian range."""
@inline function _pair_lags(s::AbstractSeparableSchedule, su::UniformLagSchedule{Dg}, I, J, r_max) where {Dg}
    lims = lag_limits(s, I, J, r_max)
    return CartesianIndices(ntuple(d -> lag_range(su, d, lims[d]), Val(Dg)))
end

"""The operator on one pair, once per frame, as the transport defines the increment."""
@inline function _lag_values(
    ::IdentityTransport, sf, frames::NTuple{M, <:NamedTuple}, data, k, kp, r2,
    ::Val{D}, ::Val{V}, ::Val{K}, ::Type{T},
) where {M, D, V, K, T}
    δu = _lag_increment(Val(D), Val(V), Val(K), data, k, kp)
    return ntuple(@inline(m -> T(SFT.flat_pair_value(sf, δu, frames[m].dir, r2))), Val(M))
end

@inline function _lag_values(
    ::FrameTransport, sf, frames::NTuple{M, <:NamedTuple}, data::AbstractMatrix{UT}, k, kp, r2,
    ::Val{D}, ::Val{V}, ::Val{K}, ::Type{T},
) where {M, UT, D, V, K, T}
    W = V * D + K
    uA = SA.SVector{W, UT}(ntuple(@inline(c -> @inbounds(data[c, k])), Val(W)))
    uB = SA.SVector{W, UT}(ntuple(@inline(c -> @inbounds(data[c, kp])), Val(W)))
    return ntuple(@inline(m -> _frame_value(sf, @inbounds(frames[m]), uA, uB, Val(D), Val(V), Val(K), T)), Val(M))
end

"""A pair's operator value under one frame's transport."""
@inline _frame_value(sf, f, uA, uB, vD::Val, vV::Val, vK::Val, ::Type{T}) where {T} =
    T(sf(_split_increment(vD, vV, vK, f.B * uB - f.A * uA), f.dir))

@inline function _lag_values(
    ::IdentityTransport, ::SFT.SinglePassInvariants, frames::NTuple{M, <:NamedTuple}, data, k, kp, r2,
    ::Val{D}, ::Val{1}, ::Val{0}, ::Type{T},
) where {M, D, T}
    δu = _lag_increment(Val(D), Val(1), Val(0), data, k, kp)
    r = sqrt(r2)
    return ntuple(@inline(m -> SA.SVector{SINGLE_PASS_N, T}(single_pass_invariants(
        SFH.increment_invariants(SFH.FlatGeometry{D}(), frames[m].dir, r, δu)...))), Val(M))
end

@inline function _lag_values(
    ::FrameTransport, ::SFT.SinglePassInvariants, frames::NTuple{M, <:NamedTuple}, data::AbstractMatrix{UT}, k,
    kp, r2, ::Val{D}, ::Val{1}, ::Val{0}, ::Type{T},
) where {M, UT, D, T}
    uA = SA.SVector{D, UT}(ntuple(@inline(c -> @inbounds(data[c, k])), Val(D)))
    uB = SA.SVector{D, UT}(ntuple(@inline(c -> @inbounds(data[c, kp])), Val(D)))
    return ntuple(@inline(m -> _frame_invariants(@inbounds(frames[m]), uA, uB, T)), Val(M))
end

"""The six single-pass invariants of a pair under one frame's transport."""
@inline function _frame_invariants(f, uA, uB, ::Type{T}) where {T}
    δu = f.B * uB - f.A * uA
    return SA.SVector{SINGLE_PASS_N, T}(single_pass_invariants(SFH.fma_dot(δu, f.dir), SFH.fma_dot(δu, δu)))
end

"""The zero of one pair's value under `sf`: a number, or the six single-pass invariants."""
@inline _zero_value(sf, ::Type{T}) where {T} = zero(T)
@inline _zero_value(::SFT.SinglePassInvariants, ::Type{T}) where {T} = zero(SA.SVector{SINGLE_PASS_N, T})

"""The leading axes a pair's value under `sf` takes in a histogram: none, or the six single-pass rows."""
@inline _value_dims(sf) = ()
@inline _value_dims(::SFT.SinglePassInvariants) = (SINGLE_PASS_N,)

"""The leading index of row `q` of a value under `sf` in a histogram: none, or `(q,)`."""
@inline _value_row(sf, q::Int) = ()
@inline _value_row(::SFT.SinglePassInvariants, q::Int) = (q,)

"""Throw unless `sums` and `counts` both have the shape of a histogram of `sf` over `cells`."""
function _check_hist_shape(sums, counts, sf, cells::Int...)
    want = (_value_dims(sf)..., cells...)
    size(sums) == want && size(counts) == want || throw(DimensionMismatch(
        "sums and counts must be $want; got $(size(sums)) and $(size(counts))",
    ))
    return nothing
end

"""
    _bin_add!(add!, sums, counts, v, n, I...)

Add a lag's value `v` and its pair count `n` to cell `I` through `add!(array, x, index...)`: one cell for a
number, and for a vector of values one cell per row `q`, at `(q, I...)`, each row counting the pairs.
"""
@inline function _bin_add!(add!::F, sums, counts, v::Number, n, I::Vararg{Int}) where {F}
    add!(sums, eltype(sums)(v), I...)
    add!(counts, eltype(counts)(n), I...)
    return nothing
end

@inline function _bin_add!(add!::F, sums, counts, v::SA.SVector{Q}, n, I::Vararg{Int}) where {F, Q}
    for q in 1:Q
        add!(sums, eltype(sums)(v[q]), q, I...)
        add!(counts, eltype(counts)(n), q, I...)
    end
    return nothing
end

"""
    _lag_reduce(transport, sf, data, valid, weights, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{Dg}, uniform, strides, h,
                frames, r2, half_dim, baseI, baseJ) -> (totals, n_pairs)

Sum the operator over every pair one lag names between the slabs starting at columns `baseI` and
`baseJ`, once per frame in `frames`, with the squared separation `r2` already fixed. With `weights`
each pair carries `w_k · w_kp` in the totals and in the pair count.

The lag's cells are swept as boxes of constant flat offset, so the innermost loop is unit-stride and
the whole lag contributes one number per frame. A wrapped direction contributes a second box rather
than a per-cell branch.
"""
@inline function _lag_reduce(
    transport::AbstractLagTransport, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix{UT}, valid, weights, ::Val{D}, ::Val{V}, ::Val{K}, ::Val{Dg}, su::UniformLagSchedule,
    strides::NTuple{Dg, Int}, h::NTuple{Dg, Int}, frames::NTuple{M, <:NamedTuple}, r2, half_dim::Int,
    baseI::Int, baseJ::Int,
) where {UT, D, V, K, Dg, M}
    T = float(promote_type(UT, eltype(frames[1].dir)))
    totals = ntuple(_ -> _zero_value(sf, T), Val(M))
    n_pairs = _zero_count(weights)
    @inbounds for combo in 0:((1 << Dg) - 1)
        segs = _box_segments(su, h, half_dim, combo)
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
                k = baseI + base + i1
                kp = baseJ + base + i1 + off
                ok = valid[k] & valid[kp]
                w = _pair_weight(weights, k, kp)
                vals = _lag_values(transport, sf, frames, data, k, kp, r2, Val(D), Val(V), Val(K), T)
                totals = _accumulate(totals, vals, ok, w)
                n_pairs = _count_pair(n_pairs, ok, w)
            end
        end
    end
    return totals, n_pairs
end

# Selected, never multiplied by `ok`: an empty cell may hold NaN, and NaN * 0 is NaN. A function of its
# own so the running totals are never a captured variable that is reassigned, which Julia would box.
# An unweighted pair carries the weight `true`, which the compiler folds away.
@inline _accumulate(totals::NTuple{M}, vals::NTuple{M}, ok::Bool, ::Bool) where {M} =
    ntuple(@inline(m -> @inbounds(totals[m] + (ok ? vals[m] : zero(vals[m])))), Val(M))
@inline _accumulate(totals::NTuple{M}, vals::NTuple{M}, ok::Bool, w) where {M} =
    ntuple(@inline(m -> @inbounds(totals[m] + (ok ? w * vals[m] : zero(vals[m])))), Val(M))

@inline _pair_weight(::NoWeights, k::Int, kp::Int) = true
@inline _pair_weight(w::AbstractVector, k::Int, kp::Int) = @inbounds w[k] * w[kp]
@inline _zero_count(::NoWeights) = 0
@inline _zero_count(w::AbstractVector) = zero(eltype(w))
@inline _count_pair(n::Int, ok::Bool, ::Bool) = n + ok
@inline _count_pair(n, ok::Bool, w) = n + (ok ? w : zero(w))

"""
    gridded_lag_sweep!(sums, counts, sf, u, schedule, distance_bins[, axis_bins], ::Val{D}; valid, weights, backend, second_axis)
    gridded_lag_sweep!(sums, counts, sf, fields, schedule, distance_bins[, axis_bins]; valid, weights, backend, second_axis)

Accumulate every pair `schedule` names into the distance histogram `sums`/`counts`, or with
`axis_bins` into the joint histogram over separation and `second_axis` — each pair's own value
(`InvariantValueAxis`, on every schedule) or the angle a `SeparationAngleAxis` reads from each lag's
direction (on a flat schedule) — `sums` and `counts` then being `(n_distance, n_axis)`. With
`sf = SinglePassInvariants()` the distance histogram holds the six single-pass invariants of one vector
field, `sums` and `counts` `(6, n_distance)`, each row counting every pair.

`u` is stored `(component, cells...)` with its trailing axes matching the schedule, and `D` is its
component count, which may exceed the grid's dimension — a lag then lies in the grid's directions
and is zero along the rest. A a multi-field carries several fields the same way. `valid` says
which cells hold a datum; a pair counts only when both of its ends do (see [`field_validity`](@ref)).
`weights`, one finite weight per cell (`nothing` for none), multiplies each pair by `w_k · w_kp` in both
`sums` and `counts`, so the bin average is `Σ w w v / Σ w w` and `counts` must then be floating point;
[`cell_measure`](@ref) supplies a grid's cell areas. `backend` is the hardware, as on the unstructured
entry; a threaded backend splits the slab pairs
across tasks, and the lags of a schedule with a single slab.

Exact, not approximate: each unordered pair is counted once, because lags are enumerated one per
distinct separation and within a slab only one of `±lag` is kept. A lag equal to its own reverse is
swept over half of one direction, which is one representative per pair.

Where a periodic direction has an even cell count, its half-period offset joins two cells by two
minimal paths of equal length and opposite sign, so the separation direction is not unique there.
The operator is averaged over those equal-length displacements, which is the only choice that does
not favour one of them; an operator odd in the separation direction therefore vanishes on the pairs
whose every offset half-turns, as it must when no direction is preferred. In the joint histogram
such a lag's pairs are split between their images' bins in equal halves, so `counts` must then hold
a floating-point type.
"""
function gridded_lag_sweep!(
    sums::AbstractArray, counts::AbstractArray,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K};
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
) where {D, V, K}
    _require_backend(backend)
    _check_grid_field(sf, data, s, Val(D), Val(V), Val(K))
    w = _pair_weights(weights, size(data, 2), float(eltype(data)))
    _assert_counts_can_accumulate(counts, size(data, 2), w)
    plan = squared_digitize_plan(dist_be)
    nb = n_histogram_bins(plan)
    _check_hist_shape(sums, counts, sf, nb)
    su = uniform_axes(s)
    T = eltype(su.spacing)
    r_max = _cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    dp, vp, wp = separable_layout(s, data, valid, w)
    transport = lag_transport(s)
    if backend isa CB.AbstractGPUBackend
        device_lag_sweep!(reshape(sums, size(sums)..., 1), reshape(counts, size(counts)..., 1), backend, sf, s, su,
                          _one_slice(dp), _one_slice_valid(vp), wp, plan, nb, r_max, transport, nothing, Val(D),
                          Val(V), Val(K))
        return sums, counts
    end
    items = sweep_items(s, r_max, sweep_tasks(backend), true)
    body! = (ls, lc, it, _) -> _sweep_item!(ls, lc, sf, s, su, dp, vp, wp, it, plan, nb, r_max, transport,
                                             Val(D), Val(V), Val(K))
    sweep_reduce!(sums, counts, backend, items, () -> nothing, body!)
    return sums, counts
end

"""
    device_lag_sweep!(sums, counts, backend, sf, s, su, data, valid, weights, plan, nb, r_max, transport, axis,
                      ::Val{D}, ::Val{V}, ::Val{K})

Accumulate the direct lag sweep on a device into `sums`/`counts` `(nb, nt)` — `(6, nb, nt)` for the
single-pass invariants — or `(nb, n_axis, nt)` with `axis = (axis_plan, n_axis, second_axis)`: one work item
per (slab pair, lag), each reducing over the
cells of the uniform directions for every slice of `data` `(W, cells, nt)`, laid out as the schedule's
slabs, with `valid` `AllValid()` or `(cells, nt)`. Supplied by the KernelAbstractions extension; this is
the route a non-polynomial operator and a value histogram take on a grid, which the transform cannot
express.
"""
function device_lag_sweep! end

"""A slice's field or validity as the one slice of a batch."""
@inline _one_slice(data::AbstractMatrix) = reshape(data, size(data, 1), size(data, 2), 1)
@inline _one_slice_valid(::AllValid) = AllValid()
@inline _one_slice_valid(v::AbstractVector) = reshape(v, length(v), 1)

"""The slices of a `(W, cells, nt)` batch laid out as the schedule's slabs, as one `(W, cells, nt)` array,
with the validity as `(cells, nt)` and the weights laid out once."""
function _batch_layout(s::AbstractSeparableSchedule, data::AbstractArray{<:Any, 3}, valid, weights)
    dps, vps, wp = _batch_slices(s, data, valid, weights)
    laid = similar(data, size(data, 1), size(data, 2), length(dps))
    for t in eachindex(dps)
        laid[:, :, t] .= dps[t]
    end
    valid isa AllValid && return laid, valid, wp
    v = Matrix{Bool}(undef, size(data, 2), length(vps))
    for t in eachindex(vps)
        v[:, t] .= vps[t]
    end
    return laid, v, wp
end

function _sweep_item!(
    sums::AbstractArray, counts::AbstractArray, sf, s, su::UniformLagSchedule{Dg}, data, valid, weights,
    item::NTuple{4, Int}, plan, nb, r_max, transport, ::Val{D}, ::Val{V}, ::Val{K},
) where {Dg, D, V, K}
    I, J, part, n_parts = item
    lags = _pair_lags(s, su, I, J, r_max)
    strides = grid_strides(su)
    Nu = n_cells(su)
    baseI, baseJ = (I - 1) * Nu, (J - 1) * Nu
    @inbounds for lin in part:n_parts:length(lags)
        h = Tuple(lags[lin])
        v = _lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
        totals, n_pairs = with_frames(transport, geometry) do frames
            _lag_reduce(transport, sf, data, valid, weights, Val(D), Val(V), Val(K), Val(Dg), su, strides, h,
                        frames, r2, half_dim, baseI, baseJ)
        end
        _bin_add!(_plain_add!, sums, counts, factor * sum(totals) / length(totals), n_pairs, b)
    end
    return nothing
end

function gridded_lag_sweep!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K};
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    second_axis::AbstractSecondAxisSource,
) where {OT, CT, D, V, K}
    _require_backend(backend)
    _check_grid_field(sf, data, s, Val(D), Val(V), Val(K))
    _require_axis(second_axis, s)
    _check_half_turn_counts(CT, s, dist_be)
    w = _pair_weights(weights, size(data, 2), float(eltype(data)))
    _assert_counts_can_accumulate(counts, size(data, 2), w)
    plan = squared_digitize_plan(dist_be)
    nb = n_histogram_bins(plan)
    axis_edges = digitize_plan(axis_be)
    na = n_histogram_bins(axis_edges)
    size(sums) == (nb, na) && size(counts) == (nb, na) || throw(DimensionMismatch(
        "sums and counts must be ($nb, $na); got $(size(sums)) and $(size(counts))",
    ))
    su = uniform_axes(s)
    T = eltype(su.spacing)
    r_max = _cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    dp, vp, wp = separable_layout(s, data, valid, w)
    transport = lag_transport(s)
    if backend isa CB.AbstractGPUBackend
        device_lag_sweep!(reshape(sums, nb, na, 1), reshape(counts, nb, na, 1), backend, sf, s, su, _one_slice(dp),
                          _one_slice_valid(vp), wp, plan, nb, r_max, transport, (axis_edges, na, second_axis),
                          Val(D), Val(V), Val(K))
        return sums, counts
    end
    items = sweep_items(s, r_max, sweep_tasks(backend), true)
    body! = (ls, lc, it, _) -> _sweep_item!(ls, lc, sf, s, su, dp, vp, wp, it, plan, nb, r_max, transport,
                                             axis_edges, na, second_axis, Val(D), Val(V), Val(K))
    sweep_reduce!(sums, counts, backend, items, () -> nothing, body!)
    return sums, counts
end

function _sweep_item!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, sf, s, su::UniformLagSchedule{Dg}, data, valid, weights,
    item::NTuple{4, Int}, plan, nb, r_max, transport, axis_edges, na, second_axis::SeparationAngleAxis,
    ::Val{D}, ::Val{V}, ::Val{K},
) where {OT, CT, Dg, D, V, K}
    I, J, part, n_parts = item
    lags = _pair_lags(s, su, I, J, r_max)
    strides = grid_strides(su)
    Nu = n_cells(su)
    baseI, baseJ = (I - 1) * Nu, (J - 1) * Nu
    @inbounds for lin in part:n_parts:length(lags)
        h = Tuple(lags[lin])
        v = _lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
        with_frames(transport, geometry) do frames
            totals, n_pairs = _lag_reduce(transport, sf, data, valid, weights, Val(D), Val(V), Val(K), Val(Dg), su,
                                          strides, h, frames, r2, half_dim, baseI, baseJ)
            _scatter_joint!(sums, counts, b, factor, totals, n_pairs, map(f -> f.dir, frames), r2,
                            axis_edges, na, second_axis)
        end
    end
    return nothing
end

function _sweep_item!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, sf, s, su::UniformLagSchedule{Dg}, data, valid, weights,
    item::NTuple{4, Int}, plan, nb, r_max, transport, value_plan, nv, ::InvariantValueAxis,
    ::Val{D}, ::Val{V}, ::Val{K},
) where {OT, CT, Dg, D, V, K}
    I, J, part, n_parts = item
    lags = _pair_lags(s, su, I, J, r_max)
    strides = grid_strides(su)
    Nu = n_cells(su)
    baseI, baseJ = (I - 1) * Nu, (J - 1) * Nu
    @inbounds for lin in part:n_parts:length(lags)
        h = Tuple(lags[lin])
        v = _lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
        with_frames(transport, geometry) do frames
            _lag_scatter!(_plain_add!, sums, counts, (), transport, sf, data, valid, weights, Val(D), Val(V), Val(K),
                          Val(Dg), su, strides, h, frames, r2, half_dim, baseI, baseJ, b, factor, value_plan, nv)
        end
    end
    return nothing
end

"""`A[I...] += x`: the host's accumulation, where a device's is atomic."""
@inline _plain_add!(A, x, I::Vararg{Int}) = (@inbounds A[I...] += x; nothing)

"""
    _lag_scatter!(add!, sums, counts, tail, transport, sf, data, valid, weights, ::Val{D}, ::Val{V}, ::Val{K},
                  ::Val{Dg}, uniform, strides, h, frames, r2, half_dim, baseI, baseJ, b, factor, value_plan, nv)

Add every pair one lag names between the slabs starting at columns `baseI` and `baseJ` to row `b` of
the joint histogram at the bin of its own value, once per frame with a `1/M` share of the pair, through
`add!(array, value, b, value_bin, tail...)`: the cells swept as [`_lag_reduce`](@ref) sweeps them.
"""
@inline function _lag_scatter!(
    add!::F, sums::AbstractArray{OT}, counts::AbstractArray{CT}, tail::Tuple, transport::AbstractLagTransport,
    sf::SFT.AbstractPairwiseStructureFunctionType, data::AbstractMatrix{UT}, valid, weights, ::Val{D}, ::Val{V},
    ::Val{K}, ::Val{Dg}, su::UniformLagSchedule, strides::NTuple{Dg, Int}, h::NTuple{Dg, Int},
    frames::NTuple{M, <:NamedTuple}, r2, half_dim::Int, baseI::Int, baseJ::Int, b::Int, factor, value_plan,
    nv::Int,
) where {F, OT, CT, UT, D, V, K, Dg, M}
    T = float(promote_type(UT, eltype(frames[1].dir)))
    @inbounds for combo in 0:((1 << Dg) - 1)
        segs = _box_segments(su, h, half_dim, combo)
        ranges = map(first, segs)
        any(isempty, ranges) && continue
        off = 0
        for d in 1:Dg
            off += segs[d][2] * strides[d]
        end
        for J in CartesianIndices(Base.tail(ranges))
            base = 0
            for d in 2:Dg
                base += (J[d - 1] - 1) * strides[d]
            end
            for i1 in ranges[1]
                k = baseI + base + i1
                _scatter_pair!(add!, sums, counts, tail, transport, sf, frames, data, valid, weights, k,
                               baseJ + base + i1 + off, k, r2, b, factor, value_plan, nv, Val(D), Val(V), Val(K), T)
            end
        end
    end
    return nothing
end

"""
    _scatter_pair!(add!, sums, counts, tail, transport, sf, frames, data, valid, weights, k, kp, kw, r2, b, factor,
                   value_plan, nv, ::Val{D}, ::Val{V}, ::Val{K}, T)

Add the pair of data columns `k` and `kp` to row `b` of the joint histogram at the bin of its own value, once
per frame with a `1/M` share of the pair's weight, through `add!(array, value, b, value_bin, tail...)`.
`kw` is the pair's first cell, which the weights are indexed by.
"""
@inline function _scatter_pair!(
    add!::F, sums::AbstractArray{OT}, counts::AbstractArray{CT}, tail::Tuple, transport, sf,
    frames::NTuple{M, <:NamedTuple}, data, valid, weights, k::Int, kp::Int, kw::Int, r2, b::Int, factor, value_plan,
    nv::Int, ::Val{D}, ::Val{V}, ::Val{K}, ::Type{T},
) where {F, OT, CT, M, D, V, K, T}
    (@inbounds(valid[k]) & @inbounds(valid[kp])) || return nothing
    share = _frame_share(_pair_weight(weights, kw, kp - k + kw), Val(M))
    vals = _lag_values(transport, sf, frames, data, k, kp, r2, Val(D), Val(V), Val(K), T)
    for m in 1:M
        value = factor * vals[m]
        vb = SFH.digitize(value, value_plan)
        1 <= vb <= nv || continue
        add!(sums, OT(share * value), b, vb, tail...)
        add!(counts, CT(share), b, vb, tail...)
    end
    return nothing
end

"""
    _lag_fold(op, acc, uniform, strides, h, half_dim, baseI, baseJ, lane, n_lanes) -> acc

Fold `acc = op(acc, k, kp)` over the pairs one lag names between the slabs starting at columns `baseI`
and `baseJ` that fall to lane `lane` of `n_lanes`: each of the lag's boxes, as [`_lag_reduce`](@ref)
sweeps them, flattened with direction 1 fastest, at the positions `lane, lane + n_lanes, …`, so
consecutive lanes read consecutive cells.
"""
@inline function _lag_fold(
    op::F, acc, su::UniformLagSchedule, strides::NTuple{Dg, Int}, h::NTuple{Dg, Int}, half_dim::Int,
    baseI::Int, baseJ::Int, lane::Int, n_lanes::Int,
) where {F, Dg}
    for combo in 0:((1 << Dg) - 1)
        segs = _box_segments(su, h, half_dim, combo)
        lens = map(sg -> length(first(sg)), segs)
        any(iszero, lens) && continue
        off = baseJ - baseI
        k0 = baseI + 1
        for d in 1:Dg
            off += segs[d][2] * strides[d]
            k0 += (first(segs[d][1]) - 1) * strides[d]
        end
        pos = _mixed_digits(lane - 1, lens)
        step = _mixed_digits(n_lanes, lens)
        while @inbounds(pos[Dg]) < @inbounds(lens[Dg])
            k = k0
            for d in 1:Dg
                k += @inbounds(pos[d]) * @inbounds(strides[d])
            end
            acc = op(acc, k, k + off)
            pos = _mixed_add(pos, step, lens)
        end
    end
    return acc
end

"""The `(cells, offset)` segment along each direction of box `combo` of lag `h` (its bit `d - 1` choosing the
direction's second segment), as [`_lag_segments`](@ref) gives them."""
@inline _box_segments(su::UniformLagSchedule, h::NTuple{Dg, Int}, half_dim::Int, combo::Int) where {Dg} =
    ntuple(@inline(d -> _lag_segments(su, d, h[d], d == half_dim)[1 + ((combo >> (d - 1)) & 1)]), Val(Dg))

"""Digits of `f` in the mixed radix `lens`, direction 1 fastest; the last digit is unbounded."""
@inline _mixed_digits(f::Int, lens::NTuple{Dg, Int}) where {Dg} = ntuple(@inline(d -> _mixed_digit(f, lens, d)), Val(Dg))

"""Digit `d` of `f` in the mixed radix `lens`."""
@inline function _mixed_digit(f::Int, lens::NTuple{Dg, Int}, d::Int) where {Dg}
    q = f
    for e in 1:(d - 1)
        q ÷= @inbounds lens[e]
    end
    return d == Dg ? q : q % @inbounds(lens[d])
end

"""The digits `a + b` in the mixed radix `lens` (every digit of both below its radix but the last)."""
@inline function _mixed_add(a::NTuple{Dg, Int}, b::NTuple{Dg, Int}, lens::NTuple{Dg, Int}) where {Dg}
    out = a
    carry = 0
    for d in 1:Dg
        s = @inbounds(a[d]) + @inbounds(b[d]) + carry
        carry = (d < Dg && s >= @inbounds(lens[d])) ? 1 : 0
        out = Base.setindex(out, s - carry * @inbounds(lens[d]), d)
    end
    return out
end

"""A pair's weight split over its `M` equal-length images; one image keeps the weight as given."""
@inline _frame_share(w, ::Val{1}) = w
@inline _frame_share(w, ::Val{M}) where {M} = w / M

"""Throw unless `second_axis` is defined on `schedule`'s pairs: an angle needs a directional schedule."""
_require_axis(::SeparationAngleAxis, s::AbstractSeparableSchedule) = _require_directional(s)
_require_axis(::InvariantValueAxis, ::AbstractSeparableSchedule) = nothing

# A pair's direction on a sphere lives in its own geodesic frame, so an angle to one fixed reference
# axis is not a property of the pair.
_require_directional(s::AbstractSeparableSchedule) = _require_directional(lag_transport(s))
_require_directional(::IdentityTransport) = nothing
_require_directional(::FrameTransport) = throw(ArgumentError(
    "a joint histogram over the separation angle is not defined on a sphere: each pair's " *
    "direction lives in its own geodesic frame, so an angle to one fixed reference axis is not a " *
    "property of the pair. Ask for the 1-D histogram.",
))

"""
    _check_half_turn_counts(CT, schedule, distance_bins)

Throw unless counts of type `CT` can hold a joint histogram over angle of `schedule`'s lags up to the
last edge: a lag that half-turns a periodic direction of even length splits each of its pairs in
halves between two directions, which an integer count cannot hold.
"""
function _check_half_turn_counts(::Type{CT}, s::AbstractSeparableSchedule, dist_be) where {CT}
    CT <: Integer || return nothing
    su = uniform_axes(s)
    T = eltype(su.spacing)
    lims = lag_limits(s, _cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be))))
    any(d -> su.periodic[d] && iseven(su.dims[d]) && lims[d] >= su.dims[d] ÷ 2, eachindex(su.dims)) || return nothing
    throw(ArgumentError(
        "a lag that half-turns a periodic direction splits each pair between its two directions, so a " *
        "joint histogram over angle needs a floating-point count type; got $CT",
    ))
end

"""
    _assert_grid_counts(schedule, counts, n_cells, weights)

The count check of a grid entry: a soft-binned schedule's counts are a pair mass and need a
floating-point type; an exact one's need room for every pair on top of what they hold.
"""
_assert_grid_counts(s, counts::AbstractArray, n_cells::Int, weights) =
    _soft_binned(s) ? _assert_mass_counts(eltype(counts)) : _assert_counts_can_accumulate(counts, n_cells, weights)

# ---------------------------------------------------------------------------------------------------
# Batches over a trailing slice axis
# ---------------------------------------------------------------------------------------------------

"""
    batch_validity(u[, cell_mask]) -> Bool matrix or AllValid

Which cells hold a usable datum in each slice of `u`, stored `(components, cells..., slices)`: one
column of cell validity per slice, in the array family of `u`. `cell_mask`, where a grid carries one,
marks the cells that exist.

Returns [`AllValid`](@ref) when nothing is excluded, as [`field_validity`](@ref) does for one slice.
"""
function batch_validity(u::AbstractArray, cell_mask = nothing)
    v = dropdims(mapreduce(isfinite, &, reshape(u, size(u, 1), :, size(u)[end]); dims = 1); dims = 1)
    cell_mask === nothing || (v .&= cell_mask)
    return all(v) ? AllValid() : v
end

"""Slice `t`'s cell validity: its own column, or the complete-field value for every slice."""
@inline _valid_slice(::AllValid, ::Int) = AllValid()
@inline _valid_slice(v::AbstractMatrix, t::Int) = view(v, :, t)

"""
    batch_shares_lag_geometry(schedule) -> Bool

Whether a transform batch visits each lag once and contracts every slice against it, holding all the
slices' inverted columns of a slab pair at the same time.

A schedule answering `false` takes its slices one at a time, with the inverse scratch of a single
slice. Curved schedules answer `true`: their per-lag geometry is a geodesic frame and `W×W` transport
matrices built from the schedule alone, and their uniform directions are a slab's, so the columns of
every slice fit together.
"""
batch_shares_lag_geometry(s::AbstractSeparableSchedule) = lag_transport(s) isa FrameTransport

"""
    _batch_slices(schedule, data, valid, weights) -> (data_slices, valid_slices, weights)

Each slice of a `(components, cells, slices)` batch laid out as the schedule's slabs. The weights
belong to the cells, so one set serves every slice.
"""
function _batch_slices(s::AbstractSeparableSchedule, data::AbstractArray{<:Any, 3}, valid, weights)
    laid = map(t -> separable_layout(s, view(data, :, :, t), _valid_slice(valid, t), weights), 1:size(data, 3))
    return map(first, laid), map(x -> x[2], laid), laid[1][3]
end

"""
    _check_batch(sf, data, schedule, valid, ::Val{D}, ::Val{V}, ::Val{K}) -> Int

The slice count of a `(V·D + K, cells, slices)` batch, with the field checked against the schedule
and the validity against both.
"""
function _check_batch(
    sf, data::AbstractArray{<:Any, 3}, s, valid, ::Val{D}, ::Val{V}, ::Val{K},
) where {D, V, K}
    nt = size(data, 3)
    nt >= 1 || throw(ArgumentError("a batch holds at least one slice; got $nt"))
    _check_grid_field(sf, view(data, :, :, 1), s, Val(D), Val(V), Val(K))
    valid isa AllValid || size(valid) == (size(data, 2), nt) || throw(DimensionMismatch(
        "valid must be $((size(data, 2), nt)) — one cell validity per slice; got $(size(valid))",
    ))
    return nt
end

"""
    gridded_sweep_batch!(sums, counts, sf, u, schedule, distance_bins[, axis_bins], ::Val{D}, spectral_backend; valid, weights, backend)

Accumulate the histogram of every pair `schedule` names, for every slice of a batch, by the algorithm
`spectral_backend` selects. `sums` and `counts` are `(n_distance, n_slices)`, or with `axis_bins`
`(n_distance, n_angle, n_slices)`.

`u` is `(component, cells..., slices)` with the cells matching the schedule; the packed form is
`(V·D + K, cells, slices)`, which is how a multi-field time series is passed. `valid`, `weights` and
`backend` are as on [`gridded_lag_sweep_batch!`](@ref).

Each pair's geometry is fixed by `schedule` alone, so the lag sweep visits every lag once and reduces
all the slices against it. A transform shares each lag between the slices on the schedules that
answer [`batch_shares_lag_geometry`](@ref) and takes one slice at a time on the rest.
"""
gridded_sweep_batch!(sums, counts, sf, data::AbstractArray{<:Any, 3}, schedule, distance_bins, ::Val{D}, ::Val{V},
                     ::Val{K}, ::Union{SB.AbstractDirectSumSpectralBackend, SB.AbstractAutoSpectralBackend};
                     kwargs...) where {D, V, K} =
    gridded_lag_sweep_batch!(sums, counts, sf, data, schedule, distance_bins, Val(D), Val(V), Val(K); kwargs...)

gridded_sweep_batch!(sums, counts, sf, data::AbstractArray{<:Any, 3}, schedule, distance_bins, axis_bins, ::Val{D},
                     ::Val{V}, ::Val{K},
                     ::Union{SB.AbstractDirectSumSpectralBackend, SB.AbstractAutoSpectralBackend};
                     kwargs...) where {D, V, K} =
    gridded_lag_sweep_batch!(sums, counts, sf, data, schedule, distance_bins, axis_bins, Val(D), Val(V), Val(K);
                             kwargs...)

gridded_sweep_batch!(sums, counts, sf, ::AbstractArray{<:Any, 3}, schedule, distance_bins, ::Val{D}, ::Val{V},
                     ::Val{K}, tag::SB.AbstractSpectralBackend; kwargs...) where {D, V, K} =
    _no_transform_loaded(tag, schedule)

gridded_sweep_batch!(sums, counts, sf, ::AbstractArray{<:Any, 3}, schedule, distance_bins, axis_bins, ::Val{D},
                     ::Val{V}, ::Val{K}, tag::SB.AbstractSpectralBackend; kwargs...) where {D, V, K} =
    _no_transform_loaded(tag, schedule)

gridded_sweep_batch!(sums, counts, sf, ::AbstractArray{<:Any, 3}, schedule, distance_bins, ::Val{D}, ::Val{V},
                     ::Val{K}, spectral_backend; kwargs...) where {D, V, K} = _not_a_spectral_tag(spectral_backend)

gridded_sweep_batch!(sums, counts, sf, ::AbstractArray{<:Any, 3}, schedule, distance_bins, axis_bins, ::Val{D},
                     ::Val{V}, ::Val{K}, spectral_backend; kwargs...) where {D, V, K} =
    _not_a_spectral_tag(spectral_backend)

# The array forms of both batch sweeps route through the packed layout, as the single-slice forms do.
function gridded_sweep_batch!(sums, counts, sf, u::AbstractArray, schedule, distance_bins, ::Val{D},
                              spectral_backend; kwargs...) where {D}
    return gridded_sweep_batch!(sums, counts, sf, _packed_batch(u, Val(D)), schedule, distance_bins, Val(D), Val(1),
                                Val(0), spectral_backend; kwargs...)
end

function gridded_sweep_batch!(sums, counts, sf, u::AbstractArray, schedule, distance_bins, axis_bins, ::Val{D},
                              spectral_backend; kwargs...) where {D}
    return gridded_sweep_batch!(sums, counts, sf, _packed_batch(u, Val(D)), schedule, distance_bins, axis_bins,
                                Val(D), Val(1), Val(0), spectral_backend; kwargs...)
end

function gridded_lag_sweep_batch!(sums, counts, sf, u::AbstractArray, schedule, distance_bins, ::Val{D};
                                  kwargs...) where {D}
    return gridded_lag_sweep_batch!(sums, counts, sf, _packed_batch(u, Val(D)), schedule, distance_bins, Val(D),
                                    Val(1), Val(0); kwargs...)
end

function gridded_lag_sweep_batch!(sums, counts, sf, u::AbstractArray, schedule, distance_bins, axis_bins, ::Val{D};
                                  kwargs...) where {D}
    return gridded_lag_sweep_batch!(sums, counts, sf, _packed_batch(u, Val(D)), schedule, distance_bins, axis_bins,
                                    Val(D), Val(1), Val(0); kwargs...)
end

"""The `(D, cells, slices)` layout of a batch stored `(D, cells..., slices)`, one vector field of width `D`."""
@inline function _packed_batch(u::AbstractArray, ::Val{D}) where {D}
    size(u, 1) == D || throw(DimensionMismatch("field has $(size(u, 1)) components, declared $D"))
    ndims(u) >= 3 || throw(DimensionMismatch(
        "a batch is stored (component, cells..., slices) and so has at least three axes; got $(size(u))",
    ))
    return reshape(u, D, :, size(u)[end])
end

"""
    gridded_lag_sweep_batch!(sums, counts, sf, u, schedule, distance_bins[, axis_bins], ::Val{D}; valid, weights, backend)

Sweep the lags of `schedule` once and reduce every slice of a batch against each of them, into
`sums`/`counts` of shape `(n_distance, n_slices)` — with `axis_bins`, `(n_distance, n_axis, n_slices)`;
for `sf = SinglePassInvariants()`, `(6, n_distance, n_slices)`.

`u` is `(component, cells..., slices)`, the cells matching the schedule. `valid` is `AllValid()` or
one column of cell validity per slice ([`batch_validity`](@ref)); `weights` is one finite weight per
cell, shared by the slices, so a cell measure is given once. `backend` names the hardware, as on
[`gridded_lag_sweep!`](@ref), whose pair enumeration, reading rule and half-turn averaging this
follows exactly: slice `t` of the output is that sweep on slice `t` of the field.
"""
function gridded_lag_sweep_batch!(
    sums::AbstractArray, counts::AbstractArray, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractArray{<:Any, 3}, s::AbstractSeparableSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K};
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
) where {D, V, K}
    _require_backend(backend)
    nt = _check_batch(sf, data, s, valid, Val(D), Val(V), Val(K))
    w = _pair_weights(weights, size(data, 2), float(eltype(data)))
    _assert_counts_can_accumulate(counts, size(data, 2), w)
    plan = squared_digitize_plan(dist_be)
    nb = n_histogram_bins(plan)
    _check_hist_shape(sums, counts, sf, nb, nt)
    su = uniform_axes(s)
    T = eltype(su.spacing)
    r_max = _cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    transport = lag_transport(s)
    if backend isa CB.AbstractGPUBackend
        laid, lv, lw = _batch_layout(s, data, valid, w)
        device_lag_sweep!(sums, counts, backend, sf, s, su, laid, lv, lw, plan, nb, r_max, transport, nothing,
                          Val(D), Val(V), Val(K))
        return sums, counts
    end
    dps, vps, wp = _batch_slices(s, data, valid, w)
    items = sweep_items(s, r_max, sweep_tasks(backend), true)
    body! = (ls, lc, it, _) -> _sweep_item_batch!(ls, lc, sf, s, su, dps, vps, wp, it, plan, nb, r_max, transport,
                                                  Val(D), Val(V), Val(K))
    sweep_reduce!(sums, counts, backend, items, () -> nothing, body!)
    return sums, counts
end

function _sweep_item_batch!(
    sums::AbstractArray, counts::AbstractArray, sf, s, su::UniformLagSchedule{Dg}, dps, vps, weights,
    item::NTuple{4, Int}, plan, nb, r_max, transport, ::Val{D}, ::Val{V}, ::Val{K},
) where {Dg, D, V, K}
    I, J, part, n_parts = item
    lags = _pair_lags(s, su, I, J, r_max)
    strides = grid_strides(su)
    Nu = n_cells(su)
    baseI, baseJ = (I - 1) * Nu, (J - 1) * Nu
    @inbounds for lin in part:n_parts:length(lags)
        h = Tuple(lags[lin])
        v = _lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
        with_frames(transport, geometry) do frames
            for t in eachindex(dps)
                totals, n_pairs = _lag_reduce(transport, sf, dps[t], vps[t], weights, Val(D), Val(V), Val(K),
                                              Val(Dg), su, strides, h, frames, r2, half_dim, baseI, baseJ)
                _bin_add!(_plain_add!, sums, counts, factor * sum(totals) / length(totals), n_pairs, b, t)
            end
        end
    end
    return nothing
end

function gridded_lag_sweep_batch!(
    sums::AbstractArray{OT, 3}, counts::AbstractArray{CT, 3}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractArray{<:Any, 3}, s::AbstractSeparableSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K};
    valid = AllValid(), weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    second_axis::AbstractSecondAxisSource,
) where {OT, CT, D, V, K}
    _require_backend(backend)
    nt = _check_batch(sf, data, s, valid, Val(D), Val(V), Val(K))
    _require_axis(second_axis, s)
    _check_half_turn_counts(CT, s, dist_be)
    w = _pair_weights(weights, size(data, 2), float(eltype(data)))
    _assert_counts_can_accumulate(counts, size(data, 2), w)
    plan = squared_digitize_plan(dist_be)
    nb = n_histogram_bins(plan)
    axis_edges = digitize_plan(axis_be)
    na = n_histogram_bins(axis_edges)
    size(sums) == (nb, na, nt) && size(counts) == (nb, na, nt) || throw(DimensionMismatch(
        "sums and counts must be ($nb, $na, $nt); got $(size(sums)) and $(size(counts))",
    ))
    su = uniform_axes(s)
    T = eltype(su.spacing)
    r_max = _cull_is_unbounded(dist_be) ? T(Inf) : T(float(last(dist_be)))
    transport = lag_transport(s)
    if backend isa CB.AbstractGPUBackend
        laid, lv, lw = _batch_layout(s, data, valid, w)
        device_lag_sweep!(sums, counts, backend, sf, s, su, laid, lv, lw, plan, nb, r_max, transport,
                          (axis_edges, na, second_axis), Val(D), Val(V), Val(K))
        return sums, counts
    end
    dps, vps, wp = _batch_slices(s, data, valid, w)
    items = sweep_items(s, r_max, sweep_tasks(backend), true)
    body! = (ls, lc, it, _) -> _sweep_item_batch!(ls, lc, sf, s, su, dps, vps, wp, it, plan, nb, r_max, transport,
                                                  axis_edges, na, second_axis, Val(D), Val(V), Val(K))
    sweep_reduce!(sums, counts, backend, items, () -> nothing, body!)
    return sums, counts
end

function _sweep_item_batch!(
    sums::AbstractArray{OT, 3}, counts::AbstractArray{CT, 3}, sf, s, su::UniformLagSchedule{Dg}, dps, vps, weights,
    item::NTuple{4, Int}, plan, nb, r_max, transport, value_plan, nv, ::InvariantValueAxis, ::Val{D}, ::Val{V},
    ::Val{K},
) where {OT, CT, Dg, D, V, K}
    I, J, part, n_parts = item
    lags = _pair_lags(s, su, I, J, r_max)
    strides = grid_strides(su)
    Nu = n_cells(su)
    baseI, baseJ = (I - 1) * Nu, (J - 1) * Nu
    @inbounds for lin in part:n_parts:length(lags)
        h = Tuple(lags[lin])
        v = _lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
        with_frames(transport, geometry) do frames
            for t in eachindex(dps)
                _lag_scatter!(_plain_add!, sums, counts, (t,), transport, sf, dps[t], vps[t], weights, Val(D), Val(V),
                              Val(K), Val(Dg), su, strides, h, frames, r2, half_dim, baseI, baseJ, b, factor,
                              value_plan, nv)
            end
        end
    end
    return nothing
end

function _sweep_item_batch!(
    sums::AbstractArray{OT, 3}, counts::AbstractArray{CT, 3}, sf, s, su::UniformLagSchedule{Dg}, dps, vps, weights,
    item::NTuple{4, Int}, plan, nb, r_max, transport, axis_edges, na, second_axis::SeparationAngleAxis,
    ::Val{D}, ::Val{V}, ::Val{K},
) where {OT, CT, Dg, D, V, K}
    I, J, part, n_parts = item
    lags = _pair_lags(s, su, I, J, r_max)
    strides = grid_strides(su)
    Nu = n_cells(su)
    baseI, baseJ = (I - 1) * Nu, (J - 1) * Nu
    @inbounds for lin in part:n_parts:length(lags)
        h = Tuple(lags[lin])
        v = _lag_visit(sf, s, su, I, J, h, plan, nb, Val(D), Val(V), Val(K))
        v === nothing && continue
        b, r2, factor, geometry, self_reverse = v
        half_dim = self_reverse ? findfirst(!iszero, h)::Int : 0
        with_frames(transport, geometry) do frames
            dirs = map(f -> f.dir, frames)
            for t in eachindex(dps)
                totals, n_pairs = _lag_reduce(transport, sf, dps[t], vps[t], weights, Val(D), Val(V), Val(K),
                                              Val(Dg), su, strides, h, frames, r2, half_dim, baseI, baseJ)
                _scatter_joint!(view(sums, :, :, t), view(counts, :, :, t), b, factor, totals, n_pairs, dirs, r2,
                                axis_edges, na, second_axis)
            end
        end
    end
    return nothing
end

"""
    device_transform_sweep_batch!(sums, counts, backend, sf, data, schedule, distance_bins, plan, nb, ::Val{D}, ::Val{V}, ::Val{K}, valid, weights, tag, axis, workspace)

The transform engine on a device backend for a batch over a trailing slice axis, supplied by the
KernelAbstractions extension together with an AbstractFFTs implementation for the device's arrays.
`axis` is `nothing` for the distance histogram or `(axis_edges, n_angle, second_axis)` for the joint
one; `workspace` is a [`TransformWorkspace`](@ref) or `nothing`.
"""
function device_transform_sweep_batch! end

# Each equal-length image carries its share of the lag's pairs to its own angle bin.
@inline function _scatter_joint!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, b, factor, totals::NTuple{M}, n_pairs, images,
    r2, axis_edges, na, second_axis,
) where {OT, CT, M}
    @inbounds for m in 1:M
        bθ = SFH.digitize(axis_quantity(second_axis, images[m], r2), axis_edges)
        1 <= bθ <= na || continue
        sums[b, bθ] += OT(factor * totals[m] / M)
        counts[b, bθ] += CT(n_pairs / M)
    end
    return nothing
end
