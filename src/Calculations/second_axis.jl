# What the joint histogram's second axis bins. The distance axis is always the separation; the second
# axis is a choice, and the kernels are already parameterised over how it is digitized — so a new
# choice is a new source of the quantity, not a new histogram.

"""
    AbstractSecondAxisSource

What the second axis of a joint histogram bins.

The first axis is always the separation. The second is a choice: the operator's own value, giving
the distribution of the structure function at each separation; or the separation's **direction**,
giving the structure function resolved by angle. Both are digitized the same way into the same
histogram, so a source is where the binned quantity comes from, not a new kind of output.
"""
abstract type AbstractSecondAxisSource end

"""
    InvariantValueAxis()

Bin the operator's own value: the joint distribution of separation and structure-function value.
"""
struct InvariantValueAxis <: AbstractSecondAxisSource end

"""
    SeparationAngleAxis(reference_axis)

Bin the angle between the pair's separation and `reference_axis`, giving `S(r, θ)` — the structure
function resolved by direction, which is what an anisotropic flow needs.

The angle is folded so that a pair and its reverse give the same value, which they must: swapping the
two ends flips the separation, and no structure function distinguishes the two. In two dimensions the
signed azimuth is taken modulo `π`, giving `[0, π)`; in three or more the polar angle to the axis is
taken from `|cos|`, giving `[0, π/2]`.

`reference_axis` need not be normalized.
"""
struct SeparationAngleAxis{V} <: AbstractSecondAxisSource
    reference_axis::V
end

"""
Whether the source needs its own per-pair buffer, or reads the operator value the kernel already
computed. Constant-folded, so binning the operator's value costs no extra store.
"""
@inline needs_axis_buffer(::InvariantValueAxis) = false
@inline needs_axis_buffer(::SeparationAngleAxis) = true

"""
    axis_quantity(source, dx, r2) -> value

The second-axis quantity for one pair, from its separation vector.
"""
@inline axis_quantity(::InvariantValueAxis, dx, r2) = zero(eltype(dx))

@inline function axis_quantity(s::SeparationAngleAxis, dx::SA.SVector{2}, r2)
    e = s.reference_axis
    ex, ey = e[1], e[2]
    # the azimuth measured from the reference axis, folded by π so a pair and its reverse agree
    return mod(atan(dx[2] * ex - dx[1] * ey, dx[1] * ex + dx[2] * ey), oftype(r2, π))
end

@inline function axis_quantity(s::SeparationAngleAxis, dx::SA.SVector{D}, r2) where {D}
    e = s.reference_axis
    en2 = zero(r2)
    proj = zero(r2)
    @inbounds for d in 1:D
        en2 += e[d] * e[d]
        proj += dx[d] * e[d]
    end
    # |cos| folds the polar angle into [0, π/2], so a pair and its reverse agree
    c = abs(proj) / sqrt(r2 * en2)
    return acos(min(one(c), c))
end

"""
    axis_key(source, valbuf, axbuf, j) -> value

The quantity to digitize onto the second axis for pair `j`: the operator value the kernel already
buffered, or the source's own buffered quantity.
"""
@inline axis_key(::InvariantValueAxis, valbuf, axbuf, j) = @inbounds valbuf[j]
@inline axis_key(::SeparationAngleAxis, valbuf, axbuf, j) = @inbounds axbuf[j]

"""
    axis_bounds(source) -> (lo, hi)

The range the source's quantity spans, for callers building bins over it. The operator's value has no
bound the source knows.
"""
axis_bounds(::InvariantValueAxis) = throw(ArgumentError(
    "the operator's value has no range known ahead of the calculation; supply value bins, or scan " *
    "the data as the auto-binning entry does",
))
axis_bounds(::SeparationAngleAxis{V}) where {V} = (0.0, Float64(π))
