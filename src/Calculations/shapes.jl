# Array shape contract for public calculation APIs.

"""
    AbstractFieldShape{D}

Which array-rank pattern a validated `(x, u)` pair forms, with `D` the number of velocity components
on axis 1 of `u`. Backends dispatch on this, so `D` reaches the kernels as a static method parameter.

- [`PointField`](@ref): `u` is `(D, N)`.
- [`SharedPositionField`](@ref): `u` is `(D, N, auxiliary...)` — one set of positions, many fields.
- [`VaryingPositionField`](@ref): `x` and `u` both carry matching auxiliary axes.

The coordinate count on axis 1 of `x` is deliberately **not** here: it is fixed by the metric's
geometry (see `HelperFunctions.coordinate_width`), so it is read off that geometry once `D` is
static rather than being restated as a shape parameter.
"""
abstract type AbstractFieldShape{D} end
struct PointField{D} <: AbstractFieldShape{D} end
struct SharedPositionField{D} <: AbstractFieldShape{D} end
struct VaryingPositionField{D} <: AbstractFieldShape{D} end

"""Velocity components on axis 1 of `u`."""
@inline spatial_dimension(::AbstractFieldShape{D}) where {D} = D

"""
    _pair_dims(metric, D) -> (Val{W}, Val{D})

Coordinate and velocity widths as type parameters, for the kernels that take `x`/`u` as `Array`s
(whose axis-1 lengths are values, not type parameters). `D` is already restricted to 2 or 3 by
[`_validate_spatial_dimension`](@ref), so dispatching on those two makes both results concrete;
`W` comes from the metric's geometry, which is what defines it.
"""
@inline function _pair_dims(distance_metric, D::Int)
    D == 2 && return (SFH.coordinate_width(SFH.pair_geometry_for(distance_metric, Val(2))), Val(2))
    D == 3 && return (SFH.coordinate_width(SFH.pair_geometry_for(distance_metric, Val(3))), Val(3))
    return _validate_spatial_dimension(D)
end
@inline has_auxiliary_axes(::PointField) = false
@inline has_auxiliary_axes(::SharedPositionField) = true
@inline has_auxiliary_axes(::VaryingPositionField) = true

function _unsupported_tuple_input()
    throw(
        ArgumentError(
            "tuple-of-component-vector inputs are not part of the stabilized public API; " *
            "pass arrays with shape (D, N) or (D, N, auxiliary...) instead",
        ),
    )
end

function _validate_spatial_dimension(D::Integer)
    D == 2 || D == 3 ||
        throw(DimensionMismatch("expected spatial/velocity dimension D=2 or D=3 on axis 1; got D=$D"))
    return nothing
end

@inline _val_int(::Val{W}) where {W} = W

"""
    _validate_array_shape(x, u, distance_metric) -> AbstractFieldShape

Axis-1 of `u` is the velocity dimension `D`; axis-1 of `x` is however many coordinates the
metric's geometry needs to locate a point, which is **not** always `D`. On a sphere a point takes
two coordinates whether or not the velocity carries a third, radial, component — the shell radius
belongs to the geometry, not to each point — so `x` is `(2, N)` while `u` may be `(3, N)`.
"""
function _validate_array_shape(x::AbstractArray, u::AbstractArray, distance_metric)
    ndims(x) >= 2 ||
        throw(DimensionMismatch("x must have shape (Dx, N) or (Dx, N, auxiliary...); got ndims(x)=$(ndims(x))"))
    ndims(u) >= 2 ||
        throw(DimensionMismatch("u must have shape (D, N) or (D, N, auxiliary...); got ndims(u)=$(ndims(u))"))

    D = size(u, 1)
    _validate_spatial_dimension(D)
    W = _val_int(first(_pair_dims(distance_metric, D)))
    size(x, 1) == W || throw(
        DimensionMismatch(
            "this geometry locates a point with $W coordinate(s) on axis 1 of x, but got " *
            "size(x,1)=$(size(x, 1)) (velocity dimension D=$D from size(u,1))",
        ),
    )
    size(u, 2) == size(x, 2) ||
        throw(DimensionMismatch("x and u must share axis-2 point count N; got size(x,2)=$(size(x, 2)) and size(u,2)=$(size(u, 2))"))

    # Classify by array rank. The ternaries make `D` a literal type parameter, so the backend
    # methods that dispatch on `PointField{D}` etc. specialize on it.
    if ndims(x) == 2 && ndims(u) == 2
        return D == 2 ? PointField{2}() : PointField{3}()
    elseif ndims(x) == 2 && ndims(u) >= 3
        return D == 2 ? SharedPositionField{2}() : SharedPositionField{3}()
    elseif ndims(x) >= 3 && ndims(u) >= 3
        ndims(x) == ndims(u) ||
            throw(DimensionMismatch("varying-position inputs must have the same rank; got ndims(x)=$(ndims(x)) and ndims(u)=$(ndims(u))"))
        size(x)[3:end] == size(u)[3:end] ||
            throw(DimensionMismatch("varying-position inputs must share auxiliary axes; got $(size(x)[3:end]) and $(size(u)[3:end])"))
        return D == 2 ? VaryingPositionField{2}() : VaryingPositionField{3}()
    else
        throw(
            DimensionMismatch(
                "unsupported shape combination: x has shape $(size(x)), u has shape $(size(u)); " *
                "valid forms are (Dx,N)/(D,N), (Dx,N)/(D,N,auxiliary...), or matched (Dx,N,auxiliary...) arrays",
            ),
        )
    end
end
