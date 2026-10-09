"""
    AbstractFieldShape{D}

Which array-rank pattern a validated `(x, u)` pair forms, with `D` the number of velocity components
on axis 1 of `u`. Backends dispatch on this, so `D` reaches the kernels as a static method parameter.

- `PointField`: `u` is `(D, N)`.
- `SharedPositionField`: `u` is `(D, N, auxiliary...)` — one set of positions, many fields.
- `VaryingPositionField`: `x` and `u` both carry matching auxiliary axes.

The coordinate count on axis 1 of `x` is fixed by the metric's geometry (see
`HelperFunctions.coordinate_width`).
"""
abstract type AbstractFieldShape{D} end
struct PointField{D} <: AbstractFieldShape{D} end
struct SharedPositionField{D} <: AbstractFieldShape{D} end
struct VaryingPositionField{D} <: AbstractFieldShape{D} end

"""Velocity components on axis 1 of `u`."""
@inline spatial_dimension(::AbstractFieldShape{D}) where {D} = D

"""
    BatchLeading(u)

Wrap a velocity/position array stored **batch-leading**, shape `(B, D, N)` (batch axis
innermost/contiguous). CPU batch kernels run on it zero-copy. A plain `(D, N, B...)` array is
transposed once internally.
"""
struct BatchLeading{A <: AbstractArray}
    data::A
end

"""Position/velocity input a batch driver accepts: a plain `(D, N, B…)` array, or a
[`BatchLeading`](@ref) wrapper around a `(B, D, N)` one."""
const BatchInput = Union{AbstractArray, BatchLeading}

"""Batch input in the `(D, N, B…)` layout of the shape contract; a batch-leading `(B, D, N)` one is viewed, not copied."""
_contract_layout(a::AbstractArray) = a
_contract_layout(a::BatchLeading) = PermutedDimsArray(a.data, (2, 3, 1))

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

"""
    _validate_spatial_dimension(D)

Check the velocity dimension on axis 1 of `u`.

The isotropic invariants are built from `δu_L` and `‖δu‖²`, both defined for any `D ≥ 1`.
"""
function _validate_spatial_dimension(D::Integer)
    D >= 1 ||
        throw(DimensionMismatch("expected a velocity dimension D >= 1 on axis 1; got D=$D"))
    return nothing
end

@inline _val_int(::Val{W}) where {W} = W

"""
    _shape_kind(x, u) -> Type

Which shape family the inputs form, from their **ranks** alone: `ndims` is a property of the array type, so this
constant-folds.
"""
@inline _shape_kind(x::AbstractArray, u::AbstractArray) =
    ndims(x) == 2 && ndims(u) == 2 ? PointField :
    ndims(x) == 2 ? SharedPositionField : VaryingPositionField

"""
    _by_width(g, D)

`g(Val(D))` for a velocity width `D` of at least 1, through one call at run time ([`_at_width`](@ref)): a call compiles
`g` at its own width only. The result has the type [`_width_result_type`](@ref) gives, which no width changes.
"""
@inline function _by_width(g, D::Int)
    _validate_spatial_dimension(D)
    return _at_width(g, Val(D))::_width_result_type(g)
end

"""The type `g(Val(W))` returns at the first width `W` of 2, 3 and 1 at which it returns."""
@inline function _width_result_type(g)
    T = Base.promote_op(g, Val{2})
    T === Union{} || return T
    T = Base.promote_op(g, Val{3})
    T === Union{} || return T
    return Base.promote_op(g, Val{1})
end

"""`g(vD)` at a width read from array sizes: the one dynamic dispatch on such a width ([`_by_width`](@ref))."""
_at_width(g, vD::Val) = g(vD)

"""
    _shaped(f, S, D, distance_metric)

`f(shape, geometry)` with the field shape `S{D}()` of velocity width `D` and the pair geometry of `distance_metric` at
that width, both concretely typed ([`_by_width`](@ref)).
"""
@inline _shaped(f, ::Type{S}, D::Int, distance_metric) where {S} =
    _by_width(vD -> f(S{_val_int(vD)}(), SFH.pair_geometry_for(distance_metric, vD)), D)

"""The width at which the CPU SIMD pair kernels run `geometry`: flat coordinates of width 2 or 3, else `nothing`."""
@inline _simd_width(::SFH.FlatGeometry{2}) = Val(2)
@inline _simd_width(::SFH.FlatGeometry{3}) = Val(3)
@inline _simd_width(_) = nothing

"""
    _validate_array_shape(x, u, distance_metric)

Throw unless `x` and `u` form one of the shapes of [`AbstractFieldShape`](@ref) under `distance_metric`. Axis 1 of `u`
is the velocity dimension `D`; axis 1 of `x` is however many coordinates the metric's geometry needs to locate a point,
which is not always `D`: on a sphere a point takes two coordinates whether or not the velocity carries a third, radial,
component, so `x` is `(2, N)` while `u` may be `(3, N)`.
"""
function _validate_array_shape(x::AbstractArray, u::AbstractArray, distance_metric)
    ndims(x) >= 2 ||
        throw(DimensionMismatch("x must have shape (Dx, N) or (Dx, N, auxiliary...); got ndims(x)=$(ndims(x))"))
    ndims(u) >= 2 ||
        throw(DimensionMismatch("u must have shape (D, N) or (D, N, auxiliary...); got ndims(u)=$(ndims(u))"))

    D = size(u, 1)
    W = _shaped((_, geometry) -> _val_int(SFH.input_coordinate_width(geometry)), PointField, D, distance_metric)
    size(x, 1) == W || throw(
        DimensionMismatch(
            "this geometry locates a point with $W coordinate(s) on axis 1 of x, but got " *
            "size(x,1)=$(size(x, 1)) (velocity dimension D=$D from size(u,1))",
        ),
    )
    size(u, 2) == size(x, 2) ||
        throw(DimensionMismatch("x and u must share axis-2 point count N; got size(x,2)=$(size(x, 2)) and size(u,2)=$(size(u, 2))"))

    if ndims(x) == 2
        return nothing
    elseif ndims(u) >= 3
        ndims(x) == ndims(u) ||
            throw(DimensionMismatch("varying-position inputs must have the same rank; got ndims(x)=$(ndims(x)) and ndims(u)=$(ndims(u))"))
        size(x)[3:end] == size(u)[3:end] ||
            throw(DimensionMismatch("varying-position inputs must share auxiliary axes; got $(size(x)[3:end]) and $(size(u)[3:end])"))
        return nothing
    else
        throw(
            DimensionMismatch(
                "unsupported shape combination: x has shape $(size(x)), u has shape $(size(u)); " *
                "valid forms are (Dx,N)/(D,N), (Dx,N)/(D,N,auxiliary...), or matched (Dx,N,auxiliary...) arrays",
            ),
        )
    end
end
