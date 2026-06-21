# Array shape contract for public calculation APIs.

abstract type AbstractFieldShape{D} end
struct PointField{D} <: AbstractFieldShape{D} end
struct SharedPositionField{D} <: AbstractFieldShape{D} end
struct VaryingPositionField{D} <: AbstractFieldShape{D} end

@inline spatial_dimension(::AbstractFieldShape{D}) where {D} = D
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

function _validate_array_shape(x::AbstractArray, u::AbstractArray)
    ndims(x) >= 2 ||
        throw(DimensionMismatch("x must have shape (D, N) or (D, N, auxiliary...); got ndims(x)=$(ndims(x))"))
    ndims(u) >= 2 ||
        throw(DimensionMismatch("u must have shape (D, N) or (D, N, auxiliary...); got ndims(u)=$(ndims(u))"))

    D = size(x, 1)
    _validate_spatial_dimension(D)
    size(u, 1) == D ||
        throw(DimensionMismatch("x and u must share axis-1 spatial dimension D; got size(x,1)=$D and size(u,1)=$(size(u, 1))"))
    size(u, 2) == size(x, 2) ||
        throw(DimensionMismatch("x and u must share axis-2 point count N; got size(x,2)=$(size(x, 2)) and size(u,2)=$(size(u, 2))"))

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
                "valid forms are (D,N)/(D,N), (D,N)/(D,N,auxiliary...), or matched (D,N,auxiliary...) arrays",
            ),
        )
    end
end
