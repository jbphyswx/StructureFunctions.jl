"""
Generic helper functions for the other routines in this package
"""
module HelperFunctions

using LinearAlgebra: LinearAlgebra as LA
import LinearAlgebra: normalize # import directly ONLY if extending

using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
@inline create_svector(x) = SA.SVector{length(x)}(x) # helper function to create a static vector from a vector without splatting


export digitize,
    δu_longitudinal, δu_l,
    δu_transverse, δu_t,
    magnitude_δu_longitudinal, mδu_l,
    magnitude_δu_transverse, mδu_t,
    transverse_norm2,
    transverse_component_norm2,
    transverse_component,
    transverse_basis,
    transverse_basis_vector,
    AbstractTransverseBasisConvention,
    CanonicalTransverseBasis,
    ReferenceAxisTransverseBasis,
    CoordinateGaugeTransverseBasis,
    UserTransverseBasis,
    r̂,
    n̂,
    δr,
    midpoints,
    flatten_data,
    remove_nans

"""
    flatten_data(vecs::Tuple)

Flatten multi-dimensional arrays in the tuple to vectors. 
Useful for converting grid data to point lists.
"""
function flatten_data(vecs::Tuple)
    return ntuple(i -> vec(vecs[i]), length(vecs))
end

"""
    remove_nans(x_mat::AbstractMatrix, u_mat::AbstractMatrix)

Remove points (columns) where either the position `x_mat` or the velocity `u_mat` 
contains a `NaN`. Returns `(x_mat_clean, u_mat_clean)`.
"""
function remove_nans(x_mat::AbstractMatrix{FT}, u_mat::AbstractMatrix{FT}) where {FT}
    # Find columns that have NO NaNs in either matrix
    nan_in_x = any(isnan, x_mat, dims = 1)
    nan_in_u = any(isnan, u_mat, dims = 1)

    # Combined mask (vec to turn 1xN matrix into N-vector)
    valid_mask = vec(.!(nan_in_x .| nan_in_u))

    return x_mat[:, valid_mask], u_mat[:, valid_mask]
end

@inline function digitize(x, bins::AbstractVector)
    """
    Return the index of the bin that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    Note that the bins are right inclusive, the bins are (a,b]
    """
    searchsortedfirst(bins, x) - 1
end

"""
    midpoints(edges::AbstractVector{<:Number})

Bin midpoints from flat edges `[e₀, e₁, …, eₙ]` (length `n+1` → `n` midpoints).
"""
function midpoints(edges::AbstractVector{T}) where {T}
    n = length(edges) - 1
    return T[(edges[i] + edges[i + 1]) / 2 for i in 1:n]
end

@inline midpoints(edges::SA.SVector{N,T}) where {N,T} = SA.SVector{N-1,T}(ntuple(i -> (edges[i] + edges[i+1]) / 2, N-1))
@inline midpoints(edges::AbstractRange{T}) where {T} = range((first(edges) + last(edges)) / 2, length = length(edges) - 1, step = step(edges))

@inline function digitize(x::AbstractVector, bins::AbstractVector)
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(x, Ref(bins))
end

## `digitize` dispatches on `AbstractBinEdges` via `searchsortedfirst` overrides in `BinEdges.jl`.



@inline LA.normalize(x::Tuple{T, Vararg{T}}) where {T} =
    NTuple{length(x), T}(LA.normalize(SA.SVector(x)))

@inline function δr(x1, x2)
    """
    Return the vector from x1 to x2
    """
    return x2 .- x1
end

@inline function r̂(x1, x2)
    """
    Return the longitudinal (parallel) unit vector from x1 to x2
    """
    return LA.normalize(δr(x1, x2))
end

@inline r̂(x1, x2, ::DI.Euclidean, distance) = δr(x1, x2) / distance
# NOTE: LA.normalize is fast here because the vector is a StaticArray.SVector.
# If dynamic Vectors are ever used, LA.normalize would be ~2.5x slower due to scaling checks.
@inline r̂(x1, x2, ::DI.PreMetric, distance) = LA.normalize(δr(x1, x2))


@inline function n̂(r_hat::AbstractVector{FT}) where {FT}
    """
    Return the transverse (perpendicular) unit vector given the longitudinal unit vector r_hat.
    In 2D: n̂ = [r_hat[2], -r_hat[1]]
    In 3D: n̂ = normalize(cross(r_hat, k_hat)) where k_hat = [0,0,1]
    """
    ND::Int = length(r_hat)

    if ND == 2
        return SA.SVector{2, FT}(r_hat[2], -r_hat[1]) # assume normalized
    elseif ND == 3
        k_hat = SA.SVector{3, FT}(0, 0, 1)
        # Lindberg and Cho defined this order in NH and opposite in SH but we're just doing the same for both
        return LA.normalize(
            LA.cross(SA.SVector{3, FT}(r_hat[1], r_hat[2], r_hat[3]), k_hat),
        )
    else
        error("Only 2D and 3D supported")
    end
end

@inline n̂(r_hat::SA.SVector{2, T}) where {T} = SA.SVector{2, T}(r_hat[2], -r_hat[1])
@inline n̂(r_hat::SA.SVector{3, T}) where {T} = LA.normalize(
    LA.cross(SA.SVector{3, T}(r_hat[1], r_hat[2], r_hat[3]), SA.SVector{3, T}(0, 0, 1)),
)


@inline n̂(r_hat::NTuple{2, T}) where {T} = (r_hat[2], -r_hat[1])
# @inline n̂(r_hat::NTuple{3, T}) where {T} = error("see if we can make this stay in tuple land without any burden") LA.normalize(
    # LA.cross(SA.SVector{3, T}(r_hat[1], r_hat[2], r_hat[3]), SA.SVector{3, T}(0, 0, 1)),
# )


@inline function n̂(x1, x2)
    """
    Return the transverse (perpendicular) unit vector from  x to y
    Calling this  n̂ is opposite of Lindberg and Cho notation, but idk...
    This is defined as the cross between the unit vector in the vertical direction and the longitudinal unit vector
    """
    return n̂(r̂(x1, x2))
end

abstract type AbstractTransverseBasisConvention end

"""
    CanonicalTransverseBasis()

The canonical oriented transverse basis in 2D, equivalent to [`n̂`](@ref).
It is intentionally undefined for 3D because there is no unique oriented
transverse direction without extra information.
"""
struct CanonicalTransverseBasis <: AbstractTransverseBasisConvention end

"""
    ReferenceAxisTransverseBasis(axis; parallel_tol=1e-12)

Project a physical reference axis into the plane perpendicular to `r̂` and
normalize it to get the first transverse basis vector. In 3D, the second
transverse vector is `cross(r̂, e₁)`.

If the reference axis is parallel or nearly parallel to `r̂`, construction of
the per-pair basis throws `ArgumentError`. The tolerance is explicit because
this is a physical convention, not a hidden computational replacement.
"""
struct ReferenceAxisTransverseBasis{A, T} <: AbstractTransverseBasisConvention
    axis::A
    parallel_tol::T
end

ReferenceAxisTransverseBasis(axis; parallel_tol = 1e-12) =
    ReferenceAxisTransverseBasis(axis, parallel_tol)

"""
    CoordinateGaugeTransverseBasis()

An always-defined computational gauge for 3D: choose the coordinate axis least
aligned with `r̂`, project it into the perpendicular plane, and complete a
right-handed basis. This is useful for deterministic component diagnostics,
but the chosen direction can jump discontinuously as `r̂` changes.
"""
struct CoordinateGaugeTransverseBasis <: AbstractTransverseBasisConvention end

"""
    UserTransverseBasis(f)

Use `f(r̂)` as the transverse basis provider. The function must return either
a single unit vector or a tuple of unit vectors perpendicular to `r̂`.
"""
struct UserTransverseBasis{F} <: AbstractTransverseBasisConvention
    basis_function::F
end

@inline function _sum_abs2(x)
    out = zero(eltype(x))
    @inbounds for i in eachindex(x)
        out += x[i] * x[i]
    end
    return out
end

@inline function _as_svector_like(r_hat, x)
    return SA.SVector{length(r_hat), eltype(r_hat)}(ntuple(i -> x[i], length(r_hat)))
end

@inline function _project_reference_axis(axis, r_hat, parallel_tol)
    a = _as_svector_like(r_hat, axis)
    projected = a - LA.dot(a, r_hat) * r_hat
    projected_norm2 = _sum_abs2(projected)
    tol2 = parallel_tol * parallel_tol
    projected_norm2 > tol2 || throw(
        ArgumentError(
            "reference axis is parallel or nearly parallel to r̂; " *
            "choose a different axis or use CoordinateGaugeTransverseBasis()",
        ),
    )
    return projected / sqrt(projected_norm2)
end

@inline function transverse_basis(::CanonicalTransverseBasis, r_hat::SA.SVector{2})
    return (n̂(r_hat),)
end

@inline function transverse_basis(::CanonicalTransverseBasis, r_hat)
    length(r_hat) == 2 || throw(ArgumentError("CanonicalTransverseBasis is only defined in 2D"))
    return (n̂(r_hat),)
end

@inline function transverse_basis(basis::ReferenceAxisTransverseBasis, r_hat::SA.SVector{2})
    e1 = _project_reference_axis(basis.axis, r_hat, basis.parallel_tol)
    return (e1,)
end

@inline function transverse_basis(basis::ReferenceAxisTransverseBasis, r_hat::SA.SVector{3})
    e1 = _project_reference_axis(basis.axis, r_hat, basis.parallel_tol)
    e2 = LA.cross(r_hat, e1)
    return (e1, e2)
end

@inline function transverse_basis(basis::ReferenceAxisTransverseBasis, r_hat)
    D = length(r_hat)
    D == 2 && return transverse_basis(basis, SA.SVector{2, eltype(r_hat)}(r_hat))
    D == 3 && return transverse_basis(basis, SA.SVector{3, eltype(r_hat)}(r_hat))
    throw(ArgumentError("ReferenceAxisTransverseBasis currently supports D=2 or D=3"))
end

@inline function transverse_basis(::CoordinateGaugeTransverseBasis, r_hat::SA.SVector{3, T}) where {T}
    ax = abs(r_hat[1])
    ay = abs(r_hat[2])
    az = abs(r_hat[3])
    axis = ax <= ay && ax <= az ? SA.SVector{3, T}(1, 0, 0) :
           ay <= az ? SA.SVector{3, T}(0, 1, 0) :
           SA.SVector{3, T}(0, 0, 1)
    e1 = axis - LA.dot(axis, r_hat) * r_hat
    e1 = e1 / sqrt(_sum_abs2(e1))
    e2 = LA.cross(r_hat, e1)
    return (e1, e2)
end

@inline function transverse_basis(::CoordinateGaugeTransverseBasis, r_hat::SA.SVector{2})
    return (n̂(r_hat),)
end

@inline function transverse_basis(basis::CoordinateGaugeTransverseBasis, r_hat)
    D = length(r_hat)
    D == 2 && return transverse_basis(basis, SA.SVector{2, eltype(r_hat)}(r_hat))
    D == 3 && return transverse_basis(basis, SA.SVector{3, eltype(r_hat)}(r_hat))
    throw(ArgumentError("CoordinateGaugeTransverseBasis currently supports D=2 or D=3"))
end

@inline transverse_basis(basis::UserTransverseBasis, r_hat) = basis.basis_function(r_hat)

@inline function transverse_basis_vector(r_hat, basis::AbstractTransverseBasisConvention, basis_index::Integer = 1)
    basis_vectors = transverse_basis(basis, r_hat)
    1 <= basis_index <= length(basis_vectors) ||
        throw(BoundsError(basis_vectors, basis_index))
    return basis_vectors[basis_index]
end


@inline function magnitude_δu_longitudinal(δu, r_hat)
    """
    Return the longitudinal component of u (along the vector)
    Left to the user to ensure r_hat has norm 1
    """
    return LA.dot(δu, r_hat) # r_hat is unit vector so just dot product
end

@inline function δu_longitudinal(δu, r_hat)
    """
    Return the longitudinal component of u (along the vector)
    Left to the user to ensure r_hat has norm 1
    """
    return magnitude_δu_longitudinal(δu, r_hat) * r_hat
end

@inline function magnitude_δu_transverse(δu, r_hat)
    """
    Return the magnitude of the transverse component of u (perpendicular to the vector) relative to the normal vector...
    Left to the user to ensure r_hat has norm 1
    """
    # This instead of  LA.norm(δu .- δu_longitudinal(δu, r_hat)) because we want the signed magnitude relative to the normal vector...
    return LA.dot(δu, n̂(r_hat))
end

@inline function magnitude_δu_transverse(
    δu,
    r_hat,
    basis::AbstractTransverseBasisConvention,
    basis_index::Integer = 1,
)
    return transverse_component(δu, r_hat, basis, basis_index)
end

@inline function δu_transverse(δu, r_hat)
    """
    Return the transverse component of u (perpendicular to the vector)
    Left to the user to ensure r_hat (and local_unit_vertical) has norm 1

    -- note, it appears these two methods turned out to be identical -- see if we can simplify...
    """

    return δu .- δu_longitudinal(δu, r_hat) # I think this is faster than magnitude_δu_transverse(δu, r_hat) * n̂(δu, r_hat)

end

"""
    transverse_norm2(δu, r̂)

Squared norm of the full transverse vector ``δu - (δu⋅r̂)r̂``. This is the
invariant transverse energy used by `T2SF` and `L1T2SF`.
"""
@inline function transverse_norm2(δu, r_hat)
    du_l = magnitude_δu_longitudinal(δu, r_hat)
    return _sum_abs2(δu) - du_l * du_l
end

"""
    transverse_component_norm2(δu, r̂)

Per-component transverse energy under isotropic transverse averaging,
``||δu_t||² / (D - 1)``.
"""
@inline function transverse_component_norm2(δu, r_hat)
    D = length(r_hat)
    D > 1 || throw(ArgumentError("transverse_component_norm2 requires D >= 2"))
    return transverse_norm2(δu, r_hat) / (D - 1)
end

"""
    transverse_component(δu, r̂, basis, basis_index=1)

Component of `δu` along an explicit transverse basis vector. This is the
quantity to use for basis-dependent odd/component diagnostics such as `T3SF`
or `L2T1SF`.
"""
@inline function transverse_component(
    δu,
    r_hat,
    basis::AbstractTransverseBasisConvention,
    basis_index::Integer = 1,
)
    e = transverse_basis_vector(r_hat, basis, basis_index)
    return LA.dot(δu, e)
end


const δu_l = δu_longitudinal
const δu_t = δu_transverse

const mδu_l = magnitude_δu_longitudinal
const mδu_t = magnitude_δu_transverse






end # end Module
