"""
Generic helper functions for the other routines in this package
"""
module HelperFunctions

using LinearAlgebra: LinearAlgebra as LA
import LinearAlgebra: normalize # import directly ONLY if extending

using StaticArrays: StaticArrays as SA
@inline create_svector(x) = SA.SVector{length(x)}(x) # helper function to create a static vector from a vector without splatting


export digitize,
    δu_longitudinal, δu_l,
    δu_transverse, δu_t,
    magnitude_δu_longitudinal, mδu_l,
    magnitude_δu_transverse, mδu_t,
    r̂,
    n̂,
    δr

@inline function digitize(x, bins::AbstractVector)
    """
    Return the index of the bin that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    Note that the bins are right inclusive, the bins are (a,b]
    """
    searchsortedfirst(bins, x) - 1
end
@inline function digitize(x::AbstractVector, bins::AbstractVector)
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(x, Ref(bins))
end

@inline function digitize(x::AbstractVector, bins::Tuple)
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(x, Ref(bins))
end



@inline LA.normalize(x::Tuple{T, Vararg{T}}) where {T} = NTuple{length(x), T}(LA.normalize(SA.SVector(x)))

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


@inline function n̂(r_hat::AbstractVector{FT}) where {FT}
    """
    Return the transverse (perpendicular) unit vector given the longitudinal unit vector r_hat.
    In 2D: n̂ = [r_hat[2], -r_hat[1]]
    In 3D: n̂ = normalize(cross(r_hat, k_hat)) where k_hat = [0,0,1]
    """
    ND::Int = length(r_hat)

    if ND == 2
        return LA.normalize(typeof(r_hat)([r_hat[2], -r_hat[1]]))
    elseif ND == 3
        k_hat = SA.@SVector FT[0, 0, 1]
        # Lindberg and Cho defined this order in NH and opposite in SH but we're just doing the same for both
        return LA.normalize(LA.cross(SA.SVector{3, FT}(r_hat...), k_hat))
    else
        error("Only 2D and 3D supported")
    end
end

@inline function n̂(x1, x2)
    """
    Return the transverse (perpendicular) unit vector from  x to y
    Calling this  n̂ is opposite of Lindberg and Cho notation, but idk...
    This is defined as the cross between the unit vector in the vertical direction and the longitudinal unit vector
    """
    return n̂(r̂(x1, x2))
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

@inline function δu_transverse(δu, r_hat)
    """
    Return the transverse component of u (perpendicular to the vector)
    Left to the user to ensure r_hat (and local_unit_vertical) has norm 1

    -- note, it appears these two methods turned out to be identical -- see if we can simplify...
    """

    return δu .- δu_longitudinal(δu, r_hat) # I think this is faster than magnitude_δu_transverse(δu, r_hat) * n̂(δu, r_hat)

end


const δu_l = δu_longitudinal
const δu_t = δu_transverse

const mδu_l = magnitude_δu_longitudinal
const mδu_t = magnitude_δu_transverse






end # end Module
