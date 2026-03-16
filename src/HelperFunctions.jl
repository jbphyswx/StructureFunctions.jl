"""
Generic helper functions for the other routines in this package
"""
module HelperFunctions

using LinearAlgebra: dot, cross, norm, normalize #, normalize! # would noormalize! work for reducing allocations? (no bc it returns nothing anyway)
import LinearAlgebra: normalize # import direclty so can be extended for tuples

using StaticArrays: SVector, @SVector

create_svector(x) = SVector{length(x)}(x) # helper function to create a static vector from a vector without splatting


export digitize,
    δu_longitudinal, δu_l,
    δu_transverse, δu_t,
    magnitude_δu_longitudinal, mδu_l,
    magnitude_δu_transverse, mδu_t,
    r̂,
    n̂,
    δr

@inline function digitize(x::Real, bins::Union{AbstractVector{<:Real}, SVector{N2, <:Real}}) where {N2}
    """
    Return the index of the bin that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    Note that the bins are right inclusive, the bins are (a,b]
    """
    searchsortedfirst(bins, x) - 1
end
@inline function digitize(
    x::Union{AbstractVector{<:Real}, SVector{N, <:Real}},
    bins::Union{AbstractVector{<:Real}, SVector{N2, <:Real}},
) where {N, N2}
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(x, Ref(bins))
end

@inline function digitize(x::Union{AbstractVector{<:Real}, SVector{N, <:Real}}, bins::Tuple{Real}) where {N}
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(x, Ref(bins))
end



normalize(x::NTuple{N, T}) where {N, T} = NTuple{N, T}(normalize(SVector(x...))) # normalize doesnt work for tuples so use this

@inline function δr(
    x1::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
    x2::Union{AbstractVector{FT2}, SVector{N, FT2}, NTuple{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the vector from  x to y
    """
    return x2 .- x1
end

@inline function r̂(
    x1::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
    x2::Union{AbstractVector{FT}, SVector{N, FT2}, NTuple{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the longitudinal (parallel) unit vector from  x to y
    """
    return normalize( δr(x1, x2)) # note this doesnt work with tuple.... can't divide...
end


@inline function n̂(
    r_hat::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
) where {FT <: Real, N}
    """
    Return the transverse (perpendicular) unit vector given the longitudinal unit vector r_hat.
    In 2D: n̂ = [r_hat[2], -r_hat[1]]
    In 3D: n̂ = normalize(cross(r_hat, k_hat)) where k_hat = [0,0,1]
    """
    ND::Int = length(r_hat)

    if ND == 2
        return normalize(typeof(r_hat)([r_hat[2], -r_hat[1]]))
    elseif ND == 3
        k_hat = @SVector FT[0, 0, 1]
        # Lindberg and Cho defined this order in NH and opposite in SH but we're just doing the same for both
        return normalize(cross(SVector{3, FT}(r_hat...), k_hat))
    else
        error("Only 2D and 3D supported")
    end
end

@inline function n̂(
    x1::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
    x2::Union{AbstractVector{FT}, SVector{N, FT2}, NTuple{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the transverse (perpendicular) unit vector from  x to y
    Calling this  n̂ is opposite of Lindberg and Cho notation, but idk...
    This is defined as the cross between the unit vector in the vertical direction and the longitudinal unit vector
    """
    return n̂(r̂(x1, x2))
end


@inline function magnitude_δu_longitudinal(
    δu::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
    r_hat::Union{AbstractVector{FT2}, SVector{N, FT2}, NTuple{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the longitudinal component of u (along the vector)
    Left to the user to ensure r_hat has norm 1
    """
    return dot(δu, r_hat) # r_hat is unit vector so just dot product
end

@inline function δu_longitudinal(
    δu::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
    r_hat::Union{AbstractVector{FT2}, SVector{N, FT2}, NTuple{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the longitudinal component of u (along the vector)
    Left to the user to ensure r_hat has norm 1
    """
    return magnitude_δu_longitudinal(δu, r_hat) * r_hat
end

@inline function magnitude_δu_transverse(
    δu::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
    r_hat::Union{AbstractVector{FT2}, SVector{N, FT2}, NTuple{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the magnitude of the transverse component of u (perpendicular to the vector) relative to the normal vector...
    Left to the user to ensure r_hat has norm 1
    """
    # This instead of  norm(δu .- δu_longitudinal(δu, r_hat)) because we want the signed magnitude relative to the normal vector...
    return dot(δu, n̂(r_hat)) 
end

@inline function δu_transverse(
    δu::Union{AbstractVector{FT}, SVector{N, FT}, NTuple{N, FT}},
    r_hat::Union{AbstractVector{FT}, SVector{N, FT2},   NTuple{N, FT2}},;
    # local_unit_vertical = [0, 0, 1],
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the transverse component of u (perpendicular to the vector)
    Left to the user to ensure r_hat (and local_unit_vertical) has norm 1

    -- note, it appears these two methods turned out to be identical -- see if we can simplify...
    """

    return δu .- δu_longitudinal(δu, r_hat) # I think this is faster than magnitude_δu_transverse(δu, r_hat) * n̂(δu, r_hat)

end


δu_l = δu_longitudinal
δu_t = δu_transverse

mδu_l = magnitude_δu_longitudinal
mδu_t = magnitude_δu_transverse






end # end Module
