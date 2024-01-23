"""
Generic helper functions for the other routines in this package
"""
module HelperFunctions

using LinearAlgebra: dot, cross, norm, normalize

using StaticArrays: SVector, @SVector

export digitize, δu_longitudinal, δu_l, δu_transverse, δu_t

function digitize(x::Real, bins::Union{Vector{<:Real}, SVector{N2, <:Real}}) where {N2}
    """
    Return the index of the bin that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    Note that the bins are right inclusive, the bins are (a,b]
    """
    searchsortedfirst(bins, x) - 1
end

function digitize(
    x::Union{Vector{<:Real}, SVector{N, <:Real}},
    bins::Union{Vector{<:Real}, SVector{N2, <:Real}},
) where {N, N2}
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(Ref(bins), x)
end

function digitize(x::Union{Vector{<:Real}, SVector{N, <:Real}}, bins::Tuple{<:Real}) where {N}
    """
    Return the indices of the bins that x belongs to
    (see np.digitize and https://discourse.julialang.org/t/find-the-index-of-a-bin-where-a-value-between-two-bin-value/32080/2?u=jbphyswx )
    """
    digitize.(Ref(bins), x)
end




function δr(
    x1::Union{Vector{FT}, SVector{N, FT}},
    x2::Union{Vector{FT2}, SVector{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the vector from  x to y
    """
    return x2 .- x1
end

function r̂(
    x1::Union{Vector{FT}, SVector{N, FT}},
    x2::Union{Vector{FT}, SVector{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the unit vector from  x to y
    """
    return normalize(δr(x1, x2))
end



function δu_longitudinal(
    δu::Union{Vector{FT}, SVector{N, FT}},
    r_hat::Union{Vector{FT2}, SVector{N, FT2}},
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the longitudinal component of u (along the vector)
    Left to the user to ensure r_hat has norm 1
    """
    return dot(δu, r_hat)
end


function δu_transverse(
    δu::Union{Vector{FT}, SVector{N, FT}},
    r_hat::Union{Vector{FT}, SVector{N, FT2}};
    local_unit_vertical = [0, 0, 1],
) where {FT <: Real, FT2 <: Real, N}
    """
    Return the transverse component of u (perpendicular to the vector)
    Left to the user to ensure r_hat (and local_unit_vertical) has norm 1

    -- note, it appears these two methods turned out to be identical -- see if we can simplify...
    """

    if isa(δu, SVector)
        δu = SVector{length(δu)}(δu) # convert to SVector bc cross is defined in 2D and 3D for StaticArrays but only 3D for normal Arrays
    end
    if isa(r_hat, SVector)
        r_hat = SVector{length(r_hat)}(r_hat)
    end

    if isnothing(local_unit_vertical)
        return δu .- δu_longitudinal(δu, r_hat) * r_hat # not sure, i think you're supposed to project onto a direction perpendicular to k
    else
        _cross = cross(δu, r_hat)
        if length(_cross) == 1 # if 2d, staticarrays just returns a scalar.... the k̂ in the third dimension is just 1 so ignore dot...
            return _cross
        else
            return dot(local_unit_vertical, cross(δu, r_hat)) # cross is only defined for 3d vectors --- cross on 2d vectors is implemented in StaticArrays though...
        end
    end
end


δu_l = δu_longitudinal
δu_t = δu_transverse






end # end Module
