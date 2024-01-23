module StructureFunctions # `using StructureFunctions`` should work `@everywhere` automatically... hopefully the methods and extensinos below follow...

using StaticArrays # for speed, SVectors should be useful...
abstract type AbstractStructureFunction end

# using Distributed
# @everywhere include("ParallelCalculations.jl") # this works w/ include("src/StructureFunctions.jl") but not w/ using StructureFunctions, and the former dumps directly into Main...
# using NaNStatistics # consider making this a strong dependency for easier use, needed for parallel but I'm sure we'll need it once we work with real data...

include("HelperFunctions.jl")
include("StructureFunctionTypes.jl")
include("Calculations.jl")


# Define structure function type that holds structure functions of different orders for diffrent radii or radii bins
# the order defines the order of the structure function
# the distance is the distance is either a vector of distances or a vector of tuples of distance bins (if bins are used)

# struct StructureFunction{FT, FT2, BT<:(Union{Vector{FT2}, Vector{Tuple{FT2,FT2}}})} <: StructureFunctionType where {FT<:Real, FT2<:Real} # this FT,FT2 type seems to be ignored, have to enforce in constructor 
# # struct StructureFunction{FT, BT::Vector{FT2}} <: StructureFunctionType where {FT<:Real, FT2<:Real} # this FT,FT2 type seems to be ignored, have to enforce in constructor, see https://discourse.julialang.org/t/nested-parametric-type/58009/2?u=jbphyswx
#     order::Int
#     distance::Union{Vector{FT2}, Vector{Tuple{FT2,FT2}}} # could we make some sort of bintype for this? FT complicates that but...
#     values::Vector{FT}

#     StructureFunction(order, distance, values) = begin
#         @assert length(distance) == length(values)
#         # enforce FT and FT2 types? the where constraints above are just ignored and never enforced in the constructor
#         return new{eltype(values), eltype(eltype(distance)), typeof(distance)}(order, distance, values)
#         # return new{eltype(values), typeof(distance)}(order, distance, values) # I want to use this but no way to make just FT2 part of the struct type without initializing it in the parametric type
#     end
# end

struct StructureFunction{FT, BT <: Vector} <: AbstractStructureFunction where {FT <: Real}
    order::Int
    distance::BT
    values::Vector{FT}

    function StructureFunction(order, distance::Vector{<:Real}, values::Vector{FT}) where {FT}
        @assert length(distance) == length(values)
        return new{FT, typeof(distance)}(order, distance, values)
    end

    function StructureFunction(order, distance::Vector{Tuple{T, T}}, values::Vector{FT}) where {FT} where {T <: Real}
        @assert length(distance) == length(values)
        return new{FT, typeof(distance)}(order, distance, values)
    end
end


# function __init__()
#     # @require Distributed="8ba89e20-285c-5b6f-9357-94700520ee1b" include("ParallelCalculations.jl")
#     @require Distributed="8ba89e20-285c-5b6f-9357-94700520ee1b" begin # maybe this always works because Distributed is in the extension?
#         println(pkgdir(Distributed))
#         include("ParallelCalculations.jl") # This seems to be almost twice as fast as the extension, not sure why...
#     end
# end

end
