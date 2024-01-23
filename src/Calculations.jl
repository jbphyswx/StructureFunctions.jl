"""
The workhorse of this package
"""
module Calculations
import ProgressMeter: @showprogress
import Distances # from JuliaStats
import ..HelperFunctions

using StaticArrays
using LinearAlgebra

using SharedArrays

using ..StructureFunctionTypes # how can we make these natively available?
# print(eval(Symbol("StructureFunctionTypes.SecondOrderStructureFunction"))) # doesn't work

# using LoopVectorization # can't use w/ [bin] indexing below, see https://github.com/JuliaSIMD/LoopVectorization.jl/issues/331

export calculate_structure_function, parallel_calculate_structure_function


###########
# Add methods for dispatching vectors/tuples to the static SVector versions...
########

"""
Consider using some sort of Intervals thingy for the intervals
    worked on arrays of size 1e4 in 12 seconds, was about a true half and half timewise, so w/ saved bins would be about 6 seconds...
    note we still need to turn values to vectors for u,v,w, etc and have a function for the SF calculation that's not just order...
    and we need a parallel version, maybe parallelize over the `i` loop, aggregate the results separately, then bin and average from the workers...
    - probably shouldn't use sharedarrays as those can possibly fail w/ concurrent reads/writes
"""
function calculate_structure_function(
    x_vecs::NTuple{N, <:Union{SharedVector{FT1}, SVector{N2, FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{SharedVector{FT2}, SVector{N2, FT2}}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins::SVector{N3, Tuple{FT3, FT3}},
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType; # add N here?
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1 <: Real, FT2 <: Real, FT3 <: Real, N, N2, N3}
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    output = zeros(N3)
    counts = zeros(N3)

    distance_bins_vec = SVector{length(distance_bins) + 1, FT3}(
        [[distance_bin[1] for distance_bin in distance_bins]; [distance_bins[end][2]]]...,
    ) # the start of each bin plus the ending

    if verbose
        @info("calculating structure function")
    end


    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    @showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _output, _counts = calculate_structure_function_i(
            i,
            x_vecs,
            u_vecs,
            distance_bins_vec,
            structure_function_type;
            distance_metric = distance_metric,
        )
        output .+= _output
        counts .+= _counts

        # X1 = SVector{N,FT1}((x_vec[i] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
        # U1 = SVector{N,FT2}((u_vec[i] for u_vec in u_vecs)...)
        # @simd for j in iter_inds
        #     if i != j # this has δx and δu = 0, so we skip it
        #         X2 = SVector{N,FT1}((x_vec[j] for x_vec in x_vecs)...)
        #         U2 = SVector{N,FT2}((u_vec[j] for u_vec in u_vecs)...)

        #         @inbounds distance = distance_metric(X1, X2) # this is the slow part
        #         bin = HelperFunctions.digitize(distance, distance_bins_vec) # this will fail when the the distance value equals the smallest bin's smallest edge due to the way digitize works, we could add a handler here but we extended the smallest bin to be the next smallest float so this shouldn't happen
        #         @inbounds output[bin] += structure_function_type.method(U2 - U1, HelperFunctions.r̂(X1, X2))
        #         @inbounds counts[bin] += 1
        #     end
        # end
    end

    if return_sums_and_counts # just return the sums and the counts, don't take the mean in each bin...
        return (output, counts), distance_bins
    else # do the mean in each bin.
        counts[counts .== 0] .= NaN # replace 0s with NaNs to avoid divide by 0 giving Inf
        output ./= counts
        return output, distance_bins
    end
end



function calculate_structure_function_i(
    i::Int,
    x_vecs::NTuple{N, <:Union{SharedVector{FT1}, SVector{N2, FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{SharedVector{FT2}, SVector{N2, FT2}}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins_vec::SVector{N3, FT3},
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType; # add N here?
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
) where {FT1 <: Real, FT2 <: Real, FT3 <: Real, N, N2, N3}


    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    output = zeros(N3 - 1)
    counts = zeros(N3 - 1)


    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    X1 = SVector{N, FT1}((x_vec[i] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
    U1 = SVector{N, FT2}((u_vec[i] for u_vec in u_vecs)...)
    @simd for j in iter_inds
        if i != j # this has δx and δu = 0, so we skip it
            X2 = SVector{N, FT1}((x_vec[j] for x_vec in x_vecs)...)
            U2 = SVector{N, FT2}((u_vec[j] for u_vec in u_vecs)...)

            @inbounds distance = distance_metric(X1, X2) # this is the slow part
            bin = HelperFunctions.digitize(distance, distance_bins_vec) # this will fail when the the distance value equals the smallest bin's smallest edge due to the way digitize works, we could add a handler here but we extended the smallest bin to be the next smallest float so this shouldn't happen
            @inbounds output[bin] += structure_function_type.method(U2 - U1, HelperFunctions.r̂(X1, X2))
            @inbounds counts[bin] += 1
        end
    end
    return output, counts
end


function calculate_structure_function(
    x_vecs::NTuple{N, <:Union{SharedVector{FT1}, SVector{N2, FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{SharedVector{FT2}, SVector{N2, FT2}}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins::Int,
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType;
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
    bin_spacing = :logarithmic,
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1 <: Real, FT2 <: Real, N, N2} # You can't dispatch on N, N2 since they're Ints not types
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.

    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    @showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _min_distance, _max_distance = minmax_i(i, x_vecs, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)

        # X1 = SVector{N,FT1}((x_vec[i] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
        # @simd for j in iter_inds
        #     if i != j # don't compare to self (will add a distance of 0 to our distances)
        #         X2 = SVector{N,FT1}((x_vec[j] for x_vec in x_vecs)...)
        #         # update the min and max distances
        #         @inbounds distance = distance_metric(X1, X2) # this is the slow part
        #         if distance < min_distance
        #             min_distance = distance
        #         elseif distance > max_distance
        #             max_distance = distance
        #         end
        #     end
        # end
    end

    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin (note this is needed so things matching min_distance don't get assigned bin '0', alternative is to check for bin 0 every time... which sounds slower
    if bin_spacing == :linear
        distance_bins = range(min_distance, max_distance, length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
    elseif bin_spacing ∈ (:logarithmic, :log)
        distance_bins = 10 .^ range(log10(min_distance), log10(max_distance), length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
        distance_bins[1] = min_distance # combat floating point errors (may have rounded up or down during operations)
        distance_bins[end] = max_distance # combat floating point errors (may have rounded up or down during operations)
    else
        error("bin_spacing must be :linear or :logarithmic/:log")
    end
    FT3 = eltype(distance_bins)
    distance_bins = SVector{n_distance_bins, Tuple{FT3, FT3}}(
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins]...,
    ) # convert to tuples of the bin edges
    return calculate_structure_function(
        x_vecs,
        u_vecs,
        distance_bins,
        structure_function_type;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        return_sums_and_counts = return_sums_and_counts,
    )
end


function minmax_i(
    i::Int,
    x_vecs::NTuple{N, <:Union{SharedVector{FT}, SVector{N2, FT}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    distance_metric = Distances.Euclidean(),
) where {FT <: Real, N, N2} # You can't dispatch on N, N2 since they're Ints not types
    """ calculate, bin, and mean the pairwise distances """


    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = SVector{N, FT}((x_vec[i] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
    for j in eachindex(x_vecs[1])
        if i != j
            @inbounds X2 = SVector{N, FT}((x_vec[j] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
            # update the min and max distances
            distance = distance_metric(X1, X2) # this is the slow part
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end





function parallel_calculate_structure_function end # placeholder for parallel extension


end
