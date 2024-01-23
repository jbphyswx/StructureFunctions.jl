"""
The workhorse of this package
"""

module ParallelCalculations
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###

using Distributed
using ProgressMeter: @showprogress
include("HelperFunctions.jl") # We need to load these because this has to happen on all workers in the parallel case... (idk how to make this submodule a separate module though...)
include("Calculations.jl")
# import ..StructureFunction # ideally
import Distances # from JuliaStats
import NaNStatistics
using StaticArrays
using LinearAlgebra

import ..Calculations
import ..HelperFunctions

export parallel_calculate_structure_function #consider changing this to just calculate_structure function to match the non-parallel name

@info(@__DIR__)


@info("On Not Ext: ", workers()) # This shows this required code is reaching all workers


# import Distances
# import NaNMath
# import ..Calculations
# import ..HelperFunctions
# thisdir = @_
# @info(Calculations)

"""
Consider using some sort of Intervals thingy for the intervals
    worked on arrays of size 1e4 in 12 seconds, was about a true half and half timewise, so w/ saved bins would be about 6 seconds...
    note we still need to turn values to vectors for u,v,w, etc and have a function for the SF calculation that's not just order...
    and we need a parallel version, maybe parallelize over the `i` loop, aggregate the results separately, then bin and average from the workers...
    - probably shouldn't use sharedarrays as those can possibly fail w/ concurrent reads/writes

    parallel version took 4 seconds, 2 for bin calculation, 2 for SF w/ 5 workers (was 12 w/o parallel so 3x speedup)
"""
function parallel_calculate_structure_function_i(
    i::Int,
    x_vecs::NTuple{N, SVector{N2, FT}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, SVector{N2, FT}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins_vec::SVector{N3, FT2}; # add N here?
    order::Int = 2,
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
) where {FT <: Real, FT2 <: Real, N, N2, N3}

    # preallocate output as vector of length of distance_bins
    output = zeros(length(distance_bins_vec) - 1)
    counts = zeros(length(distance_bins_vec) - 1)

    @inbounds X1 = SVector{N, FT}((x_vec[i] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
    @inbounds U1 = SVector{N, FT}((u_vec[i] for u_vec in u_vecs)...)

    for j in eachindex(x_vecs[1])
        @inbounds X2 = SVector{N, FT}((x_vec[j] for x_vec in x_vecs)...)
        @inbounds U2 = SVector{N, FT}((u_vec[j] for u_vec in u_vecs)...)

        # find the bin that this distance belongs to
        distance = distance_metric(X1, X2) # this is the slow part
        bin = HelperFunctions.digitize(distance, distance_bins_vec) # this will fail when the the distance value equals the largest bin's largest edge due to the way digitize works, we could add a handler here to put that value back into the largest bin...
        output[bin] += norm((U2 - U1))^order # replace w/ actual function
        counts[bin] += 1
    end
    counts[counts .== 0] .= NaN
    return output, counts
end

function parallel_calculate_structure_function(
    x_vecs::NTuple{N, SVector{N2, FT}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, SVector{N2, FT}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins::SVector{N3, Tuple{FT2, FT2}}; # add N here?
    order::Int = 2,
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
) where {FT <: Real, FT2 <: Real, N, N2, N3}

    distance_bins_vec = SVector{length(distance_bins) + 1, FT2}(
        [[distance_bin[1] for distance_bin in distance_bins]; [distance_bins[end][2]]]...,
    ) # the start of each bin plus the ending

    outputs_counts = @showprogress pmap(
        i -> parallel_calculate_structure_function_i(
            i,
            x_vecs,
            u_vecs,
            distance_bins_vec;
            order = order,
            distance_metric = distance_metric,
        ),
        eachindex(x_vecs[1]),
    )

    # unzip output
    @inbounds outputs, counts = ([x[1] for x in outputs_counts], [x[2] for x in outputs_counts])

    outputs = mapreduce(permutedims, vcat, outputs) # turn vector of vectors to matrix
    counts = mapreduce(permutedims, vcat, counts)

    output = NaNStatistics.nansum(outputs, dims = 1)
    counts = NaNStatistics.nansum(counts, dims = 1)
    output ./= counts
    return output, distance_bins
end



"""
Function to the bins and and i value and calculate the outputs and counts for that bin 
"""

function parallel_calculate_structure_function(
    x_vecs::NTuple{N, SVector{N2, FT}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, SVector{N2, FT}},
    distance_bins::Int;
    order::Int = 2,
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
) where {FT <: Real, N, N2} # You can't dispatch on N, N2 since they're Ints not types
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """

    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.


    # calculate the bins
    @info("Calculating min and max distances and generating bins")

    minmax_distances = @showprogress pmap(i -> minmax_i(i, x_vecs, distance_metric), eachindex(x_vecs[1]))
    minvals, maxvals = ([x[1] for x in minmax_distances], [x[2] for x in minmax_distances])
    min_distance = minimum(minvals)
    max_distance = maximum(maxvals)

    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin
    distance_bins = range(min_distance, max_distance, length = distance_bins + 1) # +1 to get the right number of bins since these are the edge
    FT2 = eltype(distance_bins)
    distance_bins = SVector{n_distance_bins, Tuple{FT2, FT2}}(
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins]...,
    ) # convert to tuples of the bin edges

    @info("calculating structure function")
    return parallel_calculate_structure_function(
        x_vecs,
        u_vecs,
        distance_bins;
        order = order,
        distance_metric = distance_metric,
    )
end


function minmax_i(
    i::Int,
    x_vecs::NTuple{N, SVector{N2, FT}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    distance_metric = Distances.Euclidean(),
) where {FT <: Real, N, N2} # You can't dispatch on N, N2 since they're Ints not types
    """ calculate, bin, and mean the pairwise distances """

    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = SVector{N, FT}((x_vec[i] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
    for j in eachindex(x_vecs[1])
        @inbounds X2 = SVector{N, FT}((x_vec[j] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
        # update the min and max distances
        distance = distance_metric(X1, X2) # this is the slow part
        if distance < min_distance
            min_distance = distance
        elseif distance > max_distance
            max_distance = distance
        end
    end
    return min_distance, max_distance
end

### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###
### NOTE: This is deprecated, and we're trying to rely on the extension instead for parallel capability ###


end
