"""
The workhorse of this package
For speed, pretty much all your inputs seems to need to be SharedArrays...
"""
module ParallelCalculationsExt
using Distributed: Distributed
using ProgressMeter: ProgressMeter as PM
import Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using SharedArrays: SharedArrays as ShA
import StructureFunctions as SF # Use alias to avoid name conflict if any

const Calculations = SF.Calculations
const HelperFunctions = SF.HelperFunctions
const StructureFunctionTypes = SF.StructureFunctionTypes


export parallel_calculate_structure_function 

function Calculations.parallel_calculate_structure_function(
    x_vecs::NTuple{N, <:Union{ShA.SharedVector{FT1}, SA.SVector{N2, FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{ShA.SharedVector{FT2}, SA.SVector{N2, FT2}}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins::SA.SVector{N3, Tuple{FT3, FT3}},
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType; # add N here?
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1, FT2, FT3, N, N2, N3}


    distance_bins_vec = SA.SVector{length(distance_bins) + 1, FT3}(
        [[distance_bin[1] for distance_bin in distance_bins]; [distance_bins[end][2]]]...,
    ) # the start of each bin plus the ending

    if verbose
        @info("calculating structure function")
    end

    output, counts =
        PM.@showprogress enabled = show_progress Distributed.@distributed ((x, y) -> ((x[1] .+ y[1]), (x[2] .+ y[2]))) for i in
                                                                                                            eachindex(
            x_vecs[1],
        )
            Calculations.calculate_structure_function_i(
                i,
                x_vecs,
                u_vecs,
                distance_bins_vec,
                structure_function_type;
                distance_metric = distance_metric,
            )
        end

    if return_sums_and_counts
        return (output, counts), distance_bins
    else
        counts[counts .== 0] .= NaN # replace 0s with NaNs to avoid divide by 0 giving Inf
        output ./= counts
        return output, distance_bins
    end
end


function Calculations.parallel_calculate_structure_function(
    x_vecs::NTuple{N, <:Union{ShA.SharedVector{FT1}, SA.SVector{N2, FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{ShA.SharedVector{FT2}, SA.SVector{N2, FT2}}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins::Int,
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing = :logarithmic,
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1, FT2, N, N2}
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """


    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.

    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end


    min_distance, max_distance =
        PM.@showprogress enabled = show_progress Distributed.@distributed ((x, y) -> (min(x[1], y[1]), max(x[2], y[2]))) for i in
                                                                                                              eachindex(
            x_vecs[1],
        )
            Calculations.minmax_i(i, x_vecs, distance_metric)
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
    distance_bins = SA.SVector{n_distance_bins, Tuple{FT3, FT3}}(
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins]...,
    ) # convert to tuples of the bin edges


    return Calculations.parallel_calculate_structure_function(
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

end
