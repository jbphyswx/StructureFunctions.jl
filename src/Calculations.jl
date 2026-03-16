"""
The workhorse of this package
"""
module Calculations
import ProgressMeter: ProgressMeter as PM
import Distances: Distances as DI
import ..HelperFunctions: HelperFunctions as HF
import ..StructureFunctionTypes: StructureFunctionTypes as SFT

using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using SharedArrays: SharedArrays as ShA
using Base.Threads: Threads
using LoopVectorization: LoopVectorization as LV

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
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, ShA.SharedVector{FT1}, SA.SVector{N2, FT1}, AbstractVector{FT1}}},
    u_vecs::NTuple{N, <:Union{NTuple{N2, FT2}, ShA.SharedVector{FT2}, SA.SVector{N2, FT2}, AbstractVector{FT2}}},
    distance_bins::SA.SVector{N3, Tuple{FT3, FT3}},
    structure_function_type::SFT.AbstractStructureFunctionType;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1, FT2, FT3, N, N2, N3}
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    # preallocate output as vector of length of distance_bins
    # Use promote_type and float to ensure output can hold float results/NaN
    OT = promote_type(float(FT1), float(FT2)) 
    output = zeros(OT, N3)
    counts = zeros(OT, N3)

    distance_bins_vec = SA.SVector{length(distance_bins) + 1, FT3}(
        [[distance_bin[1] for distance_bin in distance_bins]; [distance_bins[end][2]]]...,
    ) # the start of each bin plus the ending

    if verbose
        @info("calculating structure function")
    end

    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
    Threads.@threads for i::Int64 in iter_inds # is this the fast order?
    # @threads for i::Int64 in 1:length(x_vecs[1])::Int64 # is this the fast order?
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
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, ShA.SharedVector{FT1}, SA.SVector{N2, FT1}, AbstractVector{FT1}}},
    u_vecs::NTuple{N, <:Union{NTuple{N2, FT2}, ShA.SharedVector{FT2}, SA.SVector{N2, FT2}, AbstractVector{FT2}}},
    distance_bins_vec::SA.SVector{N3, FT3},
    structure_function_type::SFT.AbstractStructureFunctionType;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {FT1, FT2, FT3, N, N2, N3}



    # Commit A2: Fixed "double calculating" by iterating only over unique pairs (j > i).
    # Removed BadImplementationError as this path is now considered correct.

    # preallocate output as vector of length of distance_bins
    OT = promote_type(float(FT1), float(FT2))
    output = zeros(OT, N3 - 1)
    counts = zeros(OT, N3 - 1)

    iter_inds = eachindex(x_vecs[1]) 
    X1 = SA.SVector{N, FT1}((x_vec[i] for x_vec in x_vecs)...)
    U1 = SA.SVector{N, FT2}((u_vec[i] for u_vec in u_vecs)...)
    
    # Iterate only over unique pairs where j > i to avoid double calculation
    LV.@simd for j in (i+1):last(iter_inds)
        X2 = SA.SVector{N, FT1}((x_vec[j] for x_vec in x_vecs)...)
        U2 = SA.SVector{N, FT2}((u_vec[j] for u_vec in u_vecs)...)

        @inbounds distance = distance_metric(X1, X2)
        bin = HF.digitize(distance, distance_bins_vec)
        @inbounds output[bin] += structure_function_type.method(U2 - U1, HF.r̂(X1, X2))
        @inbounds counts[bin] += 1
    end
    return output, counts
end




function calculate_structure_function(
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, ShA.SharedVector{FT1}, SA.SVector{N2, FT1}, AbstractVector{FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{NTuple{N2, FT2}, ShA.SharedVector{FT2}, SA.SVector{N2, FT2}, AbstractVector{FT2}}}, # Tuple{Vararg{Vector{FT},N}}
    distance_bins::Int,
    structure_function_type::SFT.AbstractStructureFunctionType;
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
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop


    min_distance, max_distance = Inf, 0
    n_distance_bins = distance_bins # save the number of bins since we'll overwrite distance_bins with the actual bins later.

    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    iter_inds = eachindex(x_vecs[1]) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _min_distance, _max_distance = minmax_i(i, x_vecs, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
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
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT}, ShA.SharedVector{FT}, SA.SVector{N2, FT}, AbstractVector{FT}}},
    distance_metric = DI.Euclidean(),
) where {FT, N, N2}
    """ calculate, bin, and mean the pairwise distances """


    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = SA.SVector{N, FT}((x_vec[i] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
    for j in eachindex(x_vecs[1])
        if i != j
            @inbounds X2 = SA.SVector{N, FT}((x_vec[j] for x_vec in x_vecs)...) # StaticArrays.sacollect(SVector{N, FT}, x_vec[i] for x_vec in x_vecs ) could be faster
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


### ====================================================== ###
### ====================================================== ###
### ====================================================== ###
"""
Array version (is slower lol)
"""
# array version seems to be slower lol, idk why (and we're even using views!)

function calculate_structure_function(
    x_arr::T1,
    u_arr::T2,
    distance_bins::SA.SVector{N3, Tuple{FT3, FT3}},
    structure_function_type::SFT.AbstractStructureFunctionType; # add N here?
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1 <: Real, FT2 <: Real, FT3 <: Real, N3, T1 <:Union{ShA.SharedArray{FT1}, AbstractArray{FT1}}, T2 <:Union{ShA.SharedArray{FT2}, AbstractArray{FT2}}}
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop

    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    output = zeros(N3)
    counts = zeros(N3)

    distance_bins_vec = SA.SVector{length(distance_bins) + 1, FT3}(
        [[distance_bin[1] for distance_bin in distance_bins]; [distance_bins[end][2]]]...,
    ) # the start of each bin plus the ending

    if verbose
        @info("calculating structure function")
    end

    iter_inds = axes(x_arr,2) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # iter_inds = axes(x_arr,1) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _output, _counts = calculate_structure_function_i(
            i,
            x_arr,
            u_arr,
            distance_bins_vec,
            structure_function_type;
            distance_metric = distance_metric,
        )
        output .+= _output
        counts .+= _counts
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
    x_arr::T1,
    u_arr::T2,
    distance_bins_vec::SA.SVector{N3, FT3},
    structure_function_type::SFT.AbstractStructureFunctionType; # add N here?
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {FT1 <: Real, FT2 <: Real, FT3 <: Real, N3, T1 <:Union{ShA.SharedArray{FT1}, AbstractArray{FT1}}, T2 <:Union{ShA.SharedArray{FT2}, AbstractArray{FT2}}}


    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    output = zeros(N3 - 1)
    counts = zeros(N3 - 1)


    iter_inds = axes(x_arr,2) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # iter_inds = axes(x_arr,1) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    X1 = @view(x_arr[:, i]) # column major faster right?
    U1 = @view(u_arr[:, i])
    # X1 = @view(x_arr[i, :]) # column major faster right?
    # U1 = @view(u_arr[i, :])


    LV.@simd for j in iter_inds
        if i != j # this has δx and δu = 0, so we skip it
            X2 = @view(x_arr[:, j])
            U2 = @view(u_arr[:, j])
            # X2 = @view(x_arr[j, :])
            # U2 = @view(u_arr[j, :])

            @inbounds distance = distance_metric(X1, X2) # this is the slow part
            bin = HF.digitize(distance, distance_bins_vec) # this will fail when the the distance value equals the smallest bin's smallest edge due to the way digitize works, we could add a handler here but we extended the smallest bin to be the next smallest float so this shouldn't happen
            @inbounds output[bin] += structure_function_type.method(U2 - U1, HF.r̂(X1, X2))
            @inbounds counts[bin] += 1
        end
    end
    return output, counts
end



function calculate_structure_function(
    x_arr::T1,
    u_arr::T2,
    distance_bins::Int,
    structure_function_type::SFT.AbstractStructureFunctionType;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing = :logarithmic,
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {FT1 <: Real, FT2 <: Real, T1 <:Union{ShA.SharedArray{FT1}, AbstractArray{FT1}}, T2 <:Union{ShA.SharedArray{FT2}, AbstractArray{FT2}}}
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

    iter_inds = axes(x_arr,2) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # iter_inds = axes(x_arr,1) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    PM.@showprogress enabled = show_progress for i in iter_inds # is this the fast order?
        _min_distance, _max_distance = minmax_i(i, x_arr, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
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
    return calculate_structure_function(
        x_arr,
        u_arr,
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
    x_arr::T1,
    distance_metric = DI.Euclidean(),
) where {FT <: Real, T1 <:Union{ShA.SharedArray{FT}, AbstractArray{FT}}}
    """ calculate, bin, and mean the pairwise distances """


    # preallocate output as vector of length of distance_bins
    min_distance, max_distance = Inf, 0

    @inbounds X1 = @view(x_arr[:, i])
    # @inbounds X1 = @view(x_arr[i, :])

    for j in axes(x_arr,2)
    # for j in axes(x_arr,1)
        if i != j
            @inbounds X2 = @view(x_arr[:, j]) 
            # @inbounds X2 = @view(x_arr[j, :]) 

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





































### ====================================================== ###
### ====================================================== ###
### ====================================================== ###


function parallel_calculate_structure_function end # placeholder for parallel extension


end
