""" calculate, bin, and mean the pairwise distances """

# THIS VERSION WILL WORK IN 3D -- YOU WILL HAVE LIST OF ARRAYS IN TIME AT EACH POINT, CALCULATE THE BIN THEY GO IN AND THEN GET ALL THE SUM (nansum for safety?) OF ALL THE DISTANCES AT ONCE and add to that bin/counts
# hopefully this makes it faster than the previous version, but the real advantage should come with integration with python etc and other external programs that would repeatedly have to call into this fcn.... :)

# Then we coould one day do, say, by month or by year and store counts and sums and only do the full hourly when we feel like we have many hours to burn lol

###


"""
The workhorse of this package
"""
module AlternateCalculations3D
import ProgressMeter: @showprogress
import Distances # from JuliaStats
import ..HelperFunctions

using StaticArrays
using LinearAlgebra

using SharedArrays
using Base.Threads

using ..StructureFunctionTypes # how can we make these natively available?
# print(eval(Symbol("StructureFunctionTypes.SecondOrderStructureFunction"))) # doesn't work

# using LoopVectorization # can't use w/ [bin] indexing below, see https://github.com/JuliaSIMD/LoopVectorization.jl/issues/331

export calculate_structure_function, parallel_calculate_structure_function, calculate_distance_bins


###########
# Add methods for dispatching vectors/tuples to the static SVector versions...
########




using LoopVectorization



### ====================================================== ###
### ====================================================== ###
""" 
NTuple of Vectors/Svectrs etc... but not by coordinate but y point
I think this is best bc the number of dimensions are short and thus are fine as svectors and additionally, selecting points is just an easy selectoin with no reconstruction...
Since we do need linear algebra operations to calculate like distance and stuff, maybe SVectors are better? (though i think those methods can handle tuples idk...)
"""

function calculate_structure_function(
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, SharedVector{FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{ SVector{NT, NTuple{N2, FT2}}, NTuple{NT, NTuple{N2, FT2}}, AbstractVector{NTuple{N2, FT2}} }}, # NT is number of time points (or whatever 3rd dim is) -- should we do this or allow vectors and then get NT from the object?
    distance_bins::SVector{N3, Tuple{FT3, FT3}},
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType; # add N here?
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
    return_sums_and_counts::Bool = false,
    check_nan::Bool= false, # checking is slower but if you have many times binned together, you may want to check for NaNs before calling the method fcns... (is that faster then just calling and then not adding nans? idk...)
    threaded::Bool = true,
    combine_third_dimension::Bool = true, # we can either keep the results from the third dimensions separate (i.e. a list of hourly SFs, or combine them (e.g. one daily SF from hourly data))
) where {FT1 <: Real, FT2 <: Real, FT3 <: Real, N, N2, N3, NT}
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop

    # check if NT exists if not get from length of object, use new name cause cant overwrite static param
    if !@isdefined(NT)
        NTT::Int64 = length(u_vecs[1]) # type annotate for speed
        # @info("NT not defined, setting to length of u_vecs[1] = $NTT")
    else
        NTT = NT
    end

    # preallocate output as vector of length of distance_bins (vector so it's mutable)

    if combine_third_dimension
        output::MVector = MVector{N3, FT2}(zeros(FT2, N3)) # allocation checking
        counts::MVector = MVector{N3, Int64}(zeros(Int64,N3))
    else # we need an array with one for each time point in NTT
        output = MVector{NTT, MVector{N3, FT2}}(MVector{N3, FT2}(zeros(FT2, N3)) for i in 1:NTT) # allocation checking
        counts = MVector{NTT, MVector{N3, Int64}}(MVector{N3, Int64}(zeros(Int64,N3)) for i in 1:NTT)
    end

    distance_bins_vec::SVector{length(distance_bins) + 1, FT3} = SVector{length(distance_bins) + 1, FT3}(
        [[distance_bin[1] for distance_bin in distance_bins]; [distance_bins[end][2]]]...,
    ) # the start of each bin plus the ending

    if verbose
        @info("calculating structure function")
    end

    iter_inds = eachindex(x_vecs) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    # @showprogress enabled = show_progress for i in iter_inds # is this the fast order?
    if threaded
        @threads for i::Int64 in iter_inds # is this the fast order?
            _output, _counts = calculate_structure_function_i(
                i,
                x_vecs,
                u_vecs,
                distance_bins_vec,
                structure_function_type;
                distance_metric = distance_metric,
                check_nan = check_nan,
                combine_third_dimension = combine_third_dimension,
            )
            if combine_third_dimension
                output .+= _output
                counts .+= _counts
            else
                for k in Base.OneTo(NTT::Int64)# iterate over all pairs (annotating makes a huge difference in allocations, reduces by factor of 6!, though we did annotate above)
                    output[k] .+= _output[k]
                    counts[k] .+= _counts[k]
                end
            end
        end
    else
        @showprogress enabled = show_progress for i in iter_inds # is this the fast order?
            _output, _counts = calculate_structure_function_i(
                i,
                x_vecs,
                u_vecs,
                distance_bins_vec,
                structure_function_type;
                distance_metric = distance_metric,
                check_nan = check_nan,
                combine_third_dimension = combine_third_dimension,
            )
            if combine_third_dimension
                output .+= _output
                counts .+= _counts
            else
                for k in Base.OneTo(NTT::Int64)# iterate over all pairs (annotating makes a huge difference in allocations, reduces by factor of 6!, though we did annotate above)
                    output[k] .+= _output[k]
                    counts[k] .+= _counts[k]
                end
            end
        end
    end

    if return_sums_and_counts # just return the sums and the counts, don't take the mean in each bin...
        return (output, counts), distance_bins
    else # do the mean in each bin.
        # counts[counts .== 0] .= eltype(output)(NaN) # replace 0s with NaNs to avoid divide by 0 giving Inf
        counts = (count == 0 ? NaN : count for count in counts) # replace 0s with NaNs to avoid divide by 0 giving Inf
        output ./= counts
        return output, distance_bins
    end
end



function calculate_structure_function_i(
    i::Int,
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, SharedVector{FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{ SVector{NT, NTuple{N2, FT2}}, NTuple{NT, NTuple{N2, FT2}}, AbstractVector{NTuple{N2, FT2}}  }}, # NT is number of time points (or whatever 3rd dim is) -- should we do this or allow vectors and then get NT from the object?
    distance_bins_vec::SVector{N3, FT3},
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType; # add N here?
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
    check_nan::Bool = false, # checking is slower but if you have many times binned together, you may want to check for NaNs before calling the method fcns... (is that faster then just calling and then not adding nans? idk...)
    combine_third_dimension::Bool = true, # we can either keep the results from the third dimensions separate (i.e. a list of hourly SFs, or combine them (e.g. one daily SF from hourly data))
) where {FT1 <: Real, FT2 <: Real, FT3 <: Real, N, N2, N3, NT}


    # preallocate output as vector of length of distance_bins (vector so it's mutable)
    # output = zeros(N3 - 1)
    # counts = zeros(N3 - 1)

    # check if NT exists if not get from length of object, use new name cause cant overwrite static param
    if !@isdefined(NT)
        NTT::Int64 = length(u_vecs[1]) # type annotate for speed
    else
        NTT = NT
    end

    if combine_third_dimension
        output::MVector = MVector{N3-1, FT2}(zeros(FT2, N3-1)) # allocation checking
        counts::MVector = MVector{N3-1, Int64}(zeros(Int64, N3-1))
    else
        output = MVector{NTT, MVector{N3-1, FT2}}(MVector{N3-1, FT2}(zeros(FT2, N3-1)) for i in 1:NTT::Int64) # allocation checking
        counts = MVector{NTT, MVector{N3-1, Int64}}(MVector{N3-1, Int64}(zeros(Int64,N3-1)) for i in 1:NTT::Int64)
    end

    iter_inds = Iterators.rest( eachindex(x_vecs), i) # get only the rest of the indices (i.e. skip before the i'th index)
     
    for j::Int64 in iter_inds
        @inbounds distance::FT1 = distance_metric(x_vecs[i], x_vecs[j]) # this is the slow part (has an allocation too...)
        bin::Int64 = HelperFunctions.digitize(distance, distance_bins_vec) # this will fail when the the distance value equals the smallest bin's smallest edge due to the way digitize works, we could add a handler here but we extended the smallest bin to be the next smallest float so this shouldn't happen
        
        r_hat::typeof(x_vecs[i]) = HelperFunctions.r̂(x_vecs[i],x_vecs[j]) # save

        for k::Int64 in Base.OneTo(NTT::Int64)# iterate over all pairs (annotating makes a huge difference in allocations, reduces by factor of 6!, though we did annotate above)
            if check_nan # checking slows code down by almost double
                if ! (any(isnan, u_vecs[i][k]) || any(isnan, u_vecs[j][k])) # skip NaNs (cause if we bin times together we can't pre-remove all NaNs)
                # if ! (fast_any_isnan(u_vecs[i][k]) || fast_any_isnan(u_vecs[j][k])) # skip NaNs (cause if we bin times together we can't pre-remove all NaNs)
                    if combine_third_dimension
                        @inbounds output[bin] += structure_function_type.method(u_vecs[j][k] .- u_vecs[i][k], r_hat) # the work happens here, with allocations.... this is the slow part
                        @inbounds counts[bin] += 1
                    else
                        @inbounds output[k][bin] += structure_function_type.method(u_vecs[j][k] .- u_vecs[i][k], r_hat) # the work happens here, with allocations.... this is the slow part
                        @inbounds counts[k][bin] += 1
                    end
                end
            else
                if combine_third_dimension
                    @inbounds output[bin] += structure_function_type.method(u_vecs[j][k] .- u_vecs[i][k], r_hat) # the work happens here, with allocations.... this is the slow part
                    @inbounds counts[bin] += 1
                else
                    @inbounds output[k][bin] += structure_function_type.method(u_vecs[j][k] .- u_vecs[i][k], r_hat) # the work happens here, with allocations.... this is the slow part
                    @inbounds counts[k][bin] += 1
                end
            end
        end

    end
    return output, counts
end

# fast_any_isnan(x) = isnan(sum(x))  # faster than any(isnan,x)


function calculate_structure_function(
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, SharedVector{FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    u_vecs::NTuple{N, <:Union{ SVector{NT, NTuple{N2, FT2}}, NTuple{NT, NTuple{N2, FT2}}, AbstractVector{NTuple{N2, FT2}}  } }, # NT is number of time points (or whatever 3rd dim is) -- should we do this or allow vectors and then get NT from the object?
    distance_bins::Int,
    structure_function_type::StructureFunctionTypes.AbstractStructureFunctionType;
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
    bin_spacing = :logarithmic,
    verbose::Bool = true,
    show_progress::Bool = true,
    return_sums_and_counts::Bool = false,
    check_nan::Bool = false, # checking is slower but if you have many times binned together, you may want to check for NaNs before calling the method fcns... (is that faster then just calling and then not adding nans? idk...)
    threaded::Bool = true,
    combine_third_dimension::Bool = true, # we can either keep the results from the third dimensions separate (i.e. a list of hourly SFs, or combine them (e.g. one daily SF from hourly data))
) where {FT1 <: Real, FT2 <: Real, N, N2, NT} # You can't dispatch on N, N2 since they're Ints not types
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """

    return calculate_structure_function(
        x_vecs,
        u_vecs,
        calculate_distance_bins(x_vecs; distance_metric = distance_metric, bin_spacing = bin_spacing, n_distance_bins = distance_bins, verbose = verbose),
        structure_function_type;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        return_sums_and_counts = return_sums_and_counts,
        check_nan = check_nan,
        threaded = threaded,
        combine_third_dimension = combine_third_dimension,
    )
end


function minmax_i(
    i::Int,
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT}, SharedVector{FT}, SVector{N2, FT}, AbstractVector{FT}}}, # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
) where {FT <: Real, N <: Int64, N2 <: Int64} # You can't dispatch on N, N2 since they're Ints not types
    """ calculate, bin, and mean the pairwise distances """

    # preallocate output as vector of length of distance_bins
    min_distance::FT, max_distance::FT = FT(Inf), FT(0)

    iter_inds = Iterators.rest( eachindex(x_vecs), i) # get only the rest of the indices (i.e. skip before the i'th index)

    for j::Int64 in iter_inds
        # update the min and max distances
        # distance = distance_metric(X1, X2) # this is the slow part
        @inbounds distance::FT = distance_metric(x_vecs[i], x_vecs[j]) # this is the slow part
        if distance < min_distance
            min_distance = distance
        elseif distance > max_distance
            max_distance = distance
        end
    end

    return min_distance, max_distance
end



function calculate_min_max_distances(
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, SharedVector{FT1}, SVector{N2, FT1}, AbstractVector{FT1}}}; # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
    show_progress::Bool = true,
    ) where {FT1 <: Real, N <: Int64, N2 <: Int64}

    """ To do, add threaded option """
    min_distance::FT1, max_distance::FT1 = FT1(Inf), FT1(0)

    iter_inds = eachindex(x_vecs) # these should all match..., idk if doing 1:N2 is faster but the indexing could be shifted...
    @showprogress enabled = show_progress for i::Int64 in iter_inds # is this the fast order?
        _min_distance::FT1, _max_distance::FT1 = minmax_i(i, x_vecs, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
    end

    return min_distance, max_distance
end

function calculate_distance_bins(
    x_vecs::NTuple{N, <:Union{NTuple{N2, FT1}, SharedVector{FT1}, SVector{N2, FT1},  StaticArrays.MVector{N2, FT1}, AbstractVector{FT1}}}; # Tuple{Vararg{Vector{FT},N}} # I think this is faster than array cause then we have columns only
    distance_metric::Distances.PreMetric = Distances.Euclidean(),
    bin_spacing = :logarithmic,
    n_distance_bins::Int = 50,
    verbose::Bool = true,
    show_progress::Bool = true,
    ) where {FT1 <: Real, N <: Int64, N2 <:Int64}


    # calculate the bins
    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    min_distance::FT1, max_distance::FT1 = calculate_min_max_distances(x_vecs; distance_metric=distance_metric, show_progress=show_progress)

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

    FT3::DataType = eltype(distance_bins)
    distance_bins = SVector{n_distance_bins, Tuple{FT3, FT3}}( 
        [(distance_bins[i], distance_bins[i + 1]) for i in 1:n_distance_bins::Int64]...,
    ) # convert to tuples of the bin edges, I think this is okay bc n_distance_bins is fixed so the method will precompile once...

    return distance_bins
end

### ====================================================== ###
### ====================================================== ###
### ====================================================== ###


function parallel_calculate_structure_function end # placeholder for parallel extension


end
