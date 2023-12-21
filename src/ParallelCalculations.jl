"""
The workhorse of this package
"""

@everywhere module ParallelCalculations
    using Distributed
    using ProgressMeter: @showprogress
    include("HelperFunctions.jl") # We need to load these because this has to happen on all workers in the parallel case... (idk how to make this submodule a separate module though...)
    include("Calculations.jl")
    # import ..StructureFunction # ideally
    import Distances # from JuliaStats
    import NaNStatistics
    import ..Calculations
    import ..HelperFunctions
    @info(Calculations)

    export parallel_calculate_structure_function #consider changing this to just calculate_structure function to match the non-parallel name

    @info(@__DIR__)

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
    function parallel_calculate_structure_function_i(i::Int, X::Vector{FT}, Y::Vector{FT}, values::Vector{FT2}, distance_bins_vec::Vector{FT3}; order=2, distance_metric=Distances.Euclidean()) where {FT<:Real, FT2<:Real, FT3<:Real}
        # calculate and bin and mean the pairwise distances
        # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop
        # bin the distances

        # preallocate output as vector of length of distance_bins
        output = zeros(length(distance_bins_vec)-1)
        counts = zeros(length(distance_bins_vec)-1)

        @assert length(X) == length(Y) == length(values) # for speed, X,Y, and could could be put in an array to skip this check...
        n = length(X)

        for j in 1:n
            # find the bin that this distance belongs to
            distance = distance_metric([X[i], Y[i]], [X[j], Y[j]])
            bin = HelperFunctions.digitize(distance, distance_bins_vec) # this will fail when the the distance value equals the largest bin's largest edge due to the way digitize works, we could add a handler here to put that value back into the largest bin...
            output[bin] += (values[i]-values[j]) ^ order
            counts[bin] += 1
        end
        counts[counts .== 0] .= NaN
        return output, counts
    end

    function parallel_calculate_structure_function(X::Vector{FT}, Y::Vector, values::Vector{FT}, distance_bins::Vector{Tuple{FT2,FT2}}; order=2, distance_metric=Distances.Euclidean()) where {FT<:Real, FT2<:Real}
        # calculate and bin and mean the pairwise distances
        # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop
        # bin the distances

        # preallocate output as vector of length of distance_bins

        @assert length(X) == length(Y) == length(values) # for speed, X,Y, and could could be put in an array to skip this check...
        n = length(X)

        # distance_bins_vec = Tuple([[x[1] for x in distance_bins] ; [distance_bins[end][2]] ]) # the start of each bin plus the ending
        distance_bins_vec = [[x[1] for x in distance_bins] ; [distance_bins[end][2]] ] # the start of each bin plus the ending

        outputs_counts = @showprogress pmap(i -> parallel_calculate_structure_function_i(i, X, Y, values, distance_bins_vec; order=order, distance_metric=distance_metric), 1:n)
        
        # outputs = [x[1] for x in outputs]
        # counts = [x[2] for x in outputs]
        
        # unzip output
        outputs, counts = ([x[1] for x in outputs_counts], [x[2] for x in outputs_counts]   )

        outputs =  mapreduce(permutedims, vcat, outputs) # turn vector of vectors to matrix
        counts =  mapreduce(permutedims, vcat, counts)

        output = NaNStatistics.nansum(outputs, dims=1)
        counts = NaNStatistics.nansum(counts, dims=1)
        output ./= counts
        return output, distance_bins
    end



    """
    Function to the bins and and i value and calculate the outputs and counts for that bin 
    """

    function parallel_calculate_structure_function(X::Vector{FT}, Y::Vector{FT}, values::Vector{FT}, distance_bins::Int; order=1, distance_metric=Distances.Euclidean()) where FT <: Real
    """
    Here we assume that the distance bins are evenly spaced
    However, we assume we cant store all the output pairs in memory (cause goes as len(x)^2
    so we make two passes, one to find the closest and furthest points, and then one to calculate
    """
        # calculate and bin and mean the pairwise distances
        # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop
        # bin the distances

        min_distance, max_distance = Inf, 0
        @assert length(X) == length(Y) == length(values) # for speed, X,Y, and could could be put in an array to skip this check...
        n=length(X)

        # calculate the bins
        @info("Calculating min and max distances and generating bins")

        minmax_distances = @showprogress pmap(i -> minmax_i(i, X, Y, distance_metric), 1:n)
        minvals, maxvals = ([x[1] for x in minmax_distances], [x[2] for x in minmax_distances]   )
        min_distance = minimum(minvals)
        max_distance = maximum(maxvals)

        # for i in 1:n
        #     for j in 1:n
        #         # update the min and max distances
        #         distance = distance_metric([X[i], Y[i]], [X[j], Y[j]])
        #         if distance < min_distance
        #             min_distance = distance
        #         elseif distance > max_distance
        #             max_distance = distance
        #         end
        #     end
        # end



        min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin
        distance_bins = range(min_distance, max_distance, length=distance_bins+1) # +1 to get the right number of bins since these are the edges
        distance_bins = [ (distance_bins[i], distance_bins[i+1]) for i in 1:length(distance_bins)-1 ] # convert to tuples of the bin edges


        @info("calculating structure function")
        return parallel_calculate_structure_function(X, Y, values, distance_bins; order=order, distance_metric=distance_metric)
    end


    function minmax_i(i::Int, X::Vector{FT}, Y::Vector{FT}, distance_metric=Distances.Euclidean()) where FT <: Real
        # calculate and bin and mean the pairwise distances
        # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop
        # bin the distances

        # preallocate output as vector of length of distance_bins
        min_distance, max_distance = Inf, 0

        @assert length(X) == length(Y) # for speed, X,Y, and could could be put in an array to skip this check...
        n = length(X)

        for j in 1:n
            # update the min and max distances
            distance = distance_metric([X[i], Y[i]], [X[j], Y[j]])
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
        return min_distance, max_distance
    end

end

