"""
The workhorse of this package
"""
module Calculations
import ProgressMeter: @showprogress
import Distances # from JuliaStats
import ..HelperFunctions

export calculate_structure_function

"""
Consider using some sort of Intervals thingy for the intervals
    worked on arrays of size 1e4 in 12 seconds, was about a true half and half timewise, so w/ saved bins would be about 6 seconds...
    note we still need to turn values to vectors for u,v,w, etc and have a function for the SF calculation that's not just order...
    and we need a parallel version, maybe parallelize over the `i` loop, aggregate the results separately, then bin and average from the workers...
    - probably shouldn't use sharedarrays as those can possibly fail w/ concurrent reads/writes
"""
function calculate_structure_function(
    X::Vector,
    Y::Vector,
    values::Vector,
    distance_bins::Vector{Tuple{FT, FT}};
    order = 2,
    distance_metric = Distances.Euclidean(),
) where {FT}
    # calculate and bin and mean the pairwise distances
    # distances = pairwise(distance_metric, X, Y, dims=2) # will blow up if too big... so we're just doing the loop
    # bin the distances

    # preallocate output as vector of length of distance_bins
    output = zeros(length(distance_bins))
    counts = zeros(length(distance_bins))

    @assert length(X) == length(Y) == length(values) # for speed, X,Y, and could could be put in an array to skip this check...
    n = length(X)

    # distance_bins_vec = Tuple([[x[1] for x in distance_bins] ; [distance_bins[end][2]] ]) # the start of each bin plus the ending
    distance_bins_vec = [[x[1] for x in distance_bins]; [distance_bins[end][2]]] # the start of each bin plus the ending

    @info("calculating structure function")
    @showprogress for i in 1:n
        for j in 1:n
            # find the bin that this distance belongs to
            distance = distance_metric([X[i], Y[i]], [X[j], Y[j]])
            bin = HelperFunctions.digitize(distance, distance_bins_vec) # this will fail when the the distance value equals the largest bin's largest edge due to the way digitize works, we could add a handler here to put that value back into the largest bin...
            output[bin] += (values[i] - values[j])^order
            counts[bin] += 1
        end
    end
    counts[counts .== 0] .= NaN
    output ./= counts
    return output, distance_bins
end

function calculate_structure_function(
    X::Vector,
    Y::Vector,
    values::Vector,
    distance_bins::Int;
    order = 1,
    distance_metric = Distances.Euclidean(),
) where {FT}
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
    n = length(X)

    # calculate the bins
    @info("Calculating min and max distances and generating bins")
    @showprogress for i in 1:n
        for j in 1:n
            # update the min and max distances
            distance = distance_metric([X[i], Y[i]], [X[j], Y[j]])
            if distance < min_distance
                min_distance = distance
            elseif distance > max_distance
                max_distance = distance
            end
        end
    end
    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin
    distance_bins = range(min_distance, max_distance, length = distance_bins + 1) # +1 to get the right number of bins since these are the edges
    distance_bins = [(distance_bins[i], distance_bins[i + 1]) for i in 1:(length(distance_bins) - 1)] # convert to tuples of the bin edges

    return calculate_structure_function(X, Y, values, distance_bins; order = order, distance_metric = distance_metric)
end

end
