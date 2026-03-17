"""
The workhorse of this package
For speed, pretty much all your inputs seems to need to be SharedArrays...
"""
module StructureFunctionsParallelCalculationsExt
using Distributed: Distributed
using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using SharedArrays: SharedArrays
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, HelperFunctions as SFH, StructureFunctionTypes as SFT


export parallel_calculate_structure_function 

function SFC.parallel_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}};
    return_sums_and_counts::Bool = false,
    kwargs...,
) where {T1, T2, FT3}
    return SFC.parallel_calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        Val(return_sums_and_counts);
        kwargs...,
    )
end

function SFC.parallel_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    ::Val{RSAC};
    kwargs...,
) where {T1, T2, FT3, RSAC}
    return _parallel_calculate_structure_function_core(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        Val(RSAC);
        kwargs...,
    )
end

function _parallel_calculate_structure_function_core(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector{<:Tuple{FT3, FT3}},
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {T1, T2, FT3, RSAC}
    N = length(x_vecs)
    N3 = length(distance_bins)


    # Create a stable Vector for bins edges
    distance_bins_vec = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        distance_bins_vec[k] = distance_bins[k][1]
    end
    distance_bins_vec[end] = distance_bins[end][2]

    if verbose
        @info("calculating structure function")
    end

    output, counts =
        PM.@showprogress enabled = show_progress Distributed.@distributed ((x, y) -> ((x[1] .+ y[1]), (x[2] .+ y[2]))) for i in
                                                                                                            eachindex(
            x_vecs[1],
        )
            SFC.calculate_structure_function_i(
                structure_function_type,
                i,
                x_vecs,
                u_vecs,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
        end

    if RSAC
        return SF.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, output, counts)
    else
        counts_safe = copy(counts)
        counts_safe[counts_safe .== 0] .= NaN # replace 0s with NaNs to avoid divide by 0 giving Inf
        output_div = output ./ counts_safe
        return SF.StructureFunction(structure_function_type, distance_bins, output_div)
    end
end


function SFC.parallel_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing = :logarithmic,
    verbose = true,
    show_progress = true,
    return_sums_and_counts = false,
) where {T1, T2}
    N = length(x_vecs)
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
            SFC.minmax_i(i, x_vecs, distance_metric)
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
    )


    return SFC.parallel_calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        return_sums_and_counts = return_sums_and_counts,
    )
end

end
