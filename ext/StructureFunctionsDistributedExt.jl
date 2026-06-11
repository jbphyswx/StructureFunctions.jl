"""
The workhorse of this package
For speed, pretty much all your inputs seems to need to be SharedArrays...
"""
module StructureFunctionsDistributedExt
using Distributed: Distributed
using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using SharedArrays: SharedArrays
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionTypes as SFT,
    AbstractBinEdges, LinearBinEdges, LogBinEdges


export parallel_calculate_structure_function

SFC.distributed_workers_available(::Val{:distributed}) = Distributed.nworkers() > 1

function SFC.parallel_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    return_sums_and_counts::Bool = false,
    kwargs...,
)
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
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
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
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {RSAC}
    if verbose
        @info("calculating structure function (distributed reduction)")
    end

    # Use the result-object wrapper for clean distributed reduction using package types
    sums_and_counts =
        PM.@showprogress enabled = show_progress Distributed.@distributed (+) for i in
                                                                                  eachindex(
            x_vecs[1],
        )
            SFC.calculate_structure_function_i(
                structure_function_type,
                i,
                x_vecs,
                u_vecs,
                distance_bins;
                distance_metric = distance_metric,
            )
        end

    if RSAC
        return sums_and_counts
    else
        counts_safe = copy(sums_and_counts.counts)
        counts_safe[counts_safe .== 0] .= NaN # replace 0s with NaNs to avoid divide by 0 giving Inf
        output_div = sums_and_counts.sums ./ counts_safe
        return SF.StructureFunction(structure_function_type, distance_bins, output_div)
    end
end


function SFC.parallel_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing::Type{<:AbstractBinEdges} = LogBinEdges,
    verbose = true,
    show_progress = true,
    return_sums_and_counts::Bool = false,
    kwargs...,
)
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
        PM.@showprogress enabled = show_progress Distributed.@distributed (
            (x, y) -> (min(x[1], y[1]), max(x[2], y[2]))
        ) for i in
              eachindex(
            x_vecs[1],
        )
            SFC.minmax_i(i, x_vecs, distance_metric)
        end


    min_distance = prevfloat(min_distance) # go to the next smallest float from the min distance to make sure the true smallest distance can fit in the first bin (note this is needed so things matching min_distance don't get assigned bin '0', alternative is to check for bin 0 every time... which sounds slower
    if bin_spacing === LinearBinEdges
        distance_bins = range(min_distance, max_distance, length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
    elseif bin_spacing === LogBinEdges
        distance_bins =
            10 .^
            range(log10(min_distance), log10(max_distance), length = n_distance_bins + 1) # +1 to get the right number of bins since these are the edges
        distance_bins[1] = min_distance # combat floating point errors (may have rounded up or down during operations)
        distance_bins[end] = max_distance # combat floating point errors (may have rounded up or down during operations)
    else
        throw(ArgumentError("bin_spacing must be LinearBinEdges or LogBinEdges"))
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
        kwargs...,
    )
end

"""
    SFC._dispatch_single_pass(::DistributedBackend, x, u, distance_bins; kwargs...)

Calculates single-pass structure functions utilizing multi-process Distributed execution.
Uses a process-parallelized loop via `@distributed (+)` and reduces results element-wise.
"""
function SFC._dispatch_single_pass(
    ::SF.DistributedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_points = size(x, 2)
    
    # We distribute the outer points loop.
    # The reduction operator `+` works element-wise on the returned Float64 matrix of shape (16, n_bins).
    combined_reduced = Distributed.@distributed (+) for i in 1:n_points
        local_combined = zeros(Float64, 16, n_bins)
        x_i = SA.SVector{2, FT1}(x[1, i], x[2, i])
        u_i = SA.SVector{2, FT2}(u[1, i], u[2, i])
        
        for j in (i+1):n_points
            x_j = SA.SVector{2, FT1}(x[1, j], x[2, j])
            
            r = distance_metric(x_i, x_j)
            bin_idx = SFH.digitize(r, distance_bins)
            
            if 1 <= bin_idx <= n_bins
                u_j = SA.SVector{2, FT2}(u[1, j], u[2, j])
                du = u_j - u_i
                
                rh = SFH.r̂(x_i, x_j, distance_metric, r)
                nh = SFH.n̂(rh)
                
                du_L = LA.dot(du, rh)
                du_T = LA.dot(du, nh)
                
                du_L2 = du_L * du_L
                du_T2 = du_T * du_T
                
                local_combined[1, bin_idx] += du_L2 + du_T2
                local_combined[2, bin_idx] += du_L2
                local_combined[3, bin_idx] += du_T2
                local_combined[4, bin_idx] += du_L * (du_L2 + du_T2)
                local_combined[5, bin_idx] += du_L * du_L2
                local_combined[6, bin_idx] += du_L2 * du_T
                local_combined[7, bin_idx] += du_L * du_T2
                local_combined[8, bin_idx] += du_T * du_T2
                
                for t in 9:16
                    local_combined[t, bin_idx] += 1.0
                end
            end
        end
        local_combined
    end
    
    sums = OT.(combined_reduced[1:8, :])
    counts = Int64.(combined_reduced[9:16, :])
    
    return SFC.postprocess_single_pass_results(sums, counts, distance_bins)
end

function SFC.parallel_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
)
    if verbose
        @info("calculating 2D joint structure function (distributed reduction)")
    end

    sums_and_counts =
        PM.@showprogress enabled = show_progress Distributed.@distributed (+) for i in eachindex(x_vecs[1])
            SFC.calculate_structure_function_2d_i(
                structure_function_type,
                i,
                x_vecs,
                u_vecs,
                distance_bins,
                value_bins;
                distance_metric = distance_metric,
            )
        end
    return sums_and_counts
end

function SFC.parallel_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC.parallel_calculate_structure_function(
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function SFC.parallel_calculate_structure_function!(
    sums::AbstractVector{OT},
    counts::AbstractVector{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    kwargs...,
) where {OT}
    result = SFC.parallel_calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        Val(true);
        kwargs...,
    )
    sums .+= result.sums
    counts .+= result.counts
    return nothing
end

function SFC.parallel_calculate_structure_function!(
    sums::AbstractVector{OT},
    counts::AbstractVector{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}};
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC.parallel_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        kwargs...,
    )
end

function SFC.parallel_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {OT}
    result = SFC.parallel_calculate_structure_function(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        kwargs...,
    )
    sums_2d .+= result.sums
    counts_2d .+= result.counts
    return nothing
end

function SFC.parallel_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{OT},
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    value_bins::AbstractVector;
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number, FT3 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC.parallel_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function SFC._dispatch_single_pass_2d(
    ::SF.DistributedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins_by_type::AbstractVector{<:AbstractVector};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = length(value_bins_by_type[1]) - 1
    n_points = size(x, 2)

    combined_reduced = Distributed.@distributed (+) for i in 1:n_points
        local_combined = zeros(Float64, 16, n_bins, n_val)
        x_i = SA.SVector{2, FT1}(x[1, i], x[2, i])
        u_i = SA.SVector{2, FT2}(u[1, i], u[2, i])

        for j in (i + 1):n_points
            x_j = SA.SVector{2, FT1}(x[1, j], x[2, j])

            r = distance_metric(x_i, x_j)
            bin_idx = SFH.digitize(r, distance_bins)

            if 1 <= bin_idx <= n_bins
                u_j = SA.SVector{2, FT2}(u[1, j], u[2, j])
                du = u_j - u_i

                rh = SFH.r̂(x_i, x_j, distance_metric, r)
                nh = SFH.n̂(rh)

                du_L = LA.dot(du, rh)
                du_T = LA.dot(du, nh)

                du_L2 = du_L * du_L
                du_T2 = du_T * du_T

                vals = (
                    du_L2 + du_T2,
                    du_L2,
                    du_T2,
                    du_L * (du_L2 + du_T2),
                    du_L * du_L2,
                    du_L2 * du_T,
                    du_L * du_T2,
                    du_T * du_T2,
                )

                for t in 1:8
                    vbin = SFH.digitize(vals[t], value_bins_by_type[t])
                    n_val_t = length(value_bins_by_type[t]) - 1
                    if 1 <= vbin <= n_val_t && vbin <= n_val
                        local_combined[t, bin_idx, vbin] += vals[t]
                        local_combined[t + 8, bin_idx, vbin] += 1.0
                    end
                end
            end
        end
        local_combined
    end

    sums = OT.(combined_reduced[1:8, :, :])
    counts = Int64.(combined_reduced[9:16, :, :])
    return sums, counts
end

end
