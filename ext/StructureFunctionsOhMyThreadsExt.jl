module StructureFunctionsOhMyThreadsExt

using Distances: Distances as DI
using OhMyThreads: OhMyThreads as OMT
using StructureFunctions:
    StructureFunctions as SF,
    Calculations as SFC,
    StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT

function SFC.threaded_calculate_structure_function(
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
        @info("calculating structure function (threaded reduction via OhMyThreads)")
    end

    # OhMyThreads currently does not integrate with our progress display path.
    _ = show_progress

    sums_and_counts = OMT.tmapreduce(+, eachindex(x_vecs[1])) do i
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
    end

    counts_safe = copy(sums_and_counts.counts)
    counts_safe[counts_safe .== 0] .= NaN
    output_div = sums_and_counts.sums ./ counts_safe
    return SF.StructureFunction(structure_function_type, distance_bins, output_div)
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector{Tuple{FT3, FT3}},
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, RSAC}
    if verbose
        @info("calculating structure function (threaded reduction via OhMyThreads)")
    end

    # OhMyThreads currently does not integrate with our progress display path.
    _ = show_progress

    N3 = length(distance_bins)
    OT = promote_type(float(FT1), float(FT2))

    distance_bins_vec = Vector{FT3}(undef, N3 + 1)
    for k in 1:N3
        distance_bins_vec[k] = distance_bins[k][1]
    end
    distance_bins_vec[end] = distance_bins[end][2]

    N = size(x_arr, 1)
    if !(N in (1, 2, 3))
        throw(ArgumentError("Threaded array backend supports only 1D, 2D, or 3D inputs."))
    end

    if N == 1
        sums_and_counts = OMT.tmapreduce(+, axes(x_arr, 2)) do i
            local_output = zeros(OT, N3)
            local_counts = zeros(OT, N3)
            SFC.calculate_structure_function_i!(
                local_output,
                local_counts,
                Val(1),
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
            SFO.StructureFunctionSumsAndCounts(
                structure_function_type,
                distance_bins,
                local_output,
                local_counts,
            )
        end
    elseif N == 2
        sums_and_counts = OMT.tmapreduce(+, axes(x_arr, 2)) do i
            local_output = zeros(OT, N3)
            local_counts = zeros(OT, N3)
            SFC.calculate_structure_function_i!(
                local_output,
                local_counts,
                Val(2),
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
            SFO.StructureFunctionSumsAndCounts(
                structure_function_type,
                distance_bins,
                local_output,
                local_counts,
            )
        end
    else
        sums_and_counts = OMT.tmapreduce(+, axes(x_arr, 2)) do i
            local_output = zeros(OT, N3)
            local_counts = zeros(OT, N3)
            SFC.calculate_structure_function_i!(
                local_output,
                local_counts,
                Val(3),
                structure_function_type,
                i,
                x_arr,
                u_arr,
                distance_bins_vec;
                distance_metric = distance_metric,
            )
            SFO.StructureFunctionSumsAndCounts(
                structure_function_type,
                distance_bins,
                local_output,
                local_counts,
            )
        end
    end

    if RSAC
        return sums_and_counts
    end

    counts_safe = copy(sums_and_counts.counts)
    counts_safe[counts_safe .== 0] .= NaN
    output_div = sums_and_counts.sums ./ counts_safe
    return SF.StructureFunction(structure_function_type, distance_bins, output_div)
end

end
