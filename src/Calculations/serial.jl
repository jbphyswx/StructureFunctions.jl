# Serial 1D CPU Reduction Kernels

function serial_calculate_structure_function!(
    output::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, T1, T2}
    distance_bins = BinEdges(distance_bins)

    if verbose
        @info("calculating structure function (serial reduction)")
    end

    iter_inds = eachindex(x_vecs[1])
    # Pass Val(length(x_vecs)) to make it a type parameter in the work function
    vN = Val(length(x_vecs))
    PM.@showprogress enabled = show_progress for i in iter_inds
        calculate_structure_function_i!(
            output,
            counts,
            vN,
            structure_function_type,
            i,
            x_vecs,
            u_vecs,
            distance_bins;
            distance_metric = distance_metric,
        )
    end
    return nothing
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
    ::Val{RSAC};
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {T1, T2, RSAC, CT}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(CT, N3)

    serial_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        kwargs...,
    )

    if RSAC # just return the sums and the counts, don't take the mean in each bin...
        return SFO.StructureFunctionSumsAndCounts(
            structure_function_type,
            distance_bins,
            output,
            counts,
        )
    else # do the mean in each bin.
        output_div = similar(output)
        for k in eachindex(output)
            c = counts[k]
            output_div[k] = iszero(c) ? OT(NaN) : output[k] / c
        end
        return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
    end
end

function serial_calculate_structure_function!(
    sums::AbstractVector{OT},
    counts::AbstractVector,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector;
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return serial_calculate_structure_function!(
        sums,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        kwargs...,
    )
end

function calculate_structure_function_i!(
    output::AbstractVector{OT},
    counts::AbstractVector,
    ::Val{N},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, N, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)

    X1 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][i], Val(N)))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][i], Val(N)))

    iter_inds = eachindex(x_vecs[1])
    for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][j], Val(N)))
        U2 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][j], Val(N)))

        distance = distance_metric(X1, X2)
        bin = SFH.digitize(distance, distance_bins)
        if 1 <= bin < N3
            rh = SFH.r̂(X1, X2, distance_metric, distance)
            @inbounds output[bin] += structure_function_type(U2 - U1, rh)
            @inbounds counts[bin] += 1
        end
    end
    return nothing
end

function calculate_structure_function_i(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    N3 = n_histogram_bins(distance_bins)
    local_output = zeros(OT, N3)
    local_counts = zeros(CT, N3)
    calculate_structure_function_i!(
        local_output, local_counts,
        Val(length(x_vecs)),
        structure_function_type, i, x_vecs, u_vecs, BinEdges(distance_bins);
        distance_metric = distance_metric,
    )
    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        local_output,
        local_counts,
    )
end
