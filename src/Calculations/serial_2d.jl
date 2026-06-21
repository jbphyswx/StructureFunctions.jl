# Serial 2D CPU Joint Reduction Kernels

function serial_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {OT, CT, T1, T2}
    distance_bins = BinEdges(distance_bins)
    value_bins = BinEdges(value_bins)

    if verbose
        @info("calculating 2D joint structure function (serial reduction)")
    end

    iter_inds = eachindex(x_vecs[1])
    vN = Val(length(x_vecs))
    
    PM.@showprogress enabled = show_progress for i in iter_inds
        calculate_structure_function_2d_i!(
            sums_2d,
            counts_2d,
            vN,
            structure_function_type,
            i,
            x_vecs,
            u_vecs,
            distance_bins,
            value_bins;
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
    value_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {T1, T2, CT}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(CT, N3, N4)

    serial_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        kwargs...,
    )

    return SFO.StructureFunction2D(
        structure_function_type,
        distance_bins,
        value_bins,
        sums_2d,
        counts_2d,
    )
end

function serial_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return serial_calculate_structure_function!(
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

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number}
    if ndims(u_arr) >= 3
        FT = promote_type(float(FT1), float(FT2))
        n_dist = length(distance_bins) - 1
        n_val = length(value_bins) - 1
        bdims = batch_dims(u_arr)
        sums = zeros(FT, n_dist, n_val, bdims...)
        counts = zeros(UInt32, n_dist, n_val, bdims...)
        auxiliary_joint2d!(sums, counts, structure_function_type, x_arr, u_arr, distance_bins, value_bins; kwargs...)
        return SFO.StructureFunction2D(structure_function_type, distance_bins, value_bins, sums, counts)
    end
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return serial_calculate_structure_function(
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

function calculate_structure_function_2d_i!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix,
    ::Val{N},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple{T1, Vararg{T1}},
    u_vecs::Tuple{T2, Vararg{T2}},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {OT, N, T1, T2}
    FT1 = eltype(T1)
    FT2 = eltype(T2)
    N3 = length(distance_bins)
    N4 = length(value_bins)

    X1 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][i], Val(N)))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][i], Val(N)))

    iter_inds = eachindex(x_vecs[1])
    for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{N, FT1}(ntuple(k -> x_vecs[k][j], Val(N)))
        U2 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][j], Val(N)))

        distance = distance_metric(X1, X2)
        dist_bin = SFH.digitize(distance, distance_bins)
        if 1 <= dist_bin < N3
            rh = SFH.r̂(X1, X2, distance_metric, distance)
            val = structure_function_type(U2 - U1, rh)
            val_bin = SFH.digitize(val, value_bins)
            
            if 1 <= val_bin < N4
                @inbounds sums_2d[dist_bin, val_bin] += val
                @inbounds counts_2d[dist_bin, val_bin] += 1
            end
        end
    end
    return nothing
end

function calculate_structure_function_2d_i(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    i::Int,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    N = length(x_vecs)
    FT1 = eltype(x_vecs[1])
    FT2 = eltype(u_vecs[1])
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)
    OT = promote_type(float(FT1), float(FT2))
    local_sums = zeros(OT, N3, N4)
    local_counts = zeros(CT, N3, N4)

    vN = Val(N)
    calculate_structure_function_2d_i!(
        local_sums,
        local_counts,
        vN,
        structure_function_type,
        i,
        x_vecs,
        u_vecs,
        BinEdges(distance_bins),
        BinEdges(value_bins);
        distance_metric = distance_metric,
    )

    return SFO.StructureFunction2D(
        structure_function_type,
        distance_bins,
        value_bins,
        local_sums,
        local_counts,
    )
end
