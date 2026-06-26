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

    # Fast path: Euclidean + D ∈ (2,3) via the SIMD compute/scatter split (distance + SF value
    # vectorize over j; the 2D (dist,value) scatter stays scalar).
    D = length(x_vecs)
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _pf_2d_simd_run!(sums_2d, counts_2d, structure_function_type, x_vecs, u_vecs,
                         distance_bins, value_bins, D == 2 ? Val(2) : Val(3))
        return nothing
    end

    vN = Val(D)
    PM.@showprogress enabled = show_progress for i in eachindex(x_vecs[1])
        calculate_structure_function_2d_i!(
            sums_2d, counts_2d, vN, structure_function_type, i, x_vecs, u_vecs,
            distance_bins, value_bins; distance_metric = distance_metric,
        )
    end
    return nothing
end

"""
    _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, dist_be, val_be, ::Val{D}, distbuf, valbuf, irange)

2D-joint point-field SIMD compute/scatter kernel over outer indices `irange`: `@simd` over `j>i`
computes distance + SF value into buffers (no scatter ⇒ vectorizes), then a scalar loop digitizes
both axes and scatters into the (dist,value) cell. Loop lives in this one kernel (called once per
range) so the `@simd` vectorizes; shared by serial + threaded.
"""
function _pf_2d_simd_pairs!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    xc::NTuple{D}, uc::NTuple{D}, dist_be, val_be, ::Val{D},
    distbuf::AbstractVector, valbuf::AbstractVector, irange,
) where {OT, CT, D}
    N = length(xc[1])
    n_dist = n_histogram_bins(dist_be)
    n_val = n_histogram_bins(val_be)
    FTx = eltype(xc[1])
    @inbounds for i in irange
        Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
        Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
        @simd for j in (i + 1):N
            Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
            dx = Xj - Xi
            dist = sqrt(LA.dot(dx, dx))
            rh = dx / dist
            Uj = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D)))
            distbuf[j] = dist
            valbuf[j] = sf(Uj - Ui, rh)
        end
        for j in (i + 1):N
            dbin = SFH.digitize(distbuf[j], dist_be)
            if 1 <= dbin <= n_dist
                vbin = SFH.digitize(valbuf[j], val_be)
                if 1 <= vbin <= n_val
                    sums2d[dbin, vbin] += valbuf[j]
                    counts2d[dbin, vbin] += one(CT)
                end
            end
        end
    end
    return nothing
end

function _pf_2d_simd_run!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, val_be, ::Val{D},
) where {OT, CT, D}
    xc = ntuple(d -> collect(x_vecs[d]), Val(D))
    uc = ntuple(d -> collect(u_vecs[d]), Val(D))
    N = length(xc[1])
    distbuf = Vector{eltype(xc[1])}(undef, N)
    valbuf = Vector{OT}(undef, N)
    _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, dist_be, val_be, Val(D), distbuf, valbuf, 1:(N - 1))
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

    return SFO.StructureFunction2DSumsAndCounts(
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
        return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums, counts)
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

    return SFO.StructureFunction2DSumsAndCounts(
        structure_function_type,
        distance_bins,
        value_bins,
        local_sums,
        local_counts,
    )
end
