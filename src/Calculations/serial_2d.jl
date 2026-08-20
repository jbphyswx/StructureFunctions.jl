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
    D = length(u_vecs)
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
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, val_be, ::Val{D},
    keybuf::AbstractVector, valbuf::AbstractVector, idxbuf::AbstractVector{Int32}, irange,
) where {OT, CT, D}
    N = length(xc[1])
    n_dist = n_histogram_bins(plan)
    n_val = n_histogram_bins(val_be)
    FTx = eltype(xc[1])
    @inbounds for i in irange
        Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
        Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
        @simd for j in (i + 1):N
            Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
            dx = Xj - Xi
            r2 = LA.dot(dx, dx)
            Uj = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D)))
            keybuf[j] = digitize_key(plan, r2)
            valbuf[j] = SFT._sf_raw(sf, Uj - Ui, dx, r2)
            if has_vector_index(plan)
                idxbuf[j] = squared_approx_index(plan, r2)
            end
        end
        for j in (i + 1):N
            dbin = squared_bin(plan, keybuf[j], idxbuf[j])
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
    N = length(x_vecs[1])
    return _pf_2d_simd_partial!(sums2d, counts2d, sf, x_vecs, u_vecs, dist_be, val_be, Val(D), 1:(N - 1))
end

"""
    _pf_2d_simd_partial!(sums2d, counts2d, sf, x_vecs, u_vecs, dist_be, val_be, ::Val{D}, ilist)

Run [`_pf_2d_simd_pairs!`](@ref) over an explicit outer-index list, materializing the contiguous
component vectors and scratch buffers this worker needs. Shared by the serial, distributed and MPI
drivers, whose inputs arrive as strided views.
"""
function _pf_2d_simd_partial!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple, u_vecs::Tuple, dist_be, val_be, ::Val{D}, ilist,
) where {OT, CT, D}
    xc = ntuple(d -> collect(x_vecs[d]), Val(D))
    uc = ntuple(d -> collect(u_vecs[d]), Val(D))
    N = length(xc[1])
    keybuf = Vector{eltype(xc[1])}(undef, N)
    valbuf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    plan = squared_digitize_plan(dist_be)
    _pf_2d_simd_pairs!(sums2d, counts2d, sf, xc, uc, plan, val_be, Val(D), keybuf, valbuf, idxbuf, ilist)
    return nothing
end

"""
    _partial_2d_sums_counts(inner, sf_type, x_vecs, u_vecs, distance_bins, value_bins, ilist; kwargs...)

Partial 2D-joint sums/counts over an explicit outer-index list `ilist`, the 2D analogue of
[`_partial_sums_counts`](@ref). Euclidean `D ∈ {2,3}` takes the SIMD compute/scatter kernel; other
metrics fall back to the scalar per-`i` kernel. Returns `(sums, counts)`.
"""
function _partial_2d_sums_counts(
    ::CB.AbstractExecutionBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ilist;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    _assert_counts_representable(CT, length(x_vecs[1]))
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nd = n_histogram_bins(distance_bins)
    nv = n_histogram_bins(value_bins)
    sums = zeros(OT, nd, nv)
    counts = zeros(CT, nd, nv)
    dist_be = BinEdges(distance_bins)
    val_be = BinEdges(value_bins)
    D = length(u_vecs)

    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        vD = D == 2 ? Val(2) : Val(3)
        _pf_2d_simd_partial!(sums, counts, structure_function_type, x_vecs, u_vecs, dist_be, val_be, vD, ilist)
        return sums, counts
    end

    vN = Val(D)
    for i in ilist
        calculate_structure_function_2d_i!(
            sums, counts, vN, structure_function_type, i, x_vecs, u_vecs, dist_be, val_be;
            distance_metric = distance_metric,
        )
    end
    return sums, counts
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
    _assert_counts_representable(CT, length(x_vecs[1]))
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
    x_tuple = ntuple(k -> view(x_arr, k, :), size(x_arr, 1))
    u_tuple = ntuple(k -> view(u_arr, k, :), size(u_arr, 1))
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
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    if ndims(u_arr) >= 3
        FT = promote_type(float(FT1), float(FT2))
        n_dist = length(distance_bins) - 1
        n_val = length(value_bins) - 1
        bdims = batch_dims(u_arr)
        sums = zeros(FT, n_dist, n_val, bdims...)
        # Consumed here, where `counts` is allocated. `auxiliary_joint2d!` takes already-allocated
        # buffers, so forwarding `count_eltype` into it is a MethodError.
        counts = zeros(CT, n_dist, n_val, bdims...)
        auxiliary_joint2d!(sums, counts, structure_function_type, x_arr, u_arr, distance_bins, value_bins; kwargs...)
        return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums, counts)
    end
    x_tuple = ntuple(k -> view(x_arr, k, :), size(x_arr, 1))
    u_tuple = ntuple(k -> view(u_arr, k, :), size(u_arr, 1))
    return serial_calculate_structure_function(
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins,
        value_bins;
        count_eltype = count_eltype,
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

    # `N` is the velocity dimension; the coordinate count is the x tuple's own (static) length.
    W = length(x_vecs)
    vW = Val(W)
    X1 = SA.SVector{W, FT1}(ntuple(k -> x_vecs[k][i], vW))
    U1 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][i], Val(N)))

    iter_inds = eachindex(x_vecs[1])
    geom = SFH.pair_geometry_for(distance_metric, Val(N))
    for j in (i + 1):last(iter_inds)
        X2 = SA.SVector{W, FT1}(ntuple(k -> x_vecs[k][j], vW))
        U2 = SA.SVector{N, FT2}(ntuple(k -> u_vecs[k][j], Val(N)))

        ok, distance, frame = SFH.pair_frame(geom, X1, X2)
        dist_bin = SFH.digitize(distance, distance_bins)
        if ok && 1 <= dist_bin < N3
            δu, rh = SFH.pair_increments(geom, frame, distance, X1, X2, U1, U2)
            val = structure_function_type(δu, rh)
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
    N = length(u_vecs)
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
