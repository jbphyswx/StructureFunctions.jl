module StructureFunctionsOhMyThreadsExt

using Distances: Distances as DI
using OhMyThreads: OhMyThreads as OMT
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using StructureFunctions:
    StructureFunctions as SF,
    Calculations as SFC,
    StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT,
    HelperFunctions as SFH,
    BinEdges,
    n_histogram_bins

# Signal to AutoBackend that threading is genuinely available (see backends.jl). Set in
# __init__ (load time) via a Ref — overriding the method here would be an illegal
# method-overwrite during this extension's precompilation.
function __init__()
    SFC._OHMYTHREADS_LOADED[] = true
    return nothing
end

"""
    _triangle_outer_chunks(indices, n_tasks)

Partition outer loop indices for O(N²) pair kernels where work for index `i` is
`(N - i)`. `OMT.chunks` defaults to `Consecutive()`, which assigns equal-width
contiguous blocks and severely load-imbalances this loop (~T× skew). Round-robin
(`OMT.RoundRobin()`) balances pair work across tasks.
"""
@inline _triangle_outer_chunks(indices, n_tasks::Integer) =
    OMT.chunks(indices; n = n_tasks, split = OMT.RoundRobin())

# --- 1D Tuple thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {OT, CT}
    if verbose
        @info("calculating structure function (threaded reduction via OhMyThreads)")
    end
    _ = show_progress

    # Chunked tmapreduce: O(n_tasks) allocations instead of O(N_points)
    n_bins = n_histogram_bins(distance_bins)
    distance_bins = BinEdges(distance_bins)
    result = OMT.tmapreduce(+, _triangle_outer_chunks(eachindex(x_vecs[1]), Threads.nthreads())) do chunk
        local_output = zeros(OT, n_bins)
        local_counts = zeros(CT, n_bins)
        vN = Val(length(x_vecs))
        for i in chunk
            SFC.calculate_structure_function_i!(
                local_output,
                local_counts,
                vN,
                structure_function_type,
                i,
                x_vecs,
                u_vecs,
                distance_bins;
                distance_metric = distance_metric,
            )
        end
        SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, local_output, local_counts)
    end
    output_sums .+= result.sums
    output_counts .+= result.counts
    return nothing
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {CT}
    FT1 = eltype(x_vecs[1])
    FT2 = eltype(u_vecs[1])
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(CT, N3)

    SFC.threaded_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        kwargs...,
    )

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        output,
        counts,
    )
end

# --- 1D Array thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    output_sums::AbstractVector{OT},
    output_counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {OT, CT, FT1 <: Number, FT2 <: Number}
    if verbose
        @info("calculating structure function (threaded reduction via OhMyThreads)")
    end
    _ = show_progress

    N3 = n_histogram_bins(distance_bins)
    distance_bins = BinEdges(distance_bins)

    N = size(x_arr, 1)
    if !(N in (1, 2, 3))
        throw(ArgumentError("Threaded array backend supports only 1D, 2D, or 3D inputs."))
    end

    # Fast path: Euclidean + D ∈ (2,3) threads the SIMD compute/scatter-split kernel over
    # round-robin i-chunks (per-task buffers + local accumulators; contiguous components shared).
    if distance_metric isa DI.Euclidean && N == 2
        return _threaded_pf_simd!(output_sums, output_counts, structure_function_type, x_arr, u_arr, distance_bins, Val(2))
    elseif distance_metric isa DI.Euclidean && N == 3
        return _threaded_pf_simd!(output_sums, output_counts, structure_function_type, x_arr, u_arr, distance_bins, Val(3))
    end

    # Fallback: scalar per-i over chunks
    x_tuple = ntuple(k -> view(x_arr, k, :), N)
    u_tuple = ntuple(k -> view(u_arr, k, :), N)
    result = OMT.tmapreduce(+, _triangle_outer_chunks(axes(x_arr, 2), Threads.nthreads())) do chunk
        local_output = zeros(OT, N3)
        local_counts = zeros(CT, N3)
        for i in chunk
            SFC.calculate_structure_function_i!(
                local_output, local_counts, Val(N), structure_function_type, i,
                x_tuple, u_tuple, distance_bins; distance_metric = distance_metric,
            )
        end
        SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, local_output, local_counts)
    end

    output_sums .+= result.sums
    output_counts .+= result.counts
    return nothing
end

# Threaded point-field SIMD compute/scatter split: contiguous component vectors materialized
# once (shared, read-only), per-task histogram + distbuf/valbuf, round-robin i-chunks.
function _threaded_pf_simd!(
    output_sums::AbstractVector{OT}, output_counts::AbstractVector{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType, x_arr, u_arr, dist_be, ::Val{D},
) where {OT, CT, D}
    xc = ntuple(d -> collect(view(x_arr, d, :)), Val(D))
    uc = ntuple(d -> collect(view(u_arr, d, :)), Val(D))
    Np = size(x_arr, 2)
    nb = n_histogram_bins(dist_be)
    FTx = eltype(xc[1])
    result = OMT.tmapreduce(+, _triangle_outer_chunks(1:(Np - 1), Threads.nthreads())) do chunk
        local_output = zeros(OT, nb)
        local_counts = zeros(CT, nb)
        distbuf = Vector{FTx}(undef, Np)
        valbuf = Vector{OT}(undef, Np)
        SFC._pf_simd_pairs!(local_output, local_counts, sf, xc, uc, dist_be, Val(D), distbuf, valbuf, chunk)
        SFO.StructureFunctionSumsAndCounts(sf, dist_be, local_output, local_counts)
    end
    output_sums .+= result.sums
    output_counts .+= result.counts
    return nothing
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    output = zeros(OT, N3)
    counts = zeros(CT, N3)

    SFC.threaded_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_arr,
        u_arr,
        distance_bins;
        kwargs...,
    )

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        output,
        counts,
    )
end

# --- 2D Tuple thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
) where {OT, CT}
    if verbose
        @info("calculating 2D joint structure function (threaded reduction via OhMyThreads)")
    end
    _ = show_progress

    distance_bins = BinEdges(distance_bins)
    value_bins = BinEdges(value_bins)
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)

    # Fast path: Euclidean + D ∈ (2,3) threads the 2D SIMD compute/scatter kernel.
    D = length(x_vecs)
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _threaded_2d_simd!(sums_2d, counts_2d, structure_function_type, x_vecs, u_vecs,
                           distance_bins, value_bins, D == 2 ? Val(2) : Val(3))
        return nothing
    end

    # Chunked tmapreduce: O(n_tasks) allocations instead of O(N_points)
    result = OMT.tmapreduce(+, _triangle_outer_chunks(eachindex(x_vecs[1]), Threads.nthreads())) do chunk
        local_sums = zeros(OT, N3, N4)
        local_counts = zeros(CT, N3, N4)
        vN = Val(length(x_vecs))
        for i in chunk
            SFC.calculate_structure_function_2d_i!(
                local_sums, local_counts, vN, structure_function_type, i,
                x_vecs, u_vecs, distance_bins, value_bins; distance_metric = distance_metric,
            )
        end
        SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, local_sums, local_counts)
    end

    sums_2d .+= result.sums
    counts_2d .+= result.counts
    return nothing
end

# Threaded 2D-joint point-field SIMD: contiguous components shared, per-task buffers + local
# (n_dist,n_val) accumulators, round-robin i-chunks reduced by +.
function _threaded_2d_simd!(
    sums2d::AbstractMatrix{OT}, counts2d::AbstractMatrix{CT},
    sf::SFT.AbstractPairwiseStructureFunctionType, x_vecs, u_vecs, dist_be, val_be, ::Val{D},
) where {OT, CT, D}
    xc = ntuple(d -> collect(x_vecs[d]), Val(D))
    uc = ntuple(d -> collect(u_vecs[d]), Val(D))
    Np = length(xc[1])
    n_dist = n_histogram_bins(dist_be)
    n_val = n_histogram_bins(val_be)
    FTx = eltype(xc[1])
    result = OMT.tmapreduce(+, _triangle_outer_chunks(1:(Np - 1), Threads.nthreads())) do chunk
        local_sums = zeros(OT, n_dist, n_val)
        local_counts = zeros(CT, n_dist, n_val)
        distbuf = Vector{FTx}(undef, Np)
        valbuf = Vector{OT}(undef, Np)
        SFC._pf_2d_simd_pairs!(local_sums, local_counts, sf, xc, uc, dist_be, val_be, Val(D), distbuf, valbuf, chunk)
        SFO.StructureFunction2DSumsAndCounts(sf, dist_be, val_be, local_sums, local_counts)
    end
    sums2d .+= result.sums
    counts2d .+= result.counts
    return nothing
end

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {CT}
    FT1 = eltype(x_vecs[1])
    FT2 = eltype(u_vecs[1])
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(CT, N3, N4)

    SFC.threaded_calculate_structure_function!(
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

# --- 2D Array thread-safe chunked implementation ---

function SFC.threaded_calculate_structure_function!(
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {OT, FT1 <: Number, FT2 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC.threaded_calculate_structure_function!(
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

function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    N3 = n_histogram_bins(distance_bins)
    N4 = n_histogram_bins(value_bins)

    sums_2d = zeros(OT, N3, N4)
    counts_2d = zeros(CT, N3, N4)

    SFC.threaded_calculate_structure_function!(
        sums_2d,
        counts_2d,
        structure_function_type,
        x_arr,
        u_arr,
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

# --- Threaded single-pass via OhMyThreads tmapreduce ---
#
# IMPORTANT: We use `OMT.tmapreduce` with task-local buffers instead of
# `OMT.tforeach` + `Threads.threadid()` indexing.
#
# Julia tasks are NON-STICKY: they can migrate between OS threads at any yield
# point. This means `Threads.threadid()` is NOT guaranteed to remain constant
# within a single task. Using it to index into shared per-thread buffers causes
# data races → glibc heap corruption (malloc_consolidate / corrupted size).
#
# The correct OhMyThreads pattern (used by all other threaded methods above)
# is to give each chunk its own task-local buffer, then reduce via summation.
# Outer `i` chunks use `_triangle_outer_chunks` (RoundRobin split) because
# contiguous equal-size blocks imbalance O(N²) triangle pair loops.
#
# References:
#   - OhMyThreads thread-safe storage docs:
#     https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
#   - Julia manual on task migration:
#     https://docs.julialang.org/en/v1/manual/multi-threading/#man-task-migration
#   - OhMyThreads FAQ on threadid():
#     https://juliafolds2.github.io/OhMyThreads.jl/stable/translation/

function SFC._dispatch_single_pass(
    ::SF.ThreadedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_points = size(x, 2)
    D = size(x, 1)
    vD = Val(D)

    # Fast path: Euclidean + D ∈ (2,3) threads the SIMD compute/scatter single-pass kernel.
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins)
        counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins)
        _threaded_sp_simd!(sums, counts, x, u, BinEdges(distance_bins), D == 2 ? Val(2) : Val(3))
        return (sums = sums, counts = counts)  # raw 6-row; public wrapper adds Helmholtz once
    end

    # tmapreduce: each chunk gets its own task-local (sums, counts) buffers.
    # The reducer `+` merges partial results via element-wise addition.
    # This produces O(nthreads) allocations total — not O(n_points).
    (sums, counts) = OMT.tmapreduce(
        ((s1, c1), (s2, c2)) -> (s1 .+ s2, c1 .+ c2),
        _triangle_outer_chunks(1:n_points, Threads.nthreads())
    ) do chunk
        local_sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins)
        local_counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins)

        for i in chunk
            x_i = SA.SVector{D, FT1}(ntuple(d -> x[d, i], vD))
            u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))

            for j in (i + 1):n_points
                x_j = SA.SVector{D, FT1}(ntuple(d -> x[d, j], vD))

                r = distance_metric(x_i, x_j)
                bin_idx = SFH.digitize(r, distance_bins)

                if 1 <= bin_idx <= n_bins
                    u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                    du = u_j - u_i

                    rh = SFH.r̂(x_i, x_j, distance_metric, r)
                    du_L = LA.dot(du, rh)
                    du_L2 = du_L * du_L
                    du_norm2 = LA.dot(du, du)
                    du_T2 = du_norm2 - du_L2

                    # Accumulate the six invariant bulk structure functions.
                    @inbounds local_sums[1, bin_idx] += du_norm2                 # S2SF
                    @inbounds local_sums[2, bin_idx] += du_L2                    # L2SF
                    @inbounds local_sums[3, bin_idx] += du_T2                    # T2SF
                    @inbounds local_sums[4, bin_idx] += du_L * du_norm2          # S3SF
                    @inbounds local_sums[5, bin_idx] += du_L * du_L2             # L3SF
                    @inbounds local_sums[6, bin_idx] += du_L * du_T2             # L1T2SF

                    @inbounds for t in 1:SFC.SINGLE_PASS_N
                        local_counts[t, bin_idx] += one(CT)
                    end
                end
            end
        end
        (local_sums, local_counts)
    end

    return (sums = sums, counts = counts)  # raw 6-row; public wrapper adds Helmholtz once
end

function SFC._dispatch_single_pass!(
    ::SF.ThreadedBackend,
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    n_bins = length(distance_bins) - 1
    n_points = size(x, 2)
    D = size(x, 1)
    vD = Val(D)

    # Fast path: Euclidean + D ∈ (2,3) threads the SIMD compute/scatter single-pass kernel.
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _threaded_sp_simd!(sums, counts, x, u, BinEdges(distance_bins), D == 2 ? Val(2) : Val(3))
        return sums, counts
    end

    chunk_sums, chunk_counts = OMT.tmapreduce(
        ((s1, c1), (s2, c2)) -> (s1 .+ s2, c1 .+ c2),
        _triangle_outer_chunks(1:n_points, Threads.nthreads()),
    ) do chunk
        local_sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins)
        local_counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins)

        for i in chunk
            x_i = SA.SVector{D, FT1}(ntuple(d -> x[d, i], vD))
            u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))

            for j in (i + 1):n_points
                x_j = SA.SVector{D, FT1}(ntuple(d -> x[d, j], vD))

                r = distance_metric(x_i, x_j)
                bin_idx = SFH.digitize(r, distance_bins)

                if 1 <= bin_idx <= n_bins
                    u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                    du = u_j - u_i

                    rh = SFH.r̂(x_i, x_j, distance_metric, r)
                    du_L = LA.dot(du, rh)
                    du_L2 = du_L * du_L
                    du_norm2 = LA.dot(du, du)
                    du_T2 = du_norm2 - du_L2

                    @inbounds local_sums[1, bin_idx] += du_norm2
                    @inbounds local_sums[2, bin_idx] += du_L2
                    @inbounds local_sums[3, bin_idx] += du_T2
                    @inbounds local_sums[4, bin_idx] += du_L * du_norm2
                    @inbounds local_sums[5, bin_idx] += du_L * du_L2
                    @inbounds local_sums[6, bin_idx] += du_L * du_T2

                    @inbounds for t in 1:SFC.SINGLE_PASS_N
                        local_counts[t, bin_idx] += one(CT)
                    end
                end
            end
        end
        (local_sums, local_counts)
    end

    sums .+= chunk_sums
    counts .+= chunk_counts
    return sums, counts
end

# Threaded single-pass SIMD: contiguous components shared, per-task buffers + local (6,nb)
# accumulators, round-robin i-chunks reduced by +.
function _threaded_sp_simd!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, x, u, dist_be, ::Val{D},
) where {OT, CT, D}
    xc = ntuple(d -> collect(view(x, d, :)), Val(D))
    uc = ntuple(d -> collect(view(u, d, :)), Val(D))
    Np = size(x, 2)
    nb = n_histogram_bins(dist_be)
    FTx = eltype(xc[1])
    cs, cc = OMT.tmapreduce(
        ((s1, c1), (s2, c2)) -> (s1 .+ s2, c1 .+ c2),
        _triangle_outer_chunks(1:(Np - 1), Threads.nthreads()),
    ) do chunk
        ls = zeros(OT, SFC.SINGLE_PASS_N, nb)
        lc = zeros(CT, SFC.SINGLE_PASS_N, nb)
        distbuf = Vector{FTx}(undef, Np)
        duLbuf = Vector{OT}(undef, Np)
        dn2buf = Vector{OT}(undef, Np)
        SFC._pf_sp_simd_pairs!(ls, lc, xc, uc, dist_be, Val(D), distbuf, duLbuf, dn2buf, chunk)
        (ls, lc)
    end
    sums .+= cs
    counts .+= cc
    return nothing
end

function SFC._dispatch_single_pass_2d(
    ::SF.ThreadedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    vb0 = SFC._sp2d_value_bin_at(value_bins, 1)
    n_val = length(vb0) - 1
    n_points = size(x, 2)
    D = size(x, 1)
    vD = Val(D)

    (sums, counts) = OMT.tmapreduce(
        ((s1, c1), (s2, c2)) -> (s1 .+ s2, c1 .+ c2),
        _triangle_outer_chunks(1:n_points, Threads.nthreads()),
    ) do chunk
        local_sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins, n_val)
        local_counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins, n_val)

        for i in chunk
            x_i = SA.SVector{D, FT1}(ntuple(d -> x[d, i], vD))
            u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))

            for j in (i + 1):n_points
                x_j = SA.SVector{D, FT1}(ntuple(d -> x[d, j], vD))

                r = distance_metric(x_i, x_j)
                bin_idx = SFH.digitize(r, distance_bins)

                if 1 <= bin_idx <= n_bins
                    u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                    du = u_j - u_i

                    rh = SFH.r̂(x_i, x_j, distance_metric, r)
                    du_L = LA.dot(du, rh)
                    du_L2 = du_L * du_L
                    du_norm2 = LA.dot(du, du)
                    du_T2 = du_norm2 - du_L2

                    vals = (
                        du_norm2,
                        du_L2,
                        du_T2,
                        du_L * du_norm2,
                        du_L * du_L2,
                        du_L * du_T2,
                    )

                    for t in 1:SFC.SINGLE_PASS_N
                        vb = SFC._sp2d_value_bin_at(value_bins, t)
                        vbin = SFH.digitize(vals[t], vb)
                        n_val_t = length(vb) - 1
                        if 1 <= vbin <= n_val_t && vbin <= n_val
                            @inbounds local_sums[t, bin_idx, vbin] += vals[t]
                            @inbounds local_counts[t, bin_idx, vbin] += 1
                        end
                    end
                end
            end
        end
        (local_sums, local_counts)
    end

    return sums, counts
end

function SFC._dispatch_single_pass_2d!(
    ::SF.ThreadedBackend,
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    n_bins = length(distance_bins) - 1
    n_val = size(sums_3d, 3)
    n_points = size(x, 2)
    D = size(x, 1)
    vD = Val(D)

    chunk_sums, chunk_counts = OMT.tmapreduce(
        ((s1, c1), (s2, c2)) -> (s1 .+ s2, c1 .+ c2),
        _triangle_outer_chunks(1:n_points, Threads.nthreads()),
    ) do chunk
        local_sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins, n_val)
        local_counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins, n_val)

        for i in chunk
            x_i = SA.SVector{D, FT1}(ntuple(d -> x[d, i], vD))
            u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))

            for j in (i + 1):n_points
                x_j = SA.SVector{D, FT1}(ntuple(d -> x[d, j], vD))

                r = distance_metric(x_i, x_j)
                bin_idx = SFH.digitize(r, distance_bins)

                if 1 <= bin_idx <= n_bins
                    u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                    du = u_j - u_i

                    rh = SFH.r̂(x_i, x_j, distance_metric, r)
                    du_L = LA.dot(du, rh)
                    du_L2 = du_L * du_L
                    du_norm2 = LA.dot(du, du)
                    du_T2 = du_norm2 - du_L2

                    vals = (
                        du_norm2,
                        du_L2,
                        du_T2,
                        du_L * du_norm2,
                        du_L * du_L2,
                        du_L * du_T2,
                    )

                    for t in 1:SFC.SINGLE_PASS_N
                        vb = SFC._sp2d_value_bin_at(value_bins, t)
                        vbin = SFH.digitize(vals[t], vb)
                        n_val_t = length(vb) - 1
                        if 1 <= vbin <= n_val_t && vbin <= n_val
                            @inbounds local_sums[t, bin_idx, vbin] += vals[t]
                            @inbounds local_counts[t, bin_idx, vbin] += 1
                        end
                    end
                end
            end
        end
        (local_sums, local_counts)
    end

    sums_3d .+= chunk_sums
    counts_3d .+= chunk_counts
    return sums_3d, counts_3d
end

# ============================================================================================
# Threaded CPU BATCH paths — parallelize over the OUTER pair index i (geometry computed once).
#
# Round-robin i-chunks (triangle load balance, like the point-field path) with thread-local
# accumulators reduced by elementwise +. No `threadid()`. Reuses the same `_bl_run_*!` drivers
# + `_bl_*!` kernels as serial; only the executor differs. These `::AbstractArray` methods are
# more specialized than the generic core serial-fallback stubs, so they win dispatch here.
# ============================================================================================

@inline _bl_accum_reduce(a, b) = (a[1] .+ b[1], a[2] .+ b[2])

# per-chunk work as a named function so the assigned `acc` is not a boxed closure capture
# (OhMyThreads rejects boxed captures; see its boxing docs).
@inline function _bl_run_one_chunk(make_accum, run_chunk!, isub)
    acc = make_accum()
    run_chunk!(acc, isub)
    return acc
end

# executor: round-robin chunks of the outer i-range; per-chunk thread-local accumulators reduced by +.
function _bl_threaded_exec(make_accum, run_chunk!, ifull)
    nt = Threads.nthreads()
    (nt <= 1 || length(ifull) <= 1) && return _bl_run_one_chunk(make_accum, run_chunk!, ifull)
    return OMT.tmapreduce(_bl_accum_reduce, OMT.chunks(ifull; n = nt, split = OMT.RoundRobin())) do isub
        _bl_run_one_chunk(make_accum, run_chunk!, isub)
    end
end

function SFC.auxiliary_structure_function_threaded!(
    sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...,
)
    SFC._bl_run_1d!(sums, counts, sf_type, x, u, BinEdges(distance_bins), _bl_threaded_exec)
end

function SFC.auxiliary_joint2d_threaded!(
    sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins; kwargs...,
)
    SFC._bl_run_joint2d!(sums, counts, sf_type, x, u, BinEdges(distance_bins), BinEdges(value_bins), _bl_threaded_exec)
end

function SFC.threaded_calculate_structure_functions_single_pass!(
    sums::AbstractArray, counts::AbstractArray, x, u, distance_bins; kwargs...,
)
    SFC._bl_run_sp1d!(sums, counts, x, u, BinEdges(distance_bins), _bl_threaded_exec)
end

function SFC.threaded_calculate_structure_functions_single_pass_2d!(
    sums::AbstractArray, counts::AbstractArray, x, u, distance_bins,
    value_bins::SFC.SinglePass2DValueBins; kwargs...,
)
    SFC._bl_run_sp2d!(sums, counts, x, u, BinEdges(distance_bins), value_bins, _bl_threaded_exec)
end

# Batched (ndims(u) >= 3) non-mutating joint-2D. The AbstractMatrix method earlier is more
# specific and handles the point-field case; this is entered only for batched inputs (the
# non-mutating ThreadedBackend dispatch routes value_bins here as the trailing argument).
function SFC.threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractArray{FT1},
    u_arr::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_dist = n_histogram_bins(distance_bins)
    n_val = n_histogram_bins(value_bins)
    bdims = size(u_arr)[3:end]
    sums = zeros(OT, n_dist, n_val, bdims...)
    counts = zeros(CT, n_dist, n_val, bdims...)
    SFC.auxiliary_joint2d_threaded!(sums, counts, structure_function_type, x_arr, u_arr, distance_bins, value_bins; kwargs...)
    return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums, counts)
end

# Threaded partial over an explicit outer-index list (hybrid distributed+threaded: a worker
# threads over its assigned i-list). Round-robin chunks for triangle balance; reduce by +.
function SFC._partial_sums_counts(
    ::SFC.ThreadedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ilist;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {CT}
    OT = promote_type(float(eltype(eltype(x_vecs))), float(eltype(eltype(u_vecs))))
    nb = n_histogram_bins(distance_bins)
    be = BinEdges(distance_bins)
    vN = Val(length(x_vecs))
    result = OMT.tmapreduce(+, _triangle_outer_chunks(ilist, Threads.nthreads())) do chunk
        local_sums = zeros(OT, nb)
        local_counts = zeros(CT, nb)
        for i in chunk
            SFC.calculate_structure_function_i!(
                local_sums, local_counts, vN, structure_function_type, i, x_vecs, u_vecs, be;
                distance_metric = distance_metric,
            )
        end
        SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, local_sums, local_counts)
    end
    return result
end

end # module
