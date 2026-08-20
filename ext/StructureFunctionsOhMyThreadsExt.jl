module StructureFunctionsOhMyThreadsExt

using Distances: Distances as DI
using OhMyThreads: OhMyThreads as OMT
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using ComputationalBackends: ComputationalBackends as CB
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
        vN = Val(length(u_vecs))
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
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
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
    x_tuple = ntuple(k -> view(x_arr, k, :), size(x_arr, 1))
    u_tuple = ntuple(k -> view(u_arr, k, :), size(u_arr, 1))
    vD = Val(size(u_arr, 1))
    result = OMT.tmapreduce(+, _triangle_outer_chunks(axes(x_arr, 2), Threads.nthreads())) do chunk
        local_output = zeros(OT, N3)
        local_counts = zeros(CT, N3)
        for i in chunk
            SFC.calculate_structure_function_i!(
                local_output, local_counts, vD, structure_function_type, i,
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
    plan = SF.squared_digitize_plan(dist_be)     # built once; read-only, shared across tasks
    result = OMT.tmapreduce(+, _triangle_outer_chunks(1:(Np - 1), Threads.nthreads())) do chunk
        local_output = zeros(OT, nb)
        local_counts = zeros(CT, nb)
        r2buf = Vector{FTx}(undef, Np)
        valbuf = Vector{OT}(undef, Np)
        idxbuf = Vector{Int32}(undef, Np)
        SFC._pf_simd_pairs!(local_output, local_counts, sf, xc, uc, plan, Val(D), r2buf, valbuf, idxbuf, chunk)
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
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
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
    D = length(u_vecs)
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _threaded_2d_simd!(sums_2d, counts_2d, structure_function_type, x_vecs, u_vecs,
                           distance_bins, value_bins, D == 2 ? Val(2) : Val(3))
        return nothing
    end

    # Chunked tmapreduce: O(n_tasks) allocations instead of O(N_points)
    result = OMT.tmapreduce(+, _triangle_outer_chunks(eachindex(x_vecs[1]), Threads.nthreads())) do chunk
        local_sums = zeros(OT, N3, N4)
        local_counts = zeros(CT, N3, N4)
        vN = Val(length(u_vecs))
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
    j2d_plan = SF.squared_digitize_plan(dist_be)   # built once; read-only, shared across tasks
    result = OMT.tmapreduce(+, _triangle_outer_chunks(1:(Np - 1), Threads.nthreads())) do chunk
        local_sums = zeros(OT, n_dist, n_val)
        local_counts = zeros(CT, n_dist, n_val)
        keybuf = Vector{FTx}(undef, Np)
        valbuf = Vector{OT}(undef, Np)
        idxbuf = Vector{Int32}(undef, Np)
        SFC._pf_2d_simd_pairs!(local_sums, local_counts, sf, xc, uc, j2d_plan, val_be, Val(D), keybuf, valbuf, idxbuf, chunk)
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
    x_tuple = ntuple(k -> view(x_arr, k, :), size(x_arr, 1))
    u_tuple = ntuple(k -> view(u_arr, k, :), size(u_arr, 1))
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
    ::CB.AbstractThreadedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    dist_be = BinEdges(distance_bins)
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_points = size(x, 2)
    D = size(u, 1)
    vW, vD = SFC._pair_dims(distance_metric, D)

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
        ((s1, c1), (s2, c2)) -> (s1 .+= s2; c1 .+= c2; (s1, c1)),
        _triangle_outer_chunks(1:n_points, Threads.nthreads())
    ) do chunk
        local_sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins)
        local_counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins)
        SFC._sp1d_pairs!(local_sums, local_counts, x, u, dist_be, vW, vD,
            distance_metric, n_bins, chunk)
        (local_sums, local_counts)
    end

    return (sums = sums, counts = counts)  # raw 6-row; public wrapper adds Helmholtz once
end

function SFC._dispatch_single_pass!(
    ::CB.AbstractThreadedBackend,
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    dist_be = BinEdges(distance_bins)
    n_bins = length(distance_bins) - 1
    n_points = size(x, 2)
    D = size(u, 1)
    vW, vD = SFC._pair_dims(distance_metric, D)

    # Fast path: Euclidean + D ∈ (2,3) threads the SIMD compute/scatter single-pass kernel.
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _threaded_sp_simd!(sums, counts, x, u, BinEdges(distance_bins), D == 2 ? Val(2) : Val(3))
        return sums, counts
    end

    chunk_sums, chunk_counts = OMT.tmapreduce(
        ((s1, c1), (s2, c2)) -> (s1 .+= s2; c1 .+= c2; (s1, c1)),
        _triangle_outer_chunks(1:n_points, Threads.nthreads()),
    ) do chunk
        local_sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins)
        local_counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins)
        SFC._sp1d_pairs!(local_sums, local_counts, x, u, dist_be, vW, vD,
            distance_metric, n_bins, chunk)
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
    sp_plan = SF.squared_digitize_plan(dist_be)   # built once; read-only, shared across tasks
    cs, cc = OMT.tmapreduce(
        ((s1, c1), (s2, c2)) -> (s1 .+= s2; c1 .+= c2; (s1, c1)),
        _triangle_outer_chunks(1:(Np - 1), Threads.nthreads()),
    ) do chunk
        ls = zeros(OT, SFC.SINGLE_PASS_N, nb)
        lc = zeros(CT, SFC.SINGLE_PASS_N, nb)
        keybuf = Vector{FTx}(undef, Np)
        duLbuf = Vector{OT}(undef, Np)
        dn2buf = Vector{OT}(undef, Np)
        idxbuf = Vector{Int32}(undef, Np)
        SFC._pf_sp_simd_pairs!(ls, lc, xc, uc, sp_plan, Val(D), keybuf, duLbuf, dn2buf, idxbuf, chunk)
        (ls, lc)
    end
    sums .+= cs
    counts .+= cc
    return nothing
end

function SFC._dispatch_single_pass_2d(
    ::CB.AbstractThreadedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = SFC.n_histogram_bins(distance_bins)
    n_val = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins, n_val)
    _threaded_sp2d!(sums, counts, x, u, BinEdges(distance_bins), value_bins,
        distance_metric, n_bins, n_val)
    return sums, counts
end

function SFC._dispatch_single_pass_2d!(
    ::CB.AbstractThreadedBackend,
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _threaded_sp2d!(sums_3d, counts_3d, x, u, BinEdges(distance_bins), value_bins,
        distance_metric, SFC.n_histogram_bins(distance_bins), size(sums_3d, 3))
    return sums_3d, counts_3d
end

# Round-robin outer-index chunks, thread-local accumulators reduced in place; each chunk runs the
# same single-pass 2D kernel the serial and distributed drivers use. The reduction is over the
# compact interleaved accumulator, so the unpack to (6, n_bins, n_val) runs once, not per chunk.
function _threaded_sp2d!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, value_bins, distance_metric,
    n_bins::Int, n_val::Int,
) where {OT, CT}
    n_points = size(x, 2)
    h = OMT.tmapreduce(
        (a, b) -> (a .+= b; a),
        _triangle_outer_chunks(1:n_points, Threads.nthreads()),
    ) do chunk
        hloc = SFC._sp2d_histogram(OT, n_bins, n_val)
        SFC._sp2d_fill!(hloc, x, u, dist_be, value_bins, distance_metric, n_bins, n_val, chunk)
        hloc
    end
    SFC._sp2d_unpack!(sums_3d, counts_3d, h, n_bins, n_val)
    return nothing
end

# ============================================================================================
# Threaded CPU BATCH paths — parallelize over the OUTER pair index i (geometry computed once).
#
# Round-robin i-chunks (triangle load balance, like the point-field path) with thread-local
# accumulators reduced by elementwise +. No `threadid()`. Reuses the same `_bl_run_*!` drivers
# + `_bl_*!` kernels as serial; only the executor differs. These `::AbstractArray` methods are
# more specialized than the generic core serial-fallback stubs, so they win dispatch here.
# ============================================================================================

# Executor: partition the (i, b) index space. `i` is split round-robin (triangular load balance);
# `b` is split only as far as the accumulator budget demands, because each batch chunk recomputes
# the pair geometry. Tasks in different batch chunks own disjoint output slices, so a task writes
# only its own accumulator and the slices are summed into the result afterwards.
function _bl_threaded_exec(make_accum, run_chunk!, ifull, B, accum_bytes, ws)
    nt = Threads.nthreads()
    if nt <= 1 || length(ifull) <= 1
        acc = SFC._bl_zero_accum!(SFC._bl_accum_pool(ws, make_accum, [B])[1])
        run_chunk!(acc, ifull, 1:B)
        return acc
    end

    bchunks, n_ichunks = SFC._bl_partition(B, nt, accum_bytes)
    ichunks = collect(OMT.chunks(ifull; n = n_ichunks, split = OMT.RoundRobin()))
    tasks = [(bc, isub) for bc in bchunks for isub in ichunks]
    pool = SFC._bl_accum_pool(ws, make_accum, [length(t[1]) for t in tasks])
    length(pool) == length(tasks) || throw(ArgumentError(
        "CPUSFWorkspace holds $(length(pool)) accumulators; this call needs $(length(tasks))"))

    OMT.tforeach(eachindex(tasks)) do k
        brange, isub = tasks[k]
        _bl_run_one_chunk(pool[k], run_chunk!, isub, brange)
    end

    result = SFC._bl_result_accum(ws, make_accum, B)
    for (k, (brange, _)) in enumerate(tasks)
        selectdim(result[1], 1, brange) .+= pool[k][1]
        selectdim(result[2], 1, brange) .+= pool[k][2]
    end
    return result
end

# Named function so the pooled accumulator is concretely typed inside the task (the pool is
# heterogeneous in batch width, so indexing it is a dynamic call — once per task, not per pair).
@inline function _bl_run_one_chunk(acc, run_chunk!, isub, brange)
    SFC._bl_zero_accum!(acc)
    run_chunk!(acc, isub, brange)
    return acc
end

SFC._bl_executor(::CB.AbstractThreadedBackend) = _bl_threaded_exec

function SFC.auxiliary_structure_function_threaded!(
    sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins;
    workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_1d!(sums, counts, sf_type, x, u, BinEdges(distance_bins), distance_metric,
        _bl_threaded_exec, workspace)
end

function SFC.auxiliary_joint2d_threaded!(
    sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins;
    workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_joint2d!(sums, counts, sf_type, x, u, BinEdges(distance_bins), BinEdges(value_bins),
        distance_metric, _bl_threaded_exec, workspace)
end

function SFC.threaded_calculate_structure_functions_single_pass!(
    sums::AbstractArray, counts::AbstractArray, x, u, distance_bins;
    workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_sp1d!(sums, counts, x, u, BinEdges(distance_bins), distance_metric,
        _bl_threaded_exec, workspace)
end

function SFC.threaded_calculate_structure_functions_single_pass_2d!(
    sums::AbstractArray, counts::AbstractArray, x, u, distance_bins,
    value_bins::SFC.SinglePass2DValueBins;
    workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_sp2d!(sums, counts, x, u, BinEdges(distance_bins), value_bins, distance_metric,
        _bl_threaded_exec, workspace)
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
    ::CB.AbstractThreadedBackend,
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
    D = length(u_vecs)
    vN = Val(D)
    simd = distance_metric isa DI.Euclidean && (D == 2 || D == 3)
    result = OMT.tmapreduce(+, _triangle_outer_chunks(ilist, Threads.nthreads())) do chunk
        local_sums = zeros(OT, nb)
        local_counts = zeros(CT, nb)
        if simd
            D == 2 ?
                SFC._pf_simd_partial!(local_sums, local_counts, structure_function_type, x_vecs, u_vecs, be, Val(2), chunk) :
                SFC._pf_simd_partial!(local_sums, local_counts, structure_function_type, x_vecs, u_vecs, be, Val(3), chunk)
        else
            for i in chunk
                SFC.calculate_structure_function_i!(
                    local_sums, local_counts, vN, structure_function_type, i, x_vecs, u_vecs, be;
                    distance_metric = distance_metric,
                )
            end
        end
        SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, local_sums, local_counts)
    end
    return result
end

end # module
