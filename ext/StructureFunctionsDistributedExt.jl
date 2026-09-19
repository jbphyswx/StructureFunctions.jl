"""
Distributed execution backend for structure functions utilizing Distributed.jl.
"""
module StructureFunctionsDistributedExt

using Distributed: Distributed
using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO,
    AbstractBinEdges, LinearBinEdges, LogBinEdges

# A `LocalManager` worker is another process on this node, competing for the cores the threaded
# backend already has; any other manager placed the worker somewhere this process cannot reach.
SFC.distributed_adds_hardware(::Val{:distributed}) =
    Distributed.nworkers() > 1 &&
    any(w -> !(Distributed.worker_from_id(w).manager isa Distributed.LocalManager),
        Distributed.workers())

# --- Non-Mutating 1D Dispatch (returns the raw accumulator; public boundary finalizes) ---
function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    backend = nothing,
    kwargs...,
)
    return _parallel_calculate_structure_function_core(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        inner = CB.local_backend(db),
        kwargs...,
    )
end

function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
)
    geom, x_vecs, u_vecs = SFC._prepared_tuples(distance_metric, x_arr, u_arr)
    return SFC._dispatch_execution_backend(
        db,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        geometry = geom,
        kwargs...,
    )
end

function _parallel_calculate_structure_function_core(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    geometry = SFC.default_geometry(u_vecs),
    verbose = true,
    show_progress = true,
    count_eltype::Type{CT} = UInt32,
    inner::CB.AbstractLocalBackend = CB.SerialBackend(),
    weights = nothing,
    kwargs...,
) where {CT}
    if verbose
        @info("calculating structure function (distributed reduction, inner=$(nameof(typeof(inner))))")
    end

    # One balanced i-list per worker; each worker computes its partial via `inner` (Serial, or
    # Threaded for hybrid distributed+threaded). Collect the partials and accumulate them into a
    # preallocated, concretely-typed buffer. (We deliberately avoid `@distributed (+)`, whose
    # reduction is inferred as `Any` and would force a return-type assertion and make
    # AutoBackend+Distributed type-unstable. `pmap`-into-typed-buffer mirrors the batched path
    # and infers natively.)
    N = length(x_vecs[1])
    nw = max(1, Distributed.nworkers())
    chunks = SFC._balanced_index_chunks(N, nw)
    OT0 = promote_type(float(eltype(x_vecs[1])), float(eltype(u_vecs[1])))
    w = SFC._pair_weights(weights, N, OT0)

    # A polynomial operator on a line has the exact `O(N log N)` route, and that sweep executes
    # through `sweep_reduce!`, which this extension implements — so it distributes here without a
    # second decomposition. Reaching it through the pair loop instead costs `O(N²)`.
    if SFC._on_a_line(geometry, structure_function_type)
        SFC._cull_reject_unsupported(get(kwargs, :culling, SFC.AutoCulling()),
                                     "the sorted line route")
        nb0 = SFC.n_histogram_bins(distance_bins)
        lsums = zeros(OT0, nb0)
        lcounts = zeros(CT, nb0)
        SFC.sorted_line_sweep!(lsums, lcounts, structure_function_type, x_vecs[1],
            reshape(collect(u_vecs[1]), 1, :), distance_bins, Val(1), Val(1), Val(0);
            weights = w, backend = CB.DistributedBackend(inner))
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins,
                                                  lsums, lcounts)
    end

    partials = Distributed.pmap(chunks) do ch
        SFC._partial_sums_counts(
            inner, structure_function_type, x_vecs, u_vecs, distance_bins, ch;
            geometry = geometry, count_eltype = count_eltype, weights = w,
        )
    end

    OT = promote_type(float(eltype(x_vecs[1])), float(eltype(u_vecs[1])))
    nb = SFC.n_histogram_bins(distance_bins)
    sums = zeros(OT, nb)
    counts = zeros(CT, nb)
    for p in partials
        sums .+= p.sums
        counts .+= p.counts
    end
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
end

"""
    _dist_bl_exec(inner_exec) -> executor

The batch-leading executor for the Distributed backend: split the **outer pair index** across
workers, run the batch-leading kernel on each share, and add the partials.

Splitting the outer index rather than the slice axis is what keeps the amortisation the
batch-leading kernels exist for — a pair's geometry is computed once and reused across all `B`
slices inside the kernel. A slice-wise split recomputes every pair's frame, distance and bin once
per slice, which costs `B` times the geometry and measured 6.4× the serial total at `B = 8`.
Same shape as the MPI extension's `_mpi_bl_exec`, with `pmap` and a sum where that has an
`Allreduce!`.
"""
function _dist_bl_exec(inner_exec)
    return function (make_accum, run_chunk!, ifull, B, accum_bytes, ws)
        nw = max(1, Distributed.nworkers())
        chunks = SFC._balanced_index_chunks(length(ifull), nw)
        shares = [ifull[c] for c in chunks if !isempty(c)]
        length(shares) <= 1 && return inner_exec(make_accum, run_chunk!, ifull, B, accum_bytes, ws)
        parts = Distributed.pmap(shares) do share
            acc = inner_exec(make_accum, run_chunk!, share, B, accum_bytes, nothing)
            (Array(acc[1]), Array(acc[2]))
        end
        total = parts[1]
        for k in 2:length(parts)
            total[1] .+= parts[k][1]
            total[2] .+= parts[k][2]
        end
        return total
    end
end

SFC._bl_executor(b::CB.AbstractDistributedBackend) = _dist_bl_exec(SFC._bl_executor(CB.local_backend(b)))

# --- Auxiliary-axis, batch and single-pass entries ---
# Every one runs the batch-leading kernel over an outer-index split through `_dist_bl_exec`.
# Matches ndims(u) >= 3 only (the AbstractMatrix point-field methods above are more specific).

"""Accumulate a distributed auxiliary-axis sweep into the `(n_bins, B)` reshape of `sums`/`counts`."""
function _dist_accumulate_1d!(
    sums, counts, db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector, B::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    weights = SFC.NoWeights(),
) where {CT}
    # The batch-leading kernel over an outer-index split, not a slice split: a pair's geometry is
    # computed once and reused across all `B` slices, which a slice-wise split would repeat `B`
    # times. `sums`/`counts` arrive as the `(n_bins, B)` reshape `_bl_run_1d!` accumulates into.
    SFC._bl_run_1d!(sums, counts, structure_function_type, x, u,
        SFC.BinEdges(distance_bins), distance_metric, _dist_bl_exec(SFC._bl_executor(CB.local_backend(db)));
        weights = weights)
    return nothing
end

"""Accumulate a distributed joint slice sweep into the `(n_dist, n_val, B)` reshape."""
function _dist_accumulate_2d!(
    sums, counts, db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector, value_bins::AbstractVector,
    B::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    weights = SFC.NoWeights(),
) where {CT}
    SFC._bl_run_joint2d!(sums, counts, structure_function_type, x, u,
        SFC.BinEdges(distance_bins), SFC.BinEdges(value_bins), distance_metric,
        _dist_bl_exec(SFC._bl_executor(CB.local_backend(db))); weights = weights)
    return nothing
end

function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
    count_eltype::Type{CT} = UInt32,
    backend = nothing,
    kwargs...,
) where {CT}
    verbose && @info("calculating batched structure function (distributed over slices, inner=$(nameof(typeof(CB.local_backend(db)))))")
    bdims = size(u)[3:end]
    B = prod(bdims)
    nb = SFC.n_histogram_bins(distance_bins)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, nb, B)
    counts = zeros(CT, nb, B)
    _dist_accumulate_1d!(
        sums, counts, db, structure_function_type, x, u, distance_bins, B;
        distance_metric = distance_metric, count_eltype = CT,
        weights = get(kwargs, :weights, SFC.NoWeights()),
    )

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type, distance_bins,
        reshape(sums, nb, bdims...), reshape(counts, nb, bdims...),
    )
end

# --- Auto-Binning 1D Dispatch ---
function SFC._dispatch_execution_backend(
    ::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing::Type{<:AbstractBinEdges} = LogBinEdges,
    verbose = true,
    show_progress = true,
    backend = nothing,
    kwargs...,
)
    min_distance, max_distance = Inf, 0.0
    n_distance_bins = distance_bins

    if verbose
        @info("Calculating min and max distances and generating bins")
    end

    min_distance, max_distance =
        PM.@showprogress enabled = show_progress Distributed.@distributed (
            (x, y) -> (min(x[1], y[1]), max(x[2], y[2]))
        ) for i in eachindex(x_vecs[1])
            SFC.minmax_i(i, x_vecs, distance_metric)
        end

    min_distance = prevfloat(min_distance)
    if bin_spacing === LinearBinEdges
        actual_bins = LinearBinEdges(range(min_distance, max_distance, length = n_distance_bins + 1))
    elseif bin_spacing === LogBinEdges
        edge_vec = 10 .^ range(log10(min_distance), log10(max_distance), length = n_distance_bins + 1)
        edge_vec[1] = min_distance
        edge_vec[end] = max_distance
        actual_bins = LogBinEdges(edge_vec)
    else
        throw(ArgumentError("bin_spacing must be LinearBinEdges or LogBinEdges"))
    end

    return SFC._dispatch_execution_backend(
        CB.DistributedBackend(),
        structure_function_type,
        x_vecs,
        u_vecs,
        actual_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
        kwargs...,
    )
end

# --- Non-Mutating 2D Dispatch ---
function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
    count_eltype::Type{CT} = UInt32,
    backend = nothing,
    kwargs...,
) where {CT}
    if verbose
        @info("calculating 2D joint structure function (distributed reduction)")
    end

    # Each worker accumulates its stride-`nw` share of the outer indices into a local 2D buffer,
    # then we sum the partials into a preallocated typed buffer. (Same rationale as the 1D core:
    # avoids `@distributed (+)`'s `Any`-typed reduction / the return-type assertion.)
    N = length(x_vecs[1])
    nw = max(1, Distributed.nworkers())
    chunks = SFC._balanced_index_chunks(N, nw)

    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    OT = promote_type(float(eltype(x_vecs[1])), float(eltype(u_vecs[1])))
    inner = CB.local_backend(db)

    geom = get(kwargs, :geometry, SFC.default_geometry(u_vecs))
    w2 = SFC._pair_weights(get(kwargs, :weights, nothing), N, OT)
    axis2 = get(kwargs, :second_axis, SFC.InvariantValueAxis())
    partials = Distributed.pmap(chunks) do ichunk
        SFC._partial_2d_sums_counts(
            inner, structure_function_type, x_vecs, u_vecs, distance_bins, value_bins, ichunk;
            geometry = geom, count_eltype = CT, weights = w2, second_axis = axis2,
        )
    end

    sums = zeros(OT, nd, nv)
    counts = zeros(CT, nd, nv)
    for (ls, lc) in partials
        sums .+= ls
        counts .+= lc
    end
    return SFO.StructureFunction2DSumsAndCounts(structure_function_type, distance_bins, value_bins, sums, counts)
end

function SFC._dispatch_execution_backend(
    ::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
)
    geom, x_vecs, u_vecs = SFC._prepared_tuples(
        get(kwargs, :distance_metric, DI.Euclidean()), x_arr, u_arr,
    )
    return SFC._dispatch_execution_backend(
        CB.DistributedBackend(),
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        geometry = geom,
        SFC._without_metric(kwargs)...,
    )
end

# --- Single Pass Dispatch ---
function SFC._dispatch_single_pass(
    ::CB.AbstractDistributedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = SFC.n_histogram_bins(distance_bins)
    n_points = size(x, 2)

    # One accumulator per worker chunk, not per outer index: the old `@distributed (+)` body
    # allocated a (12, n_bins) matrix for every `i` and reduced O(N) full matrices. Chunks also let
    # each worker take the SIMD kernel, and the accumulator is typed from the inputs.
    chunks = SFC._balanced_index_chunks(n_points, max(1, Distributed.nworkers()))
    partials = Distributed.pmap(chunks) do ch
        SFC._partial_single_pass_1d(
            x, u, distance_bins, ch;
            distance_metric = distance_metric, count_eltype = CT,
        )
    end

    sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins)
    counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins)
    for (s, c) in partials
        sums .+= s
        counts .+= c
    end
    return (sums = sums, counts = counts)  # raw 6-row; public wrapper adds Helmholtz once
end

function SFC._dispatch_single_pass_2d(
    ::CB.AbstractDistributedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = SFC.n_histogram_bins(distance_bins)
    n_val = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    n_points = size(x, 2)

    # One accumulator per worker chunk, not per outer index: the old `@distributed (+)` body
    # allocated a (12, n_bins, n_val) matrix for every `i` and reduced O(N) of them.
    chunks = SFC._balanced_index_chunks(n_points, max(1, Distributed.nworkers()))
    partials = Distributed.pmap(chunks) do ch
        SFC._partial_single_pass_2d(
            x, u, distance_bins, value_bins, ch;
            distance_metric = distance_metric, count_eltype = CT,
        )
    end

    sums = zeros(OT, SFC.SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SFC.SINGLE_PASS_N, n_bins, n_val)
    for (s, c) in partials
        sums .+= s
        counts .+= c
    end
    return (sums = sums, counts = counts)
end

# --- Mutating 1D Dispatch ---
function SFC._dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend,
    sums::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    kwargs...,
) where {OT, CT}
    result = SFC._dispatch_execution_backend(
        CB.DistributedBackend(),
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        kwargs...,
    )
    sums .+= result.sums
    counts .+= result.counts
    return nothing
end

function SFC._dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend,
    sums::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector;
    kwargs...,
) where {OT, CT, FT1 <: Number, FT2 <: Number}
    geom, x_tuple, u_tuple = SFC._prepared_tuples(
        get(kwargs, :distance_metric, DI.Euclidean()), x_arr, u_arr,
    )
    return SFC._dispatch_execution_backend!(
        CB.DistributedBackend(),
        sums,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        geometry = geom,
        SFC._without_metric(kwargs)...,
    )
end

# --- Mutating 2D Dispatch ---
function SFC._dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend,
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {OT, CT}
    result = SFC._dispatch_execution_backend(
        CB.DistributedBackend(),
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

function SFC._dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend,
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {OT, CT, FT1 <: Number, FT2 <: Number}
    geom, x_tuple, u_tuple = SFC._prepared_tuples(
        get(kwargs, :distance_metric, DI.Euclidean()), x_arr, u_arr,
    )
    return SFC._dispatch_execution_backend!(
        CB.DistributedBackend(),
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


# --- Non-mutating joint dispatch over auxiliary axes ---
function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
    count_eltype::Type{CT} = UInt32,
    backend = nothing,
    kwargs...,
) where {CT}
    verbose && @info("calculating batched 2D joint structure function (distributed over slices, inner=$(nameof(typeof(CB.local_backend(db)))))")
    bdims = size(u)[3:end]
    B = prod(bdims)
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, nd, nv, B)
    counts = zeros(CT, nd, nv, B)
    _dist_accumulate_2d!(
        sums, counts, db, structure_function_type, x, u, distance_bins, value_bins, B;
        distance_metric = distance_metric, count_eltype = CT,
        weights = get(kwargs, :weights, SFC.NoWeights()),
    )
    return SFO.StructureFunction2DSumsAndCounts(
        structure_function_type, distance_bins, value_bins,
        reshape(sums, nd, nv, bdims...), reshape(counts, nd, nv, bdims...),
    )
end

# --- Mutating dispatch over auxiliary axes ---
function SFC._dispatch_execution_backend!(
    db::CB.AbstractDistributedBackend,
    sums::AbstractArray, counts::AbstractArray,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
)
    B = prod(size(u)[3:end])
    nb = SFC.n_histogram_bins(distance_bins)
    _dist_accumulate_1d!(
        reshape(sums, nb, B), reshape(counts, nb, B), db, structure_function_type, x, u,
        distance_bins, B; distance_metric = distance_metric, count_eltype = eltype(counts),
        weights = get(kwargs, :weights, SFC.NoWeights()),
    )
    return nothing
end

function SFC._dispatch_execution_backend!(
    db::CB.AbstractDistributedBackend,
    sums::AbstractArray, counts::AbstractArray,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector, value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
)
    B = prod(size(u)[3:end])
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    _dist_accumulate_2d!(
        reshape(sums, nd, nv, B), reshape(counts, nd, nv, B), db, structure_function_type, x, u,
        distance_bins, value_bins, B; distance_metric = distance_metric,
        count_eltype = eltype(counts), weights = get(kwargs, :weights, SFC.NoWeights()),
    )
    return nothing
end

# --- Slice batch drivers ---
function SFC._dispatch_batch!(
    db::CB.AbstractDistributedBackend, sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...,
)
    B = prod(size(u)[3:end])
    nb = SFC.n_histogram_bins(distance_bins)
    _dist_accumulate_1d!(
        reshape(sums, nb, B), reshape(counts, nb, B), db, sf_type, x, u, distance_bins, B;
        distance_metric = distance_metric, count_eltype = eltype(counts),
        weights = get(kwargs, :weights, SFC.NoWeights()),
    )
    return nothing
end

function SFC._dispatch_2d_batch!(
    db::CB.AbstractDistributedBackend, sums, counts, sf_type, x, u, distance_bins, value_bins;
    distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...,
)
    B = prod(size(u)[3:end])
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    _dist_accumulate_2d!(
        reshape(sums, nd, nv, B), reshape(counts, nd, nv, B), db, sf_type, x, u,
        distance_bins, value_bins, B; distance_metric = distance_metric,
        count_eltype = eltype(counts), weights = get(kwargs, :weights, SFC.NoWeights()),
    )
    return nothing
end

function SFC._dispatch_single_pass_batch!(
    db::CB.AbstractDistributedBackend, sums, counts, x, u, distance_bins;
    distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...,
)
    # Outer-index split through the batch-leading kernel, so a pair's geometry is computed once
    # and reused across the slices; see `_dist_bl_exec`.
    SFC._bl_run_sp1d!(sums, counts, x, u, SFC.BinEdges(distance_bins), distance_metric,
        _dist_bl_exec(SFC._bl_executor(CB.local_backend(db)));
        weights = get(kwargs, :weights, SFC.NoWeights()))
    return nothing
end

function SFC._dispatch_single_pass_2d_batch!(
    db::CB.AbstractDistributedBackend, sums, counts, x, u, distance_bins,
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...,
)
    SFC._bl_run_sp2d!(sums, counts, x, u, SFC.BinEdges(distance_bins), value_bins,
        distance_metric, _dist_bl_exec(SFC._bl_executor(CB.local_backend(db)));
        weights = get(kwargs, :weights, SFC.NoWeights()))
    return nothing
end

# --- Mutating single-pass over a point list ---
function SFC._dispatch_single_pass!(
    db::CB.AbstractDistributedBackend, sums::AbstractMatrix, counts::AbstractMatrix,
    x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...,
)
    r = SFC._dispatch_single_pass(db, x, u, distance_bins; count_eltype = eltype(counts), kwargs...)
    sums .+= r.sums
    counts .+= r.counts
    return sums, counts
end

function SFC._dispatch_single_pass_2d!(
    db::CB.AbstractDistributedBackend, sums_3d::AbstractArray, counts_3d::AbstractArray,
    x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins; kwargs...,
)
    r = SFC._dispatch_single_pass_2d(
        db, x, u, distance_bins, value_bins; count_eltype = eltype(counts_3d), kwargs...,
    )
    sums_3d .+= r.sums
    counts_3d .+= r.counts
    return sums_3d, counts_3d
end

# --- Single pass over auxiliary axes ---
# One slice per work item, like every other auxiliary-axis entry here, so `AutoBackend` has a
# distributed method to choose and never has to refuse.
function SFC._dispatch_single_pass(
    db::CB.AbstractDistributedBackend,
    ::Union{SFC.SharedPositionField, SFC.VaryingPositionField},
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(), count_eltype::Type{CT} = UInt32, kwargs...,
) where {CT}
    bdims = size(u)[3:end]
    B = prod(bdims)
    nb = SFC.n_histogram_bins(distance_bins)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, SFC.SINGLE_PASS_N, nb, B)
    counts = zeros(CT, SFC.SINGLE_PASS_N, nb, B)
    SFC._bl_run_sp1d!(sums, counts, x, u, SFC.BinEdges(distance_bins), distance_metric,
        _dist_bl_exec(SFC._bl_executor(CB.local_backend(db)));
        weights = get(kwargs, :weights, SFC.NoWeights()))
    return (sums = reshape(sums, SFC.SINGLE_PASS_N, nb, bdims...),
            counts = reshape(counts, SFC.SINGLE_PASS_N, nb, bdims...))
end

function SFC._dispatch_single_pass_2d(
    db::CB.AbstractDistributedBackend,
    ::Union{SFC.SharedPositionField, SFC.VaryingPositionField},
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(), count_eltype::Type{CT} = UInt32, kwargs...,
) where {CT}
    bdims = size(u)[3:end]
    B = prod(bdims)
    nb = SFC.n_histogram_bins(distance_bins)
    nv = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, SFC.SINGLE_PASS_N, nb, nv, B)
    counts = zeros(CT, SFC.SINGLE_PASS_N, nb, nv, B)
    SFC._bl_run_sp2d!(sums, counts, x, u, SFC.BinEdges(distance_bins), value_bins,
        distance_metric, _dist_bl_exec(SFC._bl_executor(CB.local_backend(db)));
        weights = get(kwargs, :weights, SFC.NoWeights()))
    return (sums = reshape(sums, SFC.SINGLE_PASS_N, nb, nv, bdims...),
            counts = reshape(counts, SFC.SINGLE_PASS_N, nb, nv, bdims...))
end

# --- Harmonic pseudo-coefficients ---
function SFC._direct_coefficients(db::CB.AbstractDistributedBackend, f, θ, φ, s, lmax)
    chunks = SFC._balanced_index_chunks(length(f), max(1, Distributed.nworkers()))
    parts = Distributed.pmap(chunks) do ch
        SFC.direct_coefficients_partial(f, θ, φ, s, lmax, ch)
    end
    return reduce(+, parts)
end

# --- Gridded sweeps ---
# Each worker takes a balanced share of the sweep's work items into its own histograms, which add
# because a histogram is order-independent. `sweep_items` is asked for enough parts to feed every
# worker and, under a threaded inner backend, every task inside one, so a one-slab schedule splits
# its lags rather than leaving all but one worker idle.
SFC.sweep_tasks(db::CB.AbstractDistributedBackend) =
    max(1, Distributed.nworkers()) * SFC.sweep_tasks(CB.local_backend(db))

function SFC.sweep_reduce!(
    sums, counts, db::CB.AbstractDistributedBackend, items, make_scratch, body!,
)
    inner = CB.local_backend(db)
    chunks = SFC._balanced_index_chunks(length(items), max(1, Distributed.nworkers()))
    partials = Distributed.pmap(chunks) do ch
        local_sums = zero(sums)
        local_counts = zero(counts)
        SFC.sweep_reduce!(local_sums, local_counts, inner, [items[i] for i in ch],
                          make_scratch, body!)
        (local_sums, local_counts)
    end
    for (ls, lc) in partials
        sums .+= ls
        counts .+= lc
    end
    return nothing
end

# --- Tensor structure functions ---

# Each worker takes a balanced share of the outer index and returns its own accumulators, which add
# because a histogram is order-independent. `pmap` takes the chunk list as items, so the inputs are
# serialised once per item; a closure capture would re-serialise them on every remotecall.
function SFC.distributed_calculate_structure_function_tensor!(
    sums::AbstractArray, counts::AbstractArray, order::Val{P},
    shape::SFC.AbstractFieldShape{D}, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(), axis = nothing,
    weights = SFC.NoWeights(),
) where {P, D}
    N = size(u, 2)
    chunks = SFC._balanced_index_chunks(N, max(Distributed.nworkers(), 1))
    CT = eltype(counts)
    partials = Distributed.pmap(chunks) do chunk
        SFC.tensor_partial(order, shape, x, u, distance_bins, chunk;
                           distance_metric, count_eltype = CT, axis, weights)
    end
    for (ps, pc) in partials
        sums .+= ps
        counts .+= pc
    end
    return sums, counts
end


# --- Multi-field (`Fields`) sweeps ---

function SFC.distributed_calculate_structure_function!(
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::SFC.MF.Fields, distance_bins;
    verbose::Bool = true, show_progress::Bool = true, kwargs...,
)
    _ = show_progress
    verbose && @info("calculating multi-field structure function (distributed)")
    N = size(SFC.MF.packed(f), 2)
    chunks = SFC._balanced_index_chunks(N - 1, max(Distributed.nworkers(), 1))
    CT = eltype(counts)
    partials = Distributed.pmap(chunks) do chunk
        SFC.field_partial(sf, x, f, distance_bins, chunk; count_eltype = CT, kwargs...)
    end
    for (ps, pc) in partials
        sums .+= ps
        counts .+= pc
    end
    return nothing
end

end
