"""
Distributed execution backend for structure functions utilizing Distributed.jl.
"""
module StructureFunctionsDistributedExt

using Distributed: Distributed
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO

function __init__()
    SFC._DISTRIBUTED_LOADED[] = true
    return nothing
end

# A `LocalManager` worker shares this node's cores, which a threaded process already uses; any other manager placed
# the worker where this process cannot reach.
SFC.distributed_adds_hardware(::Val{:distributed}) =
    Distributed.nprocs() > 1 &&
    (Threads.nthreads() == 1 ||
     any(w -> !(Distributed.worker_from_id(w).manager isa Distributed.LocalManager), Distributed.workers()))

"""One share `(w, k)` of the outer indices per worker, `k` the worker count; each worker resolves its share against
the cull grid it builds (`SFC._share_indices`)."""
_worker_shares() = (k = max(1, Distributed.nworkers()); [(w, k) for w in 1:k])

# --- Non-Mutating 1D Dispatch (returns the raw accumulator; public boundary finalizes) ---
function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ::Type{CT};
    kwargs...,
) where {CT}
    return _parallel_calculate_structure_function_core(
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        CT;
        inner = CB.local_backend(db),
        kwargs...,
    )
end

function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {CT}
    geom, x_vecs, u_vecs = SFC._prepared_tuples(distance_metric, x_arr, u_arr)
    return SFC._dispatch_execution_backend(
        db,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        CT;
        geometry = geom,
        kwargs...,
    )
end

function _parallel_calculate_structure_function_core(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    ::Type{CT};
    geometry = SFC.default_geometry(u_vecs),
    inner::CB.AbstractLocalBackend = CB.SerialBackend(),
    weights = SFC.NoWeights(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {CT}
    OT0 = promote_type(float(eltype(x_vecs[1])), float(eltype(u_vecs[1])))

    # A polynomial operator on a line has the exact `O(N log N)` route, and that sweep executes
    # through `sweep_reduce!`, which this extension implements — so it distributes here without a
    # second decomposition. Reaching it through the pair loop instead costs `O(N²)`.
    if SFC._on_a_line(geometry, structure_function_type)
        nb0 = SFC.n_histogram_bins(distance_bins)
        lsums = zeros(OT0, nb0)
        lcounts = zeros(CT, nb0)
        SFC.sorted_line_sweep!(lsums, lcounts, structure_function_type, x_vecs[1],
            reshape(collect(u_vecs[1]), 1, :), distance_bins, Val(1), Val(1), Val(0);
            weights, backend = CB.DistributedBackend(inner))
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins,
                                                  lsums, lcounts)
    end

    partials = Distributed.pmap(_worker_shares()) do share
        SFC._partial_sums_counts(
            inner, structure_function_type, x_vecs, u_vecs, distance_bins, share, CT;
            geometry = geometry, culling = culling, weights,
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
workers (`SFC._outer_shares`), run the batch-leading kernel on each share, and add the partials.
Each pair's geometry is then computed once for all `B` slices inside the kernel.
"""
function _dist_bl_exec(inner_exec)
    return function (make_accum, make_scratch, run_chunk!, ifull, grid, B, accum_bytes, ws)
        shares = filter(!isempty, SFC._outer_shares(grid, ifull, Distributed.nworkers()))
        length(shares) <= 1 && return inner_exec(make_accum, make_scratch, run_chunk!, ifull, grid, B, accum_bytes, ws)
        parts = Distributed.pmap(shares) do share
            acc = inner_exec(make_accum, make_scratch, run_chunk!, share, grid, B, accum_bytes, nothing)
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
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_1d!(sums, counts, structure_function_type, x, u, distance_bins, distance_metric,
        _dist_bl_exec(SFC._bl_executor(CB.local_backend(db))); weights, culling)
    return nothing
end

"""Accumulate a distributed joint slice sweep into the `(n_dist, n_val, B)` reshape."""
function _dist_accumulate_2d!(
    sums, counts, db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector, value_bins::AbstractVector,
    B::Int;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_joint2d!(sums, counts, structure_function_type, x, u, distance_bins, value_bins, distance_metric,
        _dist_bl_exec(SFC._bl_executor(CB.local_backend(db))); weights, culling, second_axis)
    return nothing
end

function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {CT}
    bdims = size(u)[3:end]
    B = prod(bdims)
    nb = SFC.n_histogram_bins(distance_bins)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, nb, B)
    counts = zeros(CT, nb, B)
    _dist_accumulate_1d!(
        sums, counts, db, structure_function_type, x, u, distance_bins, B;
        distance_metric, culling, weights,
    )

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type, distance_bins,
        reshape(sums, nb, bdims...), reshape(counts, nb, bdims...),
    )
end

# --- Non-Mutating 2D Dispatch ---
function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ::Type{CT};
    geometry = SFC.default_geometry(u_vecs),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {CT}
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    OT = promote_type(float(eltype(x_vecs[1])), float(eltype(u_vecs[1])))
    inner = CB.local_backend(db)

    partials = Distributed.pmap(_worker_shares()) do share
        SFC._partial_2d_sums_counts(
            inner, structure_function_type, x_vecs, u_vecs, distance_bins, value_bins, share, CT;
            geometry, culling, weights, second_axis,
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
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {CT}
    geom, x_vecs, u_vecs = SFC._prepared_tuples(distance_metric, x_arr, u_arr)
    return SFC._dispatch_execution_backend(
        db,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins,
        CT;
        geometry = geom,
        kwargs...,
    )
end

# --- Single Pass Dispatch ---
function SFC._dispatch_single_pass(
    db::CB.AbstractDistributedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = SFC.n_histogram_bins(distance_bins)
    inner = CB.local_backend(db)
    partials = Distributed.pmap(_worker_shares()) do share
        SFC._partial_single_pass_1d(
            inner, x, u, distance_bins, share, CT;
            distance_metric, culling, weights,
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
    db::CB.AbstractDistributedBackend,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SFC.SinglePass2DValueBins,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = SFC.n_histogram_bins(distance_bins)
    n_val = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    inner = CB.local_backend(db)
    partials = Distributed.pmap(_worker_shares()) do share
        SFC._partial_single_pass_2d(
            inner, x, u, distance_bins, value_bins, share, CT;
            distance_metric, culling, weights,
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

# --- Mutating point-field dispatch: the allocating sweep, added into the caller's buffers ---
function SFC._dispatch_execution_backend!(
    db::CB.AbstractDistributedBackend,
    sums::AbstractVector,
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector;
    kwargs...,
) where {CT}
    result = SFC._dispatch_execution_backend(db, structure_function_type, x_arr, u_arr, distance_bins, CT; kwargs...)
    sums .+= result.sums
    counts .+= result.counts
    return nothing
end

function SFC._dispatch_execution_backend!(
    db::CB.AbstractDistributedBackend,
    sums_2d::AbstractMatrix,
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {CT}
    result = SFC._dispatch_execution_backend(db, structure_function_type, x_arr, u_arr, distance_bins, value_bins,
                                             CT; kwargs...)
    sums_2d .+= result.sums
    counts_2d .+= result.counts
    return nothing
end


# --- Non-mutating joint dispatch over auxiliary axes ---
function SFC._dispatch_execution_backend(
    db::CB.AbstractDistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    weights = SFC.NoWeights(),
) where {CT}
    bdims = size(u)[3:end]
    B = prod(bdims)
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, nd, nv, B)
    counts = zeros(CT, nd, nv, B)
    _dist_accumulate_2d!(
        sums, counts, db, structure_function_type, x, u, distance_bins, value_bins, B;
        distance_metric, culling, second_axis, weights,
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
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    B = prod(size(u)[3:end])
    nb = SFC.n_histogram_bins(distance_bins)
    _dist_accumulate_1d!(
        reshape(sums, nb, B), reshape(counts, nb, B), db, structure_function_type, x, u,
        distance_bins, B; distance_metric, culling, weights,
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
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    weights = SFC.NoWeights(),
)
    B = prod(size(u)[3:end])
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    _dist_accumulate_2d!(
        reshape(sums, nd, nv, B), reshape(counts, nd, nv, B), db, structure_function_type, x, u,
        distance_bins, value_bins, B; distance_metric, culling, second_axis, weights,
    )
    return nothing
end

# --- Slice batch drivers ---
function SFC._dispatch_batch!(
    db::CB.AbstractDistributedBackend, sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    B = prod(size(u)[3:end])
    nb = SFC.n_histogram_bins(distance_bins)
    _dist_accumulate_1d!(
        reshape(sums, nb, B), reshape(counts, nb, B), db, sf_type, x, u, distance_bins, B;
        distance_metric, culling, weights,
    )
    return nothing
end

function SFC._dispatch_2d_batch!(
    db::CB.AbstractDistributedBackend, sums, counts, sf_type, x, u, distance_bins, value_bins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    weights = SFC.NoWeights(),
)
    B = prod(size(u)[3:end])
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    _dist_accumulate_2d!(
        reshape(sums, nd, nv, B), reshape(counts, nd, nv, B), db, sf_type, x, u,
        distance_bins, value_bins, B; distance_metric, culling, second_axis, weights,
    )
    return nothing
end

function SFC._dispatch_single_pass_batch!(
    db::CB.AbstractDistributedBackend, sums, counts, x, u, distance_bins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_sp1d!(sums, counts, x, u, distance_bins, distance_metric,
        _dist_bl_exec(SFC._bl_executor(CB.local_backend(db))); weights, culling)
    return nothing
end

function SFC._dispatch_single_pass_2d_batch!(
    db::CB.AbstractDistributedBackend, sums, counts, x, u, distance_bins,
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins,
        distance_metric, _dist_bl_exec(SFC._bl_executor(CB.local_backend(db))); weights, culling)
    return nothing
end

# --- Mutating single-pass over a point list ---
function SFC._dispatch_single_pass!(
    db::CB.AbstractDistributedBackend, sums::AbstractMatrix, counts::AbstractMatrix,
    x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...,
)
    r = SFC._dispatch_single_pass(db, x, u, distance_bins, eltype(counts); kwargs...)
    sums .+= r.sums
    counts .+= r.counts
    return sums, counts
end

function SFC._dispatch_single_pass_2d!(
    db::CB.AbstractDistributedBackend, sums_3d::AbstractArray, counts_3d::AbstractArray,
    x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins; kwargs...,
)
    r = SFC._dispatch_single_pass_2d(db, x, u, distance_bins, value_bins, eltype(counts_3d); kwargs...)
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
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector, ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(), culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {CT}
    bdims = size(u)[3:end]
    B = prod(bdims)
    nb = SFC.n_histogram_bins(distance_bins)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, SFC.SINGLE_PASS_N, nb, B)
    counts = zeros(CT, SFC.SINGLE_PASS_N, nb, B)
    SFC._bl_run_sp1d!(sums, counts, x, u, distance_bins, distance_metric,
        _dist_bl_exec(SFC._bl_executor(CB.local_backend(db))); weights, culling)
    return (sums = reshape(sums, SFC.SINGLE_PASS_N, nb, bdims...),
            counts = reshape(counts, SFC.SINGLE_PASS_N, nb, bdims...))
end

function SFC._dispatch_single_pass_2d(
    db::CB.AbstractDistributedBackend,
    ::Union{SFC.SharedPositionField, SFC.VaryingPositionField},
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins, ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(), culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {CT}
    bdims = size(u)[3:end]
    B = prod(bdims)
    nb = SFC.n_histogram_bins(distance_bins)
    nv = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, SFC.SINGLE_PASS_N, nb, nv, B)
    counts = zeros(CT, SFC.SINGLE_PASS_N, nb, nv, B)
    SFC._bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins,
        distance_metric, _dist_bl_exec(SFC._bl_executor(CB.local_backend(db))); weights, culling)
    return (sums = reshape(sums, SFC.SINGLE_PASS_N, nb, nv, bdims...),
            counts = reshape(counts, SFC.SINGLE_PASS_N, nb, nv, bdims...))
end

# --- Harmonic pseudo-coefficients ---
function SFC._direct_coefficients(db::CB.AbstractDistributedBackend, f, θ, φ, s, lmax)
    inner = CB.local_backend(db)
    shares = [(f[ch], θ[ch], φ[ch]) for ch in SFC._outer_shares(nothing, 1:length(f), Distributed.nworkers())]
    parts = Distributed.pmap(shares) do (fs, θs, φs)
        SFC._direct_coefficients(inner, fs, θs, φs, s, lmax)
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
    chunks = SFC._outer_shares(nothing, 1:length(items), Distributed.nworkers())
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

# Each worker takes its share of the outer index and returns its own accumulators, which add because a histogram is
# order-independent.
function SFC.distributed_calculate_structure_function_tensor!(
    inner::CB.AbstractExecutionBackend, sums::AbstractArray, counts::AbstractArray, order::Val{P},
    shape::SFC.AbstractFieldShape{D}, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(), axis = nothing,
    culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
) where {P, D}
    CT = eltype(counts)
    partials = Distributed.pmap(_worker_shares()) do share
        SFC.tensor_partial(inner, order, shape, x, u, distance_bins, share, CT; distance_metric, axis, culling, weights)
    end
    for (ps, pc) in partials
        sums .+= ps
        counts .+= pc
    end
    return sums, counts
end


# --- Multi-field (`Fields`) sweeps ---

function SFC.distributed_calculate_structure_function!(
    inner::CB.AbstractExecutionBackend, sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::SFC.MF.Fields, distance_bins; kwargs...,
)
    CT = eltype(counts)
    partials = Distributed.pmap(_worker_shares()) do share
        SFC.field_partial(inner, sf, x, f, distance_bins, share, CT; kwargs...)
    end
    for (ps, pc) in partials
        sums .+= ps
        counts .+= pc
    end
    return nothing
end

end
