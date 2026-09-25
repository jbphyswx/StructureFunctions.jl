"""
MPI execution backend for structure functions (offered for multi-node adoption).

Mirrors the parametric `DistributedBackend{Inner}`: each MPI rank computes a balanced share
of the pairs with the `inner` backend (Serial/Threaded), then partial histograms are combined
with `MPI.Allreduce!` so every rank holds the full result. Run under `mpiexec` with `MPI.Init()`.

Covers point-field and batched inputs for all four entry families (1D, joint 2D, single-pass 1D,
single-pass 2D). Every rank must call with identical bins and input shapes: `Allreduce!` is
collective, so a rank that takes a different path deadlocks.
"""
module StructureFunctionsMPIExt

using MPI: MPI
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT, n_histogram_bins

@inline _comm(b::CB.AbstractMPIBackend) = isnothing(b.comm) ? MPI.COMM_WORLD : b.comm

# Round-robin outer-index share: work for index i is ~ N - i, so a strided subset of the
# triangular loop carries ~equal work on every rank.
@inline function _rank_share(comm, ifull)
    return (first(ifull) + MPI.Comm_rank(comm)):MPI.Comm_size(comm):last(ifull)
end

@inline function _allreduce_pair!(comm, sums, counts)
    MPI.Allreduce!(sums, +, comm)
    MPI.Allreduce!(counts, +, comm)
    return sums, counts
end

# `Allreduce!` needs contiguous, mutable buffers; a partial kernel may hand back a view.
@inline _dense(a::Array) = a
@inline _dense(a::AbstractArray) = Array(a)

# Batch-leading executor: run this rank's share through the inner backend's executor, then
# Allreduce so every rank holds the full histogram before it is permuted into the caller's buffer.
function _mpi_bl_exec(comm, inner_exec)
    return function (make_accum, run_chunk!, ifull, B, accum_bytes, ws)
        acc = inner_exec(make_accum, run_chunk!, _rank_share(comm, ifull), B, accum_bytes, ws)
        return _allreduce_pair!(comm, _dense(acc[1]), _dense(acc[2]))
    end
end

@inline _inner_exec(b::CB.AbstractMPIBackend) = SFC._bl_executor(CB.local_backend(b))

# --- 1D ---
function SFC._dispatch_execution_backend(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    verbose = false,
    show_progress = false,
) where {CT}
    if SFC.has_auxiliary_axes(shape)
        SFC._cull_reject_unsupported(culling, "the auxiliary-axis batch kernels")
        OT = promote_type(float(eltype(x)), float(eltype(u)))
        nb = n_histogram_bins(distance_bins)
        bdims = size(u)[3:end]
        sums = zeros(OT, nb, bdims...)
        counts = zeros(CT, nb, bdims...)
        SFC._bl_run_1d!(sums, counts, structure_function_type, x, u,
            SFC.digitize_plan(distance_bins), distance_metric, _mpi_bl_exec(_comm(b), _inner_exec(b)); weights)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    return _mpi_point_1d(b, structure_function_type, x, u, distance_bins, CT; distance_metric, weights, culling)
end

# Returns the raw accumulator; the public boundary applies `_finalize`.
function _mpi_point_1d(
    b::CB.AbstractMPIBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {CT}
    comm = _comm(b)
    N = size(x, 2)
    geom, x_vecs, u_vecs = SFC._prepared_tuples(distance_metric, x, u)

    # A polynomial operator on a line has the exact `O(N log N)` route, and that sweep executes
    # through `sweep_reduce!`, which this extension implements — so it splits across ranks here
    # without a second decomposition. The pair loop below would cost `O(N²)`.
    if SFC._on_a_line(geom, structure_function_type)
        SFC._cull_reject_unsupported(culling, "the sorted line route")
        nb0 = SFC.n_histogram_bins(distance_bins)
        OT0 = promote_type(float(eltype(x)), float(eltype(u)))
        lsums = zeros(OT0, nb0)
        lcounts = zeros(CT, nb0)
        SFC.sorted_line_sweep!(lsums, lcounts, structure_function_type, x_vecs[1],
            reshape(collect(u_vecs[1]), 1, :), distance_bins, Val(1), Val(1), Val(0);
            weights, backend = b)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins,
                                                  lsums, lcounts)
    end

    part = SFC._partial_sums_counts(
        CB.local_backend(b), structure_function_type, x_vecs, u_vecs, distance_bins,
        _rank_share(comm, 1:(N - 1)), CT;
        geometry = geom, culling = culling, weights,
    )
    sums, counts = _allreduce_pair!(comm, _dense(part.sums), _dense(part.counts))
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
end

# --- Joint 2D (distance x value) ---
function SFC._dispatch_execution_backend(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    verbose = false,
    show_progress = false,
) where {CT}
    comm = _comm(b)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    nd = n_histogram_bins(distance_bins)
    nv = n_histogram_bins(value_bins)

    if SFC.has_auxiliary_axes(shape)
        second_axis isa SFC.InvariantValueAxis || throw(ArgumentError(
            "the auxiliary-axis joint kernels bin each pair's operator value; $(nameof(typeof(second_axis))) " *
            "is taken by a field without auxiliary axes"))
        bdims = size(u)[3:end]
        sums = zeros(OT, nd, nv, bdims...)
        counts = zeros(CT, nd, nv, bdims...)
        SFC._bl_run_joint2d!(sums, counts, structure_function_type, x, u,
            SFC.digitize_plan(distance_bins), SFC.digitize_plan(value_bins), distance_metric,
            _mpi_bl_exec(comm, _inner_exec(b)); weights)
        return SFO.StructureFunction2DSumsAndCounts(
            structure_function_type, distance_bins, value_bins, sums, counts)
    end

    N = size(x, 2)
    geom, x_vecs, u_vecs = SFC._prepared_tuples(distance_metric, x, u)
    s, c = SFC._partial_2d_sums_counts(
        CB.local_backend(b), structure_function_type, x_vecs, u_vecs, distance_bins, value_bins,
        _rank_share(comm, 1:(N - 1)), CT;
        geometry = geom, weights, second_axis,
    )
    sums, counts = _allreduce_pair!(comm, s, c)
    return SFO.StructureFunction2DSumsAndCounts(
        structure_function_type, distance_bins, value_bins, sums, counts)
end

# --- Single pass 1D ---
function SFC._dispatch_single_pass(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    comm = _comm(b)
    OT = promote_type(float(FT1), float(FT2))
    nb = n_histogram_bins(distance_bins)

    if SFC.has_auxiliary_axes(shape)
        SFC._cull_reject_unsupported(culling, "the auxiliary-axis batch kernels")
        bdims = size(u)[3:end]
        sums = zeros(OT, SFC.SINGLE_PASS_N, nb, bdims...)
        counts = zeros(CT, SFC.SINGLE_PASS_N, nb, bdims...)
        SFC._bl_run_sp1d!(sums, counts, x, u, SFC.digitize_plan(distance_bins), distance_metric,
            _mpi_bl_exec(comm, _inner_exec(b)); weights)
        return (sums = sums, counts = counts)
    end

    s, c = SFC._partial_single_pass_1d(
        x, u, distance_bins, _rank_share(comm, 1:(size(x, 2) - 1)), CT;
        distance_metric, culling, weights,
    )
    sums, counts = _allreduce_pair!(comm, s, c)
    return (sums = sums, counts = counts)
end

# --- Single pass 2D ---
function SFC._dispatch_single_pass_2d(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins,
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    comm = _comm(b)
    OT = promote_type(float(FT1), float(FT2))
    nb = n_histogram_bins(distance_bins)
    nv = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1

    if SFC.has_auxiliary_axes(shape)
        SFC._cull_reject_unsupported(culling, "the auxiliary-axis batch kernels")
        bdims = size(u)[3:end]
        sums = zeros(OT, SFC.SINGLE_PASS_N, nb, nv, bdims...)
        counts = zeros(CT, SFC.SINGLE_PASS_N, nb, nv, bdims...)
        SFC._bl_run_sp2d!(sums, counts, x, u, SFC.digitize_plan(distance_bins), SFC.digitize_plan(value_bins),
            distance_metric, _mpi_bl_exec(comm, _inner_exec(b)); weights)
        return (sums = sums, counts = counts)
    end

    s, c = SFC._partial_single_pass_2d(
        x, u, distance_bins, value_bins, _rank_share(comm, 1:(size(x, 2) - 1)), CT;
        distance_metric, culling, weights,
    )
    rs, rc = _allreduce_pair!(comm, s, c)
    return (sums = rs, counts = rc)
end


# --- Mutating dispatch (point and auxiliary axes) ---
# The non-mutating methods Allreduce, so every rank already holds the full result when it is added
# into the caller's buffers.
function SFC._dispatch_execution_backend!(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    sums::AbstractArray,
    counts::AbstractArray,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins::AbstractVector;
    kwargs...,
)
    r = SFC._dispatch_execution_backend(
        b, shape, structure_function_type, x, u, distance_bins, eltype(counts); kwargs...,
    )
    sums .+= r.sums
    counts .+= r.counts
    return nothing
end

function SFC._dispatch_execution_backend!(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    sums::AbstractArray,
    counts::AbstractArray,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
)
    r = SFC._dispatch_execution_backend(
        b, shape, structure_function_type, x, u, distance_bins, value_bins, eltype(counts); kwargs...,
    )
    sums .+= r.sums
    counts .+= r.counts
    return nothing
end

# --- Mutating single pass ---
function SFC._dispatch_single_pass!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...,
)
    shape = SFC._validate_array_shape(x, u, distance_metric)
    r = SFC._dispatch_single_pass(b, shape, x, u, distance_bins, eltype(counts); distance_metric, kwargs...)
    sums .+= r[1]
    counts .+= r[2]
    return sums, counts
end

function SFC._dispatch_single_pass_2d!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...,
)
    shape = SFC._validate_array_shape(x, u, distance_metric)
    r = SFC._dispatch_single_pass_2d(b, shape, x, u, distance_bins, value_bins, eltype(counts);
                                     distance_metric, kwargs...)
    sums .+= r[1]
    counts .+= r[2]
    return sums, counts
end

# --- Slice batch drivers ---
# `_bl_run_*!` adds into the caller's buffers, and `_mpi_bl_exec` Allreduces each rank's share
# before the permuted add, so the batch drivers are the auxiliary-axis path on caller-owned output.
function SFC._dispatch_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_1d!(sums, counts, sf_type, x, u, SFC.digitize_plan(distance_bins), distance_metric,
        _mpi_bl_exec(_comm(b), _inner_exec(b)); weights)
    return nothing
end

function SFC._dispatch_2d_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector, value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_joint2d!(sums, counts, sf_type, x, u, SFC.digitize_plan(distance_bins),
        SFC.digitize_plan(value_bins), distance_metric, _mpi_bl_exec(_comm(b), _inner_exec(b)); weights)
    return nothing
end

function SFC._dispatch_single_pass_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_sp1d!(sums, counts, x, u, SFC.digitize_plan(distance_bins), distance_metric,
        _mpi_bl_exec(_comm(b), _inner_exec(b)); weights)
    return nothing
end

function SFC._dispatch_single_pass_2d_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = SFC.NoWeights(), verbose::Bool = true, show_progress::Bool = true,
)
    SFC._bl_run_sp2d!(sums, counts, x, u, SFC.digitize_plan(distance_bins), SFC.digitize_plan(value_bins),
        distance_metric, _mpi_bl_exec(_comm(b), _inner_exec(b)); weights)
    return nothing
end

# --- Moment tensors ---
function SFC.mpi_calculate_structure_function_tensor!(
    sums::AbstractArray, counts::AbstractArray, order::Val{P},
    shape::SFC.AbstractFieldShape{D}, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    backend::CB.AbstractMPIBackend, distance_metric::DI.PreMetric = DI.Euclidean(),
    axis = nothing, weights = SFC.NoWeights(), kwargs...,
) where {P, D}
    comm = _comm(backend)
    ps, pc = SFC.tensor_partial(order, shape, x, u, distance_bins,
        _rank_share(comm, 1:(size(u, 2) - 1)), eltype(counts);
        distance_metric = distance_metric, axis = axis, weights = weights)
    rs, rc = _allreduce_pair!(comm, _dense(ps), _dense(pc))
    sums .+= rs
    counts .+= rc
    return sums, counts
end

# --- Multi-field (`Fields`) sweeps ---
function SFC.mpi_calculate_structure_function!(
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::SFC.MF.Fields, distance_bins;
    backend::CB.AbstractMPIBackend, verbose::Bool = false, show_progress::Bool = false, kwargs...,
)
    _ = show_progress
    comm = _comm(backend)
    verbose && MPI.Comm_rank(comm) == 0 && @info("calculating multi-field structure function (MPI)")
    N = size(SFC.MF.packed(f), 2)
    ps, pc = SFC.field_partial(sf, x, f, distance_bins, _rank_share(comm, 1:(N - 1)), eltype(counts); kwargs...)
    rs, rc = _allreduce_pair!(comm, _dense(ps), _dense(pc))
    sums .+= rs
    counts .+= rc
    return nothing
end

# --- Harmonic pseudo-coefficients ---
function SFC._direct_coefficients(b::CB.AbstractMPIBackend, f, θ, φ, s, lmax)
    comm = _comm(b)
    out = SFC.direct_coefficients_partial(f, θ, φ, s, lmax, _rank_share(comm, eachindex(f)))
    MPI.Allreduce!(out, +, comm)
    return out
end

# --- Gridded sweeps ---
# One work item per rank-share of the sweep's items; `sweep_items` is asked for as many parts as
# there are ranks, so a one-slab schedule splits its lags rather than leaving ranks idle.
SFC.sweep_tasks(b::CB.AbstractMPIBackend) = MPI.Comm_size(_comm(b))

function SFC.sweep_reduce!(sums, counts, b::CB.AbstractMPIBackend, items, make_scratch, body!)
    comm = _comm(b)
    local_sums = zero(sums)
    local_counts = zero(counts)
    scratch = make_scratch()
    for i in _rank_share(comm, 1:length(items))
        body!(local_sums, local_counts, items[i], scratch)
    end
    rs, rc = _allreduce_pair!(comm, _dense(local_sums), _dense(local_counts))
    sums .+= rs
    counts .+= rc
    return nothing
end

end # module
