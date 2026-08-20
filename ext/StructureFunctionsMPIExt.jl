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
# `count_eltype` is POSITIONAL, matching the core 1D dispatch in `dispatch.jl`, which passes it as
# the 7th positional argument. Declaring it as a keyword makes this method fail to match, and the
# call then falls through to the `(::AbstractMPIBackend, args...)` stub that reports "MPI
# unavailable" — a signature mismatch disguised as a missing package.
function SFC._dispatch_execution_backend(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = false,
    show_progress = false,
    kwargs...,
) where {CT}
    if SFC.has_auxiliary_axes(shape)
        OT = promote_type(float(eltype(x)), float(eltype(u)))
        nb = n_histogram_bins(distance_bins)
        bdims = size(u)[3:end]
        sums = zeros(OT, nb, bdims...)
        counts = zeros(CT, nb, bdims...)
        SFC._bl_run_1d!(sums, counts, structure_function_type, x, u,
            SFC.BinEdges(distance_bins), distance_metric, _mpi_bl_exec(_comm(b), _inner_exec(b)))
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    return _mpi_point_1d(b, structure_function_type, x, u, distance_bins;
        distance_metric = distance_metric, count_eltype = CT, kwargs...)
end

# Returns the raw accumulator; the public boundary applies `_finalize`.
function _mpi_point_1d(
    b::CB.AbstractMPIBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {CT}
    comm = _comm(b)
    N = size(x, 2)
    x_vecs = ntuple(k -> view(x, k, :), size(x, 1))
    u_vecs = ntuple(k -> view(u, k, :), size(u, 1))

    part = SFC._partial_sums_counts(
        CB.local_backend(b), structure_function_type, x_vecs, u_vecs, distance_bins,
        _rank_share(comm, 1:(N - 1));
        distance_metric = distance_metric, count_eltype = count_eltype,
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
    value_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    verbose = false,
    show_progress = false,
    kwargs...,
) where {CT}
    comm = _comm(b)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    nd = n_histogram_bins(distance_bins)
    nv = n_histogram_bins(value_bins)

    if SFC.has_auxiliary_axes(shape)
        bdims = size(u)[3:end]
        sums = zeros(OT, nd, nv, bdims...)
        counts = zeros(CT, nd, nv, bdims...)
        SFC._bl_run_joint2d!(sums, counts, structure_function_type, x, u,
            SFC.BinEdges(distance_bins), SFC.BinEdges(value_bins), distance_metric,
            _mpi_bl_exec(comm, _inner_exec(b)))
        return SFO.StructureFunction2DSumsAndCounts(
            structure_function_type, distance_bins, value_bins, sums, counts)
    end

    N = size(x, 2)
    x_vecs = ntuple(k -> view(x, k, :), size(x, 1))
    u_vecs = ntuple(k -> view(u, k, :), size(u, 1))
    s, c = SFC._partial_2d_sums_counts(
        CB.local_backend(b), structure_function_type, x_vecs, u_vecs, distance_bins, value_bins,
        _rank_share(comm, 1:(N - 1));
        distance_metric = distance_metric, count_eltype = CT,
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
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    comm = _comm(b)
    OT = promote_type(float(FT1), float(FT2))
    nb = n_histogram_bins(distance_bins)

    if SFC.has_auxiliary_axes(shape)
        bdims = size(u)[3:end]
        sums = zeros(OT, SFC.SINGLE_PASS_N, nb, bdims...)
        counts = zeros(CT, SFC.SINGLE_PASS_N, nb, bdims...)
        SFC._bl_run_sp1d!(sums, counts, x, u, SFC.BinEdges(distance_bins), distance_metric,
            _mpi_bl_exec(comm, _inner_exec(b)))
        return (sums = sums, counts = counts)
    end

    s, c = SFC._partial_single_pass_1d(
        x, u, distance_bins, _rank_share(comm, 1:(size(x, 2) - 1));
        distance_metric = distance_metric, count_eltype = CT,
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
    value_bins::SFC.SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    comm = _comm(b)
    OT = promote_type(float(FT1), float(FT2))
    nb = n_histogram_bins(distance_bins)
    nv = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1

    if SFC.has_auxiliary_axes(shape)
        bdims = size(u)[3:end]
        sums = zeros(OT, SFC.SINGLE_PASS_N, nb, nv, bdims...)
        counts = zeros(CT, SFC.SINGLE_PASS_N, nb, nv, bdims...)
        SFC._bl_run_sp2d!(sums, counts, x, u, SFC.BinEdges(distance_bins), value_bins,
            distance_metric, _mpi_bl_exec(comm, _inner_exec(b)))
        return sums, counts
    end

    s, c = SFC._partial_single_pass_2d(
        x, u, distance_bins, value_bins, _rank_share(comm, 1:(size(x, 2) - 1));
        distance_metric = distance_metric, count_eltype = CT,
    )
    return _allreduce_pair!(comm, s, c)
end

end # module
