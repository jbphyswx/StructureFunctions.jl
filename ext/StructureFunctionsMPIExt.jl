"""
MPI execution backend for structure functions.

Mirrors the parametric `DistributedBackend{Inner}`: each MPI rank computes a balanced share
of the pairs with the `inner` backend (Serial/Threaded), then partial histograms are combined
with `MPI.Allreduce!` so every rank holds the full result. Run under `mpiexec` with `MPI.Init()`.

Covers point-field and batched inputs for all four entry families (1D, joint 2D, single-pass 1D,
single-pass 2D). Every rank must call with identical bins and input shapes: `Allreduce!` is
collective, so a rank that takes a different path deadlocks.
"""
module StructureFunctionsMPIExt

using MPI: MPI
using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT, n_histogram_bins

function __init__()
    SFC._MPI_LOADED[] = true
    return nothing
end

@inline _comm(b::CB.AbstractMPIBackend) = isnothing(b.comm) ? MPI.COMM_WORLD : b.comm

"""This rank's share `(w, k)` of a sweep's outer indices: share `w` of the `k` ranks."""
@inline _rank_part(comm) = (MPI.Comm_rank(comm) + 1, MPI.Comm_size(comm))

"""This rank's share of the outer indices `ifull` of a sweep over `grid`'s schedule."""
@inline _rank_share(comm, ifull, grid) = SFC._outer_share(grid, ifull, _rank_part(comm)...)

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
    return function (make_accum, make_scratch, run_chunk!, ifull, grid, B, accum_bytes, ws)
        acc = inner_exec(make_accum, make_scratch, run_chunk!, _rank_share(comm, ifull, grid), grid, B, accum_bytes, ws)
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
    geometry,
    weights = SFC.NoWeights(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {CT}
    if SFC.has_auxiliary_axes(shape)
        OT = promote_type(float(eltype(x)), float(eltype(u)))
        nb = n_histogram_bins(distance_bins)
        bdims = size(u)[3:end]
        sums = zeros(OT, nb, bdims...)
        counts = zeros(CT, nb, bdims...)
        SFC._bl_run_1d!(sums, counts, structure_function_type, x, u, distance_bins, geometry,
            _mpi_bl_exec(_comm(b), _inner_exec(b)); weights, culling)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    return _mpi_point_1d(b, structure_function_type, x, u, distance_bins, CT; geometry, weights, culling)
end

# Returns the raw accumulator; the public boundary applies `_finalize`.
function _mpi_point_1d(
    b::CB.AbstractMPIBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector,
    ::Type{CT};
    geometry,
    weights = SFC.NoWeights(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
) where {CT}
    comm = _comm(b)
    x_vecs, u_vecs = SFC._prepared_tuples(geometry, x, u)

    # A polynomial operator on a line takes the `O(N log N)` sorted sweep, which splits across ranks through `sweep_reduce!`.
    if SFC._on_a_line(geometry, structure_function_type)
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
        CB.local_backend(b), structure_function_type, x_vecs, u_vecs, distance_bins, _rank_part(comm), CT;
        geometry, culling, weights,
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
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
) where {CT}
    comm = _comm(b)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    nd = n_histogram_bins(distance_bins)
    nv = n_histogram_bins(value_bins)

    if SFC.has_auxiliary_axes(shape)
        bdims = size(u)[3:end]
        sums = zeros(OT, nd, nv, bdims...)
        counts = zeros(CT, nd, nv, bdims...)
        SFC._bl_run_joint2d!(sums, counts, structure_function_type, x, u, distance_bins, value_bins, geometry,
            _mpi_bl_exec(comm, _inner_exec(b)); weights, culling, second_axis)
        return SFO.StructureFunction2DSumsAndCounts(
            structure_function_type, distance_bins, value_bins, sums, counts, second_axis)
    end

    x_vecs, u_vecs = SFC._prepared_tuples(geometry, x, u)
    s, c = SFC._partial_2d_sums_counts(
        CB.local_backend(b), structure_function_type, x_vecs, u_vecs, distance_bins, value_bins, _rank_part(comm), CT;
        geometry, culling, weights, second_axis,
    )
    sums, counts = _allreduce_pair!(comm, s, c)
    return SFO.StructureFunction2DSumsAndCounts(
        structure_function_type, distance_bins, value_bins, sums, counts, second_axis)
end

# --- Single pass 1D ---
function SFC._dispatch_single_pass(
    b::CB.AbstractMPIBackend,
    shape::SFC.AbstractFieldShape,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    ::Type{CT};
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    SFC.has_auxiliary_axes(shape) || return _mpi_single_pass_1d(b, x, u, distance_bins, CT; geometry, culling, weights)
    OT = promote_type(float(FT1), float(FT2))
    nb = n_histogram_bins(distance_bins)
    bdims = size(u)[3:end]
    sums = zeros(OT, SFC.SINGLE_PASS_N, nb, bdims...)
    counts = zeros(CT, SFC.SINGLE_PASS_N, nb, bdims...)
    SFC._bl_run_sp1d!(sums, counts, x, u, distance_bins, geometry,
        _mpi_bl_exec(_comm(b), _inner_exec(b)); weights, culling)
    return (sums = sums, counts = counts)
end

function _mpi_single_pass_1d(
    b::CB.AbstractMPIBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, ::Type{CT};
    geometry, culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
) where {CT}
    comm = _comm(b)
    s, c = SFC._partial_single_pass_1d(
        CB.local_backend(b), x, u, distance_bins, _rank_part(comm), CT;
        geometry, culling, weights,
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
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    SFC.has_auxiliary_axes(shape) ||
        return _mpi_single_pass_2d(b, x, u, distance_bins, value_bins, CT; geometry, culling, weights)
    OT = promote_type(float(FT1), float(FT2))
    nb = n_histogram_bins(distance_bins)
    nv = length(SFC._sp2d_value_bin_at(value_bins, 1)) - 1
    bdims = size(u)[3:end]
    sums = zeros(OT, SFC.SINGLE_PASS_N, nb, nv, bdims...)
    counts = zeros(CT, SFC.SINGLE_PASS_N, nb, nv, bdims...)
    SFC._bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins,
        geometry, _mpi_bl_exec(_comm(b), _inner_exec(b)); weights, culling)
    return (sums = sums, counts = counts)
end

function _mpi_single_pass_2d(
    b::CB.AbstractMPIBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins, ::Type{CT};
    geometry, culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
) where {CT}
    comm = _comm(b)
    s, c = SFC._partial_single_pass_2d(
        CB.local_backend(b), x, u, distance_bins, value_bins, _rank_part(comm), CT;
        geometry, culling, weights,
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
    b::CB.AbstractMPIBackend, sums::AbstractMatrix, counts::AbstractMatrix,
    x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...,
)
    r = _mpi_single_pass_1d(b, x, u, distance_bins, eltype(counts); kwargs...)
    sums .+= r.sums
    counts .+= r.counts
    return sums, counts
end

function SFC._dispatch_single_pass_2d!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins; kwargs...,
)
    r = _mpi_single_pass_2d(b, x, u, distance_bins, value_bins, eltype(counts); kwargs...)
    sums .+= r.sums
    counts .+= r.counts
    return sums, counts
end

# --- Slice batch drivers ---
# `_bl_run_*!` adds into the caller's buffers, and `_mpi_bl_exec` Allreduces each rank's share
# before the permuted add, so the batch drivers are the auxiliary-axis path on caller-owned output.
function SFC._dispatch_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_1d!(sums, counts, sf_type, x, u, distance_bins, geometry,
        _mpi_bl_exec(_comm(b), _inner_exec(b)); weights, culling)
    return nothing
end

function SFC._dispatch_2d_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector, value_bins::AbstractVector;
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    second_axis::SFC.AbstractSecondAxisSource = SFC.InvariantValueAxis(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins, geometry,
        _mpi_bl_exec(_comm(b), _inner_exec(b)); weights, culling, second_axis)
    return nothing
end

function SFC._dispatch_single_pass_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector;
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_sp1d!(sums, counts, x, u, distance_bins, geometry,
        _mpi_bl_exec(_comm(b), _inner_exec(b)); weights, culling)
    return nothing
end

function SFC._dispatch_single_pass_2d_batch!(
    b::CB.AbstractMPIBackend, sums::AbstractArray, counts::AbstractArray,
    x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
    value_bins::SFC.SinglePass2DValueBins;
    geometry,
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
)
    SFC._bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins,
        geometry, _mpi_bl_exec(_comm(b), _inner_exec(b)); weights, culling)
    return nothing
end

# --- Moment tensors ---
function SFC.mpi_calculate_structure_function_tensor!(
    sums::AbstractArray, counts::AbstractArray, order::Val{P},
    shape::SFC.AbstractFieldShape{D}, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    backend::CB.AbstractMPIBackend, geometry,
    axis = nothing, culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(),
) where {P, D}
    comm = _comm(backend)
    ps, pc = SFC.tensor_partial(CB.local_backend(backend), order, shape, x, u, distance_bins, _rank_part(comm),
        eltype(counts); geometry, axis, culling, weights)
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
    backend::CB.AbstractMPIBackend, kwargs...,
)
    comm = _comm(backend)
    ps, pc = SFC.field_partial(CB.local_backend(backend), sf, x, f, distance_bins, _rank_part(comm), eltype(counts);
                               kwargs...)
    rs, rc = _allreduce_pair!(comm, _dense(ps), _dense(pc))
    sums .+= rs
    counts .+= rc
    return nothing
end

# --- Harmonic pseudo-coefficients ---
function SFC._direct_coefficients(b::CB.AbstractMPIBackend, f, θ, φ, s, lmax)
    comm = _comm(b)
    share = _rank_share(comm, eachindex(f), nothing)
    out = SFC._direct_coefficients(CB.local_backend(b), f[share], θ[share], φ[share], s, lmax)
    MPI.Allreduce!(out, +, comm)
    return out
end

# --- Gridded sweeps ---
# `sweep_items` is asked for as many parts as there are ranks.
SFC.sweep_tasks(b::CB.AbstractMPIBackend) = MPI.Comm_size(_comm(b))

function SFC.sweep_reduce!(sums, counts, b::CB.AbstractMPIBackend, items, make_scratch, body!)
    comm = _comm(b)
    local_sums = zero(sums)
    local_counts = zero(counts)
    scratch = make_scratch()
    for i in _rank_share(comm, 1:length(items), nothing)
        body!(local_sums, local_counts, items[i], scratch)
    end
    rs, rc = _allreduce_pair!(comm, _dense(local_sums), _dense(local_counts))
    sums .+= rs
    counts .+= rc
    return nothing
end

end # module
