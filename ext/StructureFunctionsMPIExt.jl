"""
MPI execution backend for structure functions (offered for multi-node adoption).

Mirrors the parametric `DistributedBackend{Inner}`: each MPI rank computes a balanced share
of the pairs with the `inner` backend (Serial/Threaded), then partial histograms are combined
with `MPI.Allreduce!` so every rank holds the full result. Run under `mpiexec` with `MPI.Init()`.

Currently supports point-field (non-batched) 1D inputs. Batched / 2D MPI can reuse the same
share-then-Allreduce pattern and is left as a follow-up.
"""
module StructureFunctionsMPIExt

using MPI: MPI
using Distances: Distances as DI
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT, n_histogram_bins

@inline _comm(b::SFC.MPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm

# Public entry (with shape). Point-field only for now.
function SFC._dispatch_execution_backend(
    b::SFC.MPIBackend,
    shape::SFC.AbstractFieldShape,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    distance_bins::AbstractVector,
    ::Val{RSAC};
    kwargs...,
) where {RSAC}
    SFC.has_auxiliary_axes(shape) && throw(ArgumentError(
        "MPIBackend currently supports point-field (non-batched) inputs; got a batched shape."))
    return _mpi_point_1d(b, structure_function_type, x, u, distance_bins, Val(RSAC); kwargs...)
end

function _mpi_point_1d(
    b::SFC.MPIBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector,
    ::Val{RSAC};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
    verbose = false,
    show_progress = false,
    kwargs...,
) where {RSAC, CT}
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    D, N = size(x, 1), size(x, 2)
    x_vecs = ntuple(k -> view(x, k, :), D)
    u_vecs = ntuple(k -> view(u, k, :), D)

    # round-robin outer-index share balances the triangular pair work across ranks
    ilist = (rank + 1):nranks:N
    part = SFC._partial_sums_counts(
        b.inner, structure_function_type, x_vecs, u_vecs, distance_bins, ilist;
        distance_metric = distance_metric, count_eltype = count_eltype,
    )
    sums = Array(part.sums)        # contiguous buffers for in-place Allreduce
    counts = Array(part.counts)
    MPI.Allreduce!(sums, +, comm)
    MPI.Allreduce!(counts, +, comm)

    if RSAC
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    else
        OT = eltype(sums)
        out = similar(sums)
        @inbounds for k in eachindex(sums)
            c = counts[k]
            out[k] = iszero(c) ? OT(NaN) : sums[k] / c
        end
        return SFO.StructureFunction(structure_function_type, distance_bins, out)
    end
end

end # module
