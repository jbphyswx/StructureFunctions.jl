"""
Distributed execution backend for structure functions utilizing Distributed.jl.
"""
module StructureFunctionsDistributedExt

using Distributed: Distributed
using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using SharedArrays: SharedArrays
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO,
    AbstractBinEdges, LinearBinEdges, LogBinEdges

SFC.distributed_workers_available(::Val{:distributed}) = Distributed.nworkers() > 1

# Balanced outer-index order for the triangular pair loop (work for index i is ~ N - i).
# `@distributed` splits its iterable into CONTIGUOUS per-worker blocks, so iterating 1:N
# directly gives worker 1 all the expensive low-i pairs (~2x imbalance). Interleaving
# high/low indices ([1, N, 2, N-1, ...]) makes every contiguous block carry ~equal work.
function _balanced_triangle_perm(N::Integer)
    perm = Vector{Int}(undef, N)
    lo, hi, k = 1, N, 1
    @inbounds while lo <= hi
        perm[k] = lo; k += 1; lo += 1
        if lo <= hi
            perm[k] = hi; k += 1; hi -= 1
        end
    end
    return perm
end

# --- Non-Mutating 1D Dispatch (returns the raw accumulator; public boundary finalizes) ---
function SFC._dispatch_execution_backend(
    db::SFC.DistributedBackend,
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
        inner = db.inner,
        kwargs...,
    )
end

function SFC._dispatch_execution_backend(
    db::SFC.DistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector;
    kwargs...,
)
    N_dims = size(x_arr, 1)
    x_vecs = ntuple(k -> view(x_arr, k, :), N_dims)
    u_vecs = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC._dispatch_execution_backend(
        db,
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins;
        kwargs...,
    )
end

function _parallel_calculate_structure_function_core(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose = true,
    show_progress = true,
    count_eltype::Type{CT} = UInt32,
    inner::SFC.AbstractExecutionBackend = SFC.SerialBackend(),
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
    partials = Distributed.pmap(chunks) do ch
        SFC._partial_sums_counts(
            inner, structure_function_type, x_vecs, u_vecs, distance_bins, ch;
            distance_metric = distance_metric, count_eltype = count_eltype,
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

# --- Batched (auxiliary-axis) distributed dispatch ---
# Distribute the BATCH axis across workers (per-b work is equal, so contiguous chunks are
# balanced). Each worker computes complete b-slices via the `inner` backend (serial or
# threaded batch kernels) and returns its slice; results are concatenated along the batch
# axis (disjoint b's ⇒ no reduction). With one worker pinned per NUMA node this keeps each
# process's velocity data socket-local, beating pure threading's single-socket bandwidth ceiling.
# Matches ndims(u) >= 3 only (the AbstractMatrix point-field method above is more specific).
@inline function _dist_batch_chunks(B::Int, nw::Int)
    nw = clamp(nw, 1, max(B, 1))
    return [(((w - 1) * B) ÷ nw + 1):((w * B) ÷ nw) for w in 1:nw]
end

function SFC._dispatch_execution_backend(
    db::SFC.DistributedBackend,
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
    verbose && @info("calculating batched structure function (distributed over batch axis, inner=$(nameof(typeof(db.inner))))")
    D, N = size(u, 1), size(u, 2)
    bdims = size(u)[3:end]
    B = prod(bdims)
    fixed_x = ndims(x) == 2
    u_flat = reshape(u, D, N, B)
    x_flat = fixed_x ? x : reshape(x, D, N, B)
    nb = SFC.n_histogram_bins(distance_bins)
    inner = db.inner

    chunks = _dist_batch_chunks(B, Distributed.nworkers())
    # each worker computes its contiguous b-chunk locally via the inner backend, requesting the
    # raw accumulator so partials can be concatenated; the public boundary finalizes.
    parts = Distributed.pmap(chunks) do bc
        usub = u_flat[:, :, bc]
        xsub = fixed_x ? x_flat : x_flat[:, :, bc]
        r = SFC.calculate_structure_function(
            structure_function_type, xsub, usub, distance_bins;
            backend = inner, output_type = SFO.StructureFunctionSumsAndCounts,
            verbose = false, show_progress = false,
            distance_metric = distance_metric, count_eltype = count_eltype,
        )
        (r.sums, r.counts)
    end

    OT = promote_type(float(eltype(x)), float(eltype(u)))
    sums = zeros(OT, nb, B)
    counts = zeros(CT, nb, B)
    for (w, bc) in enumerate(chunks)
        @inbounds sums[:, bc] .= reshape(parts[w][1], nb, length(bc))
        @inbounds counts[:, bc] .= reshape(parts[w][2], nb, length(bc))
    end

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type, distance_bins,
        reshape(sums, nb, bdims...), reshape(counts, nb, bdims...),
    )
end

# --- Auto-Binning 1D Dispatch ---
function SFC._dispatch_execution_backend(
    ::SFC.DistributedBackend,
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
        SFC.DistributedBackend(),
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
    ::SFC.DistributedBackend,
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

    # Round-robin the balanced triangle permutation across workers (an interleaved subset of a
    # balance-ordered list stays load-balanced); each worker accumulates its i's into a local 2D
    # buffer, then we sum the partials into a preallocated typed buffer. (Same rationale as the 1D
    # core: avoids `@distributed (+)`'s `Any`-typed reduction / the return-type assertion.)
    N = length(x_vecs[1])
    nw = max(1, Distributed.nworkers())
    perm = _balanced_triangle_perm(N)
    chunks = [perm[w:nw:length(perm)] for w in 1:nw]

    D = length(x_vecs)
    vD = Val(D)
    nd = SFC.n_histogram_bins(distance_bins)
    nv = SFC.n_histogram_bins(value_bins)
    OT = promote_type(float(eltype(x_vecs[1])), float(eltype(u_vecs[1])))
    dist_be = SFC.BinEdges(distance_bins)
    val_be = SFC.BinEdges(value_bins)

    partials = Distributed.pmap(chunks) do ichunk
        ls = zeros(OT, nd, nv)
        lc = zeros(CT, nd, nv)
        for i in ichunk
            SFC.calculate_structure_function_2d_i!(
                ls, lc, vD, structure_function_type, i, x_vecs, u_vecs, dist_be, val_be;
                distance_metric = distance_metric,
            )
        end
        (ls, lc)
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
    ::SFC.DistributedBackend,
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix,
    u_arr::AbstractMatrix,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
)
    N_dims = size(x_arr, 1)
    x_vecs = ntuple(k -> view(x_arr, k, :), N_dims)
    u_vecs = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC._dispatch_execution_backend(
        SFC.DistributedBackend(),
        structure_function_type,
        x_vecs,
        u_vecs,
        distance_bins,
        value_bins;
        kwargs...,
    )
end

# --- Single Pass Dispatch ---
function SFC._dispatch_single_pass(
    ::SFC.DistributedBackend,
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
    
    combined_reduced = Distributed.@distributed (+) for i in _balanced_triangle_perm(n_points)
        local_combined = zeros(Float64, 2 * SFC.SINGLE_PASS_N, n_bins)
        x_i = SA.SVector{D, FT1}(ntuple(d -> x[d, i], vD))
        u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))
        
        for j in (i+1):n_points
            x_j = SA.SVector{D, FT1}(ntuple(d -> x[d, j], vD))
            
            r = distance_metric(x_i, x_j)
            bin_idx = SFH.digitize(r, distance_bins)
            
            if 1 <= bin_idx <= n_bins
                u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                du = u_j - u_i
                
                rh = SFH.r̂(x_i, x_j, distance_metric, r)
                du_L = LA.dot(du, rh)
                du_L2 = du_L * du_L
                du_T2 = SFH.transverse_norm2(du, rh)
                
                local_combined[1, bin_idx] += du_L2 + du_T2
                local_combined[2, bin_idx] += du_L2
                local_combined[3, bin_idx] += du_T2
                local_combined[4, bin_idx] += du_L * (du_L2 + du_T2)
                local_combined[5, bin_idx] += du_L * du_L2
                local_combined[6, bin_idx] += du_L * du_T2
                
                for t in (SFC.SINGLE_PASS_N + 1):(2 * SFC.SINGLE_PASS_N)
                    local_combined[t, bin_idx] += 1.0
                end
            end
        end
        local_combined
    end
    
    sums = OT.(combined_reduced[1:SFC.SINGLE_PASS_N, :])
    counts = CT.(combined_reduced[(SFC.SINGLE_PASS_N + 1):(2 * SFC.SINGLE_PASS_N), :])

    return SFC.append_helmholtz_rotational_divergent_rows(sums, counts, distance_bins)
end

function SFC._dispatch_single_pass_2d(
    ::SFC.DistributedBackend,
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

    combined_reduced = Distributed.@distributed (+) for i in _balanced_triangle_perm(n_points)
        local_combined = zeros(Float64, 2 * SFC.SINGLE_PASS_N, n_bins, n_val)
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
                du_T = SFH.mδu_t(du, rh)

                du_L2 = du_L * du_L
                du_T2 = SFH.transverse_norm2(du, rh)

                vals = (
                    du_L2 + du_T2,
                    du_L2,
                    du_T2,
                    du_L * (du_L2 + du_T2),
                    du_L * du_L2,
                    du_L * du_T2,
                )

                for t in 1:SFC.SINGLE_PASS_N
                    vb = SFC._sp2d_value_bin_at(value_bins, t)
                    vbin = SFH.digitize(vals[t], vb)
                    n_val_t = length(vb) - 1
                    if 1 <= vbin <= n_val_t && vbin <= n_val
                        local_combined[t, bin_idx, vbin] += vals[t]
                        local_combined[t + SFC.SINGLE_PASS_N, bin_idx, vbin] += 1.0
                    end
                end
            end
        end
        local_combined
    end

    sums = OT.(combined_reduced[1:SFC.SINGLE_PASS_N, :, :])
    counts = CT.(combined_reduced[(SFC.SINGLE_PASS_N + 1):(2 * SFC.SINGLE_PASS_N), :, :])
    return sums, counts
end

# --- Mutating 1D Dispatch ---
function SFC._dispatch_execution_backend!(
    ::SFC.DistributedBackend,
    sums::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    kwargs...,
) where {OT, CT}
    result = SFC._dispatch_execution_backend(
        SFC.DistributedBackend(),
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
    ::SFC.DistributedBackend,
    sums::AbstractVector{OT},
    counts::AbstractVector{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector;
    kwargs...,
) where {OT, CT, FT1 <: Number, FT2 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC._dispatch_execution_backend!(
        SFC.DistributedBackend(),
        sums,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        kwargs...,
    )
end

# --- Mutating 2D Dispatch ---
function SFC._dispatch_execution_backend!(
    ::SFC.DistributedBackend,
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
        SFC.DistributedBackend(),
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
    ::SFC.DistributedBackend,
    sums_2d::AbstractMatrix{OT},
    counts_2d::AbstractMatrix{CT},
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_arr::AbstractMatrix{FT1},
    u_arr::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    kwargs...,
) where {OT, CT, FT1 <: Number, FT2 <: Number}
    N_dims = size(x_arr, 1)
    x_tuple = ntuple(k -> view(x_arr, k, :), N_dims)
    u_tuple = ntuple(k -> view(u_arr, k, :), N_dims)
    return SFC._dispatch_execution_backend!(
        SFC.DistributedBackend(),
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

end
