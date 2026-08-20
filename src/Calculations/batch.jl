# CPU Batch Calculation Drivers
#
# The batch-leading, Val{D}-specialized kernels + drivers live in batch_leading.jl (included
# first). These public functions are thin wrappers selecting the serial executor; the
# OhMyThreads extension provides threaded executors over the batch axis.

using Distances: Distances as DI

# --- Public CPU Batch APIs ---

"""
    auxiliary_structure_function!(sums, counts, sf_type, x, u, distance_bins; strip_width=32)

Dispatch CPU batch calculation based on `x` rank.
"""
function auxiliary_structure_function!(
    sums,
    counts,
    sf_type,
    x,
    u,
    distance_bins;
    kwargs...,
)
    if ndims(x) == 2
        auxiliary_shared_positions!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    else
        auxiliary_varying_positions!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    end
    return nothing
end

"""
    auxiliary_shared_positions!(sums, counts, x_mat, u_batch, sf_type, distance_bins; strip_width=32)

Fixed geometry batch: `x` is (N_dims, N), `u` has trailing batch dims.
"""
function auxiliary_shared_positions!(sums, counts, x_mat::AbstractMatrix, u_batch,
        sf_type::SFT.AbstractPairwiseStructureFunctionType, distance_bins;
        workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(),
        verbose::Bool = true, show_progress::Bool = true)
    _bl_run_1d!(sums, counts, sf_type, x_mat, u_batch, BinEdges(distance_bins), distance_metric,
        _bl_serial_exec, workspace)
end

"""
    auxiliary_varying_positions!(sums, counts, x_batch, u_batch, sf_type, distance_bins)

Varying geometry batch: `x` and `u` have matching trailing batch dims.
"""
function auxiliary_varying_positions!(sums, counts, x_batch, u_batch,
        sf_type::SFT.AbstractPairwiseStructureFunctionType, distance_bins;
        workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(),
        verbose::Bool = true, show_progress::Bool = true)
    _bl_run_1d!(sums, counts, sf_type, x_batch, u_batch, BinEdges(distance_bins), distance_metric,
        _bl_serial_exec, workspace)
end

"""
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins)

Six-invariant-type single-pass 1D batch (serial).
"""
function serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins;
        workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(),
        verbose::Bool = true, show_progress::Bool = true)
    _bl_run_sp1d!(sums, counts, x, u, BinEdges(distance_bins), distance_metric, _bl_serial_exec, workspace)
end

"""
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins)

Six-invariant-type SP2D batch (serial); output `(6, n_dist, n_val, batch…)`.
"""
function serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins,
        value_bins::SinglePass2DValueBins; workspace = nothing,
        distance_metric::DI.PreMetric = DI.Euclidean(),
        verbose::Bool = true, show_progress::Bool = true)
    _bl_run_sp2d!(sums, counts, x, u, BinEdges(distance_bins), value_bins, distance_metric,
        _bl_serial_exec, workspace)
end

"""
    auxiliary_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins)

Single-type joint 2D batch (serial); output `(n_dist, n_val, batch…)`.
"""
function auxiliary_joint2d!(sums, counts, sf_type::SFT.AbstractPairwiseStructureFunctionType,
        x, u, distance_bins, value_bins; workspace = nothing,
        distance_metric::DI.PreMetric = DI.Euclidean(),
        verbose::Bool = true, show_progress::Bool = true)
    _bl_run_joint2d!(sums, counts, sf_type, x, u, BinEdges(distance_bins), BinEdges(value_bins),
        distance_metric, _bl_serial_exec, workspace)
end

"""Loop-over-slice gold reference for batch parity."""
function cpu_slice_baseline!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    distance_bins;
    fixed_x::Bool = true,
) where {FT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    bd = batch_dims(u)
    B = batch_size(u)
    sums_f, counts_f = _flatten_sums_counts(sums, counts)
    for b in 1:B
        if fixed_x
            x_slice = x
            u_slice = batch_field_slice(u, b)
        else
            x_slice = batch_field_slice(x, b)
            u_slice = batch_field_slice(u, b)
        end
        local_output = zeros(eltype(sums), n_bins)
        local_counts = zeros(eltype(counts), n_bins)
        serial_calculate_structure_function!(
            local_output,
            local_counts,
            sf_type,
            x_slice,
            u_slice,
            distance_bins;
            verbose = false,
            show_progress = false,
        )
        sums_f[:, b] .= local_output
        counts_f[:, b] .= local_counts
    end
end

# --- Multi-threaded CPU Batch Reducers ---
#
# Threading lives in the OhMyThreads extension (StructureFunctionsOhMyThreadsExt), which
# defines more-specialized `::AbstractArray` methods that win dispatch and parallelize over
# the batch axis B (disjoint b-slices of one shared batch-leading accumulator — no threadid,
# no per-thread replication). These generic core methods are the SERIAL FALLBACK used when
# OhMyThreads is not loaded, so the threaded/auto backends stay correct (just not parallel).

auxiliary_structure_function_threaded!(sums, counts, sf_type, x, u, distance_bins; kwargs...) =
    auxiliary_structure_function!(sums, counts, sf_type, x, u, distance_bins; kwargs...)

auxiliary_joint2d_threaded!(sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...) =
    auxiliary_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins; kwargs...)

threaded_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...) =
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)

threaded_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...) =
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)

# --- Unified CPU Batch Entry Points (Methods of serial_calculate_structure_function / threaded_calculate_structure_function) ---

@inline _component_vector_views(a, ::Val{D}) where {D} =
    ntuple(k -> view(a, k, :), Val(D))

function _serial_calculate_structure_function_point(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    vD::Val{D},
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1, FT2, D, CT}
    _assert_counts_representable(CT, size(x, 2))
    # `D` is static here, so the geometry — and with it the coordinate width, which differs from `D`
    # on a shell — is a concrete type.
    x_tuple = _component_vector_views(x, SFH.coordinate_width(SFH.pair_geometry_for(distance_metric, vD)))
    u_tuple = _component_vector_views(u, vD)
    OT = promote_type(float(FT1), float(FT2))
    output = zeros(OT, n_histogram_bins(distance_bins))
    counts = zeros(CT, n_histogram_bins(distance_bins))

    serial_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
    )

    return SFO.StructureFunctionSumsAndCounts(
        structure_function_type,
        distance_bins,
        output,
        counts,
    )
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    if ndims(u) >= 3
        dist_be = BinEdges(distance_bins)
        n_bins = n_histogram_bins(dist_be)
        bdims = batch_dims(u)
        FT = promote_type(float(FT1), float(FT2))
        sums = zeros(FT, n_bins, bdims...)
        counts = zeros(CT, n_bins, bdims...)
        auxiliary_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    # Point-field route:
    D = size(u, 1)
    D == 2 && return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(2), count_eltype; kwargs...,
    )
    D == 3 && return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(3), count_eltype; kwargs...,
    )
    return _validate_spatial_dimension(D)
end

function threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32;
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    if ndims(u) >= 3
        dist_be = BinEdges(distance_bins)
        n_bins = n_histogram_bins(dist_be)
        bdims = batch_dims(u)
        FT = promote_type(float(FT1), float(FT2))
        sums = zeros(FT, n_bins, bdims...)
        counts = zeros(CT, n_bins, bdims...)
        auxiliary_structure_function_threaded!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    throw(ArgumentError("Threaded backend is unavailable for non-batch inputs. Load the OhMyThreads extension or use backend=CB.SerialBackend()."))
end
