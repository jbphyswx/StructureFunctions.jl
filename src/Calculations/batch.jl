# CPU Batch Calculation Drivers
#
# The batch-leading, Val{D}-specialized kernels + drivers live in batch_leading.jl (included
# first). These public functions are thin wrappers selecting the serial executor; the
# OhMyThreads extension provides threaded executors over the batch axis.

using Distances: Distances as DI

# --- Public CPU Batch APIs ---

"""
    auxiliary_structure_function!(sums, counts, sf_type, x, u, distance_bins; kwargs...)

The CPU batch calculation: shared positions when `x` is a `(W, N)` matrix, varying positions for a
`(W, N, B…)` array or a [`BatchLeading`](@ref) `(B, W, N)` one.
"""
function auxiliary_structure_function!(sums, counts, sf_type, x::AbstractMatrix, u, distance_bins; kwargs...)
    auxiliary_shared_positions!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    return nothing
end

function auxiliary_structure_function!(sums, counts, sf_type, x, u, distance_bins; kwargs...)
    auxiliary_varying_positions!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    return nothing
end

"""
    auxiliary_shared_positions!(sums, counts, x_mat, u_batch, sf_type, distance_bins; strip_width=32)

Fixed geometry batch: `x` is (N_dims, N), `u` has trailing batch dims.
"""
function auxiliary_shared_positions!(sums, counts, x_mat::AbstractMatrix, u_batch,
        sf_type::SFT.AbstractPairwiseStructureFunctionType, distance_bins;
        workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(), culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_1d!(sums, counts, sf_type, x_mat, u_batch, distance_bins, distance_metric,
        _bl_serial_exec, workspace; weights, culling)
end

"""
    auxiliary_varying_positions!(sums, counts, x_batch, u_batch, sf_type, distance_bins)

Varying geometry batch: `x` and `u` have matching trailing batch dims.
"""
function auxiliary_varying_positions!(sums, counts, x_batch, u_batch,
        sf_type::SFT.AbstractPairwiseStructureFunctionType, distance_bins;
        workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(), culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_1d!(sums, counts, sf_type, x_batch, u_batch, distance_bins, distance_metric,
        _bl_serial_exec, workspace; weights, culling)
end

"""
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins)

Six-invariant-type single-pass 1D batch (serial).
"""
function serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins;
        workspace = nothing, distance_metric::DI.PreMetric = DI.Euclidean(), culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_sp1d!(sums, counts, x, u, distance_bins, distance_metric, _bl_serial_exec, workspace; weights, culling)
end

"""
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins)

Six-invariant-type SP2D batch (serial); output `(6, n_dist, n_val, batch…)`.
"""
function serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins,
        value_bins::SinglePass2DValueBins; workspace = nothing,
        distance_metric::DI.PreMetric = DI.Euclidean(), culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins, distance_metric, _bl_serial_exec, workspace;
        weights, culling)
end

"""
    auxiliary_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins; second_axis)

Single-type joint 2D batch (serial); output `(n_dist, n_val, batch…)`, the second axis binning what
`second_axis` reads from each pair.
"""
function auxiliary_joint2d!(sums, counts, sf_type::SFT.AbstractPairwiseStructureFunctionType,
        x, u, distance_bins, value_bins; workspace = nothing,
        distance_metric::DI.PreMetric = DI.Euclidean(), culling::CullingPolicy = AutoCulling(),
        second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
        weights = NoWeights())
    _bl_run_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins, distance_metric, _bl_serial_exec,
        workspace; weights, culling, second_axis)
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
            distance_bins,
        )
        sums_f[:, b] .= local_output
        counts_f[:, b] .= local_counts
    end
end

# --- Multi-threaded CPU Batch Reducers ---

"""Threaded batch driver over the trailing axis; the OhMyThreads extension supplies the methods."""
function auxiliary_structure_function_threaded! end

"""Threaded joint batch driver; the OhMyThreads extension supplies the methods."""
function auxiliary_joint2d_threaded! end

"""Threaded single-pass batch driver; the OhMyThreads extension supplies the methods."""
function threaded_calculate_structure_functions_single_pass! end

"""Threaded 2D single-pass batch driver; the OhMyThreads extension supplies the methods."""
function threaded_calculate_structure_functions_single_pass_2d! end

# --- Unified CPU Batch Entry Points (Methods of serial_calculate_structure_function / threaded_calculate_structure_function) ---

@inline _component_vector_views(a, ::Val{D}) where {D} =
    ntuple(k -> view(a, k, :), Val(D))

"""
    _prepared_tuples(distance_metric, x, u) -> (geom, x_tuple, u_tuple)

Fix the geometry from the pre-conversion velocity dimension, convert both arrays into the form the
kernels index, and view them as component tuples. `size(u, 1)` is the velocity dimension only before
the conversion, so this is the one place it can be read.
"""
@inline function _prepared_tuples(distance_metric, x, u)
    geom = SFH.pair_geometry_for(distance_metric, Val(size(u, 1)))
    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    return geom,
           _component_vector_views(xk, SFH.coordinate_width(geom)),
           _component_vector_views(uk, SFH.field_width(geom))
end

"""Flat geometry of the width a component tuple carries — the default when no metric is in play."""
@inline default_geometry(u_vecs) = SFH.FlatGeometry{length(u_vecs)}()

function _serial_calculate_structure_function_point(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    vD::Val{D},
    ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1, FT2, D, CT}
    geom = SFH.pair_geometry_for(distance_metric, vD)
    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    x_tuple = _component_vector_views(xk, SFH.coordinate_width(geom))
    u_tuple = _component_vector_views(uk, SFH.field_width(geom))
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
        geometry = geom,
        culling = culling,
        weights = weights,
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
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    if ndims(u) >= 3
        n_bins = n_histogram_bins(distance_bins)
        bdims = batch_dims(u)
        FT = promote_type(float(FT1), float(FT2))
        sums = zeros(FT, n_bins, bdims...)
        counts = zeros(CT, n_bins, bdims...)
        auxiliary_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
    end
    # Point-field route:
    D = size(u, 1)
    D == 1 && return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(1), CT; kwargs...,
    )
    D == 2 && return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(2), CT; kwargs...,
    )
    D == 3 && return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(3), CT; kwargs...,
    )
    _validate_spatial_dimension(D)
    return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(D), CT; kwargs...,
    )
end

function threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    ::Type{CT};
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, CT}
    ndims(u) >= 3 || throw(MethodError(threaded_calculate_structure_function,
                                       (structure_function_type, x, u, distance_bins, CT)))
    n_bins = n_histogram_bins(distance_bins)
    bdims = batch_dims(u)
    FT = promote_type(float(FT1), float(FT2))
    sums = zeros(FT, n_bins, bdims...)
    counts = zeros(CT, n_bins, bdims...)
    auxiliary_structure_function_threaded!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
end
