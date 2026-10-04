# CPU batch drivers: wrappers over the batch_leading.jl kernels that select the serial executor.

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
    auxiliary_shared_positions!(sums, counts, x_mat, u_batch, sf_type, distance_bins; workspace, geometry, culling, weights)

Fixed geometry batch: `x` is (N_dims, N), `u` has trailing batch dims.
"""
function auxiliary_shared_positions!(sums, counts, x_mat::AbstractMatrix, u_batch,
        sf_type::SFT.AbstractPairwiseStructureFunctionType, distance_bins;
        workspace = nothing, geometry, culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_1d!(sums, counts, sf_type, x_mat, u_batch, distance_bins, geometry,
        _bl_serial_exec, workspace; weights, culling)
end

"""
    auxiliary_varying_positions!(sums, counts, x_batch, u_batch, sf_type, distance_bins)

Varying geometry batch: `x` and `u` have matching trailing batch dims.
"""
function auxiliary_varying_positions!(sums, counts, x_batch, u_batch,
        sf_type::SFT.AbstractPairwiseStructureFunctionType, distance_bins;
        workspace = nothing, geometry, culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_1d!(sums, counts, sf_type, x_batch, u_batch, distance_bins, geometry,
        _bl_serial_exec, workspace; weights, culling)
end

"""
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins)

Six-invariant-type single-pass 1D batch (serial).
"""
function serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins;
        workspace = nothing, geometry, culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_sp1d!(sums, counts, x, u, distance_bins, geometry, _bl_serial_exec, workspace; weights, culling)
end

"""
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins)

Six-invariant-type SP2D batch (serial); output `(6, n_dist, n_val, batch…)`.
"""
function serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins,
        value_bins::SinglePass2DValueBins; workspace = nothing, geometry, culling::CullingPolicy = AutoCulling(),
        weights = NoWeights())
    _bl_run_sp2d!(sums, counts, x, u, distance_bins, value_bins, geometry, _bl_serial_exec, workspace;
        weights, culling)
end

"""
    auxiliary_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins; second_axis)

Single-type joint 2D batch (serial); output `(n_dist, n_val, batch…)`, the second axis binning what
`second_axis` reads from each pair.
"""
function auxiliary_joint2d!(sums, counts, sf_type::SFT.AbstractPairwiseStructureFunctionType,
        x, u, distance_bins, value_bins; workspace = nothing,
        geometry, culling::CullingPolicy = AutoCulling(),
        second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
        weights = NoWeights())
    _bl_run_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins, geometry, _bl_serial_exec,
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
    geometry,
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
            geometry,
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

# --- Batch methods of serial_calculate_structure_function / threaded_calculate_structure_function ---

@inline _component_vector_views(a, ::Val{D}) where {D} =
    ntuple(k -> view(a, k, :), Val(D))

"""
    _prepared_tuples(geometry, x, u) -> (x_tuple, u_tuple)

`x` and `u` converted into the form the kernels of `geometry` index, viewed as component tuples.
"""
@inline function _prepared_tuples(geometry, x, u)
    xk, uk = SFH.prepare_pair_inputs(geometry, x, u)
    return _component_vector_views(xk, SFH.coordinate_width(geometry)),
           _component_vector_views(uk, SFH.field_width(geometry))
end

"""Flat geometry of the width a component tuple carries — the default when no metric is in play."""
@inline default_geometry(u_vecs) = SFH.FlatGeometry{length(u_vecs)}()

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    ::Type{CT};
    geometry,
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {FT1 <: Number, FT2 <: Number, CT}
    x_tuple, u_tuple = _prepared_tuples(geometry, x, u)
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
        geometry,
        culling,
        weights,
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
    ndims(u) >= 3 || throw(MethodError(serial_calculate_structure_function,
                                       (structure_function_type, x, u, distance_bins, CT)))
    n_bins = n_histogram_bins(distance_bins)
    bdims = batch_dims(u)
    FT = promote_type(float(FT1), float(FT2))
    sums = zeros(FT, n_bins, bdims...)
    counts = zeros(CT, n_bins, bdims...)
    auxiliary_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
    return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
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
