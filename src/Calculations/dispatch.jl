# Dispatch Entry Points and Backend Routing

# --- Public Entry Points ---

# Tuple inputs are deliberately rejected at the public calculation boundary while
# the array API is stabilized. Lower-level tuple kernels remain private helpers.
function _derived_structure_function_error(structure_function_type)
    throw(ArgumentError(
        "$(typeof(structure_function_type)) is a derived structure-function quantity, " *
        "not a pairwise operator. Use helmholtz_decompose_2d for 2D Helmholtz " *
        "rotational/divergent quantities.",
    ))
end

# --- Result finalization (output-type dispatch) ---
# Backends compute and return only the raw accumulator (`…SumsAndCounts`). The public boundary
# maps it to the requested `output_type` via dispatch on `(raw, ::Type{output})`. Asking for an
# unsupported representation (e.g. an averaged 2D result) errors cleanly via the fallback — it
# never silently ignores, and the request can never leak into a backend kernel as a stray kwarg.
_finalize(r::SFO.StructureFunctionSumsAndCounts, ::Type{<:SFO.StructureFunctionSumsAndCounts}) = r
_finalize(r::SFO.StructureFunctionSumsAndCounts, ::Type{<:SFO.StructureFunction}) =
    SFO.StructureFunction(r.operator, r.distance, _bin_average(r.sums, r.counts))
_finalize(r::SFO.StructureFunction2DSumsAndCounts, ::Type{<:SFO.StructureFunction2DSumsAndCounts}) = r
_finalize(r::SFO.StructureFunctionTensorSumsAndCounts, ::Type{<:SFO.StructureFunctionTensorSumsAndCounts}) = r
_finalize(r::SFO.StructureFunctionTensorSumsAndCounts{P}, ::Type{<:SFO.StructureFunctionTensor}) where {P} =
    SFO.StructureFunctionTensor(r.order, r.distance_bins, _tensor_bin_average(r.sums, r.counts, Val(P)))
_finalize(r, ::Type{R}) where {R} = throw(ArgumentError(
    "Cannot produce a $R from this calculation (got a $(typeof(r))). Check the `output_type` keyword.",
))

function calculate_structure_function(
    structure_function_type::SFT.AbstractDerivedStructureFunctionType,
    x,
    u,
    distance_bins;
    kwargs...,
)
    _derived_structure_function_error(structure_function_type)
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractDerivedStructureFunctionType,
    x,
    u,
    distance_bins,
    value_bins;
    kwargs...,
)
    _derived_structure_function_error(structure_function_type)
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector;
    kwargs...,
)
    _unsupported_tuple_input()
end

# 1D public entry: the backend returns the raw `StructureFunctionSumsAndCounts`; `_finalize`
# maps it to the requested `output_type` (default the averaged `StructureFunction`).
function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32
    ;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {FT1, FT2, OT, CT}
    shape = _validate_array_shape(x, u, distance_metric)
    _assert_counts_representable(CT, size(x, 2))
    raw = _dispatch_execution_backend(
        backend,
        shape,
        structure_function_type,
        x,
        u,
        distance_bins,
        count_eltype
        ;
        distance_metric,
        kwargs...,
    )
    return _finalize(raw, output_type)
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::AbstractVector,
    value_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    kwargs...,
)
    _unsupported_tuple_input()
end

# 2D joint public entry: the backend returns the raw `StructureFunction2DSumsAndCounts`;
# `_finalize` defaults to returning it as-is. There is no averaged 2D representation, so any
# other `output_type` errors cleanly via the `_finalize` fallback (never silently ignored).
function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    count_eltype::Type{CT} = UInt32
    ;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction2DSumsAndCounts,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {FT1, FT2, OT, CT}
    shape = _validate_array_shape(x, u, distance_metric)
    _assert_counts_representable(CT, size(x, 2))
    raw = _dispatch_execution_backend(
        backend,
        shape,
        structure_function_type,
        x,
        u,
        distance_bins,
        value_bins;
        distance_metric,
        count_eltype,
        kwargs...,
    )
    return _finalize(raw, output_type)
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x_vecs::Tuple,
    u_vecs::Tuple,
    distance_bins::Int,
    args...;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing::Type{<:AbstractBinEdges} = LogBinEdges,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
)
    _unsupported_tuple_input()
end

function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::Int,
    args...;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing::Type{<:AbstractBinEdges} = LogBinEdges,
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {FT1, FT2}
    shape = _validate_array_shape(x, u, distance_metric)

    if verbose
        @info("Calculating min and max distances and generating bins")
    end
    min_distance, max_distance = _minmax_for_autobins(shape, x, distance_metric, show_progress)
    actual_bins = _auto_distance_bins(min_distance, max_distance, distance_bins, bin_spacing)

    # `bin_spacing` selected these edges and means nothing downstream, so it is consumed here
    # rather than forwarded into a `kwargs...` sink that would silently swallow it.
    return calculate_structure_function(
        structure_function_type,
        x,
        u,
        actual_bins,
        args...;
        distance_metric,
        verbose,
        show_progress,
        kwargs...,
    )
end

function _auto_distance_bins(min_distance, max_distance, distance_bins::Int, ::Type{LinearBinEdges})
    min_distance = prevfloat(min_distance)
    return LinearBinEdges(range(min_distance, max_distance, length = distance_bins + 1))
end

function _auto_distance_bins(min_distance, max_distance, distance_bins::Int, ::Type{LogBinEdges})
    min_distance = prevfloat(min_distance)
    edge_vec = 10 .^ range(log10(min_distance), log10(max_distance), length = distance_bins + 1)
    edge_vec[1] = min_distance
    edge_vec[end] = max_distance
    return LogBinEdges(edge_vec)
end

function _auto_distance_bins(min_distance, max_distance, distance_bins::Int, bin_spacing)
    throw(ArgumentError("bin_spacing must be LinearBinEdges or LogBinEdges; got $bin_spacing"))
end

function _minmax_for_autobins(::PointField, x::AbstractMatrix, distance_metric, show_progress::Bool)
    return _minmax_matrix_for_autobins(x, distance_metric, show_progress)
end

function _minmax_for_autobins(::SharedPositionField, x::AbstractMatrix, distance_metric, show_progress::Bool)
    return _minmax_matrix_for_autobins(x, distance_metric, show_progress)
end

function _minmax_matrix_for_autobins(x::AbstractMatrix, distance_metric, show_progress::Bool)
    # Accumulate in the input eltype; Float64 literals here would widen the bin edges.
    FT = float(eltype(x))
    min_distance, max_distance = FT(Inf), FT(0)
    PM.@showprogress enabled = show_progress for i in axes(x, 2)
        _min_distance, _max_distance = minmax_i(i, x, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
    end
    return min_distance, max_distance
end

function _minmax_for_autobins(::VaryingPositionField, x::AbstractArray, distance_metric, show_progress::Bool)
    D, N = size(x, 1), size(x, 2)
    B = prod(size(x)[3:end])
    x_flat = reshape(x, D, N, B)
    FT = float(eltype(x))
    min_distance, max_distance = FT(Inf), FT(0)
    PM.@showprogress enabled = show_progress for b in 1:B
        x_slice = @view x_flat[:, :, b]
        for i in axes(x_slice, 2)
            _min_distance, _max_distance = minmax_i(i, x_slice, distance_metric)
            min_distance = min(min_distance, _min_distance)
            max_distance = max(max_distance, _max_distance)
        end
    end
    return min_distance, max_distance
end

# --- Auto-binning MinMax Helpers ---

"""
    minmax_i(i, x_vecs, distance_metric)

Calculate the min and max distances from point `i` to all other points `j != i`.
"""
function minmax_i(
    i::Int,
    x_vecs::Tuple,
    distance_metric = DI.Euclidean(),
)
    D = length(x_vecs)
    FT = eltype(x_vecs[1])
    X1 = SA.SVector{D, FT}(ntuple(k -> x_vecs[k][i], Val(D)))

    min_distance, max_distance = FT(Inf), FT(0.0)
    iter_inds = eachindex(x_vecs[1])
    # `j > i`: the metric is symmetric, so `j != i` measured every pair twice.
    for j in iter_inds
        if j > i
            X2 = SA.SVector{D, FT}(ntuple(k -> x_vecs[k][j], Val(D)))
            distance = distance_metric(X1, X2)
            if distance < min_distance
                min_distance = distance
            end
            if distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end

function minmax_i(
    i::Int,
    x_arr::AbstractArray{FT},
    distance_metric = DI.Euclidean(),
) where {FT <: Number}
    N_dims = size(x_arr, 1)
    X1 = SA.SVector{N_dims, FT}(ntuple(k -> x_arr[k, i], Val(N_dims)))

    min_distance, max_distance = FT(Inf), FT(0.0)
    for j in axes(x_arr, 2)
        if i != j
            X2 = SA.SVector{N_dims, FT}(ntuple(k -> x_arr[k, j], Val(N_dims)))
            distance = distance_metric(X1, X2)
            if distance < min_distance
                min_distance = distance
            end
            if distance > max_distance
                max_distance = distance
            end
        end
    end
    return min_distance, max_distance
end

# --- StructureFunction Factory Constructor ---
function SFO.StructureFunction(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x,
    u,
    bins,
    args...;
    kwargs...,
)
    return calculate_structure_function(structure_function_type, x, u, bins, args...; kwargs...)
end

# --- Backend Dispatch for Mutating API (calculate_structure_function!) ---

function calculate_structure_function!(
    sums,
    counts,
    sf_type::SFT.AbstractDerivedStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins;
    kwargs...
)
    _derived_structure_function_error(sf_type)
end

function calculate_structure_function!(
    sums_2d,
    counts_2d,
    sf_type::SFT.AbstractDerivedStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins,
    value_bins;
    kwargs...
)
    _derived_structure_function_error(sf_type)
end

function calculate_structure_function!(
    sums, counts, sf_type, x::Tuple, u::Tuple, distance_bins;
    backend=CB.SerialBackend(), kwargs...
)
    _unsupported_tuple_input()
end

function calculate_structure_function!(
    sums, counts, sf_type, x::AbstractArray, u::AbstractArray, distance_bins;
    backend=CB.SerialBackend(), distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...
)
    shape = _validate_array_shape(x, u, distance_metric)
    _assert_counts_representable(eltype(counts), size(x, 2))
    _dispatch_execution_backend!(backend, shape, sums, counts, sf_type, x, u, distance_bins;
        distance_metric, kwargs...)
    return nothing
end

function calculate_structure_function!(
    sums_2d, counts_2d, sf_type, x::Tuple, u::Tuple, distance_bins, value_bins;
    backend=CB.SerialBackend(), kwargs...
)
    _unsupported_tuple_input()
end

function calculate_structure_function!(
    sums_2d, counts_2d, sf_type, x::AbstractArray, u::AbstractArray, distance_bins, value_bins;
    backend=CB.SerialBackend(), distance_metric::DI.PreMetric = DI.Euclidean(), kwargs...
)
    shape = _validate_array_shape(x, u, distance_metric)
    _assert_counts_representable(eltype(counts_2d), size(x, 2))
    _dispatch_execution_backend!(backend, shape, sums_2d, counts_2d, sf_type, x, u, distance_bins, value_bins;
        distance_metric, kwargs...)
    return nothing
end

# # --- Backend Dispatch Layers for Mutating API ---

function _dispatch_execution_backend!(
    ::CB.AbstractSerialBackend, shape::AbstractFieldShape, sums, counts, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    if has_auxiliary_axes(shape)
        auxiliary_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        return nothing
    end
    serial_calculate_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_execution_backend!(
    ::CB.AbstractThreadedBackend, shape::AbstractFieldShape, sums, counts, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    if has_auxiliary_axes(shape)
        auxiliary_structure_function_threaded!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        return nothing
    end
    threaded_calculate_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_execution_backend!(
    backend::CB.AbstractGPUBackend, shape::PointField, sums::AbstractVector, counts::AbstractVector, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    gpu_calculate_structure_function!(sums, counts, structure_function_type, backend.backend, x, u, distance_bins; kwargs...)
    return nothing
end

function _dispatch_execution_backend!(
    backend::CB.AbstractGPUBackend, shape::AbstractFieldShape, sums, counts, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    throw(ArgumentError("in-place auxiliary-axis calculate_structure_function! is not implemented for GPUBackend"))
end

# --- Replaced AutoBackend Mutating Dispatch ---
function _dispatch_execution_backend!(
    ::CB.AbstractAutoBackend, shape::AbstractFieldShape, sums, counts, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    if has_auxiliary_axes(shape)
        if Threads.nthreads() > 1
            return _dispatch_execution_backend!(CB.ThreadedBackend(), shape, sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        end
        return _dispatch_execution_backend!(CB.SerialBackend(), shape, sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
    end

    if Threads.nthreads() > 1 &&
       _threaded_backend_available!(sums, counts, structure_function_type, x, u, distance_bins)
        return threaded_calculate_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
    end

    return serial_calculate_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
end

# --- Mutating 2D Backend Dispatch Layers ---

function _dispatch_execution_backend!(
    ::CB.AbstractSerialBackend, shape::AbstractFieldShape, sums_2d, counts_2d, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    if has_auxiliary_axes(shape)
        auxiliary_joint2d!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
        return nothing
    end
    serial_calculate_structure_function!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_execution_backend!(
    ::CB.AbstractThreadedBackend, shape::AbstractFieldShape, sums_2d, counts_2d, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    if has_auxiliary_axes(shape)
        auxiliary_joint2d_threaded!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
        return nothing
    end
    threaded_calculate_structure_function!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
    return nothing
end

function _dispatch_execution_backend!(
    backend::CB.AbstractGPUBackend, shape::AbstractFieldShape, sums_2d, counts_2d, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("In-place calculate_structure_function! is not supported on GPU backend."))
end

function _dispatch_execution_backend!(
    ::CB.AbstractAutoBackend, shape::AbstractFieldShape, sums_2d, counts_2d, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    if has_auxiliary_axes(shape)
        if Threads.nthreads() > 1
            return _dispatch_execution_backend!(CB.ThreadedBackend(), shape, sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
        end
        return _dispatch_execution_backend!(CB.SerialBackend(), shape, sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
    end

    if Threads.nthreads() > 1 &&
       _threaded_backend_available!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins)
        return threaded_calculate_structure_function!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
    end

    return serial_calculate_structure_function!(sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend, shape::AbstractFieldShape, sums, counts, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    throw(ArgumentError("calculate_structure_function! is not implemented for DistributedBackend."))
end

function _dispatch_execution_backend!(
    backend::CB.AbstractExecutionBackend, sums::AbstractArray, counts::AbstractArray, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    throw(ArgumentError("calculate_structure_function! is not implemented for backend $(typeof(backend))."))
end

function _dispatch_execution_backend!(
    ::CB.AbstractDistributedBackend, shape::AbstractFieldShape, sums_2d, counts_2d, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("calculate_structure_function! with distance_bins and value_bins is not implemented for DistributedBackend."))
end

function _dispatch_execution_backend!(
    backend::CB.AbstractExecutionBackend, sums_2d::AbstractArray, counts_2d::AbstractArray, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("calculate_structure_function! with distance_bins and value_bins is not implemented for backend $(typeof(backend))."))
end

# --- Non-Mutating Dispatch Layers ---
# 1D (distance_bins only) and 2D (distance_bins + value_bins) are distinguished by ARITY here:
# 1D methods take 6 positional args and return a raw `StructureFunctionSumsAndCounts`; 2D methods
# take a 7th `value_bins::AbstractVector` and return a raw `StructureFunction2DSumsAndCounts`.
# The public boundary applies `_finalize` to pick the representation.

# 1D
function _dispatch_execution_backend(
    ::CB.AbstractSerialBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, count_eltype::Type{CT} = UInt32; kwargs...
) where {CT}
    return serial_calculate_structure_function(structure_function_type, x, u, distance_bins, count_eltype; kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractSerialBackend, shape::PointField{D}, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, count_eltype::Type{CT} = UInt32;
    kwargs...
) where {D, CT}
    return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(D), count_eltype; kwargs...,
    )
end

function _dispatch_execution_backend(
    ::CB.AbstractThreadedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, count_eltype::Type{CT} = UInt32; kwargs...
) where {CT}
    return threaded_calculate_structure_function(structure_function_type, x, u, distance_bins, count_eltype; kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractDistributedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, count_eltype::Type{CT} = UInt32; kwargs...
) where {CT}
    return _dispatch_execution_backend(backend, structure_function_type, x, u, distance_bins; count_eltype, kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractGPUBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, count_eltype::Type{CT} = UInt32; kwargs...
) where {CT}
    if has_auxiliary_axes(shape)
        return gpu_calculate_structure_function_batch(structure_function_type, backend.backend, x, u, distance_bins; count_eltype, kwargs...)
    end
    return gpu_calculate_structure_function(structure_function_type, backend.backend, x, u, distance_bins; count_eltype, kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractAutoBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, count_eltype::Type{CT} = UInt32; kwargs...
) where {CT}
    backend = resolve_auto_backend(
        shape,
        () -> _threaded_backend_available(structure_function_type, x, u, distance_bins),
        () -> true,
    )
    return _dispatch_execution_backend(backend, shape, structure_function_type, x, u, distance_bins, count_eltype; kwargs...)
end

# 2D (joint distance×value)
function _dispatch_execution_backend(
    ::CB.AbstractSerialBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    return serial_calculate_structure_function(structure_function_type, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractThreadedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    return threaded_calculate_structure_function(structure_function_type, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractDistributedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    return _dispatch_execution_backend(backend, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractGPUBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    if has_auxiliary_axes(shape)
        return gpu_calculate_structure_function_2d_batch(structure_function_type, backend.backend, x, u, distance_bins, value_bins; kwargs...)
    end
    return gpu_calculate_structure_function_2d(structure_function_type, backend.backend, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractAutoBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    backend = resolve_auto_backend(
        shape,
        () -> _threaded_backend_available(structure_function_type, x, u, distance_bins),
        () -> true,
    )
    return _dispatch_execution_backend(backend, shape, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
end
