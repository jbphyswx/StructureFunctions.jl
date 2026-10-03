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

# --- Result finalization (result-type dispatch) ---
# Backends compute and return only the raw accumulator (`…SumsAndCounts`). The public boundary
# maps it to the requested result type `OT` via dispatch on `(raw, ::Type{OT})`. An unsupported
# representation (e.g. an averaged 2D result) raises in the fallback.
_finalize(r::SFO.StructureFunctionSumsAndCounts, ::Type{<:SFO.StructureFunctionSumsAndCounts}) = r
_finalize(r::SFO.StructureFunctionSumsAndCounts, ::Type{<:SFO.StructureFunction}) =
    SFO.StructureFunction(r.operator, r.distance, _bin_average(r.sums, r.counts))
_finalize(r::SFO.StructureFunction2DSumsAndCounts, ::Type{<:SFO.StructureFunction2DSumsAndCounts}) = r
_finalize(r::SFO.StructureFunctionTensorSumsAndCounts, ::Type{<:SFO.StructureFunctionTensorSumsAndCounts}) = r
_finalize(r::SFO.StructureFunctionTensor2DSumsAndCounts, ::Type{<:SFO.StructureFunctionTensor2DSumsAndCounts}) = r
_finalize(r::SFO.StructureFunctionTensorSumsAndCounts{P}, ::Type{<:SFO.StructureFunctionTensor}) where {P} =
    SFO.StructureFunctionTensor(r.order, r.distance_bins, _tensor_bin_average(r.sums, r.counts, Val(P)))
_finalize(r, ::Type{R}) where {R} = throw(ArgumentError(
    "Cannot produce a $R from this calculation (got a $(typeof(r))). Check the result type argument.",
))

"""Zeroed result storage of element type `T` for `backend`: host arrays here; the KernelAbstractions
extension allocates on the device of a GPU backend."""
_result_zeros(::CB.AbstractExecutionBackend, ::Type{T}, dims::Integer...) where {T} = zeros(T, dims...)

"""The function that moves an array to the memory `backend` computes in: `identity` on a host backend; the
KernelAbstractions extension adapts to a GPU backend's device."""
_adaptor(::CB.AbstractExecutionBackend) = identity

"""Throw unless mutating outputs reside where `backend` computes: nothing to check on a host backend; the
KernelAbstractions extension checks a GPU backend's."""
_require_device_outputs(::CB.AbstractExecutionBackend, sums, counts) = nothing

"""`workspace` as a keyword to forward, or none when there is no workspace."""
@inline _workspace_kw(::Nothing) = (;)
@inline _workspace_kw(workspace) = (; workspace)

calculate_structure_function(structure_function_type::SFT.AbstractDerivedStructureFunctionType, x, u, args...;
                             kwargs...) = _derived_structure_function_error(structure_function_type)

calculate_structure_function(::SFT.AbstractPairwiseStructureFunctionType, x::Tuple, u::Tuple, args...; kwargs...) =
    _unsupported_tuple_input()

"""
    calculate_structure_function(sf, x, u, distance_bins[, CT][, OT]; backend, distance_metric, weights, kwargs...)
    calculate_structure_function(sf, x, u, distance_bins, value_bins[, CT][, OT]; kwargs...)

The pair histogram of `sf` over `distance_bins`, or the joint distance × value histogram with
`value_bins`, on `backend`. `CT` is the count element type (default `$(DEFAULT_COUNT_TYPE)`), and
`OT` the result representation: `StructureFunction` (the default without `value_bins`) or
`StructureFunctionSumsAndCounts`, and `StructureFunction2DSumsAndCounts` with `value_bins`.
`weights`, one finite value per point, weight each pair by the product of its two points' weights and
need a floating-point `CT`.
"""
function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    ::Type{CT},
    ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
    kwargs...,
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), promote_type(float(eltype(x)), float(eltype(u))))
    _assert_count_type(CT, size(x, 2), w)
    # The shape carries the velocity dimension as a type parameter, but that dimension is an array
    # axis length, so the constructed type is not inferrable and types every kernel below it `Any`.
    # Re-entering through a concrete `Val` hands each branch a shape whose parameter is known: the
    # branch is chosen at runtime, everything under it is not.
    D = size(u, 1)
    S = _shape_kind(x, u)
    kw = (; distance_metric, weights = w, kwargs...)
    b = (backend, structure_function_type, x, u, distance_bins)
    raw = D == 1 ? _dw(S{1}(), b..., CT, kw) :
          D == 2 ? _dw(S{2}(), b..., CT, kw) :
          D == 3 ? _dw(S{3}(), b..., CT, kw) :
          D == 4 ? _dw(S{4}(), b..., CT, kw) :
          D == 5 ? _dw(S{5}(), b..., CT, kw) :
          D == 6 ? _dw(S{6}(), b..., CT, kw) :
          D == 7 ? _dw(S{7}(), b..., CT, kw) :
          D == 8 ? _dw(S{8}(), b..., CT, kw) : _dw(shape, b..., CT, kw)
    return _finalize(raw, OT)
end

calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
                             distance_bins::AbstractVector; kwargs...) =
    calculate_structure_function(sf, x, u, distance_bins, DEFAULT_COUNT_TYPE, SFO.StructureFunction; kwargs...)
calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
                             distance_bins::AbstractVector, ::Type{CT}; kwargs...) where {CT <: Real} =
    calculate_structure_function(sf, x, u, distance_bins, CT, SFO.StructureFunction; kwargs...)
calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
                             distance_bins::AbstractVector, ::Type{OT}; kwargs...) where {OT <: SFO.AbstractStructureFunction} =
    calculate_structure_function(sf, x, u, distance_bins, DEFAULT_COUNT_TYPE, OT; kwargs...)

"""Dispatch a validated shape through a specialization boundary.

Common small dimensions have explicit branches at the public entry point.
Other dimensions specialize once on the concrete shape type at this boundary.
"""
@inline _dw(shape, backend, sf, x, u, distance_bins, ::Type{CT}, kw::NamedTuple) where {CT} =
    _dispatch_execution_backend(backend, shape, sf, x, u, distance_bins, CT; kw...)


"""
    _shape_kind(x, u) -> Type

Which shape family the inputs form, from their **ranks** alone.

`ndims` is a property of the array type, so this constant-folds; the velocity width is an axis
length and is supplied separately. Keeping the two apart is what lets the caller build a shape whose
width parameter is a literal.
"""
@inline _shape_kind(x::AbstractArray, u::AbstractArray) =
    ndims(x) == 2 && ndims(u) == 2 ? PointField :
    ndims(x) == 2 ? SharedPositionField : VaryingPositionField

# There is no averaged joint representation, so an `OT` other than the raw histogram raises in
# `_finalize`.
function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::AbstractVector,
    ::Type{CT},
    ::Type{OT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
    kwargs...,
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), promote_type(float(eltype(x)), float(eltype(u))))
    _assert_count_type(CT, size(x, 2), w)
    raw = _dispatch_execution_backend(backend, shape, structure_function_type, x, u, distance_bins,
        value_bins, CT; distance_metric, weights = w, kwargs...)
    return _finalize(raw, OT)
end

calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
                             distance_bins::AbstractVector, value_bins::AbstractVector; kwargs...) =
    calculate_structure_function(sf, x, u, distance_bins, value_bins, DEFAULT_COUNT_TYPE,
                                 SFO.StructureFunction2DSumsAndCounts; kwargs...)
calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
                             distance_bins::AbstractVector, value_bins::AbstractVector,
                             ::Type{CT}; kwargs...) where {CT <: Real} =
    calculate_structure_function(sf, x, u, distance_bins, value_bins, CT, SFO.StructureFunction2DSumsAndCounts; kwargs...)
calculate_structure_function(sf::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
                             distance_bins::AbstractVector, value_bins::AbstractVector,
                             ::Type{OT}; kwargs...) where {OT <: SFO.AbstractStructureFunction} =
    calculate_structure_function(sf, x, u, distance_bins, value_bins, DEFAULT_COUNT_TYPE, OT; kwargs...)

function calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::Int,
    args...;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    bin_spacing::Type{<:AbstractBinEdges} = LogBinEdges,
    kwargs...,
) where {FT1, FT2}
    shape = _validate_array_shape(x, u, distance_metric)
    min_distance, max_distance = _minmax_for_autobins(shape, x, distance_metric)
    actual_bins = _auto_distance_bins(min_distance, max_distance, distance_bins, bin_spacing)

    # `bin_spacing` selected these edges and means nothing downstream, so it is consumed here.
    return calculate_structure_function(
        structure_function_type,
        x,
        u,
        actual_bins,
        args...;
        distance_metric,
        kwargs...,
    )
end

function _auto_distance_bins(min_distance, max_distance, distance_bins::Int, ::Type{LinearBinEdges})
    min_distance = prevfloat(min_distance)
    return LinearBinEdges(range(min_distance, max_distance, length = distance_bins + 1))
end

function _auto_distance_bins(min_distance, max_distance, distance_bins::Int, ::Type{LogBinEdges})
    return LogBinEdges_from_log_edges(
        range(log(prevfloat(min_distance)), log(max_distance); length = distance_bins + 1),
    )
end

function _auto_distance_bins(min_distance, max_distance, distance_bins::Int, bin_spacing)
    throw(ArgumentError("bin_spacing must be LinearBinEdges or LogBinEdges; got $bin_spacing"))
end

function _minmax_for_autobins(::PointField, x::AbstractMatrix, distance_metric)
    return _minmax_matrix_for_autobins(x, distance_metric)
end

function _minmax_for_autobins(::SharedPositionField, x::AbstractMatrix, distance_metric)
    return _minmax_matrix_for_autobins(x, distance_metric)
end

function _minmax_matrix_for_autobins(x::AbstractMatrix, distance_metric)
    # Accumulate in the input eltype; Float64 literals here would widen the bin edges.
    FT = float(eltype(x))
    min_distance, max_distance = FT(Inf), FT(0)
    for i in axes(x, 2)
        _min_distance, _max_distance = minmax_i(i, x, distance_metric)
        min_distance = min(min_distance, _min_distance)
        max_distance = max(max_distance, _max_distance)
    end
    return min_distance, max_distance
end

function _minmax_for_autobins(::VaryingPositionField, x::AbstractArray, distance_metric)
    D, N = size(x, 1), size(x, 2)
    B = prod(size(x)[3:end])
    x_flat = reshape(x, D, N, B)
    FT = float(eltype(x))
    min_distance, max_distance = FT(Inf), FT(0)
    for b in 1:B
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

calculate_structure_function!(sums, counts, sf_type::SFT.AbstractDerivedStructureFunctionType, x, u, args...;
                              kwargs...) = _derived_structure_function_error(sf_type)

calculate_structure_function!(sums, counts, ::SFT.AbstractPairwiseStructureFunctionType, x::Tuple, u::Tuple,
                              args...; kwargs...) = _unsupported_tuple_input()

"""
    calculate_structure_function!(sums, counts, sf, x, u, distance_bins[, value_bins]; backend, distance_metric, weights, kwargs...)

Add the pairs of [`calculate_structure_function`](@ref) into `sums` and `counts`, whose element types
are the result's and the count type.
"""
function calculate_structure_function!(
    sums, counts, sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing, kwargs...,
)
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), eltype(sums))
    _assert_counts_can_accumulate(counts, size(x, 2), w)
    _dispatch_execution_backend!(backend, shape, sums, counts, sf_type, x, u, distance_bins;
        distance_metric, weights = w, kwargs...)
    return nothing
end

function calculate_structure_function!(
    sums_2d, counts_2d, sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector, value_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing, kwargs...,
)
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), eltype(sums_2d))
    _assert_counts_can_accumulate(counts_2d, size(x, 2), w)
    _dispatch_execution_backend!(backend, shape, sums_2d, counts_2d, sf_type, x, u, distance_bins, value_bins;
        distance_metric, weights = w, kwargs...)
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
    gpu_calculate_structure_function!(sums, counts, structure_function_type, backend.backend, x, u, distance_bins;
        kwargs...)
    return nothing
end

function _dispatch_execution_backend!(
    backend::CB.AbstractGPUBackend, shape::AbstractFieldShape, sums, counts, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    return gpu_calculate_structure_function_batch!(sums, counts, structure_function_type,
        backend.backend, x, u, distance_bins; kwargs...)
end

_dispatch_execution_backend!(::CB.AbstractAutoBackend, shape::AbstractFieldShape, sums, counts,
                             structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins;
                             kwargs...) =
    _dispatch_execution_backend!(resolve_auto_backend(), shape, sums, counts, structure_function_type, x, u,
                                 distance_bins; kwargs...)

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
    if has_auxiliary_axes(shape)
        return gpu_calculate_structure_function_2d_batch!(sums_2d, counts_2d, structure_function_type,
            backend.backend, x, u, distance_bins, value_bins; kwargs...)
    end
    return gpu_calculate_structure_function_2d!(sums_2d, counts_2d, structure_function_type,
        backend.backend, x, u, distance_bins, value_bins; kwargs...)
end

_dispatch_execution_backend!(::CB.AbstractAutoBackend, shape::AbstractFieldShape, sums_2d, counts_2d,
                             structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins,
                             value_bins::AbstractVector; kwargs...) =
    _dispatch_execution_backend!(resolve_auto_backend(), shape, sums_2d, counts_2d, structure_function_type, x, u,
                                 distance_bins, value_bins; kwargs...)

function _dispatch_execution_backend!(
    backend::CB.AbstractDistributedBackend, shape::AbstractFieldShape, sums, counts, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins; kwargs...
)
    return _dispatch_execution_backend!(backend, sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
end

function _dispatch_execution_backend!(
    backend::CB.AbstractDistributedBackend, shape::AbstractFieldShape, sums_2d, counts_2d, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector; kwargs...
)
    return _dispatch_execution_backend!(backend, sums_2d, counts_2d, structure_function_type, x, u, distance_bins, value_bins; kwargs...)
end

# --- Non-Mutating Dispatch Layers ---
# 1D (distance_bins only) and 2D (distance_bins + value_bins) are distinguished by ARITY here:
# 1D methods take `(backend, shape, sf, x, u, distance_bins, CT)` and return a raw
# `StructureFunctionSumsAndCounts`; 2D methods take `value_bins::AbstractVector` before `CT` and return a
# raw `StructureFunction2DSumsAndCounts`. The public boundary applies `_finalize` to pick the
# representation.

# 1D
function _dispatch_execution_backend(
    ::CB.AbstractSerialBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type{CT}; kwargs...
) where {CT}
    return serial_calculate_structure_function(structure_function_type, x, u, distance_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractSerialBackend, shape::PointField{D}, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type{CT};
    kwargs...
) where {D, CT}
    return _serial_calculate_structure_function_point(
        structure_function_type, x, u, distance_bins, Val(D), CT; kwargs...,
    )
end

function _dispatch_execution_backend(
    ::CB.AbstractThreadedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type{CT}; kwargs...
) where {CT}
    return threaded_calculate_structure_function(structure_function_type, x, u, distance_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractDistributedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type{CT}; kwargs...
) where {CT}
    return _dispatch_execution_backend(backend, structure_function_type, x, u, distance_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractGPUBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type{CT}; kwargs...
) where {CT}
    if has_auxiliary_axes(shape)
        return gpu_calculate_structure_function_batch(structure_function_type, backend.backend, x, u, distance_bins, CT; kwargs...)
    end
    return gpu_calculate_structure_function(structure_function_type, backend.backend, x, u, distance_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractAutoBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type{CT}; kwargs...
) where {CT}
    return _dispatch_execution_backend(resolve_auto_backend(), shape, structure_function_type, x, u, distance_bins, CT;
                                       kwargs...)
end

# 2D (joint distance×value)
function _dispatch_execution_backend(
    ::CB.AbstractSerialBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector, ::Type{CT}; kwargs...
) where {CT}
    return serial_calculate_structure_function(structure_function_type, x, u, distance_bins, value_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractThreadedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector, ::Type{CT}; kwargs...
) where {CT}
    return threaded_calculate_structure_function(structure_function_type, x, u, distance_bins, value_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractDistributedBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector, ::Type{CT}; kwargs...
) where {CT}
    return _dispatch_execution_backend(backend, structure_function_type, x, u, distance_bins, value_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    backend::CB.AbstractGPUBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector, ::Type{CT}; kwargs...
) where {CT}
    if has_auxiliary_axes(shape)
        return gpu_calculate_structure_function_2d_batch(structure_function_type, backend.backend, x, u, distance_bins, value_bins, CT; kwargs...)
    end
    return gpu_calculate_structure_function_2d(structure_function_type, backend.backend, x, u, distance_bins, value_bins, CT; kwargs...)
end

function _dispatch_execution_backend(
    ::CB.AbstractAutoBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, value_bins::AbstractVector, ::Type{CT}; kwargs...
) where {CT}
    return _dispatch_execution_backend(resolve_auto_backend(), shape, structure_function_type, x, u, distance_bins,
                                       value_bins, CT; kwargs...)
end
