# Dispatch Entry Points and Backend Routing

# Public entry points

function _derived_structure_function_error(structure_function_type)
    throw(ArgumentError(
        "$(typeof(structure_function_type)) is a derived structure-function quantity, " *
        "not a pairwise operator. Use helmholtz_decompose_2d for 2D Helmholtz " *
        "rotational/divergent quantities.",
    ))
end

# Result finalization: maps a backend's raw accumulator to the requested result type `OT`; unsupported pairs raise.
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
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), promote_type(float(eltype(x)), float(eltype(u))))
    _assert_count_type(CT, size(x, 2), w)
    raw = _shaped(_shape_kind(x, u), size(u, 1), distance_metric) do shape, geometry
        _dispatch_execution_backend(backend, shape, structure_function_type, x, u, distance_bins, CT; geometry,
                                    weights = w, kwargs...)
    end
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
    second_axis::AbstractSecondAxisSource = InvariantValueAxis(),
    kwargs...,
) where {CT <: Real, OT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), promote_type(float(eltype(x)), float(eltype(u))))
    _assert_count_type(CT, size(x, 2), w)
    raw = _shaped(_shape_kind(x, u), size(u, 1), distance_metric) do shape, geometry
        _dispatch_execution_backend(backend, shape, structure_function_type, x, u, distance_bins, value_bins, CT;
                                    geometry, weights = w, second_axis, kwargs...)
    end
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
    _validate_array_shape(x, u, distance_metric)
    min_distance, max_distance = _shaped(_shape_kind(x, u), size(u, 1), distance_metric) do _, geometry
        _minmax_for_autobins(x, geometry)
    end
    actual_bins = _auto_distance_bins(min_distance, max_distance, distance_bins, bin_spacing)

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

# Bins are `(edges[k], edges[k+1]]`, so the first edge sits just below the least separation and the last on the greatest.
_auto_distance_bins(min_distance, max_distance, distance_bins::Int, ::Type{LinearBinEdges}) =
    LinearBinEdges(prevfloat(min_distance), max_distance, distance_bins + 1)

_auto_distance_bins(min_distance, max_distance, distance_bins::Int, ::Type{LogBinEdges}) =
    LogBinEdges(prevfloat(min_distance), max_distance, distance_bins + 1)

function _auto_distance_bins(min_distance, max_distance, distance_bins::Int, bin_spacing)
    throw(ArgumentError("bin_spacing must be LinearBinEdges or LogBinEdges; got $bin_spacing"))
end

"""Least and greatest separation of two points of `x` that `geometry`'s pair frame gives, over every slice of a
varying-position array: the separations the kernels bin."""
function _minmax_for_autobins(x::AbstractArray, geometry)
    xk = SFH.prepare_coordinates(geometry, x)
    xf = reshape(xk, _val_int(SFH.coordinate_width(geometry)), size(xk, 2), :)
    FT = float(eltype(xk))
    min_distance, max_distance = FT(Inf), FT(0)
    for b in axes(xf, 3)
        for i in axes(xf, 2)
            lo, hi = minmax_i(i, view(xf, :, :, b), geometry)
            min_distance = min(min_distance, lo)
            max_distance = max(max_distance, hi)
        end
    end
    return min_distance, max_distance
end

"""
    minmax_i(i, x, geometry)

The least and greatest separation `geometry`'s pair frame gives from point `i` of the kernel coordinates `x` to every
later point; pairs the frame refuses are skipped, as the kernels skip them.
"""
function minmax_i(i::Int, x::AbstractMatrix{FT}, geometry) where {FT <: Number}
    vW = SFH.coordinate_width(geometry)
    X1 = SA.SVector(ntuple(k -> x[k, i], vW))
    T = Base.promote_op((a, b) -> SFH.pair_frame(geometry, a, b)[2], typeof(X1), typeof(X1))
    min_distance, max_distance = T(Inf), zero(T)
    for j in (i + 1):size(x, 2)
        ok, r, _ = SFH.pair_frame(geometry, X1, SA.SVector(ntuple(k -> x[k, j], vW)))
        ok || continue
        min_distance = min(min_distance, r)
        max_distance = max(max_distance, r)
    end
    return min_distance, max_distance
end

# StructureFunction factory constructor
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

# Backend dispatch for the mutating API

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
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), eltype(sums))
    _assert_counts_can_accumulate(counts, size(x, 2), w)
    _shaped(_shape_kind(x, u), size(u, 1), distance_metric) do shape, geometry
        _dispatch_execution_backend!(backend, shape, sums, counts, sf_type, x, u, distance_bins; geometry,
                                     weights = w, kwargs...)
    end
    return nothing
end

function calculate_structure_function!(
    sums_2d, counts_2d, sf_type::SFT.AbstractPairwiseStructureFunctionType, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector, value_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(), distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing, kwargs...,
)
    _require_backend(backend)
    _validate_array_shape(x, u, distance_metric)
    w = _pair_weights(weights, size(x, 2), eltype(sums_2d))
    _assert_counts_can_accumulate(counts_2d, size(x, 2), w)
    _shaped(_shape_kind(x, u), size(u, 1), distance_metric) do shape, geometry
        _dispatch_execution_backend!(backend, shape, sums_2d, counts_2d, sf_type, x, u, distance_bins, value_bins;
                                     geometry, weights = w, kwargs...)
    end
    return nothing
end

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

# Mutating 2D backend dispatch

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

# Non-mutating dispatch: 1D methods take `(backend, shape, sf, x, u, distance_bins, CT)` and return a raw
# `StructureFunctionSumsAndCounts`; 2D methods take `value_bins` before `CT` and return a raw
# `StructureFunction2DSumsAndCounts`.

# 1D
function _dispatch_execution_backend(
    ::CB.AbstractSerialBackend, shape::AbstractFieldShape, structure_function_type::SFT.AbstractPairwiseStructureFunctionType, x, u, distance_bins, ::Type{CT}; kwargs...
) where {CT}
    return serial_calculate_structure_function(structure_function_type, x, u, distance_bins, CT; kwargs...)
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
