# Tensor structure-function calculations.

"""
    calculate_structure_function_tensor(order, x, u, distance_bins[, CT][, OT]; backend, distance_metric, weights)

The rank-`order` increment moment tensor of a point list, `sums` of shape
`(D, …, D, n_bins, auxiliary...)`. `CT` is the count element type (default `$(DEFAULT_COUNT_TYPE)`) and `OT`
the result representation, the averaged `StructureFunctionTensor` by default or the raw
`StructureFunctionTensorSumsAndCounts`.
"""
function calculate_structure_function_tensor(
    order::Val{P},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    ::Type{CT},
    ::Type{OTT};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
) where {P, FT1, FT2, CT <: Real, OTT <: SFO.AbstractStructureFunction}
    shape = _validate_array_shape(x, u, distance_metric)
    OT = promote_type(float(FT1), float(FT2))
    w = _pair_weights(weights, size(u, 2), OT)
    _assert_count_type(CT, size(u, 2), w)
    D = spatial_dimension(shape)
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
    sums = zeros(OT, ntuple(_ -> D, P)..., n_bins, auxiliary_dims...)
    counts = zeros(CT, n_bins, auxiliary_dims...)
    _dispatch_tensor!(backend, shape, sums, counts, order, x, u, distance_bins; distance_metric, weights = w)
    raw = SFO.StructureFunctionTensorSumsAndCounts(order, distance_bins, sums, counts)
    return _finalize(raw, OTT)
end

calculate_structure_function_tensor(order::Val, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector;
                                    kwargs...) =
    calculate_structure_function_tensor(order, x, u, distance_bins, DEFAULT_COUNT_TYPE, SFO.StructureFunctionTensor;
                                        kwargs...)
calculate_structure_function_tensor(order::Val, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                    ::Type{CT}; kwargs...) where {CT <: Real} =
    calculate_structure_function_tensor(order, x, u, distance_bins, CT, SFO.StructureFunctionTensor; kwargs...)
calculate_structure_function_tensor(order::Val, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                    ::Type{OTT}; kwargs...) where {OTT <: SFO.AbstractStructureFunction} =
    calculate_structure_function_tensor(order, x, u, distance_bins, DEFAULT_COUNT_TYPE, OTT; kwargs...)

"""
    calculate_structure_function_tensor(order, x, u, distance_bins, axis_bins[, CT][, OT]; second_axis, backend, distance_metric, weights)

The increment moment tensor joint in separation and the angle `second_axis` reads from each pair's
direction, `sums` of shape `(D, …, D, n_bins, n_axis)`: the tensor resolved by direction. One field
(no auxiliary axes), a flat metric, the CPU backends. `OT` is `StructureFunctionTensor2DSumsAndCounts`.
"""
function calculate_structure_function_tensor(
    order::Val{P},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    axis_bins::AbstractVector,
    ::Type{CT},
    ::Type{OTT};
    second_axis::SeparationAngleAxis,
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
) where {P, FT1, FT2, CT <: Real, OTT <: SFO.AbstractStructureFunction}
    shape = _validate_array_shape(x, u, distance_metric)
    OT = promote_type(float(FT1), float(FT2))
    w = _pair_weights(weights, size(u, 2), OT)
    _assert_count_type(CT, size(u, 2), w)
    axis = _tensor_axis(axis_bins, second_axis)
    D = spatial_dimension(shape)
    sums = zeros(OT, ntuple(_ -> D, P)..., n_histogram_bins(distance_bins), axis[2])
    counts = zeros(CT, n_histogram_bins(distance_bins), axis[2])
    _tensor_shape_check(order, shape, sums, counts, u, distance_bins, axis)
    _dispatch_tensor!(backend, shape, sums, counts, order, x, u, distance_bins; distance_metric, weights = w, axis)
    raw = SFO.StructureFunctionTensor2DSumsAndCounts(order, distance_bins, axis_bins, sums, counts)
    return _finalize(raw, OTT)
end

calculate_structure_function_tensor(order::Val, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                    axis_bins::AbstractVector; kwargs...) =
    calculate_structure_function_tensor(order, x, u, distance_bins, axis_bins, DEFAULT_COUNT_TYPE,
                                        SFO.StructureFunctionTensor2DSumsAndCounts; kwargs...)
calculate_structure_function_tensor(order::Val, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                    axis_bins::AbstractVector, ::Type{CT}; kwargs...) where {CT <: Real} =
    calculate_structure_function_tensor(order, x, u, distance_bins, axis_bins, CT,
                                        SFO.StructureFunctionTensor2DSumsAndCounts; kwargs...)
calculate_structure_function_tensor(order::Val, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector,
                                    axis_bins::AbstractVector,
                                    ::Type{OTT}; kwargs...) where {OTT <: SFO.AbstractStructureFunction} =
    calculate_structure_function_tensor(order, x, u, distance_bins, axis_bins, DEFAULT_COUNT_TYPE, OTT; kwargs...)

"""The joint tensor's second axis: its digitize plan, its bin count and the angle source."""
@inline function _tensor_axis(axis_bins, second_axis::SeparationAngleAxis)
    axis_edges = digitize_plan(axis_bins)
    return (axis_edges, n_histogram_bins(axis_edges), second_axis)
end

"""
    calculate_structure_function_tensor!(sums, counts, order, x, u, distance_bins[, axis_bins]; second_axis, backend, distance_metric)

Accumulate the rank-`order` increment moment tensor of a point list into `sums`
`(D, …, D, n_bins[, n_axis], auxiliary...)` and `counts` `(n_bins[, n_axis], auxiliary...)` on
`backend`, the in-place form of [`calculate_structure_function_tensor`](@ref). With `axis_bins` the
tensor is binned jointly in separation and in the angle `second_axis` reads from each pair.
"""
function calculate_structure_function_tensor!(
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val{P},
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
) where {P}
    shape = _validate_array_shape(x, u, distance_metric)
    _tensor_shape_check(order, shape, sums, counts, u, distance_bins, nothing)
    w = _pair_weights(weights, size(u, 2), eltype(sums))
    _assert_counts_can_accumulate(counts, size(u, 2), w)
    return _dispatch_tensor!(
        backend, shape, sums, counts, order, x, u, distance_bins;
        distance_metric = distance_metric, weights = w,
    )
end

function calculate_structure_function_tensor!(
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val{P},
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    axis_bins::AbstractVector;
    second_axis::SeparationAngleAxis,
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    weights = nothing,
) where {P}
    shape = _validate_array_shape(x, u, distance_metric)
    axis = _tensor_axis(axis_bins, second_axis)
    _tensor_shape_check(order, shape, sums, counts, u, distance_bins, axis)
    w = _pair_weights(weights, size(u, 2), eltype(sums))
    _assert_counts_can_accumulate(counts, size(u, 2), w)
    return _dispatch_tensor!(
        backend, shape, sums, counts, order, x, u, distance_bins;
        distance_metric = distance_metric, weights = w, axis = axis,
    )
end

function _dispatch_tensor!(
    ::CB.AbstractSerialBackend,
    shape::AbstractFieldShape,
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...,
)
    return serial_calculate_structure_function_tensor!(
        sums, counts, order, shape, x, u, distance_bins; kwargs...
    )
end

function _dispatch_tensor!(
    ::CB.AbstractAutoBackend,
    shape::AbstractFieldShape,
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...,
)
    return _dispatch_tensor!(
        _auto_local_backend(), shape, sums, counts, order, x, u, distance_bins; kwargs...
    )
end

function _dispatch_tensor!(
    ::CB.AbstractThreadedBackend,
    shape::AbstractFieldShape,
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...,
)
    return threaded_calculate_structure_function_tensor!(
        sums, counts, order, shape, x, u, distance_bins; kwargs...
    )
end

function _dispatch_tensor!(
    ::CB.AbstractDistributedBackend,
    shape::AbstractFieldShape,
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...,
)
    return distributed_calculate_structure_function_tensor!(
        sums, counts, order, shape, x, u, distance_bins; kwargs...
    )
end

function _dispatch_tensor!(
    backend::CB.AbstractGPUBackend,
    shape::AbstractFieldShape,
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...,
)
    return gpu_calculate_structure_function_tensor!(
        backend, sums, counts, order, shape, x, u, distance_bins; kwargs...
    )
end

"""
    threaded_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, bins; kwargs...)

Accumulate a tensor structure function across threads. Supplied by the OhMyThreads extension.
"""
function threaded_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, bins;
                                                       kwargs...)
    throw(ArgumentError(
        "the threaded tensor backend needs OhMyThreads: run `using OhMyThreads`.",
    ))
end

"""
    distributed_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, bins; kwargs...)

Accumulate a tensor structure function across worker processes. Supplied by the Distributed
extension.
"""
function distributed_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, bins;
                                                          kwargs...)
    throw(ArgumentError(
        "the distributed tensor backend needs Distributed: run `using Distributed` and add workers.",
    ))
end

"""
    mpi_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, bins; kwargs...)

Accumulate a tensor structure function across MPI ranks. Supplied by the MPI extension.
"""
function mpi_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, bins;
                                                  kwargs...)
    throw(ArgumentError(
        "the MPI tensor backend needs MPI: run `using MPI` under `mpiexec`.",
    ))
end

function _dispatch_tensor!(b::CB.AbstractMPIBackend, shape::AbstractFieldShape, sums::AbstractArray,
                           counts::AbstractArray, order::Val, x::AbstractArray, u::AbstractArray,
                           distance_bins::AbstractVector; kwargs...)
    return mpi_calculate_structure_function_tensor!(sums, counts, order, shape, x, u,
                                                    distance_bins; backend = b, kwargs...)
end

"""
    gpu_calculate_structure_function_tensor!(backend, sums, counts, order, shape, x, u, bins; kwargs...)

Accumulate a tensor structure function on a device. Supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_function_tensor!(backend, sums, counts, order, shape, x, u, bins;
                                                  kwargs...)
    throw(ArgumentError(
        "the GPU tensor backend needs KernelAbstractions: run `using KernelAbstractions` and a " *
        "device backend such as CUDA.",
    ))
end

"""
    _tensor_shape_check(order, shape, sums, counts, u, distance_bins, axis)

Validate the accumulator shapes of a tensor sweep, at the public boundary.
"""
function _tensor_shape_check(order::Val{P}, shape::AbstractFieldShape{D}, sums, counts, u,
                             distance_bins, axis) where {P, D}
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
    if axis === nothing
        expected_sums = (ntuple(_ -> D, P)..., n_bins, auxiliary_dims...)
        expected_counts = (n_bins, auxiliary_dims...)
    else
        isempty(auxiliary_dims) || throw(ArgumentError(
            "the joint tensor over the separation angle takes one field; sweep the auxiliary slices one by one",
        ))
        expected_sums = (ntuple(_ -> D, P)..., n_bins, axis[2])
        expected_counts = (n_bins, axis[2])
    end
    size(sums) == expected_sums ||
        throw(DimensionMismatch("sums must have shape $expected_sums; got $(size(sums))"))
    size(counts) == expected_counts ||
        throw(DimensionMismatch("counts must have shape $expected_counts; got $(size(counts))"))
    return nothing
end

"""
    _tensor_setup(order, shape, x, u, distance_bins, distance_metric[, axis]) -> NamedTuple

Everything a tensor sweep needs before its first pair: the geometry, the widened coordinates and
field, the bin edges, and the flattened accumulator shapes. With `axis = (axis_edges, n_axis,
second_axis)` the sweep is joint in separation and angle, over one field on a flat metric.

Shared by every backend so the preparation happens **once**, above any task or worker loop.
"""
function _tensor_setup(
    order::Val{P}, shape::AbstractFieldShape{D}, sums, counts, x, u, distance_bins, distance_metric,
    axis = nothing, weights = NoWeights(),
) where {P, D}
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
    dist_be = digitize_plan(distance_bins)
    N = size(u, 2)
    B = isempty(auxiliary_dims) ? 1 : prod(auxiliary_dims)
    fixed_x = ndims(x) == 2

    geom = SFH.pair_geometry_for(distance_metric, Val(D))
    axis === nothing || geom isa SFH.FlatGeometry || _require_directional(FrameTransport())
    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    vW = SFH.coordinate_width(geom)
    vF = SFH.field_width(geom)
    W = _val_int(vW)
    F = _val_int(vF)

    return (; n_bins, auxiliary_dims, dist_be, N, B, fixed_x, geom, xk, uk, vW, vF, W, F,
            D = D, P = P, axis, weights)
end

"""Flattened views of the accumulators, so the kernel indexes one auxiliary axis."""
@inline _tensor_flat(sums, counts, s) = s.axis === nothing ?
    (reshape(sums, ntuple(_ -> s.D, s.P)..., s.n_bins, s.B), reshape(counts, s.n_bins, s.B)) :
    (sums, counts)

"""
    serial_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, distance_bins; distance_metric, axis)

The serial pair loop behind [`calculate_structure_function_tensor!`](@ref): every pair's increment
in its pair frame, its `order`-fold outer product added to the bin of the pair's separation (and of
its angle when `axis` names one).
"""
function serial_calculate_structure_function_tensor!(
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val{P},
    shape::AbstractFieldShape{D},
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    axis = nothing,
    weights = NoWeights(),
) where {P, D}
    s = _tensor_setup(order, shape, sums, counts, x, u, distance_bins, distance_metric, axis, weights)
    sums_flat, counts_flat = _tensor_flat(sums, counts, s)
    _tensor_pairs!(sums_flat, counts_flat, order, s, 1:s.N)
    return sums, counts
end

"""
    _tensor_pairs!(sums_flat, counts_flat, order, s, outer)

Accumulate every pair `(i, j > i)` for `i` in `outer`.

The outer list is a parameter so that one kernel serves every backend: serial passes the whole
range, threaded and distributed pass a chunk of it, and the partial results add because a histogram
is order-independent. With a second axis in `s` the pair's angle picks the second bin.
"""
function _tensor_pairs!(sums_flat, counts_flat, order::Val{P}, s, outer) where {P}
    s.axis === nothing || return _tensor_pairs_joint!(sums_flat, counts_flat, order, s, outer)
    # `s` carries the widths and the staging choice as values. They become type parameters here so
    # the `SVector`s below are statically sized and the branch is resolved at compile time; read
    # off `s` they make every pair's coordinate load a heap allocation.
    return _tensor_pairs_inner!(sums_flat, counts_flat, order, s, outer, s.vW, s.vF,
                                Val(s.fixed_x))
end

function _tensor_pairs_inner!(sums_flat, counts_flat, order::Val{P}, s, outer,
                              ::Val{W}, ::Val{F}, ::Val{FX}) where {P, W, F, FX}
    vW, vF = Val(W), Val(F)
    geom, dist_be, n_bins, N, B = s.geom, s.dist_be, s.n_bins, s.N, s.B
    x_fixed = FX ? reshape(s.xk, W, N) : nothing
    x_flat = FX ? nothing : reshape(s.xk, W, N, B)
    u_flat = reshape(s.uk, F, N, B)
    XT, UT = eltype(s.xk), eltype(s.uk)

    # The increment comes from `pair_delta`, so on a curved manifold the tensor components are in
    # the pair's own transported frame. An odd rank takes the canonical pair reading, as an odd
    # scalar increment does.
    wts = s.weights
    @inbounds for i in outer
        wi = _point_weight(wts, i)
        for j in (i + 1):N
            w = wi * _point_weight(wts, j)
            if FX
                X1 = SA.SVector{W, XT}(ntuple(d -> x_fixed[d, i], vW))
                X2 = SA.SVector{W, XT}(ntuple(d -> x_fixed[d, j], vW))
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = SFH.digitize(dist, dist_be)
                if ok && 1 <= bin <= n_bins
                    sgn = _tensor_reading(order, geom, frame)
                    for b in 1:B
                        U1 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, i, b], vF))
                        U2 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, j, b], vF))
                        du = sgn * SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                        _accumulate_tensor_pair!(sums_flat, counts_flat, du, bin, b, order, w)
                    end
                end
            else
                for b in 1:B
                    X1 = SA.SVector{W, XT}(ntuple(d -> x_flat[d, i, b], vW))
                    X2 = SA.SVector{W, XT}(ntuple(d -> x_flat[d, j, b], vW))
                    ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                    bin = SFH.digitize(dist, dist_be)
                    if ok && 1 <= bin <= n_bins
                        U1 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, i, b], vF))
                        U2 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, j, b], vF))
                        du = _tensor_reading(order, geom, frame) * SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                        _accumulate_tensor_pair!(sums_flat, counts_flat, du, bin, b, order, w)
                    end
                end
            end
        end
    end
    return sums_flat, counts_flat
end

"""
The sign a pair's increment carries into an odd-rank tensor. In a fixed Cartesian frame the increment
flips when the pair is read from its other end, so an odd rank takes the canonical reading; in a
pair's own geodesic frame the axes flip with it and the components are read the same from either
end, so no sign is taken. An even rank takes none.
"""
@inline _tensor_reading(::Val{P}, geom::SFH.FlatGeometry, frame) where {P} =
    isodd(P) ? SFH.pair_orientation(geom, frame) : 1
@inline _tensor_reading(::Val{P}, ::SFH.SphericalGeometry, frame) where {P} = 1

"""The same rule for a lag: the lag's reading factor under identity transport, none under frame transport."""
@inline _tensor_factor(::IdentityTransport, factor) = factor
@inline _tensor_factor(::FrameTransport, factor) = one(factor)

# One field on a flat metric: the pair's displacement gives its angle to the reference axis, which
# picks the second bin in place of the auxiliary slot.
_tensor_pairs_joint!(sums, counts, order::Val, s, outer) =
    _tensor_pairs_joint_inner!(sums, counts, order, s, outer, s.vW, s.vF)

function _tensor_pairs_joint_inner!(sums, counts, order::Val{P}, s, outer,
                                    ::Val{W}, ::Val{F}) where {P, W, F}
    vW, vF = Val(W), Val(F)
    geom, dist_be, n_bins, N = s.geom, s.dist_be, s.n_bins, s.N
    axis_edges, na, second_axis = s.axis
    x_fixed = reshape(s.xk, W, N)
    u_flat = reshape(s.uk, F, N)
    XT, UT = eltype(s.xk), eltype(s.uk)
    wts = s.weights
    @inbounds for i in outer
        wi = _point_weight(wts, i)
        X1 = SA.SVector{W, XT}(ntuple(d -> x_fixed[d, i], vW))
        U1 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, i], vF))
        for j in (i + 1):N
            w = wi * _point_weight(wts, j)
            X2 = SA.SVector{W, XT}(ntuple(d -> x_fixed[d, j], vW))
            ok, dist, frame = SFH.pair_frame(geom, X1, X2)
            bin = SFH.digitize(dist, dist_be)
            (ok && 1 <= bin <= n_bins) || continue
            dx = X2 - X1
            bθ = SFH.digitize(axis_quantity(second_axis, dx, dist * dist), axis_edges)
            1 <= bθ <= na || continue
            U2 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, j], vF))
            du = _tensor_reading(order, geom, frame) * SFH.pair_delta(geom, frame, X1, X2, U1, U2)
            _accumulate_tensor_pair!(sums, counts, du, bin, bθ, order, w)
        end
    end
    return sums, counts
end

"""
    tensor_partial(order, shape, x, u, distance_bins, outer, CT; distance_metric, axis, weights) -> (sums, counts)

A worker's share of a tensor sweep: the pairs whose lower index is in `outer`, in freshly allocated
accumulators.

The outer lists partition `1:N`, so the partials add to the whole sweep exactly.
"""
function tensor_partial(
    order::Val{P}, shape::AbstractFieldShape{D}, x, u, distance_bins, outer, ::Type{CT};
    distance_metric::DI.PreMetric = DI.Euclidean(), axis = nothing, weights = NoWeights(),
) where {P, D, CT}
    n_bins = n_histogram_bins(distance_bins)
    OT = promote_type(float(eltype(x)), float(eltype(u)))
    if axis === nothing
        auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
        sums = zeros(OT, ntuple(_ -> D, P)..., n_bins, auxiliary_dims...)
        counts = zeros(CT, n_bins, auxiliary_dims...)
    else
        sums = zeros(OT, ntuple(_ -> D, P)..., n_bins, axis[2])
        counts = zeros(CT, n_bins, axis[2])
    end
    s = _tensor_setup(order, shape, sums, counts, x, u, distance_bins, distance_metric, axis, weights)
    sf, cf = _tensor_flat(sums, counts, s)
    _tensor_pairs!(sf, cf, order, s, outer)
    return sums, counts
end

# One pair's `δu^{⊗P}` added at `[…, bin, b]`, the loops over the P component indices unrolled to
# the order at compile time.
@generated function _accumulate_tensor_pair!(sums, counts, du, bin::Int, b::Int, ::Val{P}, w) where {P}
    idx = [Symbol(:i, k) for k in 1:P]
    prod_ex = Expr(:call, :*, [:(du[$(idx[k])]) for k in 1:P]...)
    body = :(sums[$(idx...), bin, b] += w * $prod_ex)
    for k in P:-1:1
        body = :(for $(idx[k]) in 1:D
            $body
        end)
    end
    return quote
        D = length(du)
        @inbounds $body
        @inbounds counts[bin, b] += convert(eltype(counts), w)
        return nothing
    end
end

# ---------------------------------------------------------------------------------------------------
# Tensors on grids: the transform engine's per-lag symmetric moment store, binned without contraction.
# ---------------------------------------------------------------------------------------------------

"""
    gridded_tensor_sweep!(sums, counts, order, data, schedule, distance_bins[, axis_bins], ::Val{D}, spectral_backend; valid, weights, backend[, second_axis])

Accumulate the rank-`P` increment moment tensor of a field's vector field over every lag
`schedule` names, by the transform `spectral_backend` names, into `sums` `(D, …, D, n_bins[, n_axis])`
and `counts` `(n_bins[, n_axis])`. Supplied by the AbstractFFTs extension for every separable
schedule, the non-uniform FFT provider included.
"""
gridded_tensor_sweep!(sums, counts, order::Val, data::AbstractMatrix, schedule::AbstractSeparableSchedule,
                      distance_bins, ::Val{D}, ::SB.AbstractDirectSumSpectralBackend; kwargs...) where {D} =
    _no_tensor_sum()

gridded_tensor_sweep!(sums, counts, order::Val, data::AbstractMatrix, schedule::AbstractSeparableSchedule,
                      distance_bins, axis_bins, ::Val{D}, ::SB.AbstractDirectSumSpectralBackend;
                      kwargs...) where {D} = _no_tensor_sum()

gridded_tensor_sweep!(sums, counts, order::Val, data::AbstractMatrix, schedule::AbstractSeparableSchedule,
                      distance_bins, ::Val{D}, tag::SB.AbstractSpectralBackend; kwargs...) where {D} =
    _no_transform_loaded(tag, schedule)

gridded_tensor_sweep!(sums, counts, order::Val, data::AbstractMatrix, schedule::AbstractSeparableSchedule,
                      distance_bins, axis_bins, ::Val{D}, tag::SB.AbstractSpectralBackend; kwargs...) where {D} =
    _no_transform_loaded(tag, schedule)

gridded_tensor_sweep!(sums, counts, order::Val, data::AbstractMatrix, schedule, distance_bins, ::Val{D},
                      spectral_backend; kwargs...) where {D} = _not_a_spectral_tag(spectral_backend)

gridded_tensor_sweep!(sums, counts, order::Val, data::AbstractMatrix, schedule, distance_bins, axis_bins, ::Val{D},
                      spectral_backend; kwargs...) where {D} = _not_a_spectral_tag(spectral_backend)

# A gridded tensor has one algorithm, the transform; the pair loop over the grid's points is the point entry.
_no_tensor_sum() = throw(ArgumentError(
    "a gridded tensor structure function is computed by transform: pass FastFourierTransformSpectralBackend() or " *
    "AutoSpectralBackend() on a grid, or a non-uniform FFT tag on a ScatteredModesSchedule. The pair loop is the " *
    "point entry over the grid's points.",
))

"""Add the symmetric store `sym` `(n_entries, bins…)` into the dense tensor `dense` `(D, …, D, bins…)`."""
function _expand_symmetric!(dense::AbstractArray, sym::AbstractArray, ::Val{D}, ::Val{P}) where {D, P}
    comps = CartesianIndices(ntuple(_ -> D, Val(P)))
    rest = CartesianIndices(size(sym)[2:end])
    @inbounds for r in rest, I in comps
        dense[I, r] += sym[SFT.symmetric_rank(Val(D), Val(P), Tuple(I)), r]
    end
    return dense
end
