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
    culling::CullingPolicy = AutoCulling(),
    weights = nothing,
    workspace = nothing,
) where {P, FT1, FT2, CT <: Real, OTT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    OT = promote_type(float(FT1), float(FT2))
    w = _pair_weights(weights, size(u, 2), OT)
    _assert_count_type(CT, size(u, 2), w)
    D = spatial_dimension(shape)
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
    sums = _result_zeros(backend, OT, ntuple(_ -> D, P)..., n_bins, auxiliary_dims...)
    counts = _result_zeros(backend, CT, n_bins, auxiliary_dims...)
    _dispatch_tensor!(backend, shape, sums, counts, order, x, u, distance_bins; distance_metric, culling, weights = w,
                      _workspace_kw(workspace)...)
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
    culling::CullingPolicy = AutoCulling(),
    weights = nothing,
    workspace = nothing,
) where {P, FT1, FT2, CT <: Real, OTT <: SFO.AbstractStructureFunction}
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    OT = promote_type(float(FT1), float(FT2))
    w = _pair_weights(weights, size(u, 2), OT)
    _assert_count_type(CT, size(u, 2), w)
    axis = _tensor_axis(axis_bins, second_axis)
    D = spatial_dimension(shape)
    sums = _result_zeros(backend, OT, ntuple(_ -> D, P)..., n_histogram_bins(distance_bins), axis[2])
    counts = _result_zeros(backend, CT, n_histogram_bins(distance_bins), axis[2])
    _tensor_shape_check(order, shape, sums, counts, u, distance_bins, axis)
    _dispatch_tensor!(backend, shape, sums, counts, order, x, u, distance_bins; distance_metric, culling, weights = w,
                      axis, _workspace_kw(workspace)...)
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

"""The joint tensor's second axis: its bins, its bin count and the angle source."""
@inline _tensor_axis(axis_bins, second_axis::SeparationAngleAxis) =
    (axis_bins, n_histogram_bins(axis_bins), second_axis)

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
    culling::CullingPolicy = AutoCulling(),
    weights = nothing,
    workspace = nothing,
) where {P}
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    _tensor_shape_check(order, shape, sums, counts, u, distance_bins, nothing)
    w = _pair_weights(weights, size(u, 2), eltype(sums))
    _assert_counts_can_accumulate(counts, size(u, 2), w)
    return _dispatch_tensor!(
        backend, shape, sums, counts, order, x, u, distance_bins;
        distance_metric = distance_metric, culling, weights = w, _workspace_kw(workspace)...,
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
    culling::CullingPolicy = AutoCulling(),
    weights = nothing,
    workspace = nothing,
) where {P}
    _require_backend(backend)
    shape = _validate_array_shape(x, u, distance_metric)
    axis = _tensor_axis(axis_bins, second_axis)
    _tensor_shape_check(order, shape, sums, counts, u, distance_bins, axis)
    w = _pair_weights(weights, size(u, 2), eltype(sums))
    _assert_counts_can_accumulate(counts, size(u, 2), w)
    return _dispatch_tensor!(
        backend, shape, sums, counts, order, x, u, distance_bins;
        distance_metric = distance_metric, culling, weights = w, axis = axis, _workspace_kw(workspace)...,
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
        resolve_auto_backend(), shape, sums, counts, order, x, u, distance_bins; kwargs...
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
    db::CB.AbstractDistributedBackend,
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
        CB.local_backend(db), sums, counts, order, shape, x, u, distance_bins; kwargs...
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
function threaded_calculate_structure_function_tensor! end

"""
    distributed_calculate_structure_function_tensor!(inner, sums, counts, order, shape, x, u, bins; kwargs...)

Accumulate a tensor structure function across worker processes, each on the local backend `inner`.
Supplied by the Distributed extension.
"""
function distributed_calculate_structure_function_tensor! end

"""
    mpi_calculate_structure_function_tensor!(sums, counts, order, shape, x, u, bins; kwargs...)

Accumulate a tensor structure function across MPI ranks. Supplied by the MPI extension.
"""
function mpi_calculate_structure_function_tensor! end

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
function gpu_calculate_structure_function_tensor! end

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
    _tensor_setup(order, shape, x, u, distance_bins, distance_metric[, axis, weights, culling]) -> NamedTuple

Everything a tensor sweep needs before its first pair: the geometry, the widened coordinates and
field, the bin edges, the flattened accumulator shapes, and the cull grid with the points, the fields
and the weights permuted into it ([`_tensor_cull`](@ref)). With `axis = (axis_bins, n_axis,
second_axis)` the sweep is joint in separation and angle, over one field on a flat metric; the setup's
`axis` holds the digitize plan of `axis_bins` in their place.

Shared by every backend so the preparation happens **once**, above any task or worker loop.
"""
function _tensor_setup(
    order::Val{P}, shape::AbstractFieldShape{D}, sums, counts, x, u, distance_bins, distance_metric,
    axis = nothing, weights = NoWeights(), culling::CullingPolicy = NoCulling(),
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

    grid, xk, uk, weights = _tensor_cull(xk, uk, weights, geom, distance_bins, culling, fixed_x, W, F, N, B)
    axis_plan = axis === nothing ? nothing : (digitize_plan(axis[1]), axis[2], axis[3])
    return (; n_bins, auxiliary_dims, dist_be, N, B, fixed_x, geom, xk, uk, vW, vF, W, F,
            D = D, P = P, axis = axis_plan, weights, grid)
end

"""
    _tensor_cull(xk, uk, weights, geom, distance_bins, culling, fixed_x, W, F, N, B) -> (grid, xk, uk, weights)

The tensor sweep's inputs sorted into cull grids, or unchanged with `grid === nothing` when `culling`
declines. Shared positions are sorted once for every slice; positions varying per slice are sorted per
slice, `grid[b]` being slice `b`'s grid (`nothing` where it declines) and `weights[b]` its weights.
"""
function _tensor_cull(xk, uk, weights, geom, distance_bins, culling::CullingPolicy, fixed_x::Bool, W, F, N, B)
    _cull_enabled(culling) || return nothing, xk, uk, weights
    if fixed_x
        grid = cull_grid_for(ntuple(d -> view(xk, d, :), W), geom, distance_bins, culling)
        grid === nothing && return nothing, xk, uk, weights
        p = grid.perm
        return grid, xk[:, p], reshape(reshape(uk, F, N, B)[:, p, :], size(uk)), _permuted_point_weights(weights, p)
    end
    xs, us = reshape(xk, W, N, B), reshape(uk, F, N, B)
    grids = [cull_grid_for(ntuple(d -> view(xs, d, :, b), W), geom, distance_bins, culling) for b in 1:B]
    all(isnothing, grids) && return nothing, xk, uk, weights
    perms = [g === nothing ? collect(1:N) : g.perm for g in grids]
    xn, un = similar(xs), similar(us)
    for (b, p) in pairs(perms)
        xn[:, :, b] .= view(xs, :, p, b)
        un[:, :, b] .= view(us, :, p, b)
    end
    return grids, reshape(xn, size(xk)), reshape(un, size(uk)), [_permuted_point_weights(weights, p) for p in perms]
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
    culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
) where {P, D}
    s = _tensor_setup(order, shape, sums, counts, x, u, distance_bins, distance_metric, axis, weights, culling)
    sums_flat, counts_flat = _tensor_flat(sums, counts, s)
    _tensor_pairs!(sums_flat, counts_flat, order, s, 1:s.N)
    return sums, counts
end

"""
    _tensor_pairs!(sums_flat, counts_flat, order, s, outer)

Accumulate every pair `(i, j > i)` for `i` in `outer`, in the block pairs of `s`'s cull grid when it
has one.

The outer list is a parameter so that one kernel serves every backend: serial passes the whole
range, threaded and distributed pass a chunk of it, and the partial results add because a histogram
is order-independent. With a second axis in `s` the pair's angle picks the second bin.
"""
function _tensor_pairs!(sums_flat, counts_flat, order::Val{P}, s, outer) where {P}
    if s.grid isa AbstractVector
        for b in 1:s.B
            _tensor_pairs_inner!(sums_flat, counts_flat, order, s, pair_blocks(s.N, outer; grid = s.grid[b]),
                                 s.vW, s.vF, Val(false), b:b, s.weights[b])
        end
        return sums_flat, counts_flat
    end
    blocks = pair_blocks(s.N, outer; grid = s.grid)
    s.axis === nothing || return _tensor_pairs_joint_inner!(sums_flat, counts_flat, order, s, blocks, s.vW, s.vF)
    # `s` carries the widths and the staging choice as values. They become type parameters here so
    # the `SVector`s below are statically sized and the branch is resolved at compile time; read
    # off `s` they make every pair's coordinate load a heap allocation.
    return _tensor_pairs_inner!(sums_flat, counts_flat, order, s, blocks, s.vW, s.vF,
                                Val(s.fixed_x), 1:s.B, s.weights)
end

function _tensor_pairs_inner!(sums_flat, counts_flat, order::Val{P}, s, blocks,
                              ::Val{W}, ::Val{F}, ::Val{FX}, brange, wts) where {P, W, F, FX}
    vW, vF = Val(W), Val(F)
    geom, dist_be, n_bins, N, B = s.geom, s.dist_be, s.n_bins, s.N, s.B
    x_fixed = FX ? reshape(s.xk, W, N) : nothing
    x_flat = FX ? nothing : reshape(s.xk, W, N, B)
    u_flat = reshape(s.uk, F, N, B)
    XT, UT = eltype(s.xk), eltype(s.uk)

    # The increment comes from `pair_delta`, so on a curved manifold the tensor components are in
    # the pair's own transported frame. An odd rank takes the canonical pair reading, as an odd
    # scalar increment does.
    @inbounds for (ir, jr) in blocks, i in ir
        wi = _point_weight(wts, i)
        for j in max(i + 1, first(jr)):last(jr)
            w = wi * _point_weight(wts, j)
            if FX
                X1 = SA.SVector{W, XT}(ntuple(d -> x_fixed[d, i], vW))
                X2 = SA.SVector{W, XT}(ntuple(d -> x_fixed[d, j], vW))
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = SFH.digitize(dist, dist_be)
                if ok && 1 <= bin <= n_bins
                    sgn = _tensor_reading(order, geom, frame)
                    for b in brange
                        U1 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, i, b], vF))
                        U2 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, j, b], vF))
                        du = sgn * SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                        _accumulate_tensor_pair!(sums_flat, counts_flat, du, bin, b, order, w)
                    end
                end
            else
                for b in brange
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
function _tensor_pairs_joint_inner!(sums, counts, order::Val{P}, s, blocks,
                                    ::Val{W}, ::Val{F}) where {P, W, F}
    vW, vF = Val(W), Val(F)
    geom, dist_be, n_bins, N = s.geom, s.dist_be, s.n_bins, s.N
    axis_edges, na, second_axis = s.axis
    x_fixed = reshape(s.xk, W, N)
    u_flat = reshape(s.uk, F, N)
    XT, UT = eltype(s.xk), eltype(s.uk)
    wts = s.weights
    @inbounds for (ir, jr) in blocks, i in ir
        wi = _point_weight(wts, i)
        X1 = SA.SVector{W, XT}(ntuple(d -> x_fixed[d, i], vW))
        U1 = SA.SVector{F, UT}(ntuple(d -> u_flat[d, i], vF))
        for j in max(i + 1, first(jr)):last(jr)
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
    tensor_partial(inner, order, shape, x, u, distance_bins, share, CT; distance_metric, axis, culling, weights)
        -> (sums, counts)

A worker's share of a tensor sweep: the pairs whose lower index is in share `share = (w, k)` of the outer indices,
resolved against the cull grids the worker builds ([`_share_indices`](@ref)), in freshly allocated accumulators,
computed on the worker's local backend `inner`. The shares of `w = 1:k` partition the sweep, so the partials add to
it exactly.
"""
function tensor_partial(
    inner::CB.AbstractExecutionBackend, order::Val{P}, shape::AbstractFieldShape{D}, x, u, distance_bins,
    share::NTuple{2, Int}, ::Type{CT}; axis = nothing, kwargs...,
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
    _tensor_into!(inner, sums, counts, order, shape, x, u, distance_bins, share; axis, kwargs...)
    return sums, counts
end

"""The pairs of a tensor sweep whose lower index is in share `share` of the outer indices added into `sums`/`counts`
on the backend `inner`: serially here, threaded by the OhMyThreads extension."""
function _tensor_into!(
    ::CB.AbstractExecutionBackend, sums, counts, order::Val, shape, x, u, distance_bins, share;
    distance_metric::DI.PreMetric = DI.Euclidean(), axis = nothing, culling::CullingPolicy = AutoCulling(),
    weights = NoWeights(),
)
    s = _tensor_setup(order, shape, sums, counts, x, u, distance_bins, distance_metric, axis, weights, culling)
    sf, cf = _tensor_flat(sums, counts, s)
    _tensor_pairs!(sf, cf, order, s, _share_indices(s.grid, s.N - 1, share))
    return nothing
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
        $(Expr(:meta, :inline))
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

"""The packed component of each dense component of a rank-`P` tensor over `D` components, first index
fastest: entry `q` is the `SFT.symmetric_rank` of dense component `q`'s multi-index."""
_tensor_dense_index(::Val{D}, ::Val{P}) where {D, P} =
    [SFT.symmetric_rank(Val(D), Val(P), Tuple(I)) for I in CartesianIndices(ntuple(_ -> D, Val(P)))]

"""Add the symmetric store `sym` `(n_entries, bins…)` into the dense tensor `dense` `(D, …, D, bins…)`, by one
gather in the arrays' own memory."""
function _expand_symmetric!(dense::AbstractArray, sym::AbstractArray, ::Val{D}, ::Val{P}) where {D, P}
    idx = copyto!(similar(sym, Int, D^P), _tensor_dense_index(Val(D), Val(P)))
    R = length(sym) ÷ size(sym, 1)
    reshape(dense, D^P, R) .+= reshape(sym, size(sym, 1), R)[idx, :]
    return dense
end
