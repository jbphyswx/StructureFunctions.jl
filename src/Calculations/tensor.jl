# Tensor structure-function calculations.

function calculate_structure_function_tensor(
    order::Val{P},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    output_type::Type{OTT} = SFO.StructureFunctionTensor,
    count_eltype::Type{CT} = UInt32,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {P, FT1, FT2, FT3, OTT, CT}
    shape = _validate_array_shape(x, u, distance_metric)
    _assert_counts_representable(CT, size(x, 2))
    D = spatial_dimension(shape)
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
    OT = promote_type(float(FT1), float(FT2))
    tensor_dims = ntuple(_ -> D, P)
    sums = zeros(OT, tensor_dims..., n_bins, auxiliary_dims...)
    counts = zeros(CT, n_bins, auxiliary_dims...)
    calculate_structure_function_tensor!(
        sums, counts, order, x, u, distance_bins; backend = backend, distance_metric, kwargs...
    )
    # The backend produces the raw accumulator; `_finalize` returns it as-is or as the averaged
    # mean tensor (the default), mirroring the 1D output-type dispatch.
    raw = SFO.StructureFunctionTensorSumsAndCounts(order, distance_bins, sums, counts)
    return _finalize(raw, output_type)
end

function calculate_structure_function_tensor!(
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val{P},
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...,
) where {P}
    shape = _validate_array_shape(x, u, distance_metric)
    # Only forward knobs the kernel accepts; it has no `kwargs...` sink.
    return _dispatch_tensor!(
        backend, shape, sums, counts, order, x, u, distance_bins;
        distance_metric = distance_metric,
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
        CB.SerialBackend(), shape, sums, counts, order, x, u, distance_bins; kwargs...
    )
end

function _dispatch_tensor!(
    backend::Union{CB.AbstractThreadedBackend, CB.AbstractDistributedBackend, CB.AbstractGPUBackend},
    ::AbstractFieldShape,
    args...;
    kwargs...,
)
    throw(
        ArgumentError(
            "$(typeof(backend)) tensor backend is not implemented yet; use backend=CB.SerialBackend()",
        ),
    )
end

function serial_calculate_structure_function_tensor!(
    sums::AbstractArray,
    counts::AbstractArray,
    order::Val{P},
    shape::AbstractFieldShape{D},
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {P, D}
    P == 2 || P == 3 ||
        throw(ArgumentError("tensor order $P is not implemented yet; supported orders are 2 and 3"))

    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
    expected_sums = (ntuple(_ -> D, P)..., n_bins, auxiliary_dims...)
    expected_counts = (n_bins, auxiliary_dims...)
    size(sums) == expected_sums ||
        throw(DimensionMismatch("sums must have shape $expected_sums; got $(size(sums))"))
    size(counts) == expected_counts ||
        throw(DimensionMismatch("counts must have shape $expected_counts; got $(size(counts))"))

    dist_be = BinEdges(distance_bins)
    N = size(u, 2)
    B = isempty(auxiliary_dims) ? 1 : prod(auxiliary_dims)
    fixed_x = ndims(x) == 2

    geom = SFH.pair_geometry_for(distance_metric, Val(D))
    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    vW = SFH.coordinate_width(geom)
    vF = SFH.field_width(geom)
    W = _val_int(vW)
    F = _val_int(vF)

    x_fixed = fixed_x ? reshape(xk, W, N) : nothing
    x_flat = fixed_x ? nothing : reshape(xk, W, N, B)
    u_flat = reshape(uk, F, N, B)
    sums_flat = reshape(sums, ntuple(_ -> D, P)..., n_bins, B)
    counts_flat = reshape(counts, n_bins, B)

    # The increment comes from `pair_delta`, so on a curved manifold the tensor components are in
    # the pair's own transported frame rather than raw coordinate differences.
    @inbounds for i in 1:N
        for j in (i + 1):N
            if fixed_x
                X1 = SA.SVector{W, eltype(xk)}(ntuple(d -> x_fixed[d, i], vW))
                X2 = SA.SVector{W, eltype(xk)}(ntuple(d -> x_fixed[d, j], vW))
                ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                bin = SFH.digitize(dist, dist_be)
                if ok && 1 <= bin <= n_bins
                    for b in 1:B
                        U1 = SA.SVector{F, eltype(uk)}(ntuple(d -> u_flat[d, i, b], vF))
                        U2 = SA.SVector{F, eltype(uk)}(ntuple(d -> u_flat[d, j, b], vF))
                        du = SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                        _accumulate_tensor_pair!(sums_flat, counts_flat, du, bin, b, order)
                    end
                end
            else
                for b in 1:B
                    X1 = SA.SVector{W, eltype(xk)}(ntuple(d -> x_flat[d, i, b], vW))
                    X2 = SA.SVector{W, eltype(xk)}(ntuple(d -> x_flat[d, j, b], vW))
                    ok, dist, frame = SFH.pair_frame(geom, X1, X2)
                    bin = SFH.digitize(dist, dist_be)
                    if ok && 1 <= bin <= n_bins
                        U1 = SA.SVector{F, eltype(uk)}(ntuple(d -> u_flat[d, i, b], vF))
                        U2 = SA.SVector{F, eltype(uk)}(ntuple(d -> u_flat[d, j, b], vF))
                        du = SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                        _accumulate_tensor_pair!(sums_flat, counts_flat, du, bin, b, order)
                    end
                end
            end
        end
    end

    return sums, counts
end

@inline function _accumulate_tensor_pair!(sums, counts, du, bin::Int, b::Int, ::Val{2})
    D = length(du)
    @inbounds for a in 1:D
        dua = du[a]
        for c in 1:D
            sums[a, c, bin, b] += dua * du[c]
        end
    end
    @inbounds counts[bin, b] += one(eltype(counts))
    return nothing
end

@inline function _accumulate_tensor_pair!(sums, counts, du, bin::Int, b::Int, ::Val{3})
    D = length(du)
    @inbounds for a in 1:D
        dua = du[a]
        for c in 1:D
            duac = dua * du[c]
            for e in 1:D
                sums[a, c, e, bin, b] += duac * du[e]
            end
        end
    end
    @inbounds counts[bin, b] += one(eltype(counts))
    return nothing
end
