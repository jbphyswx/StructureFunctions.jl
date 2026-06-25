# Tensor structure-function calculations.

function calculate_structure_function_tensor(
    order::Val{P},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    backend::AbstractExecutionBackend = SerialBackend(),
    output_type::Type{OTT} = SFO.StructureFunctionTensor,
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {P, FT1, FT2, FT3, OTT, CT}
    shape = _validate_array_shape(x, u)
    D = spatial_dimension(shape)
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = has_auxiliary_axes(shape) ? size(u)[3:end] : ()
    OT = promote_type(float(FT1), float(FT2))
    tensor_dims = ntuple(_ -> D, P)
    sums = zeros(OT, tensor_dims..., n_bins, auxiliary_dims...)
    counts = zeros(CT, n_bins, auxiliary_dims...)
    calculate_structure_function_tensor!(
        sums, counts, order, x, u, distance_bins; backend = backend, kwargs...
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
    backend::AbstractExecutionBackend = SerialBackend(),
    kwargs...,
) where {P}
    shape = _validate_array_shape(x, u)
    return _dispatch_tensor!(
        backend, shape, sums, counts, order, x, u, distance_bins; kwargs...
    )
end

function _dispatch_tensor!(
    ::SerialBackend,
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
    ::AutoBackend,
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
        SerialBackend(), shape, sums, counts, order, x, u, distance_bins; kwargs...
    )
end

function _dispatch_tensor!(
    backend::Union{ThreadedBackend, DistributedBackend, GPUBackend},
    ::AbstractFieldShape,
    args...;
    kwargs...,
)
    throw(
        ArgumentError(
            "$(typeof(backend)) tensor backend is not implemented yet; use backend=SerialBackend()",
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

    fill!(sums, zero(eltype(sums)))
    fill!(counts, zero(eltype(counts)))

    dist_be = BinEdges(distance_bins)
    N = size(u, 2)
    B = isempty(auxiliary_dims) ? 1 : prod(auxiliary_dims)
    fixed_x = ndims(x) == 2
    vD = Val(D)

    x_fixed = fixed_x ? reshape(x, D, N) : nothing
    x_flat = fixed_x ? nothing : reshape(x, D, N, B)
    u_flat = reshape(u, D, N, B)
    sums_flat = reshape(sums, ntuple(_ -> D, P)..., n_bins, B)
    counts_flat = reshape(counts, n_bins, B)

    @inbounds for i in 1:N
        for j in (i + 1):N
            if fixed_x
                X1 = SA.SVector{D, eltype(x)}(ntuple(d -> x_fixed[d, i], vD))
                X2 = SA.SVector{D, eltype(x)}(ntuple(d -> x_fixed[d, j], vD))
                dist = distance_metric(X1, X2)
                bin = SFH.digitize(dist, dist_be)
                if 1 <= bin <= n_bins
                    for b in 1:B
                        U1 = SA.SVector{D, eltype(u)}(ntuple(d -> u_flat[d, i, b], vD))
                        U2 = SA.SVector{D, eltype(u)}(ntuple(d -> u_flat[d, j, b], vD))
                        _accumulate_tensor_pair!(sums_flat, counts_flat, U2 - U1, bin, b, order)
                    end
                end
            else
                for b in 1:B
                    X1 = SA.SVector{D, eltype(x)}(ntuple(d -> x_flat[d, i, b], vD))
                    X2 = SA.SVector{D, eltype(x)}(ntuple(d -> x_flat[d, j, b], vD))
                    dist = distance_metric(X1, X2)
                    bin = SFH.digitize(dist, dist_be)
                    if 1 <= bin <= n_bins
                        U1 = SA.SVector{D, eltype(u)}(ntuple(d -> u_flat[d, i, b], vD))
                        U2 = SA.SVector{D, eltype(u)}(ntuple(d -> u_flat[d, j, b], vD))
                        _accumulate_tensor_pair!(sums_flat, counts_flat, U2 - U1, bin, b, order)
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
