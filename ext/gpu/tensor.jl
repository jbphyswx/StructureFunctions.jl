# Tensor structure functions on a device: the tiled pair families with the tensor's packed symmetric
# components as the moment set, expanded to the dense tensor on the device.

"""Add the packed histogram `packed` `(n_sym, NB, S)` into the dense `sums` `(D, …, D, NB, S…)` and the
counts row of `pcnt` `(n_sym, NB, S)` into `counts` `(NB, S…)`."""
function _tensor_add_packed!(sums, counts, packed, pcnt, ::Val{D}, ::Val{P}) where {D, P}
    NB, S = size(packed, 2), size(packed, 3)
    SFC._expand_symmetric!(sums, packed, Val(D), Val(P))
    c = selectdim(pcnt, 1, 1)
    eltype(c) === eltype(counts) ? (reshape(counts, NB, S) .+= c) : (reshape(counts, NB, S) .+= eltype(counts).(c))
    return nothing
end

function SFC.gpu_calculate_structure_function_tensor!(
    backend::CB.AbstractGPUBackend,
    sums::AbstractArray, counts::AbstractArray, order::Val{P},
    shape::SFC.AbstractFieldShape{D}, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(), axis = nothing,
    culling::SFC.CullingPolicy = SFC.AutoCulling(), weights = SFC.NoWeights(), workspace = nothing,
) where {P, D}
    ka = backend.backend
    _check_gpu_residency(sums, counts, ka)
    geom = SFH.pair_geometry_for(distance_metric, Val(D))
    axis === nothing || geom isa SFH.FlatGeometry || SFC._require_directional(SFC.FrameTransport())
    fixed_x = ndims(x) == 2
    source = x
    xk, uk = SFH.prepare_pair_inputs(geom, x, u)
    N = size(uk, 2)
    B = prod(size(uk)[3:end])
    NB = SFC.n_histogram_bins(distance_bins)
    moments = TensorComponents{P, D}()
    OT, CT = eltype(sums), eltype(counts)
    if axis === nothing
        packed, pcnt, _ = _gpu_1d_unified_device(ka, xk, uk, moments, distance_bins, NB, B, fixed_x, OT, CT, geom;
                                                 weights, workspace, culling, source)
        _tensor_add_packed!(sums, counts, packed, pcnt, Val(D), order)
    else
        axis_bins, n_axis, second_axis = axis
        packed, pcnt, _ = _gpu_2d_unified_device(ka, xk, uk, moments, distance_bins, axis_bins, NB, n_axis, 1,
                                                 fixed_x, OT, CT, geom; weights, workspace, culling, source,
                                                 second_axis)
        _tensor_add_packed!(sums, counts, reshape(packed, size(packed, 1), NB * n_axis, 1),
                            reshape(pcnt, size(pcnt, 1), NB * n_axis, 1), Val(D), order)
    end
    return sums, counts
end
