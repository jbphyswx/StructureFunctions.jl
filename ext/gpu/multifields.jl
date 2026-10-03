# Multi-field (`Fields`) sweeps on a device: the tiled pair families with the operator's value over the
# packed column as the moment set, the packed field staged as the point's field column.

function SFC.gpu_calculate_structure_function_fields!(
    backend::CB.AbstractGPUBackend,
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::SFC.MF.Fields{D, V, K}, distance_bins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling::SFC.CullingPolicy = SFC.AutoCulling(),
    weights = SFC.NoWeights(),
    workspace = nothing,
) where {D, V, K}
    ka = backend.backend
    NB = SFC.n_histogram_bins(distance_bins)
    _check_gpu_outputs(sums, counts, ka, (NB,))
    geom = SFC._field_geometry(distance_metric, Val(D), Val(V), x)
    if SFC._on_a_line(geom, sf)
        sums_dev, counts_dev, direct = _accumulation_buffers(ka, eltype(sums), eltype(counts), (NB,), sums, counts)
        _gpu_sorted_line!(sums_dev, counts_dev, ka, sf, SFC._line_coordinates(x), SFC.MF.packed(f), distance_bins,
                          Val(D), Val(V), Val(K), weights)
        _add_accumulated!(sums, counts, sums_dev, counts_dev, direct)
        return nothing
    end
    xk, data, vF = SFC._kernel_fields(f, geom, x)
    moments = FieldValue(sf, vF, Val(V), Val(K))
    sums_dev, counts_dev, direct = _gpu_1d_unified_device(
        ka, xk, reshape(data, size(data, 1), size(data, 2), 1), moments, distance_bins, NB, 1, true,
        eltype(sums), eltype(counts), geom; weights, workspace, culling, source = x, sums, counts)
    _add_accumulated!(sums, counts, sums_dev, counts_dev, direct)
    return nothing
end
