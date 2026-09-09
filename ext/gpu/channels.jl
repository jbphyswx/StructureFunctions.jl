# Multi-channel (`Fields`) sweeps on a device.
#
# The bundle is already one packed `(V*F + K, N)` array, which is exactly what the scalar kernels
# stage, so the only new thing here is building the per-pair `ChannelIncrement` the operator reads.
# One thread owns an `i` and walks every `j > i`, accumulating into a global histogram.

KA.@kernel unsafe_indices = true function _channel_kernel!(
    sums, counts, @Const(x_mat), @Const(data), sf, geom, plan,
    N_points::Int, N_bins::Int, ::Val{W}, ::Val{F}, ::Val{V}, ::Val{K},
) where {W, F, V, K}
    i = @index(Global)
    if i <= N_points - 1
        XT = eltype(x_mat)
        X1 = SA.SVector{W, XT}(ntuple(d -> @inbounds(x_mat[d, i]), Val(W)))
        for j in (i + 1):N_points
            X2 = SA.SVector{W, XT}(ntuple(d -> @inbounds(x_mat[d, j]), Val(W)))
            ok, r, frame = SFH.pair_frame(geom, X1, X2)
            if ok
                bin = SFC.squared_digitize(plan, r * r)
                if 1 <= bin <= N_bins
                    val = SFC._channel_value(sf, Val(F), Val(V), Val(K), data, geom, frame, r, i, j)
                    @atomic sums[bin] += val
                    @atomic counts[bin] += one(eltype(counts))
                end
            end
        end
    end
end

function SFC.gpu_calculate_structure_function_channels!(
    backend::CB.AbstractGPUBackend,
    sums::AbstractVector, counts::AbstractVector,
    sf::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractMatrix, f::SFC.CH.Fields{D, V, K}, distance_bins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    culling = SFC.AutoCulling(),
    verbose::Bool = true, show_progress::Bool = true,
) where {D, V, K}
    _ = (verbose, show_progress)
    # Culling reorders the points on the host; the device sweep enumerates the full triangle, and
    # a permutation does not change a histogram, so the request is honoured by declining to permute.
    geom, xk, data, vF, plan, _ = SFC.channel_setup(f, x, distance_bins, distance_metric,
                                                    SFC.NoCulling())
    culling isa SFC.AlwaysCulling && throw(ArgumentError(
        "GPU multi-channel sweeps do not build a cell grid on device; use AutoCulling (which " *
        "declines here) or a CPU backend to cull.",
    ))

    ka = backend.backend
    N = size(data, 2)
    W = SFC.SFC_val_int(SFH.coordinate_width(geom))
    F = SFC.SFC_val_int(vF)
    nb = SFC.n_histogram_bins(plan)

    x_dev = KA.adapt(ka, Array(xk))
    d_dev = KA.adapt(ka, Array(data))
    s_dev = KA.adapt(ka, zeros(eltype(sums), nb))
    c_dev = KA.adapt(ka, zeros(eltype(counts), nb))

    kernel = _channel_kernel!(ka, 256)
    kernel(s_dev, c_dev, x_dev, d_dev, sf, geom, plan, N, nb,
           Val(W), Val(F), Val(V), Val(K); ndrange = N)
    KA.synchronize(ka)

    sums .+= Array(s_dev)
    counts .+= Array(c_dev)
    return nothing
end
