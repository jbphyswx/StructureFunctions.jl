# Tensor structure functions on a device.
#
# One thread owns an `i` and walks every `j > i`, so the pair set is the same upper triangle the CPU
# kernel enumerates. The accumulation is a global atomic per tensor component: a rank-`P` tensor has
# `D^P` of them per pair, which is why this is a separate kernel rather than a mode of the scalar
# ones — there is no shared-memory histogram small enough to stage it.

KA.@kernel unsafe_indices = true function _tensor_kernel!(
    sums, counts, @Const(x_mat), @Const(u_mat), geom, dist_be,
    N_points::Int, N_bins::Int, B::Int, ::Val{W}, ::Val{F}, ::Val{D}, ::Val{P},
) where {W, F, D, P}
    i = @index(Global)
    if i <= N_points - 1
        XT = eltype(x_mat)
        UT = eltype(u_mat)
        X1 = SA.SVector{W, XT}(ntuple(d -> @inbounds(x_mat[d, i]), Val(W)))
        for j in (i + 1):N_points
            X2 = SA.SVector{W, XT}(ntuple(d -> @inbounds(x_mat[d, j]), Val(W)))
            ok, dist, frame = SFH.pair_frame(geom, X1, X2)
            bin = SFH.digitize(dist, dist_be)
            if ok && 1 <= bin <= N_bins
                for b in 1:B
                    U1 = SA.SVector{F, UT}(ntuple(d -> @inbounds(u_mat[d, i, b]), Val(F)))
                    U2 = SA.SVector{F, UT}(ntuple(d -> @inbounds(u_mat[d, j, b]), Val(F)))
                    du = SFH.pair_delta(geom, frame, X1, X2, U1, U2)
                    if P == 2
                        for a in 1:D, c in 1:D
                            @atomic sums[(a - 1) * D + c, bin, b] += du[a] * du[c]
                        end
                    else
                        for a in 1:D, c in 1:D, e in 1:D
                            @atomic sums[(a - 1) * D * D + (c - 1) * D + e, bin, b] +=
                                du[a] * du[c] * du[e]
                        end
                    end
                    @atomic counts[bin, b] += one(eltype(counts))
                end
            end
        end
    end
end

function SFC.gpu_calculate_structure_function_tensor!(
    backend::CB.AbstractGPUBackend,
    sums::AbstractArray, counts::AbstractArray, order::Val{P},
    shape::SFC.AbstractFieldShape{D}, x::AbstractArray, u::AbstractArray,
    distance_bins::AbstractVector;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {P, D}
    s = SFC._tensor_setup(order, shape, sums, counts, x, u, distance_bins, distance_metric)
    s.fixed_x || throw(ArgumentError(
        "the GPU tensor kernel takes one shared position set; `x` varying per auxiliary slice is a " *
        "different staging problem and is handled by the CPU backends.",
    ))

    ka = backend.backend
    W, F, N, B, n_bins = s.W, s.F, s.N, s.B, s.n_bins
    OT = eltype(sums)
    CT = eltype(counts)

    x_dev = KA.adapt(ka, reshape(collect(s.xk), W, N))
    u_dev = KA.adapt(ka, reshape(collect(s.uk), F, N, B))
    sums_dev = KA.adapt(ka, zeros(OT, D^P, n_bins, B))
    counts_dev = KA.adapt(ka, zeros(CT, n_bins, B))

    kernel = _tensor_kernel!(ka, 256)
    kernel(sums_dev, counts_dev, x_dev, u_dev, s.geom, s.dist_be, N, n_bins, B,
           Val(W), Val(F), Val(D), order; ndrange = N)
    KA.synchronize(ka)

    host_sums = Array(sums_dev)
    host_counts = Array(counts_dev)
    sums_flat, counts_flat = SFC._tensor_flat(sums, counts, s)
    @inbounds for b in 1:B, bin in 1:n_bins
        for q in 1:(D^P)
            sums_flat[_tensor_component(q, Val(D), order)..., bin, b] += host_sums[q, bin, b]
        end
        counts_flat[bin, b] += host_counts[bin, b]
    end
    return sums, counts
end

"""Unflatten a linear tensor-component index back to its `P` subscripts."""
@inline function _tensor_component(q::Int, ::Val{D}, ::Val{P}) where {D, P}
    return ntuple(Val(P)) do k
        ((q - 1) ÷ D^(P - k)) % D + 1
    end
end
