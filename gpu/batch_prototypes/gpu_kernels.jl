# KA prototype kernels — grid-stride pair loop, inner batch accumulation.

@inline function _batch_digitize_linear(
    dist,
    first_edge,
    last_edge,
    inv_step,
    offset,
    step_val,
    n_edges,
)
    return _GPUP._gpu_digitize_linear(
        dist, first_edge, last_edge, inv_step, offset, step_val, n_edges,
    )
end

"""
Grid-stride pair kernel, Case A (fixed `x` matrix). `u_batch` is `(3, N, B)`; output `(NB, B)`.
"""
@kernel function _batch_fixed_x_pairs!(
    sums,
    counts,
    @Const(x_mat),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    B::Int,
    nworkers::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _GPUP._pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
        X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _batch_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            inv_dist = inv(dist)
            r̂ = SA.SVector{3, FT}(dX[1] * inv_dist, dX[2] * inv_dist, dX[3] * inv_dist)
            for b in 1:B
                U1 = SA.SVector{3, FT}(
                    u_batch[1, i, b], u_batch[2, i, b], u_batch[3, i, b],
                )
                U2 = SA.SVector{3, FT}(
                    u_batch[1, j, b], u_batch[2, j, b], u_batch[3, j, b],
                )
                val = sf_type(U2 - U1, r̂)
                @atomic sums[bin, b] += val
                @atomic counts[bin, b] += one(UInt32)
            end
        end
        k += nworkers
    end
end

"""
Grid-stride pair kernel, Case B (varying `x`). `x_batch`, `u_batch` are `(3, N, B)`.

One CUDA thread per `(pair, b)` (flat index) — same math as `cpu_batch_varying_x!`,
no nested `for b in 1:B` (avoids device miscompilation; exposes batch parallelism).
"""
@kernel function _batch_varying_x_pairs!(
    sums,
    counts,
    @Const(x_batch),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    B::Int,
    nworkers::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2
    total_work = total_pairs * B
    t = worker
    while t <= total_work
        b = (t - one(t)) % B + one(t)
        k = (t - one(t)) ÷ B + one(t)
        i, j = _GPUP._pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(
            x_batch[1, i, b], x_batch[2, i, b], x_batch[3, i, b],
        )
        X2 = SA.SVector{3, FT}(
            x_batch[1, j, b], x_batch[2, j, b], x_batch[3, j, b],
        )
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _batch_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            inv_dist = inv(dist)
            r̂ = SA.SVector{3, FT}(dX[1] * inv_dist, dX[2] * inv_dist, dX[3] * inv_dist)
            U1 = SA.SVector{3, FT}(
                u_batch[1, i, b], u_batch[2, i, b], u_batch[3, i, b],
            )
            U2 = SA.SVector{3, FT}(
                u_batch[1, j, b], u_batch[2, j, b], u_batch[3, j, b],
            )
            val = sf_type(U2 - U1, r̂)
            @atomic sums[bin, b] += val
            @atomic counts[bin, b] += one(UInt32)
        end
        t += nworkers
    end
end

"""
Grid-stride pair kernel for one batch index `b_idx` with 1D histogram output (CUDA-safe).
"""
@kernel function _batch_varying_x_single_b_pairs!(
    sums,
    counts,
    @Const(x_batch),
    @Const(u_batch),
    sf_type,
    N_points::Int,
    N_bins::Int,
    b_idx::Int,
    nworkers::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _GPUP._pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(
            x_batch[1, i, b_idx], x_batch[2, i, b_idx], x_batch[3, i, b_idx],
        )
        X2 = SA.SVector{3, FT}(
            x_batch[1, j, b_idx], x_batch[2, j, b_idx], x_batch[3, j, b_idx],
        )
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _batch_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            inv_dist = inv(dist)
            r̂ = SA.SVector{3, FT}(dX[1] * inv_dist, dX[2] * inv_dist, dX[3] * inv_dist)
            U1 = SA.SVector{3, FT}(
                u_batch[1, i, b_idx], u_batch[2, i, b_idx], u_batch[3, i, b_idx],
            )
            U2 = SA.SVector{3, FT}(
                u_batch[1, j, b_idx], u_batch[2, j, b_idx], u_batch[3, j, b_idx],
            )
            val = sf_type(U2 - U1, r̂)
            @atomic sums[bin] += val
            @atomic counts[bin] += one(UInt32)
        end
        k += nworkers
    end
end

"""Launch grid-stride batch kernel for fixed or varying geometry."""
function launch_batch_kernel!(
    backend,
    variant::Symbol,
    sums,
    counts,
    x_dev,
    u_dev,
    sf_type,
    N::Int,
    lp,
    B::Int;
    nworkers::Int = min(262_144, N * (N - 1) ÷ 2),
    workgroup_size::Int = 256,
    b_idx::Int = 1,
)
    FT = eltype(sums)
    n_bins = lp.n_bins
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
    if variant === :fixed_x
        kernel! = _batch_fixed_x_pairs!(backend)
        kernel!(
            sums, counts, x_dev, u_dev, sf_type,
            N, n_bins, B, nworkers,
            fe, le, is_, off, sv;
            ndrange = nworkers, workgroupsize = workgroup_size,
        )
    elseif variant === :varying_x_single_b
        total_pairs = N * (N - 1) ÷ 2
        nworkers = min(nworkers, total_pairs)
        kernel! = _batch_varying_x_single_b_pairs!(backend)
        kernel!(
            sums, counts, x_dev, u_dev, sf_type,
            N, n_bins, b_idx, nworkers,
            fe, le, is_, off, sv;
            ndrange = nworkers, workgroupsize = workgroup_size,
        )
    elseif variant === :varying_x
        total_work = N * (N - 1) ÷ 2 * B
        nworkers = min(nworkers, total_work)
        kernel! = _batch_varying_x_pairs!(backend)
        kernel!(
            sums, counts, x_dev, u_dev, sf_type,
            N, n_bins, B, nworkers,
            fe, le, is_, off, sv;
            ndrange = nworkers, workgroupsize = workgroup_size,
        )
    else
        error("unknown batch kernel variant: $variant")
    end
    KA.synchronize(backend)
    return nothing
end

"""Stage host arrays to device layout `(3, N)` or `(3, N, B)`."""
function stage_fixed_x_device(backend, x_mat::AbstractMatrix{FT}, u_batch::AbstractArray{FT}) where {FT}
    x3, u3, bd = pad3_batch(x_mat, u_batch)
    B = prod(bd)
    u_flat = reshape(u3, 3, size(x_mat, 2), B)
    return KA.adapt(backend, x3), KA.adapt(backend, u_flat)
end

function stage_varying_x_device(backend, x_batch::AbstractArray{FT}, u_batch::AbstractArray{FT}) where {FT}
    x3, u3, bd = pad3_batch_matched(x_batch, u_batch)
    B = prod(bd)
    x_flat = reshape(x3, 3, size(x_batch, 2), B)
    u_flat = reshape(u3, 3, size(u_batch, 2), B)
    return KA.adapt(backend, x_flat), KA.adapt(backend, u_flat)
end

function gpu_batch_fixed_x!(
    backend,
    sums,
    counts,
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int = min(262_144, size(x_mat, 2) * (size(x_mat, 2) - 1) ÷ 2),
) where {FT}
    lp = linear_bin_params(bin_edges)
    B = batch_size(u_batch)
    NB = lp.n_bins - 1
    sums_dev = KA.adapt(backend, zeros(FT, NB, B))
    counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
    x_dev, u_dev = stage_fixed_x_device(backend, x_mat, u_batch)
    launch_batch_kernel!(backend, :fixed_x, sums_dev, counts_dev, x_dev, u_dev, sf_type, size(x_mat, 2), lp, B; nworkers = nworkers)
    bd = batch_dims(u_batch)
    sums_host = reshape(Array(sums_dev), NB, bd...)
    counts_host = reshape(Array(counts_dev), NB, bd...)
    copy!(sums, sums_host)
    copy!(counts, counts_host)
    return nothing
end

function gpu_batch_varying_x!(
    backend,
    sums,
    counts,
    x_batch::AbstractArray{FT},
    u_batch::AbstractArray{FT},
    sf_type,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int = min(262_144, size(x_batch, 2) * (size(x_batch, 2) - 1) ÷ 2),
) where {FT}
    lp = linear_bin_params(bin_edges)
    B = batch_size(u_batch)
    NB = lp.n_bins - 1
    N = size(x_batch, 2)
    sums_dev = KA.adapt(backend, zeros(FT, NB, B))
    counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
    col_sums = KA.adapt(backend, zeros(FT, NB))
    col_counts = KA.adapt(backend, zeros(UInt32, NB))
    x_dev, u_dev = stage_varying_x_device(backend, x_batch, u_batch)
    for b in 1:B
        fill!(col_sums, zero(FT))
        fill!(col_counts, zero(UInt32))
        launch_batch_kernel!(
            backend, :varying_x_single_b, col_sums, col_counts, x_dev, u_dev, sf_type, N, lp, B;
            nworkers = nworkers, b_idx = b,
        )
        _copy_batch_column!(sums_dev, counts_dev, col_sums, col_counts, b)
        KA.synchronize(backend)
    end
    bd = batch_dims(u_batch)
    sums_host = reshape(Array(sums_dev), NB, bd...)
    counts_host = reshape(Array(counts_dev), NB, bd...)
    copy!(sums, sums_host)
    copy!(counts, counts_host)
    return nothing
end
