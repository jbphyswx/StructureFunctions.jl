# CPU Batch Calculation Drivers

using Distances: Distances as DI

# --- CPU Batch Accumulator Helpers ---

@inline function _cpu_sp1d_accum!(
    sums::AbstractArray{OT, 3},
    counts::AbstractArray{CT, 3},
    bin::Int,
    b::Int,
    du_L,
    du_T,
    du_T2,
) where {OT, CT}
    du_L2 = du_L * du_L
    @inbounds begin
        sums[1, bin, b] += du_L2 + du_T2
        sums[2, bin, b] += du_L2
        sums[3, bin, b] += du_T2
        sums[4, bin, b] += du_L * (du_L2 + du_T2)
        sums[5, bin, b] += du_L * du_L2
        sums[6, bin, b] += du_L * du_T2
        for t in 1:SINGLE_PASS_N
            counts[t, bin, b] += one(CT)
        end
    end
    return nothing
end

# --- Public CPU Batch APIs ---

"""
    auxiliary_structure_function!(sums, counts, sf_type, x, u, distance_bins; strip_width=32)

Dispatch CPU batch calculation based on `x` rank.
"""
function auxiliary_structure_function!(
    sums,
    counts,
    sf_type,
    x,
    u,
    distance_bins;
    kwargs...,
)
    if ndims(x) == 2
        auxiliary_shared_positions!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    else
        auxiliary_varying_positions!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    end
    return nothing
end

"""
    auxiliary_shared_positions!(sums, counts, x_mat, u_batch, sf_type, distance_bins; strip_width=32)

Fixed geometry batch: `x` is (N_dims, N), `u` has trailing batch dims.
"""
function auxiliary_shared_positions!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    distance_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x_mat)
    bdims = size(u_batch)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2

    u_flat = reshape(u_batch, N_dims, N, B)
    sums_flat = reshape(sums, n_bins, B)
    counts_flat = reshape(counts, n_bins, B)

    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(eltype(counts_flat)))

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    @inbounds for k in 1:total_pairs
        i, j = _pair_from_linear(k, N)
        X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_mat[d, i], vN))
        X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_mat[d, j], vN))
        dist = dist_metric(X1, X2)
        bin = SFH.digitize(dist, dist_be)
        if 1 <= bin <= n_bins
            r̂ = SFH.r̂(X1, X2, dist_metric, dist)
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                    U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                    val = sf_type(U2 - U1, r̂)
                    sums_flat[bin, b] += val
                    counts_flat[bin, b] += one(eltype(counts_flat))
                end
                b0 = b1 + 1
            end
        end
    end
    return nothing
end

"""
    auxiliary_varying_positions!(sums, counts, x_batch, u_batch, sf_type, distance_bins; strip_width=32)

Varying geometry batch: `x` and `u` have matching trailing batch dims.
"""
function auxiliary_varying_positions!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x_batch::AbstractArray{FT},
    u_batch::AbstractArray{FT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    distance_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x_batch)[1:2]
    bdims = size(u_batch)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2

    x_flat = reshape(x_batch, N_dims, N, B)
    u_flat = reshape(u_batch, N_dims, N, B)
    sums_flat = reshape(sums, n_bins, B)
    counts_flat = reshape(counts, n_bins, B)

    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(eltype(counts_flat)))

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    @inbounds for k in 1:total_pairs
        i, j = _pair_from_linear(k, N)
        b0 = 1
        while b0 <= B
            b1 = min(b0 + strip_width - 1, B)
            @inbounds @simd for b in b0:b1
                X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                dist = dist_metric(X1, X2)
                bin = SFH.digitize(dist, dist_be)
                if 1 <= bin <= n_bins
                    r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                    U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                    U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                    val = sf_type(U2 - U1, r̂)
                    sums_flat[bin, b] += val
                    counts_flat[bin, b] += one(eltype(counts_flat))
                end
            end
            b0 = b1 + 1
        end
    end
    return nothing
end

"""
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; strip_width=32)

Six-invariant-type single-pass 1D batch.
"""
function serial_calculate_structure_functions_single_pass!(
    sums::AbstractArray{FT},
    counts::AbstractArray{CT},
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT, CT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x)[1:2]
    bdims = size(u)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2
    fixed_x = ndims(x) == 2

    u_flat = reshape(u, N_dims, N, B)
    x_flat = fixed_x ? reshape(x, N_dims, N) : reshape(x, N_dims, N, B)
    sums_flat = reshape(sums, SINGLE_PASS_N, n_bins, B)
    counts_flat = reshape(counts, SINGLE_PASS_N, n_bins, B)

    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(CT))

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    @inbounds for k in 1:total_pairs
        i, j = _pair_from_linear(k, N)
        if fixed_x
            X1_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i], vN))
            X2_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j], vN))
            dist_fixed = dist_metric(X1_fixed, X2_fixed)
            bin_fixed = SFH.digitize(dist_fixed, dist_be)
            if 1 <= bin_fixed <= n_bins
                r̂_fixed = SFH.r̂(X1_fixed, X2_fixed, dist_metric, dist_fixed)
                b0 = 1
                while b0 <= B
                    b1 = min(b0 + strip_width - 1, B)
                    @inbounds @simd for b in b0:b1
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂_fixed)
                        du_T = SFH.mδu_t(du, r̂_fixed)
                        du_T2 = SFH.transverse_norm2(du, r̂_fixed)
                        _cpu_sp1d_accum!(sums_flat, counts_flat, bin_fixed, b, du_L, du_T, du_T2)
                    end
                    b0 = b1 + 1
                end
            end
        else
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                    X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                    dist = dist_metric(X1, X2)
                    bin = SFH.digitize(dist, dist_be)
                    if 1 <= bin <= n_bins
                        r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂)
                        du_T = SFH.mδu_t(du, r̂)
                        du_T2 = SFH.transverse_norm2(du, r̂)
                        _cpu_sp1d_accum!(sums_flat, counts_flat, bin, b, du_L, du_T, du_T2)
                    end
                end
                b0 = b1 + 1
            end
        end
    end
    return nothing
end

"""
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; strip_width=32)

Six-invariant-type SP2D batch; output `(6, n_dist, n_val, batch…)`.
"""
function serial_calculate_structure_functions_single_pass_2d!(
    sums::AbstractArray{FT},
    counts::AbstractArray{CT},
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins,
    value_bins::SinglePass2DValueBins;
    strip_width::Int = 32,
    kwargs...,
) where {FT, CT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x)[1:2]
    bdims = size(u)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2
    fixed_x = ndims(x) == 2
    n_val = size(sums, 3)
    _validate_value_bins!(value_bins, n_val)

    u_flat = reshape(u, N_dims, N, B)
    x_flat = fixed_x ? reshape(x, N_dims, N) : reshape(x, N_dims, N, B)
    sums_flat = reshape(sums, SINGLE_PASS_N, n_bins, n_val, B)
    counts_flat = reshape(counts, SINGLE_PASS_N, n_bins, n_val, B)

    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(CT))

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    @inbounds for k in 1:total_pairs
        i, j = _pair_from_linear(k, N)
        if fixed_x
            X1_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i], vN))
            X2_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j], vN))
            dist_fixed = dist_metric(X1_fixed, X2_fixed)
            bin_fixed = SFH.digitize(dist_fixed, dist_be)
            if 1 <= bin_fixed <= n_bins
                r̂_fixed = SFH.r̂(X1_fixed, X2_fixed, dist_metric, dist_fixed)
                b0 = 1
                while b0 <= B
                    b1 = min(b0 + strip_width - 1, B)
                    @inbounds @simd for b in b0:b1
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂_fixed)
                        du_T = SFH.mδu_t(du, r̂_fixed)
                        du_L2 = du_L * du_L
                        du_T2 = SFH.transverse_norm2(du, r̂_fixed)
                        vals = (
                            du_L2 + du_T2,
                            du_L2,
                            du_T2,
                            du_L * (du_L2 + du_T2),
                            du_L * du_L2,
                            du_L * du_T2,
                        )
                        for t in 1:SINGLE_PASS_N
                            vb = _sp2d_value_bin_at(value_bins, t)
                            vbin = SFH.digitize(vals[t], vb)
                            n_val_t = length(vb) - 1
                            if 1 <= vbin <= n_val_t && vbin <= n_val
                                sums_flat[t, bin_fixed, vbin, b] += vals[t]
                                counts_flat[t, bin_fixed, vbin, b] += one(CT)
                            end
                        end
                    end
                    b0 = b1 + 1
                end
            end
        else
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                    X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                    dist = dist_metric(X1, X2)
                    bin = SFH.digitize(dist, dist_be)
                    if 1 <= bin <= n_bins
                        r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂)
                        du_T = SFH.mδu_t(du, r̂)
                        du_L2 = du_L * du_L
                        du_T2 = SFH.transverse_norm2(du, r̂)
                        vals = (
                            du_L2 + du_T2,
                            du_L2,
                            du_T2,
                            du_L * (du_L2 + du_T2),
                            du_L * du_L2,
                            du_L * du_T2,
                        )
                        for t in 1:SINGLE_PASS_N
                            vb = _sp2d_value_bin_at(value_bins, t)
                            vbin = SFH.digitize(vals[t], vb)
                            n_val_t = length(vb) - 1
                            if 1 <= vbin <= n_val_t && vbin <= n_val
                                sums_flat[t, bin, vbin, b] += vals[t]
                                counts_flat[t, bin, vbin, b] += one(CT)
                            end
                        end
                    end
                end
                b0 = b1 + 1
            end
        end
    end
    return nothing
end

"""
    auxiliary_joint2d!(sums, counts, sf_type, x, u, distance_bins, value_bins; strip_width=32)

Single-type joint 2D batch; output `(n_dist, n_val, batch…)`.
"""
function auxiliary_joint2d!(
    sums::AbstractArray{FT},
    counts::AbstractArray{CT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins,
    value_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT, CT}
    dist_be = BinEdges(distance_bins)
    val_be = BinEdges(value_bins)
    n_dist = n_histogram_bins(dist_be)
    n_val = n_histogram_bins(val_be)
    N_dims, N = size(x)[1:2]
    bdims = size(u)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2
    fixed_x = ndims(x) == 2

    u_flat = reshape(u, N_dims, N, B)
    x_flat = fixed_x ? reshape(x, N_dims, N) : reshape(x, N_dims, N, B)
    sums_flat = reshape(sums, n_dist, n_val, B)
    counts_flat = reshape(counts, n_dist, n_val, B)

    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(CT))

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    @inbounds for k in 1:total_pairs
        i, j = _pair_from_linear(k, N)
        if fixed_x
            X1_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i], vN))
            X2_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j], vN))
            dist_fixed = dist_metric(X1_fixed, X2_fixed)
            dbin_fixed = SFH.digitize(dist_fixed, dist_be)
            if 1 <= dbin_fixed <= n_dist
                r̂_fixed = SFH.r̂(X1_fixed, X2_fixed, dist_metric, dist_fixed)
                b0 = 1
                while b0 <= B
                    b1 = min(b0 + strip_width - 1, B)
                    @inbounds @simd for b in b0:b1
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        val = sf_type(U2 - U1, r̂_fixed)
                        vbin = SFH.digitize(val, val_be)
                        if 1 <= vbin <= n_val
                            sums_flat[dbin_fixed, vbin, b] += val
                            counts_flat[dbin_fixed, vbin, b] += one(CT)
                        end
                    end
                    b0 = b1 + 1
                end
            end
        else
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                    X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                    dist = dist_metric(X1, X2)
                    dbin = SFH.digitize(dist, dist_be)
                    if 1 <= dbin <= n_dist
                        r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        val = sf_type(U2 - U1, r̂)
                        vbin = SFH.digitize(val, val_be)
                        if 1 <= vbin <= n_val
                            sums_flat[dbin, vbin, b] += val
                            counts_flat[dbin, vbin, b] += one(CT)
                        end
                    end
                end
                b0 = b1 + 1
            end
        end
    end
    return nothing
end

"""Loop-over-slice gold reference for batch parity."""
function cpu_slice_baseline!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    distance_bins;
    fixed_x::Bool = true,
) where {FT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    bd = batch_dims(u)
    B = batch_size(u)
    sums_f, counts_f = _flatten_sums_counts(sums, counts)
    for b in 1:B
        if fixed_x
            x_slice = x
            u_slice = batch_field_slice(u, b)
        else
            x_slice = batch_field_slice(x, b)
            u_slice = batch_field_slice(u, b)
        end
        local_output = zeros(eltype(sums), n_bins)
        local_counts = zeros(UInt32, n_bins)
        serial_calculate_structure_function!(
            local_output,
            local_counts,
            sf_type,
            x_slice,
            u_slice,
            distance_bins;
            verbose = false,
            show_progress = false,
        )
        sums_f[:, b] .= local_output
        counts_f[:, b] .= local_counts
    end
end
# --- Multi-threaded CPU Batch Reducers ---

function auxiliary_structure_function_threaded!(
    sums,
    counts,
    sf_type,
    x,
    u,
    distance_bins;
    kwargs...,
)
    if ndims(x) == 2
        auxiliary_shared_positions_threaded!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    else
        auxiliary_varying_positions_threaded!(sums, counts, x, u, sf_type, distance_bins; kwargs...)
    end
    return nothing
end

function auxiliary_shared_positions_threaded!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x_mat::AbstractMatrix{FT},
    u_batch::AbstractArray{FT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    distance_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x_mat)
    bdims = size(u_batch)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2

    u_flat = reshape(u_batch, N_dims, N, B)
    
    n_threads = Threads.nthreads()
    sums_thread = [zeros(FT, n_bins, B) for _ in 1:n_threads]
    counts_thread = [zeros(eltype(counts), n_bins, B) for _ in 1:n_threads]

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    Threads.@threads for k in 1:total_pairs
        tid = Threads.threadid()
        local_sums = sums_thread[tid]
        local_counts = counts_thread[tid]
        
        i, j = _pair_from_linear(k, N)
        X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_mat[d, i], vN))
        X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_mat[d, j], vN))
        dist = dist_metric(X1, X2)
        bin = SFH.digitize(dist, dist_be)
        if 1 <= bin <= n_bins
            r̂ = SFH.r̂(X1, X2, dist_metric, dist)
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                    U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                    val = sf_type(U2 - U1, r̂)
                    local_sums[bin, b] += val
                    local_counts[bin, b] += one(eltype(counts))
                end
                b0 = b1 + 1
            end
        end
    end
    
    sums_flat = reshape(sums, n_bins, B)
    counts_flat = reshape(counts, n_bins, B)
    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(eltype(counts)))
    for tid in 1:n_threads
        sums_flat .+= sums_thread[tid]
        counts_flat .+= counts_thread[tid]
    end
    return nothing
end

function auxiliary_varying_positions_threaded!(
    sums::AbstractArray{FT},
    counts::AbstractArray{<:Any},
    x_batch::AbstractArray{FT},
    u_batch::AbstractArray{FT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    distance_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x_batch)[1:2]
    bdims = size(u_batch)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2

    x_flat = reshape(x_batch, N_dims, N, B)
    u_flat = reshape(u_batch, N_dims, N, B)

    n_threads = Threads.nthreads()
    sums_thread = [zeros(FT, n_bins, B) for _ in 1:n_threads]
    counts_thread = [zeros(eltype(counts), n_bins, B) for _ in 1:n_threads]

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    Threads.@threads for k in 1:total_pairs
        tid = Threads.threadid()
        local_sums = sums_thread[tid]
        local_counts = counts_thread[tid]

        i, j = _pair_from_linear(k, N)
        b0 = 1
        while b0 <= B
            b1 = min(b0 + strip_width - 1, B)
            @inbounds @simd for b in b0:b1
                X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                dist = dist_metric(X1, X2)
                bin = SFH.digitize(dist, dist_be)
                if 1 <= bin <= n_bins
                    r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                    U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                    U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                    val = sf_type(U2 - U1, r̂)
                    local_sums[bin, b] += val
                    local_counts[bin, b] += one(eltype(counts))
                end
            end
            b0 = b1 + 1
        end
    end

    sums_flat = reshape(sums, n_bins, B)
    counts_flat = reshape(counts, n_bins, B)
    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(eltype(counts)))
    for tid in 1:n_threads
        sums_flat .+= sums_thread[tid]
        counts_flat .+= counts_thread[tid]
    end
    return nothing
end

"""
    threaded_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; strip_width=32)

Six-invariant-type single-pass 1D batch threaded.
"""
function threaded_calculate_structure_functions_single_pass!(
    sums::AbstractArray{FT, 3},
    counts::AbstractArray{CT, 3},
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT, CT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x)[1:2]
    bdims = size(u)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2
    fixed_x = ndims(x) == 2

    u_flat = reshape(u, N_dims, N, B)
    x_flat = fixed_x ? reshape(x, N_dims, N) : reshape(x, N_dims, N, B)

    n_threads = Threads.nthreads()
    sums_thread = [zeros(FT, SINGLE_PASS_N, n_bins, B) for _ in 1:n_threads]
    counts_thread = [zeros(CT, SINGLE_PASS_N, n_bins, B) for _ in 1:n_threads]

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    Threads.@threads for k in 1:total_pairs
        tid = Threads.threadid()
        local_sums = sums_thread[tid]
        local_counts = counts_thread[tid]

        i, j = _pair_from_linear(k, N)
        if fixed_x
            X1_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i], vN))
            X2_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j], vN))
            dist_fixed = dist_metric(X1_fixed, X2_fixed)
            bin_fixed = SFH.digitize(dist_fixed, dist_be)
            if 1 <= bin_fixed <= n_bins
                r̂_fixed = SFH.r̂(X1_fixed, X2_fixed, dist_metric, dist_fixed)
                b0 = 1
                while b0 <= B
                    b1 = min(b0 + strip_width - 1, B)
                    @inbounds @simd for b in b0:b1
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂_fixed)
                        du_T = SFH.mδu_t(du, r̂_fixed)
                        du_T2 = SFH.transverse_norm2(du, r̂_fixed)
                        _cpu_sp1d_accum!(local_sums, local_counts, bin_fixed, b, du_L, du_T, du_T2)
                    end
                    b0 = b1 + 1
                end
            end
        else
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                    X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                    dist = dist_metric(X1, X2)
                    bin = SFH.digitize(dist, dist_be)
                    if 1 <= bin <= n_bins
                        r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂)
                        du_T = SFH.mδu_t(du, r̂)
                        du_T2 = SFH.transverse_norm2(du, r̂)
                        _cpu_sp1d_accum!(local_sums, local_counts, bin, b, du_L, du_T, du_T2)
                    end
                end
                b0 = b1 + 1
            end
        end
    end

    sums_flat = reshape(sums, SINGLE_PASS_N, n_bins, B)
    counts_flat = reshape(counts, SINGLE_PASS_N, n_bins, B)
    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(CT))
    for tid in 1:n_threads
        sums_flat .+= sums_thread[tid]
        counts_flat .+= counts_thread[tid]
    end
    return nothing
end

"""
    threaded_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; strip_width=32)

Six-invariant-type SP2D batch threaded.
"""
function threaded_calculate_structure_functions_single_pass_2d!(
    sums::AbstractArray{FT, 4},
    counts::AbstractArray{CT, 4},
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins,
    value_bins::SinglePass2DValueBins;
    strip_width::Int = 32,
    kwargs...,
) where {FT, CT}
    dist_be = BinEdges(distance_bins)
    n_bins = n_histogram_bins(dist_be)
    N_dims, N = size(x)[1:2]
    bdims = size(u)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2
    fixed_x = ndims(x) == 2
    n_val = size(sums, 3)
    _validate_value_bins!(value_bins, n_val)

    u_flat = reshape(u, N_dims, N, B)
    x_flat = fixed_x ? reshape(x, N_dims, N) : reshape(x, N_dims, N, B)

    n_threads = Threads.nthreads()
    sums_thread = [zeros(FT, SINGLE_PASS_N, n_bins, n_val, B) for _ in 1:n_threads]
    counts_thread = [zeros(CT, SINGLE_PASS_N, n_bins, n_val, B) for _ in 1:n_threads]

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    Threads.@threads for k in 1:total_pairs
        tid = Threads.threadid()
        local_sums = sums_thread[tid]
        local_counts = counts_thread[tid]

        i, j = _pair_from_linear(k, N)
        if fixed_x
            X1_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i], vN))
            X2_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j], vN))
            dist_fixed = dist_metric(X1_fixed, X2_fixed)
            bin_fixed = SFH.digitize(dist_fixed, dist_be)
            if 1 <= bin_fixed <= n_bins
                r̂_fixed = SFH.r̂(X1_fixed, X2_fixed, dist_metric, dist_fixed)
                b0 = 1
                while b0 <= B
                    b1 = min(b0 + strip_width - 1, B)
                    @inbounds @simd for b in b0:b1
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂_fixed)
                        du_T = SFH.mδu_t(du, r̂_fixed)
                        du_L2 = du_L * du_L
                        du_T2 = SFH.transverse_norm2(du, r̂_fixed)
                        vals = (
                            du_L2 + du_T2,
                            du_L2,
                            du_T2,
                            du_L * (du_L2 + du_T2),
                            du_L * du_L2,
                            du_L * du_T2,
                        )
                        for t in 1:SINGLE_PASS_N
                            vb = _sp2d_value_bin_at(value_bins, t)
                            vbin = SFH.digitize(vals[t], vb)
                            n_val_t = length(vb) - 1
                            if 1 <= vbin <= n_val_t && vbin <= n_val
                                local_sums[t, bin_fixed, vbin, b] += vals[t]
                                local_counts[t, bin_fixed, vbin, b] += one(CT)
                            end
                        end
                    end
                    b0 = b1 + 1
                end
            end
        else
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                    X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                    dist = dist_metric(X1, X2)
                    bin = SFH.digitize(dist, dist_be)
                    if 1 <= bin <= n_bins
                        r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        du = U2 - U1
                        du_L = LA.dot(du, r̂)
                        du_T = SFH.mδu_t(du, r̂)
                        du_L2 = du_L * du_L
                        du_T2 = SFH.transverse_norm2(du, r̂)
                        vals = (
                            du_L2 + du_T2,
                            du_L2,
                            du_T2,
                            du_L * (du_L2 + du_T2),
                            du_L * du_L2,
                            du_L * du_T2,
                        )
                        for t in 1:SINGLE_PASS_N
                            vb = _sp2d_value_bin_at(value_bins, t)
                            vbin = SFH.digitize(vals[t], vb)
                            n_val_t = length(vb) - 1
                            if 1 <= vbin <= n_val_t && vbin <= n_val
                                local_sums[t, bin, vbin, b] += vals[t]
                                local_counts[t, bin, vbin, b] += one(CT)
                            end
                        end
                    end
                end
                b0 = b1 + 1
            end
        end
    end

    sums_flat = reshape(sums, SINGLE_PASS_N, n_bins, n_val, B)
    counts_flat = reshape(counts, SINGLE_PASS_N, n_bins, n_val, B)
    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(CT))
    for tid in 1:n_threads
        sums_flat .+= sums_thread[tid]
        counts_flat .+= counts_thread[tid]
    end
    return nothing
end

"""
    auxiliary_joint2d_threaded!(sums, counts, sf_type, x, u, distance_bins, value_bins; strip_width=32)

Single-type joint 2D batch threaded.
"""
function auxiliary_joint2d_threaded!(
    sums::AbstractArray{FT},
    counts::AbstractArray{CT},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT},
    u::AbstractArray{FT},
    distance_bins,
    value_bins;
    strip_width::Int = 32,
    kwargs...,
) where {FT, CT}
    dist_be = BinEdges(distance_bins)
    val_be = BinEdges(value_bins)
    n_dist = n_histogram_bins(dist_be)
    n_val = n_histogram_bins(val_be)
    N_dims, N = size(x)[1:2]
    bdims = size(u)[3:end]
    B = prod(bdims)
    total_pairs = N * (N - 1) ÷ 2
    fixed_x = ndims(x) == 2

    u_flat = reshape(u, N_dims, N, B)
    x_flat = fixed_x ? reshape(x, N_dims, N) : reshape(x, N_dims, N, B)

    n_threads = Threads.nthreads()
    sums_thread = [zeros(FT, n_dist, n_val, B) for _ in 1:n_threads]
    counts_thread = [zeros(CT, n_dist, n_val, B) for _ in 1:n_threads]

    vN = Val(N_dims)
    dist_metric = DI.Euclidean()

    Threads.@threads for k in 1:total_pairs
        tid = Threads.threadid()
        local_sums = sums_thread[tid]
        local_counts = counts_thread[tid]

        i, j = _pair_from_linear(k, N)
        if fixed_x
            X1_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i], vN))
            X2_fixed = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j], vN))
            dist_fixed = dist_metric(X1_fixed, X2_fixed)
            dbin_fixed = SFH.digitize(dist_fixed, dist_be)
            if 1 <= dbin_fixed <= n_dist
                r̂_fixed = SFH.r̂(X1_fixed, X2_fixed, dist_metric, dist_fixed)
                b0 = 1
                while b0 <= B
                    b1 = min(b0 + strip_width - 1, B)
                    @inbounds @simd for b in b0:b1
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        val = sf_type(U2 - U1, r̂_fixed)
                        vbin = SFH.digitize(val, val_be)
                        if 1 <= vbin <= n_val
                            local_sums[dbin_fixed, vbin, b] += val
                            local_counts[dbin_fixed, vbin, b] += one(CT)
                        end
                    end
                    b0 = b1 + 1
                end
            end
        else
            b0 = 1
            while b0 <= B
                b1 = min(b0 + strip_width - 1, B)
                @inbounds @simd for b in b0:b1
                    X1 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, i, b], vN))
                    X2 = SA.SVector{N_dims, FT}(ntuple(d -> x_flat[d, j, b], vN))
                    dist = dist_metric(X1, X2)
                    dbin = SFH.digitize(dist, dist_be)
                    if 1 <= dbin <= n_dist
                        r̂ = SFH.r̂(X1, X2, dist_metric, dist)
                        U1 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, i, b], vN))
                        U2 = SA.SVector{N_dims, FT}(ntuple(d -> u_flat[d, j, b], vN))
                        val = sf_type(U2 - U1, r̂)
                        vbin = SFH.digitize(val, val_be)
                        if 1 <= vbin <= n_val
                            local_sums[dbin, vbin, b] += val
                            local_counts[dbin, vbin, b] += one(CT)
                        end
                    end
                end
                b0 = b1 + 1
            end
        end
    end

    sums_flat = reshape(sums, n_dist, n_val, B)
    counts_flat = reshape(counts, n_dist, n_val, B)
    fill!(sums_flat, zero(FT))
    fill!(counts_flat, zero(CT))
    for tid in 1:n_threads
        sums_flat .+= sums_thread[tid]
        counts_flat .+= counts_thread[tid]
    end
    return nothing
end

# --- Unified CPU Batch Entry Points (Methods of serial_calculate_structure_function / threaded_calculate_structure_function) ---

@inline _component_vector_views(a, ::Val{D}) where {D} =
    ntuple(k -> view(a, k, :), Val(D))

function _serial_calculate_structure_function_point(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    vrsac::Val{RSAC},
    vD::Val{D};
    count_eltype::Type{CT} = UInt32,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
    kwargs...,
) where {FT1, FT2, RSAC, D, CT}
    x_tuple = _component_vector_views(x, vD)
    u_tuple = _component_vector_views(u, vD)
    OT = promote_type(float(FT1), float(FT2))
    output = zeros(OT, n_histogram_bins(distance_bins))
    counts = zeros(CT, n_histogram_bins(distance_bins))

    serial_calculate_structure_function!(
        output,
        counts,
        structure_function_type,
        x_tuple,
        u_tuple,
        distance_bins;
        distance_metric = distance_metric,
        verbose = verbose,
        show_progress = show_progress,
    )

    if RSAC
        return SFO.StructureFunctionSumsAndCounts(
            structure_function_type,
            distance_bins,
            output,
            counts,
        )
    end

    output_div = similar(output)
    for k in eachindex(output)
        c = counts[k]
        output_div[k] = iszero(c) ? OT(NaN) : output[k] / c
    end
    return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
end

function serial_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    ::Val{RSAC};
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, RSAC, CT}
    if ndims(u) >= 3
        dist_be = BinEdges(distance_bins)
        n_bins = n_histogram_bins(dist_be)
        bdims = batch_dims(u)
        FT = promote_type(float(FT1), float(FT2))
        sums = zeros(FT, n_bins, bdims...)
        counts = zeros(CT, n_bins, bdims...)
        auxiliary_structure_function!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        if RSAC
            return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
        else
            output_div = similar(sums)
            @inbounds for k in eachindex(sums)
                c = counts[k]
                output_div[k] = iszero(c) ? FT(NaN) : sums[k] / c
            end
            return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
        end
    end
    # Point-field route:
    D = size(x, 1)
    D == 2 && return _serial_calculate_structure_function_point(
        structure_function_type,
        x,
        u,
        distance_bins,
        Val(RSAC),
        Val(2);
        count_eltype = count_eltype,
        kwargs...,
    )
    D == 3 && return _serial_calculate_structure_function_point(
        structure_function_type,
        x,
        u,
        distance_bins,
        Val(RSAC),
        Val(3);
        count_eltype = count_eltype,
        kwargs...,
    )
    throw(DimensionMismatch("expected spatial/velocity dimension D=2 or D=3 on axis 1; got D=$D"))
end

function threaded_calculate_structure_function(
    structure_function_type::SFT.AbstractPairwiseStructureFunctionType,
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector,
    ::Val{RSAC};
    count_eltype::Type{CT} = UInt32,
    kwargs...,
) where {FT1 <: Number, FT2 <: Number, RSAC, CT}
    if ndims(u) >= 3
        dist_be = BinEdges(distance_bins)
        n_bins = n_histogram_bins(dist_be)
        bdims = batch_dims(u)
        FT = promote_type(float(FT1), float(FT2))
        sums = zeros(FT, n_bins, bdims...)
        counts = zeros(CT, n_bins, bdims...)
        auxiliary_structure_function_threaded!(sums, counts, structure_function_type, x, u, distance_bins; kwargs...)
        if RSAC
            return SFO.StructureFunctionSumsAndCounts(structure_function_type, distance_bins, sums, counts)
        else
            output_div = similar(sums)
            @inbounds for k in eachindex(sums)
                c = counts[k]
                output_div[k] = iszero(c) ? FT(NaN) : sums[k] / c
            end
            return SFO.StructureFunction(structure_function_type, distance_bins, output_div)
        end
    end
    throw(ArgumentError("Threaded backend is unavailable for non-batch inputs. Load the OhMyThreads extension or use backend=SerialBackend()."))
end
