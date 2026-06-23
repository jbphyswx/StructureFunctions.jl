# Single-Pass 1D and 2D Calculations

# --- Helmholtz Decomposition Derived Quantities ---

"""
    helmholtz_decompose_2d(distance_bins, sums, counts)

Run the 2D isotropic Helmholtz decomposition from native single-pass sums/counts.
Rows 2 and 3 must contain `L2SF` and `T2SF`.
"""
function helmholtz_decompose_2d(
    distance_bins::AbstractVector{FT3},
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
) where {OT, CT, FT3}
    return helmholtz_decompose_2d(
        distance_bins,
        @view(sums[2, :]),
        @view(counts[2, :]),
        @view(sums[3, :]),
        @view(counts[3, :]),
    )
end

"""
    helmholtz_decompose_2d(distance_bins, L2_sums, L2_counts, T2_sums, T2_counts)

Run the 2D isotropic Helmholtz decomposition using the trapezoidal rule over
binned longitudinal/transverse second-order structure functions. This implements
the cumulative integral equations described by Lindborg (JFM 2015) and
Bühler, Callies, and Ferrari (JFM 2014).
"""
function helmholtz_decompose_2d(
    distance_bins::AbstractVector{FT3},
    L2_sums::AbstractVector{OT},
    L2_counts::AbstractVector{CT},
    T2_sums::AbstractVector,
    T2_counts::AbstractVector,
) where {OT, CT, FT3}
    length(L2_sums) == length(T2_sums) ||
        throw(DimensionMismatch("L2_sums and T2_sums must have the same length"))
    length(L2_counts) == length(L2_sums) ||
        throw(DimensionMismatch("L2_counts must match L2_sums length"))
    length(T2_counts) == length(T2_sums) ||
        throw(DimensionMismatch("T2_counts must match T2_sums length"))
    n_bins = length(L2_sums)
    length(distance_bins) == n_bins + 1 ||
        throw(DimensionMismatch("distance_bins must have length n_bins + 1"))

    # Calculate bin midpoints from log-spaced edges
    min_log_dist = log(distance_bins[1])
    max_log_dist = log(distance_bins[end])
    log_step = (max_log_dist - min_log_dist) / n_bins
    
    bin_mids = zeros(FT3, n_bins)
    for k in 1:n_bins
        bin_mids[k] = exp(min_log_dist + (k - 0.5f0) * log_step)
    end
    
    # Compute normalized second-order longitudinal (2) and transverse (3) functions
    D_LL = L2_sums ./ max.(L2_counts, 1)
    D_TT = T2_sums ./ max.(T2_counts, 1)
    
    # Evaluate cumulative trapezoidal integral
    I = zeros(OT, n_bins)
    for k in 2:n_bins
        F_prev = (D_TT[k-1] - D_LL[k-1]) / bin_mids[k-1]
        F_curr = (D_TT[k] - D_LL[k]) / bin_mids[k]
        ds = bin_mids[k] - bin_mids[k-1]
        I[k] = I[k-1] + 0.5f0 * (F_prev + F_curr) * ds
    end
    
    rotational_sums = zeros(OT, n_bins)
    divergent_sums = zeros(OT, n_bins)
    rotational_counts = copy(L2_counts)
    divergent_counts = copy(L2_counts)

    for k in 1:n_bins
        D_rot = D_TT[k] + bin_mids[k] * I[k]
        D_div = D_LL[k] - bin_mids[k] * I[k]
        
        rotational_sums[k] = D_rot * rotational_counts[k]
        divergent_sums[k] = D_div * divergent_counts[k]
    end

    return SFO.HelmholtzDecomposition2D(
        distance_bins,
        rotational_sums,
        rotational_counts,
        divergent_sums,
        divergent_counts,
        collect(D_LL),
        collect(D_TT),
    )
end

"""
    append_helmholtz_rotational_divergent_rows(sums, counts, distance_bins)

Append Helmholtz-derived rotational/divergent rows to native six-row single-pass
sums and counts. Rows 1 through 6 are copied unchanged; rows 7 and 8 are
rotational and divergent second-order components.
"""
function append_helmholtz_rotational_divergent_rows(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    distance_bins::AbstractVector{FT3},
) where {OT, CT, FT3}
    n_bins = size(sums, 2)
    size(sums, 1) == SINGLE_PASS_N ||
        throw(DimensionMismatch("sums must have $SINGLE_PASS_N rows"))
    size(counts) == size(sums) ||
        throw(DimensionMismatch("counts must match sums shape"))
    decomposition = helmholtz_decompose_2d(distance_bins, sums, counts)
    final_sums = zeros(OT, SINGLE_PASS_WITH_HELMHOLTZ_N, n_bins)
    final_counts = zeros(CT, SINGLE_PASS_WITH_HELMHOLTZ_N, n_bins)

    final_sums[1:SINGLE_PASS_N, :] .= sums
    final_counts[1:SINGLE_PASS_N, :] .= counts
    final_sums[7, :] .= decomposition.rotational_sums
    final_counts[7, :] .= decomposition.rotational_counts
    final_sums[8, :] .= decomposition.divergent_sums
    final_counts[8, :] .= decomposition.divergent_counts

    return final_sums, final_counts
end

"""
    marginalize_sp2d_then_append_helmholtz_rows(sums_6, counts_6, distance_bins)

Marginalize six native invariant 2D joint histograms to 1D, then append
rotational/divergent rows via [`append_helmholtz_rotational_divergent_rows`](@ref).
Returns ``(sums_8, counts_8)``.
"""
function marginalize_sp2d_then_append_helmholtz_rows(
    sums_6::AbstractArray{OT, 3},
    counts_6::AbstractArray{CT, 3},
    distance_bins::AbstractVector{FT3},
) where {OT, CT, FT3}
    sums_1d = dropdims(sum(sums_6, dims = 3), dims = 3)
    counts_1d = dropdims(sum(counts_6, dims = 3), dims = 3)
    return append_helmholtz_rotational_divergent_rows(sums_1d, counts_1d, distance_bins)
end

# --- 1D Single Pass Functions ---

"""
    serial_calculate_structure_functions_single_pass(x, u, distance_bins, sums, counts)

Zero ``sums``/``counts`` then accumulate six invariant native 1D structure
functions on one thread.
For allocation-free reuse, prefer [`calculate_structure_functions_single_pass!`](@ref).
"""
function serial_calculate_structure_functions_single_pass(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    fill!(sums, zero(OT))
    fill!(counts, 0)
    return calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
end

"""Serial pair-loop accumulation into native ``(6, n_bins)`` buffers (no allocation)."""
function _accumulate_single_pass_1d!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    D = size(x, 1)
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    size(sums) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins); got $(size(sums))"))
    size(counts) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("counts must have shape ($SINGLE_PASS_N, n_bins); got $(size(counts))"))
    vD = Val(D)

    for i in 1:n_points
        x_i = SA.SVector{D, FT1}(ntuple(d -> x[d, i], vD))
        u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))

        for j in (i + 1):n_points
            x_j = SA.SVector{D, FT1}(ntuple(d -> x[d, j], vD))

            r = distance_metric(x_i, x_j)
            bin_idx = SFH.digitize(r, distance_bins)

            if 1 <= bin_idx <= n_bins
                u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                du = u_j - u_i

                rh = SFH.r̂(x_i, x_j, distance_metric, r)
                du_L = LA.dot(du, rh)
                du_L2 = du_L * du_L
                du_norm2 = LA.dot(du, du)
                du_T2 = du_norm2 - du_L2

                @inbounds sums[1, bin_idx] += du_norm2
                @inbounds sums[2, bin_idx] += du_L2
                @inbounds sums[3, bin_idx] += du_T2
                @inbounds sums[4, bin_idx] += du_L * du_norm2
                @inbounds sums[5, bin_idx] += du_L * du_L2
                @inbounds sums[6, bin_idx] += du_L * du_T2

                @inbounds for t in 1:SINGLE_PASS_N
                    counts[t, bin_idx] += one(CT)
                end
            end
        end
    end

    return sums, counts
end

function _dispatch_single_pass end
function _dispatch_single_pass! end

"""
    calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; backend=SerialBackend(), kwargs...)

Accumulate into pre-allocated ``(6, n_bins)`` buffers using the requested execution backend.
"""
function calculate_structure_functions_single_pass!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    backend::AbstractExecutionBackend = SerialBackend(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _validate_array_shape(x, u)
    _dispatch_single_pass!(backend, sums, counts, x, u, distance_bins; kwargs...)
    return sums, counts
end

function _dispatch_single_pass!(
    ::SerialBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    return _accumulate_single_pass_1d!(sums, counts, x, u, distance_bins; kwargs...)
end

function _dispatch_single_pass!(
    ::ThreadedBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("Threaded in-place single-pass is unavailable. Load OhMyThreads or use backend=SerialBackend()."))
end

function _dispatch_single_pass!(
    ::DistributedBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("Distributed in-place single-pass is unavailable. Load Distributed or use backend=SerialBackend()."))
end

function _dispatch_single_pass!(
    ::GPUBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("GPU in-place single-pass is unavailable. Load GPUExt or use backend=SerialBackend()."))
end

function _dispatch_single_pass!(
    ::AutoBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass!(DistributedBackend(), sums, counts, x, u, distance_bins; kwargs...)
    end
    if Threads.nthreads() > 1 && _threaded_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass!(ThreadedBackend(), sums, counts, x, u, distance_bins; kwargs...)
    end
    return _dispatch_single_pass!(SerialBackend(), sums, counts, x, u, distance_bins; kwargs...)
end

function _dispatch_single_pass(
    ::SerialBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    thread_sums = nothing,
    thread_counts = nothing,
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    
    ts = isnothing(thread_sums) ? zeros(OT, SINGLE_PASS_N, n_bins) : thread_sums
    tc = isnothing(thread_counts) ? zeros(CT, SINGLE_PASS_N, n_bins) : thread_counts
    
    # Cast to Matrix explicitly to run the serial matrix implementation
    x_mat = reshape(x, size(x, 1), size(x, 2))
    u_mat = reshape(u, size(u, 1), size(u, 2))
    sums, counts = serial_calculate_structure_functions_single_pass(x_mat, u_mat, distance_bins, ts, tc; kwargs...)
    return append_helmholtz_rotational_divergent_rows(sums, counts, distance_bins)
end

function _dispatch_single_pass(
    ::SerialBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    serial_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass(
    ::ThreadedBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    return _dispatch_single_pass(ThreadedBackend(), x, u, distance_bins; count_eltype = CT, kwargs...)
end

function _dispatch_single_pass(
    ::ThreadedBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, auxiliary_dims...)
    threaded_calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass(::ThreadedBackend, args...; kwargs...)
    throw(ArgumentError("Threaded single-pass backend is unavailable. Load the OhMyThreads extension or use backend=SerialBackend()."))
end

function _dispatch_single_pass(::DistributedBackend, args...; kwargs...)
    throw(ArgumentError("Distributed single-pass backend is unavailable. Load the Distributed/SharedArrays extension or use backend=SerialBackend()."))
end

function _dispatch_single_pass(::GPUBackend, args...; kwargs...)
    throw(ArgumentError("GPU single-pass backend is unavailable. Load the GPUExt extension or use backend=SerialBackend()."))
end

function _dispatch_single_pass(
    backend::GPUBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...
)
    return _dispatch_single_pass(backend, x, u, distance_bins; kwargs...)
end

function _dispatch_single_pass(
    backend::DistributedBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...
)
    return _dispatch_single_pass(backend, x, u, distance_bins; kwargs...)
end

_threaded_single_pass_available(x, u, distance_bins) = hasmethod(
    _dispatch_single_pass,
    Tuple{ThreadedBackend, typeof(x), typeof(u), typeof(distance_bins)}
)

_distributed_single_pass_available(x, u, distance_bins) = hasmethod(
    _dispatch_single_pass,
    Tuple{DistributedBackend, typeof(x), typeof(u), typeof(distance_bins)}
)

function _dispatch_single_pass(::AutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector; kwargs...)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass(DistributedBackend(), shape, x, u, distance_bins; kwargs...)
    end
    
    if has_auxiliary_axes(shape)
        if Threads.nthreads() > 1
            return _dispatch_single_pass(ThreadedBackend(), shape, x, u, distance_bins; kwargs...)
        end
        return _dispatch_single_pass(SerialBackend(), shape, x, u, distance_bins; kwargs...)
    end
    
    if Threads.nthreads() > 1 &&
       _threaded_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass(ThreadedBackend(), shape, x, u, distance_bins; kwargs...)
    end
    
    return _dispatch_single_pass(SerialBackend(), shape, x, u, distance_bins; kwargs...)
end

"""
    calculate_structure_functions_single_pass(x, u, distance_bins; backend=AutoBackend(), kwargs...)

Primary entrypoint for computing six invariant native structure functions in one pair pass.
"""
function calculate_structure_functions_single_pass(
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3};
    backend::AbstractExecutionBackend = AutoBackend(),
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    shape = _validate_array_shape(x, u)
    return _dispatch_single_pass(
        backend, shape, x, u, distance_bins;
        count_eltype = count_eltype,
        kwargs...,
    )
end


# --- 2D Single Pass Functions ---

function serial_calculate_structure_functions_single_pass_2d(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins,
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3};
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    fill!(sums_3d, zero(OT))
    fill!(counts_3d, 0)
    return calculate_structure_functions_single_pass_2d!(
        sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...
    )
end

function calculate_structure_functions_single_pass_2d!(
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    backend::AbstractExecutionBackend = SerialBackend(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _validate_array_shape(x, u)
    _dispatch_single_pass_2d!(
        backend, sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...
    )
    return sums_3d, counts_3d
end

"""Serial pair-loop accumulation into native ``(6, n_bins, n_val)`` buffers (no allocation)."""
function _accumulate_single_pass_2d!(
    sums_3d::AbstractArray{OT, 3},
    counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    distance_metric::DI.PreMetric = DI.Euclidean(),
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    D = size(x, 1)
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    n_val = size(sums_3d, 3)
    size(sums_3d, 1) == SINGLE_PASS_N && size(sums_3d, 2) == n_bins ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts and sums must have the same shape"))
    _validate_value_bins!(value_bins, n_val)
    vD = Val(D)

    for i in 1:n_points
        x_i = SA.SVector{D, FT1}(ntuple(d -> x[d, i], vD))
        u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))

        for j in (i + 1):n_points
            x_j = SA.SVector{D, FT1}(ntuple(d -> x[d, j], vD))

            r = distance_metric(x_i, x_j)
            bin_idx = SFH.digitize(r, distance_bins)

            if 1 <= bin_idx <= n_bins
                u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                du = u_j - u_i

                rh = SFH.r̂(x_i, x_j, distance_metric, r)
                du_L = LA.dot(du, rh)
                du_L2 = du_L * du_L
                du_norm2 = LA.dot(du, du)
                du_T2 = du_norm2 - du_L2

                vals = (
                    du_norm2,
                    du_L2,
                    du_T2,
                    du_L * du_norm2,
                    du_L * du_L2,
                    du_L * du_T2,
                )

                for t in 1:SINGLE_PASS_N
                    vb = _sp2d_value_bin_at(value_bins, t)
                    vbin = SFH.digitize(vals[t], vb)
                    n_val_t = length(vb) - 1
                    if 1 <= vbin <= n_val_t && vbin <= n_val
                        @inbounds sums_3d[t, bin_idx, vbin] += vals[t]
                        @inbounds counts_3d[t, bin_idx, vbin] += 1
                    end
                end
            end
        end
    end

    return sums_3d, counts_3d
end

function _dispatch_single_pass_2d!(
    ::SerialBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    return _accumulate_single_pass_2d!(sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d!(
    ::ThreadedBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    throw(ArgumentError("Threaded in-place 2D single-pass is unavailable. Load OhMyThreads or use backend=SerialBackend()."))
end

function _dispatch_single_pass_2d!(
    ::DistributedBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    throw(ArgumentError("Distributed in-place 2D single-pass is unavailable. Load Distributed or use backend=SerialBackend()."))
end

function _dispatch_single_pass_2d!(
    backend::GPUBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    gpu_calculate_structure_functions_single_pass_2d!(sums_3d, counts_3d, backend.backend, x, u, distance_bins, value_bins; kwargs...)
    return sums_3d, counts_3d
end

function _dispatch_single_pass_2d!(
    ::AutoBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_single_pass_2d_available(x, u, distance_bins, value_bins)
        return _dispatch_single_pass_2d!(DistributedBackend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
    end
    if Threads.nthreads() > 1 && _threaded_single_pass_2d_available(x, u, distance_bins, value_bins)
        return _dispatch_single_pass_2d!(ThreadedBackend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
    end
    return _dispatch_single_pass_2d!(SerialBackend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
end

# Specific method for matrix inputs to handle standard non-batch 2D single-pass
function _dispatch_single_pass_2d_matrix(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    thread_sums = nothing,
    thread_counts = nothing,
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = length(value_bins isa Tuple ? value_bins[1] : value_bins) - 1
    _validate_value_bins!(value_bins, n_val)

    ts = isnothing(thread_sums) ? zeros(OT, SINGLE_PASS_N, n_bins, n_val) : thread_sums
    tc = isnothing(thread_counts) ? zeros(CT, SINGLE_PASS_N, n_bins, n_val) : thread_counts

    return serial_calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins, ts, tc; kwargs...
    )
end

function _dispatch_single_pass_2d(
    ::SerialBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    return _dispatch_single_pass_2d_matrix(
        x, u, distance_bins, value_bins;
        count_eltype = count_eltype, kwargs...
    )
end

function _dispatch_single_pass_2d(
    ::SerialBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = length(value_bins isa Tuple ? value_bins[1] : value_bins) - 1
    _validate_value_bins!(value_bins, n_val)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    serial_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass_2d(::ThreadedBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    throw(ArgumentError("Threaded 2D single-pass backend is unavailable. Load the OhMyThreads extension or use backend=SerialBackend()."))
end

function _dispatch_single_pass_2d(
    backend::ThreadedBackend,
    ::PointField,
    x::AbstractMatrix,
    u::AbstractMatrix,
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins;
    kwargs...
)
    return _dispatch_single_pass_2d(backend, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d(
    ::ThreadedBackend,
    ::Union{SharedPositionField, VaryingPositionField},
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = length(distance_bins) - 1
    n_val = length(value_bins isa Tuple ? value_bins[1] : value_bins) - 1
    _validate_value_bins!(value_bins, n_val)
    auxiliary_dims = size(u)[3:end]
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val, auxiliary_dims...)
    threaded_calculate_structure_functions_single_pass_2d!(sums, counts, x, u, distance_bins, value_bins; kwargs...)
    return (sums = sums, counts = counts)
end

function _dispatch_single_pass_2d(::DistributedBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    throw(ArgumentError("Distributed 2D single-pass backend is unavailable. Load the Distributed extension or use backend=SerialBackend()."))
end

function _dispatch_single_pass_2d(
    backend::DistributedBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins;
    kwargs...
)
    return _dispatch_single_pass_2d(backend, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d(backend::GPUBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    return gpu_calculate_structure_functions_single_pass_2d(backend.backend, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d(
    backend::GPUBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins;
    kwargs...
)
    return _dispatch_single_pass_2d(backend, x, u, distance_bins, value_bins; kwargs...)
end

_threaded_single_pass_2d_available(x, u, distance_bins, value_bins) = hasmethod(
    _dispatch_single_pass_2d,
    Tuple{ThreadedBackend, typeof(x), typeof(u), typeof(distance_bins), typeof(value_bins)},
)

_distributed_single_pass_2d_available(x, u, distance_bins, value_bins) = hasmethod(
    _dispatch_single_pass_2d,
    Tuple{DistributedBackend, typeof(x), typeof(u), typeof(distance_bins), typeof(value_bins)},
)

function _dispatch_single_pass_2d(::AutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_single_pass_2d_available(x, u, distance_bins, value_bins)
        return _dispatch_single_pass_2d(DistributedBackend(), shape, x, u, distance_bins, value_bins; kwargs...)
    end
    if has_auxiliary_axes(shape)
        if Threads.nthreads() > 1
            return _dispatch_single_pass_2d(ThreadedBackend(), shape, x, u, distance_bins, value_bins; kwargs...)
        end
        return _dispatch_single_pass_2d(SerialBackend(), shape, x, u, distance_bins, value_bins; kwargs...)
    end
    if Threads.nthreads() > 1 &&
       _threaded_single_pass_2d_available(x, u, distance_bins, value_bins)
        return _dispatch_single_pass_2d(ThreadedBackend(), shape, x, u, distance_bins, value_bins; kwargs...)
    end
    return _dispatch_single_pass_2d(SerialBackend(), shape, x, u, distance_bins, value_bins; kwargs...)
end

"""
    calculate_structure_functions_single_pass_2d(x, u, distance_bins, value_bins; backend=AutoBackend(), kwargs...)

Primary entrypoint for computing six invariant 2D joint structure function histograms in one pass.
"""
function calculate_structure_functions_single_pass_2d(
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    backend::AbstractExecutionBackend = AutoBackend(),
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    shape = _validate_array_shape(x, u)
    return _dispatch_single_pass_2d(
        backend, shape, x, u, distance_bins, value_bins;
        count_eltype = count_eltype, kwargs...,
    )
end
