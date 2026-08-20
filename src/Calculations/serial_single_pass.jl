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

    # Geometric midpoint of each bin's actual edges; NaN where r <= 0 has no midpoint.
    bin_mids = zeros(FT3, n_bins)
    for k in 1:n_bins
        lo, hi = distance_bins[k], distance_bins[k + 1]
        bin_mids[k] = lo > zero(FT3) ? sqrt(lo * hi) : FT3(NaN)
    end

    # Empty bins are NaN, never 0: the integral below is cumulative.
    D_LL = _bin_average(L2_sums, L2_counts)
    D_TT = _bin_average(T2_sums, T2_counts)
    
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
    D = size(u, 1)
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    size(sums) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins); got $(size(sums))"))
    size(counts) == (SINGLE_PASS_N, n_bins) ||
        throw(DimensionMismatch("counts must have shape ($SINGLE_PASS_N, n_bins); got $(size(counts))"))
    # Fast path: Euclidean + D ∈ (2,3) via the SIMD compute/scatter split (vectorizes the
    # per-pair du_L / |du|² compute over j; the 6-way histogram scatter stays scalar).
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _sp_simd_run!(sums, counts, x, u, BinEdges(distance_bins), D == 2 ? Val(2) : Val(3))
        return sums, counts
    end

    vW, vD = _pair_dims(distance_metric, D)
    _sp1d_pairs!(sums, counts, x, u, BinEdges(distance_bins), vW, vD,
        distance_metric, n_bins, 1:n_points)
    return sums, counts
end

"""
    _sp1d_derive_rows!(sums, counts)

Fill the two derived invariant rows and replicate the shared count of a `(SINGLE_PASS_N, n_bins)`
accumulator.

`T2 = S2 - L2` and `L1T2 = S3 - L3` hold for every pair, and a bin is a sum, so the pair loops store
only rows 1, 2, 4, 5 — four stores per pair instead of six — and these two are differenced once per
call. The count is identical across all six rows, so it is accumulated into row 1 and broadcast here.

Uses `=`, never `+=`, so it is **idempotent**: correct whether a kernel runs once or many times over
the same buffer, and correct under threaded/partial reduction because the derivation is linear
(`Σ(S2-L2) = ΣS2 - ΣL2`). That is why it belongs at the end of each pair kernel rather than at the
callers' assembly points, of which there are more than twenty.
"""
@inline function _sp1d_derive_rows!(sums::AbstractMatrix, counts::AbstractMatrix)
    @inbounds for b in axes(sums, 2)
        sums[3, b] = sums[1, b] - sums[2, b]
        sums[6, b] = sums[4, b] - sums[5, b]
        c = counts[1, b]
        for t in 2:SINGLE_PASS_N
            counts[t, b] = c
        end
    end
    return nothing
end

"""
    _pf_sp_simd_pairs!(sums, counts, xc, uc, dist_be, ::Val{D}, distbuf, duLbuf, dn2buf, irange)

Single-pass (6 invariants) point-field SIMD compute/scatter kernel over outer indices `irange`.
For each `i`: `@simd` over `j>i` computes distance, `du_L = du·r̂`, and `|du|²` into buffers
(contiguous components ⇒ packed loads, no scatter ⇒ vectorizes), then a scalar loop digitizes
and scatters the 6 invariants. Like `_pf_simd_pairs!`, the loop must live in this one kernel
(not a per-`i` helper) for the `@simd` to vectorize. Shared by serial + threaded.
"""
function _pf_sp_simd_pairs!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, ::Val{D},
    keybuf::AbstractVector, duLbuf::AbstractVector, dn2buf::AbstractVector,
    idxbuf::AbstractVector{Int32}, irange,
) where {OT, CT, D}
    N = length(xc[1])
    nb = n_histogram_bins(plan)
    FTx = eltype(xc[1])
    @inbounds for i in irange
        Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
        Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
        @simd for j in (i + 1):N
            Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
            dx = Xj - Xi
            r2 = LA.dot(dx, dx)
            du = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D))) - Ui
            # δu_L needs r, so one reciprocal-sqrt stays; it vectorizes.
            inv_r = inv(sqrt(r2))
            keybuf[j] = digitize_key(plan, r2)
            duLbuf[j] = LA.dot(du, dx) * inv_r
            dn2buf[j] = LA.dot(du, du)
            if has_vector_index(plan)
                idxbuf[j] = squared_approx_index(plan, r2)
            end
        end
        for j in (i + 1):N
            bin = squared_bin(plan, keybuf[j], idxbuf[j])
            if 1 <= bin <= nb
                duL = duLbuf[j]
                dn2 = dn2buf[j]
                duL2 = duL * duL
                sums[1, bin] += dn2
                sums[2, bin] += duL2
                sums[4, bin] += duL * dn2
                sums[5, bin] += duL * duL2
                counts[1, bin] += one(CT)
            end
        end
    end
    _sp1d_derive_rows!(sums, counts)
    return nothing
end

# Serial driver: the full outer range through the same per-worker kernel.
function _sp_simd_run!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, ::Val{D},
) where {OT, CT, D}
    N = size(x, 2)
    return _sp_simd_partial!(sums, counts, x, u, dist_be, Val(D), 1:(N - 1))
end

"""
    _partial_single_pass_1d(x, u, distance_bins, ilist; distance_metric, count_eltype)

Six-invariant partial sums/counts over an explicit outer-index list, for one distributed worker or
MPI rank. Euclidean `D ∈ {2,3}` takes the SIMD kernel; other metrics use the scalar loop. The
accumulator element type comes from the inputs.
"""
function _partial_single_pass_1d(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    ilist;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    D = size(u, 1)
    nb = n_histogram_bins(distance_bins)
    sums = zeros(OT, SINGLE_PASS_N, nb)
    counts = zeros(CT, SINGLE_PASS_N, nb)
    dist_be = BinEdges(distance_bins)

    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _sp_simd_partial!(sums, counts, x, u, dist_be, D == 2 ? Val(2) : Val(3), ilist)
        return sums, counts
    end

    vW, vD = _pair_dims(distance_metric, D)
    _sp1d_pairs!(sums, counts, x, u, dist_be, vW, vD, distance_metric, nb, ilist)
    return sums, counts
end

"""
    _sp1d_pairs!(sums, counts, x, u, dist_be, ::Val{W}, ::Val{D}, metric, n_bins, ilist)

Six-invariant scalar pair loop over the outer indices `ilist`, for geometries with no SIMD fast
path. `W` is the coordinate width and `D` the velocity width; both are type parameters so the
`SVector`s stay concrete inside the loop.
"""
function _sp1d_pairs!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1}, u::AbstractMatrix{FT2},
    dist_be, ::Val{W}, ::Val{D}, distance_metric, n_bins::Int, ilist,
) where {OT, CT, FT1, FT2, W, D}
    n_points = size(x, 2)
    vW = Val(W)
    vD = Val(D)
    geom = SFH.pair_geometry_for(distance_metric, vD)
    @inbounds for i in ilist
        x_i = SA.SVector{W, FT1}(ntuple(d -> x[d, i], vW))
        u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))
        for j in (i + 1):n_points
            x_j = SA.SVector{W, FT1}(ntuple(d -> x[d, j], vW))
            ok, r, frame = SFH.pair_frame(geom, x_i, x_j)
            bin = SFH.digitize(r, dist_be)
            if ok && 1 <= bin <= n_bins
                u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                du, rh = SFH.pair_increments(geom, frame, r, x_i, x_j, u_i, u_j)
                duL = LA.dot(du, rh)
                duL2 = duL * duL
                dn2 = LA.dot(du, du)
                sums[1, bin] += dn2
                sums[2, bin] += duL2
                sums[4, bin] += duL * dn2
                sums[5, bin] += duL * duL2
                counts[1, bin] += one(CT)
            end
        end
    end
    _sp1d_derive_rows!(sums, counts)
    return nothing
end

"""Run [`_pf_sp_simd_pairs!`](@ref) over an explicit outer-index list, with this worker's buffers."""
function _sp_simd_partial!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, ::Val{D}, ilist,
) where {OT, CT, D}
    xc = ntuple(d -> collect(view(x, d, :)), Val(D))
    uc = ntuple(d -> collect(view(u, d, :)), Val(D))
    N = length(xc[1])
    FTx = eltype(xc[1])
    keybuf = Vector{FTx}(undef, N)
    duLbuf = Vector{OT}(undef, N)
    dn2buf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    plan = squared_digitize_plan(dist_be)
    _pf_sp_simd_pairs!(sums, counts, xc, uc, plan, Val(D), keybuf, duLbuf, dn2buf, idxbuf, ilist)
    return nothing
end

"""
    _partial_single_pass_2d(x, u, distance_bins, value_bins, ilist; distance_metric, count_eltype)

Six-invariant 2D joint partial sums/counts over an explicit outer-index list, for one distributed
worker or MPI rank. The accumulator element type comes from the inputs.
"""
function _partial_single_pass_2d(
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins,
    ilist;
    distance_metric::DI.PreMetric = DI.Euclidean(),
    count_eltype::Type{CT} = UInt32,
) where {FT1 <: Number, FT2 <: Number, CT}
    OT = promote_type(float(FT1), float(FT2))
    n_bins = n_histogram_bins(distance_bins)
    n_val = length(_sp2d_value_bin_at(value_bins, 1)) - 1
    _validate_value_bins!(value_bins, n_val)
    _assert_counts_representable(CT, size(x, 2))
    sums = zeros(OT, SINGLE_PASS_N, n_bins, n_val)
    counts = zeros(CT, SINGLE_PASS_N, n_bins, n_val)
    _sp2d_accumulate_range!(sums, counts, x, u, BinEdges(distance_bins), value_bins,
        distance_metric, n_bins, n_val, ilist)
    return sums, counts
end

function _dispatch_single_pass end
function _dispatch_single_pass! end

"""
    calculate_structure_functions_single_pass!(sums, counts, x, u, distance_bins; backend=CB.SerialBackend(), kwargs...)

Accumulate into pre-allocated ``(6, n_bins)`` buffers using the requested execution backend.
"""
function calculate_structure_functions_single_pass!(
    sums::AbstractMatrix{OT},
    counts::AbstractMatrix{CT},
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _validate_array_shape(x, u, distance_metric)
    _dispatch_single_pass!(backend, sums, counts, x, u, distance_bins; distance_metric, kwargs...)
    return sums, counts
end

function _dispatch_single_pass!(
    ::CB.AbstractSerialBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    return _accumulate_single_pass_1d!(sums, counts, x, u, distance_bins; kwargs...)
end

function _dispatch_single_pass!(
    ::CB.AbstractThreadedBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("Threaded in-place single-pass is unavailable. Load OhMyThreads or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass!(
    ::CB.AbstractDistributedBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("Distributed in-place single-pass is unavailable. Load Distributed or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass!(
    ::CB.AbstractGPUBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    throw(ArgumentError("GPU in-place single-pass is unavailable. Load GPUExt or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass!(
    ::CB.AbstractAutoBackend, sums::AbstractMatrix, counts::AbstractMatrix, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector; kwargs...
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass!(CB.DistributedBackend(), sums, counts, x, u, distance_bins; kwargs...)
    end
    if Threads.nthreads() > 1 && _threaded_single_pass_available(x, u, distance_bins)
        return _dispatch_single_pass!(CB.ThreadedBackend(), sums, counts, x, u, distance_bins; kwargs...)
    end
    return _dispatch_single_pass!(CB.SerialBackend(), sums, counts, x, u, distance_bins; kwargs...)
end

function _dispatch_single_pass(
    ::CB.AbstractSerialBackend,
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
    # `serial_…single_pass` fills `ts`/`tc` in place; return those concretely-typed buffers
    # (Matrix{OT}/Matrix{CT}) rather than the mutating chain's abstractly-typed return value, so
    # the result type is concrete (the chain infers `counts::Matrix`, losing the element type).
    serial_calculate_structure_functions_single_pass(x_mat, u_mat, distance_bins, ts, tc; kwargs...)
    # Return the raw six-row accumulator; the public wrapper builds the Helmholtz entry once
    # (avoids the old double-compute and the 8-row copy, and matches the batched path's shape).
    return (sums = ts, counts = tc)
end

function _dispatch_single_pass(
    ::CB.AbstractSerialBackend,
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
    ::CB.AbstractThreadedBackend,
    ::PointField,
    x::AbstractMatrix{FT1},
    u::AbstractMatrix{FT2},
    distance_bins::AbstractVector{FT3};
    count_eltype::Type{CT} = UInt32,
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, CT}
    return _dispatch_single_pass(CB.ThreadedBackend(), x, u, distance_bins; count_eltype = CT, kwargs...)
end

function _dispatch_single_pass(
    ::CB.AbstractThreadedBackend,
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

function _dispatch_single_pass(::CB.AbstractThreadedBackend, args...; kwargs...)
    throw(ArgumentError("Threaded single-pass backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass(::CB.AbstractDistributedBackend, args...; kwargs...)
    throw(ArgumentError("Distributed single-pass backend is unavailable. Load Distributed (`using Distributed`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass(::CB.AbstractGPUBackend, args...; kwargs...)
    throw(ArgumentError("GPU single-pass backend is unavailable. Load the GPUExt extension or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass(::CB.AbstractMPIBackend, args...; kwargs...)
    throw(ArgumentError("MPI single-pass backend is unavailable. Load MPI (`using MPI`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass(
    backend::CB.AbstractGPUBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector;
    kwargs...
)
    return _dispatch_single_pass(backend, x, u, distance_bins; kwargs...)
end

function _dispatch_single_pass(
    backend::CB.AbstractDistributedBackend,
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
    Tuple{CB.ThreadedBackend, typeof(x), typeof(u), typeof(distance_bins)}
)

_distributed_single_pass_available(x, u, distance_bins) = hasmethod(
    _dispatch_single_pass,
    Tuple{CB.DistributedBackend, typeof(x), typeof(u), typeof(distance_bins)}
)

function _dispatch_single_pass(::CB.AbstractAutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector; kwargs...)
    backend = resolve_auto_backend(
        shape,
        () -> _threaded_single_pass_available(x, u, distance_bins),
        () -> _distributed_single_pass_available(x, u, distance_bins),
    )
    return _dispatch_single_pass(backend, shape, x, u, distance_bins; kwargs...)
end

# --- Single-pass result collections (keyed by invariant) ---

"""
    SINGLE_PASS_OPERATORS

The six native single-pass invariants in stacked-row order, keyed by short name. Used to label
each row of the single-pass `(sums, counts)` accumulator when building the result collection.
"""
const SINGLE_PASS_OPERATORS = (
    S2   = SFT.SecondOrderStructureFunctionType(),
    L2   = SFT.LongitudinalSecondOrderStructureFunctionType(),
    T2   = SFT.TransverseSecondOrderStructureFunctionType(),
    S3   = SFT.ThirdOrderStructureFunctionType(),
    L3   = SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
    L1T2 = SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
)

# View of stacked-row `t` across all trailing axes (n_bins, auxiliary...) — zero-copy.
@inline _sp_rowview(A::AbstractArray, t::Int) = view(A, t, ntuple(_ -> Colon(), ndims(A) - 1)...)

"""
    _single_pass_collection_1d(sums, counts, distance_bins, ::Type{OT})

Wrap the stacked 1D single-pass `(sums, counts)` into a `NamedTuple` keyed by invariant
(`S2, L2, T2, S3, L3, L1T2`), each value a single-operator result of representation `OT`
(default the averaged `StructureFunction`; pass `StructureFunctionSumsAndCounts` for raw).
Entries are zero-copy views into the stacked accumulator and share the (identical-by-construction)
counts row. For point-field input (stacked with the Helmholtz rows) a
`:helmholtz => HelmholtzDecomposition2D` entry is appended.
"""
function _single_pass_collection_1d(
    sums::AbstractArray, counts::AbstractArray, distance_bins, ::Type{OT},
) where {OT}
    cc = _sp_rowview(counts, 1)   # the six invariants share one (identical) counts row
    base = (
        S2   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.S2, distance_bins, _sp_rowview(sums, 1), cc), OT),
        L2   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.L2, distance_bins, _sp_rowview(sums, 2), cc), OT),
        T2   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.T2, distance_bins, _sp_rowview(sums, 3), cc), OT),
        S3   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.S3, distance_bins, _sp_rowview(sums, 4), cc), OT),
        L3   = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.L3, distance_bins, _sp_rowview(sums, 5), cc), OT),
        L1T2 = _finalize(SFO.StructureFunctionSumsAndCounts(SINGLE_PASS_OPERATORS.L1T2, distance_bins, _sp_rowview(sums, 6), cc), OT),
    )
    # Helmholtz exists exactly for point-field input (a 2D stacked matrix); batched input is
    # ndims ≥ 3. Branch on `ndims(sums)` ALONE (compile-time) — not a runtime size check — so the
    # return type is a single concrete NamedTuple (type-stable), not a Union of 6/7-key tuples.
    if ndims(sums) == 2
        return merge(base, (; helmholtz = helmholtz_decompose_2d(distance_bins, sums, counts)))
    end
    return base
end

"""
    calculate_structure_functions_single_pass(x, u, distance_bins; backend=CB.AutoBackend(),
                                              output_type=StructureFunction, kwargs...)

Compute the six native invariant structure functions (S2, L2, T2, S3, L3, L1T2) in one pair
pass, returned as a `NamedTuple` keyed by invariant. Each entry is a single-operator result of
the requested `output_type` (default the averaged `StructureFunction`; pass
`StructureFunctionSumsAndCounts` for the raw sums+counts). For point-field input a `:helmholtz`
entry (a [`HelmholtzDecomposition2D`](@ref)) is included.

!!! note "Why only six invariants (no L2T1 / T3)"
    The single-pass set is the six **isotropic** invariants. The directional third-order
    invariants `L2T1` (`DiagonalInconsistentThirdOrderStructureFunction`) and `T3`
    (`OffDiagonalConsistentThirdOrderStructureFunction`) are intentionally excluded: they require
    choosing a basis direction for the transverse/normal component (the isotropy does not cancel),
    so they are not basis-independent and are uncommon in practice. They remain available as
    standalone operator types for an explicit [`calculate_structure_function`](@ref) call.
"""
function calculate_structure_functions_single_pass(
    x::AbstractArray{FT1},
    u::AbstractArray{FT2, M},
    distance_bins::AbstractVector{FT3};
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction,
    count_eltype::Type{CT} = UInt32,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, M, OT, CT}
    shape = _validate_array_shape(x, u, distance_metric)
    _assert_counts_representable(CT, size(x, 2))
    raw = _dispatch_single_pass(
        backend, shape, x, u, distance_bins;
        count_eltype = count_eltype,
        distance_metric,
        kwargs...,
    )
    # The accumulator's element/rank are fully determined by the inputs: the sum element type is
    # `promote_type(float(eltype(x)), float(eltype(u)))`, the count element type is `count_eltype`,
    # and the stacked accumulator's rank equals `ndims(u)` (point-field `(6, n_bins)` → rank 2;
    # batched `(6, n_bins, aux...)` → rank `M`). Asserting these recovers the concrete type that the
    # multi-backend `_dispatch_single_pass` return does not preserve through inference, making the
    # whole call type-stable without changing any computation.
    OTv = promote_type(float(FT1), float(FT2))
    sums = raw.sums::AbstractArray{OTv, M}
    counts = raw.counts::AbstractArray{CT, M}
    return _single_pass_collection_1d(sums, counts, distance_bins, output_type)
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
    backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
    distance_metric::DI.PreMetric = DI.Euclidean(),
    kwargs...
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    _validate_array_shape(x, u, distance_metric)
    _dispatch_single_pass_2d!(
        backend, sums_3d, counts_3d, x, u, distance_bins, value_bins; distance_metric, kwargs...
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
    n_points = size(x, 2)
    n_bins = length(distance_bins) - 1
    n_val = size(sums_3d, 3)
    size(sums_3d, 1) == SINGLE_PASS_N && size(sums_3d, 2) == n_bins ||
        throw(DimensionMismatch("sums must have shape ($SINGLE_PASS_N, n_bins, n_val); got $(size(sums_3d))"))
    size(counts_3d) == size(sums_3d) ||
        throw(DimensionMismatch("counts and sums must have the same shape"))
    _validate_value_bins!(value_bins, n_val)

    dist_be = BinEdges(distance_bins)
    _sp2d_accumulate_range!(sums_3d, counts_3d, x, u, dist_be, value_bins, distance_metric,
        n_bins, n_val, 1:n_points)
    return sums_3d, counts_3d
end

"""Accumulate single-pass 2D pairs for outer indices `ilist` into the caller's sums/counts."""
function _sp2d_accumulate_range!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, value_bins, distance_metric,
    n_bins::Int, n_val::Int, ilist,
) where {OT, CT}
    h = _sp2d_histogram(OT, n_bins, n_val)
    _sp2d_fill!(h, x, u, dist_be, value_bins, distance_metric, n_bins, n_val, ilist)
    return _sp2d_unpack!(sums_3d, counts_3d, h, n_bins, n_val)
end

"""
Fill the interleaved accumulator from pairs whose outer index is in `ilist`. Euclidean `D ∈ {2,3}`
takes the SIMD compute/scatter split; other metrics or dimensions take the scalar loop. The `Val{D}`
branch is a function barrier: `D` must be a type parameter inside the loop, or `SVector{D}` builds
its type per point.
"""
function _sp2d_fill!(
    h::AbstractArray{OT, 4},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, value_bins, distance_metric,
    n_bins::Int, n_val::Int, ilist,
) where {OT}
    D = size(u, 1)
    if distance_metric isa DI.Euclidean && (D == 2 || D == 3)
        _sp2d_simd_partial!(h, x, u, dist_be, value_bins, D == 2 ? Val(2) : Val(3), n_val, ilist)
        return nothing
    end
    plan = squared_digitize_plan(dist_be)
    vW, vD = _pair_dims(distance_metric, D)
    _sp2d_pairs!(h, x, u, dist_be, plan, value_bins, vW, vD,
        distance_metric, n_bins, n_val, ilist)
    return nothing
end

"""
    _sp2d_histogram(OT, n_bins, n_val) -> Array{OT,4}

The single-pass 2D accumulator, laid out `(sum|count, invariant, value_bin, distance_bin)`.

Each pair writes all six invariants at ONE distance bin but six different value bins, so putting
the value axis inside the distance axis keeps a pair's six updates inside one distance slab, and
interleaving sum with count puts each invariant's two updates on one cache line: 6 lines touched
per pair instead of 12, and the cost stops scaling with histogram size.
"""
@inline _sp2d_histogram(::Type{OT}, n_bins::Int, n_val::Int) where {OT} =
    zeros(OT, 2, SINGLE_PASS_N, n_val, n_bins)

"""Add the interleaved accumulator into the caller's `(6, n_bins, n_val)` sums/counts."""
function _sp2d_unpack!(
    sums_3d::AbstractArray{OT, 3}, counts_3d::AbstractArray{CT, 3},
    h::AbstractArray, n_bins::Int, n_val::Int,
) where {OT, CT}
    @inbounds for d in 1:n_bins, v in 1:n_val, t in 1:SINGLE_PASS_N
        sums_3d[t, d, v] += h[1, t, v, d]
        counts_3d[t, d, v] += CT(h[2, t, v, d])
    end
    return nothing
end

"""
    _sp2d_pairs!(h, x, u, dist_be, plan, value_bins, ::Val{D}, metric, n_bins, n_val, ilist)

Single-pass 2D scalar pair loop over the outer indices `ilist`, for non-Euclidean metrics.
Specialized on the spatial dimension `D` so the `SVector`s are concrete.
"""
function _sp2d_pairs!(
    h::AbstractArray{OT, 4},
    x::AbstractMatrix{FT1}, u::AbstractMatrix{FT2},
    dist_be, plan, value_bins, ::Val{W}, ::Val{D}, distance_metric, n_bins::Int, n_val::Int, ilist,
) where {OT, FT1, FT2, W, D}
    n_points = size(x, 2)
    vW = Val(W)
    vD = Val(D)
    geom = SFH.pair_geometry_for(distance_metric, vD)
    @inbounds for i in ilist
        x_i = SA.SVector{W, FT1}(ntuple(d -> x[d, i], vW))
        u_i = SA.SVector{D, FT2}(ntuple(d -> u[d, i], vD))
        for j in (i + 1):n_points
            x_j = SA.SVector{W, FT1}(ntuple(d -> x[d, j], vW))
            ok, r, frame = SFH.pair_frame(geom, x_i, x_j)
            bin_idx = SFH.digitize(r, dist_be)
            if ok && 1 <= bin_idx <= n_bins
                u_j = SA.SVector{D, FT2}(ntuple(d -> u[d, j], vD))
                du, rh = SFH.pair_increments(geom, frame, r, x_i, x_j, u_i, u_j)
                du_L = LA.dot(du, rh)
                du_L2 = du_L * du_L
                du_norm2 = LA.dot(du, du)
                du_T2 = du_norm2 - du_L2
                vals = (du_norm2, du_L2, du_T2, du_L * du_norm2, du_L * du_L2, du_L * du_T2)
                _sp2d_scatter!(h, bin_idx, vals, value_bins, n_val)
            end
        end
    end
    return nothing
end

"""Scatter the six invariants of one pair into their cells of the interleaved accumulator."""
@inline function _sp2d_scatter!(
    h::AbstractArray{OT, 4}, dbin::Int, vals::NTuple{SINGLE_PASS_N}, value_bins, n_val::Int,
) where {OT}
    @sp2d_each_invariant value_bins t vb begin
        vbin = SFH.digitize(vals[t], vb)
        if 1 <= vbin <= (length(vb) - 1) && vbin <= n_val
            @inbounds h[1, t, vbin, dbin] += vals[t]
            @inbounds h[2, t, vbin, dbin] += one(OT)
        end
    end
    return nothing
end

"""
    _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, ::Val{D}, keybuf, duLbuf, dn2buf, idxbuf, n_val, irange)

Single-pass 2D point-field SIMD compute/scatter kernel over outer indices `irange`, the 2D analogue
of [`_pf_sp_simd_pairs!`](@ref). The `@simd` half computes distance, `du_L` and `|du|²` into buffers;
the scalar half derives the six invariants from those two scalars and scatters each into its own
`(distance, value)` cell. Shared by serial + threaded.
"""
function _sp2d_simd_pairs!(
    h::AbstractArray{OT, 4},
    xc::NTuple{D}, uc::NTuple{D}, plan::AbstractSquaredDigitizePlan, value_bins, ::Val{D},
    keybuf::AbstractVector, duLbuf::AbstractVector, dn2buf::AbstractVector,
    idxbuf::AbstractVector{Int32}, n_val::Int, irange,
) where {OT, D}
    N = length(xc[1])
    nb = n_histogram_bins(plan)
    FTx = eltype(xc[1])
    @inbounds for i in irange
        Xi = SA.SVector{D, FTx}(ntuple(d -> xc[d][i], Val(D)))
        Ui = SA.SVector{D}(ntuple(d -> uc[d][i], Val(D)))
        @simd for j in (i + 1):N
            Xj = SA.SVector{D, FTx}(ntuple(d -> xc[d][j], Val(D)))
            dx = Xj - Xi
            r2 = LA.dot(dx, dx)
            du = SA.SVector{D}(ntuple(d -> uc[d][j], Val(D))) - Ui
            inv_r = inv(sqrt(r2))
            keybuf[j] = digitize_key(plan, r2)
            duLbuf[j] = LA.dot(du, dx) * inv_r
            dn2buf[j] = LA.dot(du, du)
            if has_vector_index(plan)
                idxbuf[j] = squared_approx_index(plan, r2)
            end
        end
        for j in (i + 1):N
            dbin = squared_bin(plan, keybuf[j], idxbuf[j])
            if 1 <= dbin <= nb
                duL = duLbuf[j]
                dn2 = dn2buf[j]
                duL2 = duL * duL
                duT2 = dn2 - duL2
                vals = (dn2, duL2, duT2, duL * dn2, duL * duL2, duL * duT2)
                _sp2d_scatter!(h, dbin, vals, value_bins, n_val)
            end
        end
    end
    return nothing
end

"""
    _sp2d_simd_partial!(h, x, u, dist_be, value_bins, ::Val{D}, n_val, ilist)

Run [`_sp2d_simd_pairs!`](@ref) over an explicit outer-index list, with this worker's buffers.
"""
function _sp2d_simd_partial!(
    h::AbstractArray{OT, 4},
    x::AbstractMatrix, u::AbstractMatrix, dist_be, value_bins, ::Val{D}, n_val::Int, ilist,
) where {OT, D}
    xc = ntuple(d -> collect(view(x, d, :)), Val(D))
    uc = ntuple(d -> collect(view(u, d, :)), Val(D))
    N = length(xc[1])
    keybuf = Vector{eltype(xc[1])}(undef, N)
    duLbuf = Vector{OT}(undef, N)
    dn2buf = Vector{OT}(undef, N)
    idxbuf = Vector{Int32}(undef, N)
    plan = squared_digitize_plan(dist_be)
    _sp2d_simd_pairs!(h, xc, uc, plan, value_bins, Val(D),
        keybuf, duLbuf, dn2buf, idxbuf, n_val, ilist)
    return nothing
end

function _dispatch_single_pass_2d!(
    ::CB.AbstractSerialBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    return _accumulate_single_pass_2d!(sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d!(
    ::CB.AbstractThreadedBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    throw(ArgumentError("Threaded in-place 2D single-pass is unavailable. Load OhMyThreads or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d!(
    ::CB.AbstractDistributedBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    throw(ArgumentError("Distributed in-place 2D single-pass is unavailable. Load Distributed or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d!(
    backend::CB.AbstractGPUBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    gpu_calculate_structure_functions_single_pass_2d!(sums_3d, counts_3d, backend.backend, x, u, distance_bins, value_bins; kwargs...)
    return sums_3d, counts_3d
end

function _dispatch_single_pass_2d!(
    ::CB.AbstractAutoBackend, sums_3d::AbstractArray, counts_3d::AbstractArray, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...
)
    if distributed_workers_available(Val(:distributed)) &&
       _distributed_single_pass_2d_available(x, u, distance_bins, value_bins)
        return _dispatch_single_pass_2d!(CB.DistributedBackend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
    end
    if Threads.nthreads() > 1 && _threaded_single_pass_2d_available(x, u, distance_bins, value_bins)
        return _dispatch_single_pass_2d!(CB.ThreadedBackend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
    end
    return _dispatch_single_pass_2d!(CB.SerialBackend(), sums_3d, counts_3d, x, u, distance_bins, value_bins; kwargs...)
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
    ::CB.AbstractSerialBackend,
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
    ::CB.AbstractSerialBackend,
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

function _dispatch_single_pass_2d(::CB.AbstractThreadedBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    throw(ArgumentError("Threaded 2D single-pass backend is unavailable. Load the OhMyThreads extension or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d(
    backend::CB.AbstractThreadedBackend,
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
    ::CB.AbstractThreadedBackend,
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

function _dispatch_single_pass_2d(::CB.AbstractDistributedBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    throw(ArgumentError("Distributed 2D single-pass backend is unavailable. Load the Distributed extension or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d(::CB.AbstractMPIBackend, args...; kwargs...)
    throw(ArgumentError("MPI 2D single-pass backend is unavailable. Load MPI (`using MPI`) or use backend=CB.SerialBackend()."))
end

function _dispatch_single_pass_2d(
    backend::CB.AbstractDistributedBackend,
    ::AbstractFieldShape,
    x::AbstractArray,
    u::AbstractArray,
    distance_bins::AbstractVector,
    value_bins::SinglePass2DValueBins;
    kwargs...
)
    return _dispatch_single_pass_2d(backend, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d(backend::CB.AbstractGPUBackend, x::AbstractMatrix, u::AbstractMatrix, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    return gpu_calculate_structure_functions_single_pass_2d(backend.backend, x, u, distance_bins, value_bins; kwargs...)
end

function _dispatch_single_pass_2d(
    backend::CB.AbstractGPUBackend,
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
    Tuple{CB.ThreadedBackend, typeof(x), typeof(u), typeof(distance_bins), typeof(value_bins)},
)

_distributed_single_pass_2d_available(x, u, distance_bins, value_bins) = hasmethod(
    _dispatch_single_pass_2d,
    Tuple{CB.DistributedBackend, typeof(x), typeof(u), typeof(distance_bins), typeof(value_bins)},
)

function _dispatch_single_pass_2d(::CB.AbstractAutoBackend, shape::AbstractFieldShape, x::AbstractArray, u::AbstractArray, distance_bins::AbstractVector, value_bins::SinglePass2DValueBins; kwargs...)
    backend = resolve_auto_backend(
        shape,
        () -> _threaded_single_pass_2d_available(x, u, distance_bins, value_bins),
        () -> _distributed_single_pass_2d_available(x, u, distance_bins, value_bins),
    )
    return _dispatch_single_pass_2d(backend, shape, x, u, distance_bins, value_bins; kwargs...)
end

# Per-invariant value bins: a single vector is shared across invariants; a 6-tuple is per-invariant.
@inline _sp_valuebins(vb::AbstractVector, t::Int) = vb
@inline _sp_valuebins(vb::Tuple, t::Int) = vb[t]

"""
    _single_pass_collection_2d(sums, counts, distance_bins, value_bins, ::Type{OT})

Wrap the stacked 2D single-pass `(sums, counts)` (shape `(6, n_dist, n_val, aux...)`) into a
`NamedTuple` keyed by invariant, each value a `StructureFunction2DSumsAndCounts` view into the
stacked accumulator. Unlike 1D, the per-cell counts genuinely differ per invariant (each
invariant's value lands in a different value-bin), so counts are taken per-invariant. The 2D joint
histogram has no averaged representation, so `OT` must be `StructureFunction2DSumsAndCounts`.
"""
function _single_pass_collection_2d(
    sums::AbstractArray, counts::AbstractArray, distance_bins, value_bins, ::Type{OT},
) where {OT}
    return (
        S2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.S2, distance_bins, _sp_valuebins(value_bins, 1), _sp_rowview(sums, 1), _sp_rowview(counts, 1)), OT),
        L2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L2, distance_bins, _sp_valuebins(value_bins, 2), _sp_rowview(sums, 2), _sp_rowview(counts, 2)), OT),
        T2   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.T2, distance_bins, _sp_valuebins(value_bins, 3), _sp_rowview(sums, 3), _sp_rowview(counts, 3)), OT),
        S3   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.S3, distance_bins, _sp_valuebins(value_bins, 4), _sp_rowview(sums, 4), _sp_rowview(counts, 4)), OT),
        L3   = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L3, distance_bins, _sp_valuebins(value_bins, 5), _sp_rowview(sums, 5), _sp_rowview(counts, 5)), OT),
        L1T2 = _finalize(SFO.StructureFunction2DSumsAndCounts(SINGLE_PASS_OPERATORS.L1T2, distance_bins, _sp_valuebins(value_bins, 6), _sp_rowview(sums, 6), _sp_rowview(counts, 6)), OT),
    )
end

"""
    calculate_structure_functions_single_pass_2d(x, u, distance_bins, value_bins; backend=CB.AutoBackend(),
                                                 output_type=StructureFunction2DSumsAndCounts, kwargs...)

Compute the six invariant 2D joint structure-function histograms in one pass, returned as a
`NamedTuple` keyed by invariant (`S2, L2, T2, S3, L3, L1T2`). Each entry is a
[`StructureFunction2DSumsAndCounts`](@ref) view into the stacked accumulator (the 2D joint
histogram has no averaged form, so `output_type` must be `StructureFunction2DSumsAndCounts`).
"""
function calculate_structure_functions_single_pass_2d(
    x::AbstractArray{FT1},
    u::AbstractArray{FT2},
    distance_bins::AbstractVector{FT3},
    value_bins::SinglePass2DValueBins;
    backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction2DSumsAndCounts,
    count_eltype::Type{CT} = UInt32,
    distance_metric::DI.PreMetric = DI.Euclidean(),
    verbose::Bool = true,
    show_progress::Bool = true,
) where {FT1 <: Number, FT2 <: Number, FT3 <: Number, OT, CT}
    shape = _validate_array_shape(x, u, distance_metric)
    _assert_counts_representable(CT, size(x, 2))
    # Only forward knobs the kernels accept; they have no `kwargs...` sink.
    raw = _dispatch_single_pass_2d(
        backend, shape, x, u, distance_bins, value_bins;
        count_eltype = count_eltype, distance_metric = distance_metric,
    )
    return _single_pass_collection_2d(raw[1], raw[2], distance_bins, value_bins, output_type)
end
