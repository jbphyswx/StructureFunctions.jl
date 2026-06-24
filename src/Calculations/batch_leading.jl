# Batch-leading, Val{D}-specialized CPU batch kernels.
#
# Two problems in the original (D,N,B) batch kernels (see benchmark/cpu_regimes.jl baseline):
#   1. `D = size(x,1)` was a RUNTIME value, so `SVector{D}` / `Val(D)` built the SVector TYPE
#      inside the hot loop → type instability → ~1140 allocs per pair-op → GC-bound (the 32 s).
#   2. the `@simd for b` loop read `u[d, point, b]` with stride D*N in b → no packed SIMD.
#
# Fix (proven ordering — type stability first, then layout):
#   * a single dynamic dispatch on `Val(D)` is a FUNCTION BARRIER: the kernel body is fully
#     specialized on `D` (0 allocations), the one unstable call is amortized over the whole O(N²B).
#   * kernels operate on a batch-LEADING working buffer `(B, D, N)` so `@simd for b` gets unit
#     stride; for shared positions the bin is constant across b ⇒ contiguous accumulate (no scatter).
#
# Isolated micro-bench (N=150,B=64,D=3): runtime-D 814M allocs/1906 ms → Val{D}+(B,D,N) 0/0.53 ms.

using Distances: Distances as DI

# Tag an array as already batch-leading `(B, D, N)` (CPU-optimal SoA) to skip the transpose.
# Plain arrays are the default `(D, N, B...)` contract (GPU-optimal; transposed once internally).
"""
    BatchLeading(u)

Wrap a velocity/position array that is already stored **batch-leading**, shape `(B, D, N)`
(batch axis innermost/contiguous — the CPU-optimal SoA layout). CPU batch kernels then run
zero-copy. A plain `(D, N, B...)` array (the default contract, GPU-optimal) is transposed once
internally (measured <1% of one pass). Zero-cost type tag.
"""
struct BatchLeading{A <: AbstractArray}
    data::A
end

@inline _bl_unwrap(u) = (u, false)              # (array, already_batch_leading)
@inline _bl_unwrap(u::BatchLeading) = (u.data, true)

@inline _sq(x) = x * x

# Prepare batch-leading (B,D,N) buffers. Handles the default plain `(D,N,B...)` (transposed
# once) and `BatchLeading` `(B,D,N)` (zero-copy). `x` may be fixed (D,N) or varying. Called
# once per top-level call (not in the hot loop), so the type-instability of the branch is a
# harmless function barrier. Returns (xb, ub, B, D, N, fixed_x).
function _bl_prepare(x, u)
    u_raw, u_bl = _bl_unwrap(u)
    x_raw, x_bl = _bl_unwrap(x)
    fixed_x = ndims(x_raw) == 2
    if u_bl
        B, D, N = size(u_raw)
        ub = u_raw
    else
        D, N = size(x_raw, 1), size(x_raw, 2)
        B = prod(size(u_raw)[3:end])
        ub = _to_batch_leading(reshape(u_raw, D, N, B))
    end
    xb = fixed_x ? x_raw : (x_bl ? x_raw : _to_batch_leading(reshape(x_raw, D, N, B)))
    return xb, ub, B, D, N, fixed_x
end

# (D,N,B) -> (B,D,N) materialized batch-leading buffer (lazy PermutedDimsArray would put the
# strided read back in the hot loop, so we materialize — cheap, O(D·N·B) ≪ O(N²·B)).
@inline _to_batch_leading(u_DNB) = permutedims(u_DNB, (3, 1, 2))

# ----------------------------------------------------------------------------------------
# Shared positions ("same surface"): x is (D,N) fixed; geometry computed ONCE per pair.
# ub :: (B, D, N) ; sums_bl, counts_bl :: (B, n_bins)
# ----------------------------------------------------------------------------------------
function _bl_shared_1d!(
    sums_bl::AbstractMatrix{OT},
    counts_bl::AbstractMatrix{CT},
    x::AbstractMatrix,
    ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    dist_be,
    ::Val{D},
    irange,
) where {OT, CT, D}
    B = size(ub, 1)
    N = size(ub, 3)
    nb = size(sums_bl, 2)
    @inbounds for i in irange
        Xi = SA.SVector{D}(ntuple(d -> x[d, i], Val(D)))
        for j in (i + 1):N
            Xj = SA.SVector{D}(ntuple(d -> x[d, j], Val(D)))
            dx = Xj - Xi
            dist = sqrt(LA.dot(dx, dx))
            bin = SFH.digitize(dist, dist_be)
            (1 <= bin <= nb) || continue
            rh = dx / dist
            @simd for b in 1:B
                du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                v = sf_type(du, rh)
                sums_bl[b, bin] += v
                counts_bl[b, bin] += one(CT)
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------------------
# Varying positions: x is (B,D,N) too; geometry depends on b ⇒ computed inside the b loop.
# ----------------------------------------------------------------------------------------
function _bl_varying_1d!(
    sums_bl::AbstractMatrix{OT},
    counts_bl::AbstractMatrix{CT},
    xb::AbstractArray{<:Any, 3},
    ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType,
    dist_be,
    ::Val{D},
    irange,
) where {OT, CT, D}
    B = size(ub, 1)
    N = size(ub, 3)
    nb = size(sums_bl, 2)
    @inbounds for i in irange
        for j in (i + 1):N
            @simd for b in 1:B
                dx = SA.SVector{D}(ntuple(d -> xb[b, d, j] - xb[b, d, i], Val(D)))
                dist = sqrt(LA.dot(dx, dx))
                bin = SFH.digitize(dist, dist_be)
                if 1 <= bin <= nb
                    rh = dx / dist
                    du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                    sums_bl[b, bin] += sf_type(du, rh)
                    counts_bl[b, bin] += one(CT)
                end
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------------------
# Joint 2D (single SF): output accumulator (B, n_dist, n_val). vbin varies per b (scatter on
# the value axis) ⇒ plain b-loop (still 0-alloc, type-stable). dbin constant for shared x.
# ----------------------------------------------------------------------------------------
function _bl_joint2d_shared!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType, dist_be, val_be, ::Val{D}, irange,
) where {OT, CT, D}
    B = size(ub, 1); N = size(ub, 3)
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3)
    @inbounds for i in irange
        Xi = SA.SVector{D}(ntuple(d -> x[d, i], Val(D)))
        for j in (i + 1):N
            Xj = SA.SVector{D}(ntuple(d -> x[d, j], Val(D)))
            dx = Xj - Xi
            dist = sqrt(LA.dot(dx, dx))
            dbin = SFH.digitize(dist, dist_be)
            (1 <= dbin <= n_dist) || continue
            rh = dx / dist
            for b in 1:B
                du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                val = sf_type(du, rh)
                vbin = SFH.digitize(val, val_be)
                if 1 <= vbin <= n_val
                    sums_bl[b, dbin, vbin] += val
                    counts_bl[b, dbin, vbin] += one(CT)
                end
            end
        end
    end
    return nothing
end

function _bl_joint2d_varying!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType, dist_be, val_be, ::Val{D}, irange,
) where {OT, CT, D}
    B = size(ub, 1); N = size(ub, 3)
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3)
    @inbounds for i in irange
        for j in (i + 1):N
            for b in 1:B
                dx = SA.SVector{D}(ntuple(d -> xb[b, d, j] - xb[b, d, i], Val(D)))
                dist = sqrt(LA.dot(dx, dx))
                dbin = SFH.digitize(dist, dist_be)
                (1 <= dbin <= n_dist) || continue
                rh = dx / dist
                du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                val = sf_type(du, rh)
                vbin = SFH.digitize(val, val_be)
                if 1 <= vbin <= n_val
                    sums_bl[b, dbin, vbin] += val
                    counts_bl[b, dbin, vbin] += one(CT)
                end
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------------------
# Single-pass 1D (6 invariants). Accumulator (B, 6, n_dist): for shared x the dist bin is
# constant over b, so each of the 6 writes is contiguous in b (vectorizes). Note: only du_L
# and du_norm2=⟨du,du⟩ are needed — the old `du_T = mδu_t(...)` was dead work (now removed).
# ----------------------------------------------------------------------------------------
@inline function _bl_sp1d_write!(sums_bl, counts_bl, b, bin, du_L, du_norm2, ::Type{CT}) where {CT}
    du_L2 = du_L * du_L
    du_T2 = du_norm2 - du_L2
    @inbounds begin
        sums_bl[b, 1, bin] += du_norm2          # S2  = du_L2 + du_T2
        sums_bl[b, 2, bin] += du_L2             # L2
        sums_bl[b, 3, bin] += du_T2             # T2
        sums_bl[b, 4, bin] += du_L * du_norm2   # S3
        sums_bl[b, 5, bin] += du_L * du_L2      # L3
        sums_bl[b, 6, bin] += du_L * du_T2      # L1T2
        for t in 1:SINGLE_PASS_N
            counts_bl[b, t, bin] += one(CT)
        end
    end
    return nothing
end

function _bl_sp1d_shared!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3}, dist_be, ::Val{D}, irange,
) where {OT, CT, D}
    B = size(ub, 1); N = size(ub, 3); nb = size(sums_bl, 3)
    @inbounds for i in irange
        Xi = SA.SVector{D}(ntuple(d -> x[d, i], Val(D)))
        for j in (i + 1):N
            Xj = SA.SVector{D}(ntuple(d -> x[d, j], Val(D)))
            dx = Xj - Xi
            dist = sqrt(LA.dot(dx, dx))
            bin = SFH.digitize(dist, dist_be)
            (1 <= bin <= nb) || continue
            rh = dx / dist
            @simd for b in 1:B
                du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                _bl_sp1d_write!(sums_bl, counts_bl, b, bin, LA.dot(du, rh), LA.dot(du, du), CT)
            end
        end
    end
    return nothing
end

function _bl_sp1d_varying!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3}, dist_be, ::Val{D}, irange,
) where {OT, CT, D}
    B = size(ub, 1); N = size(ub, 3); nb = size(sums_bl, 3)
    @inbounds for i in irange
        for j in (i + 1):N
            @simd for b in 1:B
                dx = SA.SVector{D}(ntuple(d -> xb[b, d, j] - xb[b, d, i], Val(D)))
                dist = sqrt(LA.dot(dx, dx))
                bin = SFH.digitize(dist, dist_be)
                if 1 <= bin <= nb
                    rh = dx / dist
                    du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                    _bl_sp1d_write!(sums_bl, counts_bl, b, bin, LA.dot(du, rh), LA.dot(du, du), CT)
                end
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------------------
# Single-pass 2D (6 invariants × per-invariant value bins). Accumulator (B, 6, n_dist, n_val);
# per-invariant value bin ⇒ scatter ⇒ plain b-loop.
# ----------------------------------------------------------------------------------------
@inline function _sp1d_vals(du_L, du_norm2)
    du_L2 = du_L * du_L
    du_T2 = du_norm2 - du_L2
    return (du_norm2, du_L2, du_T2, du_L * du_norm2, du_L * du_L2, du_L * du_T2)
end

@inline function _bl_sp2d_write!(sums_bl, counts_bl, b, dbin, vals, value_bins, n_val, ::Type{CT}) where {CT}
    @inbounds for t in 1:SINGLE_PASS_N
        vb = _sp2d_value_bin_at(value_bins, t)
        vbin = SFH.digitize(vals[t], vb)
        if 1 <= vbin <= (length(vb) - 1) && vbin <= n_val
            sums_bl[b, t, dbin, vbin] += vals[t]
            counts_bl[b, t, dbin, vbin] += one(CT)
        end
    end
    return nothing
end

function _bl_sp2d_shared!(
    sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3}, dist_be, value_bins, ::Val{D}, irange,
) where {OT, CT, D}
    B = size(ub, 1); N = size(ub, 3); nb = size(sums_bl, 3); n_val = size(sums_bl, 4)
    @inbounds for i in irange
        Xi = SA.SVector{D}(ntuple(d -> x[d, i], Val(D)))
        for j in (i + 1):N
            Xj = SA.SVector{D}(ntuple(d -> x[d, j], Val(D)))
            dx = Xj - Xi
            dist = sqrt(LA.dot(dx, dx))
            dbin = SFH.digitize(dist, dist_be)
            (1 <= dbin <= nb) || continue
            rh = dx / dist
            for b in 1:B
                du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                vals = _sp1d_vals(LA.dot(du, rh), LA.dot(du, du))
                _bl_sp2d_write!(sums_bl, counts_bl, b, dbin, vals, value_bins, n_val, CT)
            end
        end
    end
    return nothing
end

function _bl_sp2d_varying!(
    sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3}, dist_be, value_bins, ::Val{D}, irange,
) where {OT, CT, D}
    B = size(ub, 1); N = size(ub, 3); nb = size(sums_bl, 3); n_val = size(sums_bl, 4)
    @inbounds for i in irange
        for j in (i + 1):N
            for b in 1:B
                dx = SA.SVector{D}(ntuple(d -> xb[b, d, j] - xb[b, d, i], Val(D)))
                dist = sqrt(LA.dot(dx, dx))
                dbin = SFH.digitize(dist, dist_be)
                (1 <= dbin <= nb) || continue
                rh = dx / dist
                du = SA.SVector{D}(ntuple(d -> ub[b, d, j] - ub[b, d, i], Val(D)))
                vals = _sp1d_vals(LA.dot(du, rh), LA.dot(du, du))
                _bl_sp2d_write!(sums_bl, counts_bl, b, dbin, vals, value_bins, n_val, CT)
            end
        end
    end
    return nothing
end

# ========================================================================================
# Drivers: prep (transpose to batch-leading) + run kernel via an EXECUTOR + transpose back.
#
# Parallelism is over the OUTER pair index `i` (each pair's geometry is computed exactly once;
# the inner batch loop over b stays full and SIMD-vectorized). The b-work is so fast that
# parallelizing over b instead (recomputing geometry per task) was a NET SLOWDOWN — so we
# partition `i` with round-robin chunks (triangle load balance) and reduce thread-local
# accumulators, mirroring the point-field threaded path.
#
# executor(make_accum, run_chunk!, ifull) → reduced (sums_bl, counts_bl):
#   make_accum()                  → fresh zeroed (sums_bl, counts_bl)
#   run_chunk!((sums_bl,counts_bl), isub) → kernel over outer i ∈ isub
#   serial  : one accumulator over ifull
#   threaded: per-chunk accumulators, reduced by elementwise +
# ========================================================================================

@inline function _bl_serial_exec(make_accum, run_chunk!, ifull)
    acc = make_accum()
    run_chunk!(acc, ifull)
    return acc
end

function _bl_run_1d!(sums, counts, sf_type, x, u, dist_be, executor)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, N, fixed_x = _bl_prepare(x, u)
    vD = Val(D)
    make_accum() = (zeros(OT, B, n_bins), zeros(CT, B, n_bins))
    run_chunk! = fixed_x ?
        ((acc, isub) -> _bl_shared_1d!(acc[1], acc[2], xb, ub, sf_type, dist_be, vD, isub)) :
        ((acc, isub) -> _bl_varying_1d!(acc[1], acc[2], xb, ub, sf_type, dist_be, vD, isub))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1))
    permutedims!(reshape(sums, n_bins, B), sums_bl, (2, 1))
    permutedims!(reshape(counts, n_bins, B), counts_bl, (2, 1))
    return nothing
end

function _bl_run_joint2d!(sums, counts, sf_type, x, u, dist_be, val_be, executor)
    n_dist = n_histogram_bins(dist_be); n_val = n_histogram_bins(val_be)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, N, fixed_x = _bl_prepare(x, u)
    vD = Val(D)
    make_accum() = (zeros(OT, B, n_dist, n_val), zeros(CT, B, n_dist, n_val))
    run_chunk! = fixed_x ?
        ((acc, isub) -> _bl_joint2d_shared!(acc[1], acc[2], xb, ub, sf_type, dist_be, val_be, vD, isub)) :
        ((acc, isub) -> _bl_joint2d_varying!(acc[1], acc[2], xb, ub, sf_type, dist_be, val_be, vD, isub))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1))
    permutedims!(reshape(sums, n_dist, n_val, B), sums_bl, (2, 3, 1))
    permutedims!(reshape(counts, n_dist, n_val, B), counts_bl, (2, 3, 1))
    return nothing
end

function _bl_run_sp1d!(sums, counts, x, u, dist_be, executor)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, N, fixed_x = _bl_prepare(x, u)
    vD = Val(D)
    make_accum() = (zeros(OT, B, SINGLE_PASS_N, n_bins), zeros(CT, B, SINGLE_PASS_N, n_bins))
    run_chunk! = fixed_x ?
        ((acc, isub) -> _bl_sp1d_shared!(acc[1], acc[2], xb, ub, dist_be, vD, isub)) :
        ((acc, isub) -> _bl_sp1d_varying!(acc[1], acc[2], xb, ub, dist_be, vD, isub))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1))
    permutedims!(reshape(sums, SINGLE_PASS_N, n_bins, B), sums_bl, (2, 3, 1))
    permutedims!(reshape(counts, SINGLE_PASS_N, n_bins, B), counts_bl, (2, 3, 1))
    return nothing
end

function _bl_run_sp2d!(sums, counts, x, u, dist_be, value_bins, executor)
    n_bins = n_histogram_bins(dist_be)
    n_val = size(sums, 3)
    _validate_value_bins!(value_bins, n_val)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, N, fixed_x = _bl_prepare(x, u)
    vD = Val(D)
    make_accum() = (zeros(OT, B, SINGLE_PASS_N, n_bins, n_val), zeros(CT, B, SINGLE_PASS_N, n_bins, n_val))
    run_chunk! = fixed_x ?
        ((acc, isub) -> _bl_sp2d_shared!(acc[1], acc[2], xb, ub, dist_be, value_bins, vD, isub)) :
        ((acc, isub) -> _bl_sp2d_varying!(acc[1], acc[2], xb, ub, dist_be, value_bins, vD, isub))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1))
    permutedims!(reshape(sums, SINGLE_PASS_N, n_bins, n_val, B), sums_bl, (2, 3, 4, 1))
    permutedims!(reshape(counts, SINGLE_PASS_N, n_bins, n_val, B), counts_bl, (2, 3, 4, 1))
    return nothing
end
