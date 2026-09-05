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

"""Position/velocity input a batch driver accepts: a plain `(D, N, B…)` array, or a
[`BatchLeading`](@ref) wrapper around a `(B, D, N)` one."""
const BatchInput = Union{AbstractArray, BatchLeading}

@inline _bl_unwrap(u) = (u, false)              # (array, already_batch_leading)
@inline _bl_unwrap(u::BatchLeading) = (u.data, true)

# Prepare batch-leading (B,D,N) buffers. Handles the default plain `(D,N,B...)` (transposed
# once) and `BatchLeading` `(B,D,N)` (zero-copy). `x` may be fixed (D,N) or varying. Called
# once per top-level call (not in the hot loop), so the type-instability of the branch is a
# harmless function barrier. Returns (xb, ub, B, D, W, N, fixed_x, geom): `D` and `W` are the
# field and coordinate widths the kernels load, and `geom` carries the velocity dimension.
"""
Component-first view of an input, so `prepare_pair_inputs` — which reads components from axis 1 —
can convert it. Only a batch-leading `(B, W, N)` array needs permuting; the default `(W, N, B…)`
layout is already component-first. `permutedims` costs one `O(N·B)` pass per call.
"""
@inline _bl_component_first(a, is_batch_leading::Bool) =
    is_batch_leading ? permutedims(a, (2, 3, 1)) : a

function _bl_prepare(x, u, distance_metric = DI.Euclidean(), workspace = nothing)
    u_raw, u_bl = _bl_unwrap(u)
    x_raw, x_bl = _bl_unwrap(x)
    fixed_x = ndims(x_raw) == 2
    # The velocity dimension, read before any conversion — this is what fixes the geometry, and it
    # is not recoverable from the converted arrays.
    D_in = u_bl ? size(u_raw, 2) : size(u_raw, 1)
    geom = SFH.pair_geometry_for(distance_metric, Val(D_in))
    if !(geom isa SFH.FlatGeometry)
        # Convert once per call, component-first, then let the layout code below run unchanged.
        x_raw, u_raw = SFH.prepare_pair_inputs(
            geom, _bl_component_first(x_raw, x_bl), _bl_component_first(u_raw, u_bl),
        )
        x_bl = false
        u_bl = false
    end
    # `W` is the coordinate width and `D` the field width the kernels load; they differ from each
    # other and from the velocity dimension on a sphere.
    W = x_bl ? size(x_raw, 2) : size(x_raw, 1)
    if u_bl
        B, D, N = size(u_raw)
        _validate_ws_shape(workspace, B, N, D, W)
        ub = u_raw
    else
        D, N = size(u_raw, 1), size(u_raw, 2)
        B = prod(size(u_raw)[3:end])
        _validate_ws_shape(workspace, B, N, D, W)
        ub = _to_batch_leading(reshape(u_raw, D, N, B), _ws_ub(workspace))
    end
    xb = fixed_x ? x_raw :
         (x_bl ? x_raw : _to_batch_leading(reshape(x_raw, W, N, B), _ws_xb(workspace)))
    return xb, ub, B, D, W, N, fixed_x, geom
end

# Statically-sized loads for the two layouts the batch drivers hold: `(W, N)` shared positions and
# `(B, W, N)` batch-leading. The width comes from the geometry, never from the velocity rank.
Base.@propagate_inbounds _bl_pt(x::AbstractMatrix, i, ::Val{W}) where {W} =
    SA.SVector{W}(ntuple(d -> x[d, i], Val(W)))
Base.@propagate_inbounds _bl_pt(xb::AbstractArray{<:Any, 3}, b, i, ::Val{W}) where {W} =
    SA.SVector{W}(ntuple(d -> xb[b, d, i], Val(W)))
Base.@propagate_inbounds _bl_vel(ub, b, i, ::Val{D}) where {D} =
    SA.SVector{D}(ntuple(d -> ub[b, d, i], Val(D)))

"""
Throw unless the staged coordinate width matches what `geom` needs.

The batch entry points take raw `(D, N, B…)` arrays rather than going through
[`_validate_array_shape`](@ref), so this is where a mismatched `x` is caught — before the kernels
load it under `@inbounds`.
"""
@inline function _validate_bl_geometry(geom, W::Int, D::Int)
    want = _val_int(SFH.coordinate_width(geom))
    W == want || throw(
        DimensionMismatch(
            "$(nameof(typeof(geom))) locates a point with $want coordinate(s) on axis 1 of x, but " *
            "got $W (velocity dimension D=$D)",
        ),
    )
    return nothing
end

# The workspace's transpose buffers, or `nothing` for the allocate-fresh path.
@inline _ws_ub(::Nothing) = nothing
@inline _ws_xb(::Nothing) = nothing

# (D,N,B) -> (B,D,N) materialized batch-leading buffer (lazy PermutedDimsArray would put the
# strided read back in the hot loop, so we materialize — cheap, O(D·N·B) ≪ O(N²·B)).
@inline _to_batch_leading(u_DNB, ::Nothing) = permutedims(u_DNB, (3, 1, 2))
@inline _to_batch_leading(u_DNB, dest::AbstractArray) = permutedims!(dest, u_DNB, (3, 1, 2))

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
    geom,
    ::Val{D},
    irange,
    brange,
) where {OT, CT, D}
    N = size(ub, 3)
    nb = size(sums_bl, 2)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        Xi = _bl_pt(x, i, vW)
        for j in (i + 1):N
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            bin = SFH.digitize(dist, dist_be)
            (ok && 1 <= bin <= nb) || continue
            # Loop-invariant across the b strip: one frame and one direction serve every field.
            rh = SFH.pair_direction(geom, frame, dist)
            @simd for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                v = sf_type(du, rh)
                sums_bl[b - boff, bin] += v
                counts_bl[b - boff, bin] += one(CT)
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
    geom,
    ::Val{D},
    irange,
    brange,
) where {OT, CT, D}
    N = size(ub, 3)
    nb = size(sums_bl, 2)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        for j in (i + 1):N
            @simd for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = SFH.digitize(dist, dist_be)
                if ok && 1 <= bin <= nb
                    du, rh = SFH.pair_increments(geom, frame, dist, Xi, Xj,
                        _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                    sums_bl[b - boff, bin] += sf_type(du, rh)
                    counts_bl[b - boff, bin] += one(CT)
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
    sf_type::SFT.AbstractPairwiseStructureFunctionType, dist_be, val_be, geom, ::Val{D}, irange, brange,
) where {OT, CT, D}
    N = size(ub, 3)
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        Xi = _bl_pt(x, i, vW)
        for j in (i + 1):N
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            dbin = SFH.digitize(dist, dist_be)
            (ok && 1 <= dbin <= n_dist) || continue
            rh = SFH.pair_direction(geom, frame, dist)
            for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                val = sf_type(du, rh)
                vbin = SFH.digitize(val, val_be)
                if 1 <= vbin <= n_val
                    sums_bl[b - boff, dbin, vbin] += val
                    counts_bl[b - boff, dbin, vbin] += one(CT)
                end
            end
        end
    end
    return nothing
end

function _bl_joint2d_varying!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3},
    sf_type::SFT.AbstractPairwiseStructureFunctionType, dist_be, val_be, geom, ::Val{D}, irange, brange,
) where {OT, CT, D}
    N = size(ub, 3)
    n_dist = size(sums_bl, 2); n_val = size(sums_bl, 3)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        for j in (i + 1):N
            for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_be)
                (ok && 1 <= dbin <= n_dist) || continue
                du, rh = SFH.pair_increments(geom, frame, dist, Xi, Xj,
                    _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                val = sf_type(du, rh)
                vbin = SFH.digitize(val, val_be)
                if 1 <= vbin <= n_val
                    sums_bl[b - boff, dbin, vbin] += val
                    counts_bl[b - boff, dbin, vbin] += one(CT)
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
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3}, dist_be, geom, ::Val{D}, irange, brange,
) where {OT, CT, D}
    N = size(ub, 3); nb = size(sums_bl, 3)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        Xi = _bl_pt(x, i, vW)
        for j in (i + 1):N
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            bin = SFH.digitize(dist, dist_be)
            (ok && 1 <= bin <= nb) || continue
            rh = SFH.pair_direction(geom, frame, dist)
            @simd for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                _bl_sp1d_write!(sums_bl, counts_bl, b - boff, bin, LA.dot(du, rh), LA.dot(du, du), CT)
            end
        end
    end
    return nothing
end

function _bl_sp1d_varying!(
    sums_bl::AbstractArray{OT, 3}, counts_bl::AbstractArray{CT, 3},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3}, dist_be, geom, ::Val{D}, irange, brange,
) where {OT, CT, D}
    N = size(ub, 3); nb = size(sums_bl, 3)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        for j in (i + 1):N
            @simd for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                bin = SFH.digitize(dist, dist_be)
                if ok && 1 <= bin <= nb
                    du, rh = SFH.pair_increments(geom, frame, dist, Xi, Xj,
                        _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                    _bl_sp1d_write!(sums_bl, counts_bl, b - boff, bin, LA.dot(du, rh), LA.dot(du, du), CT)
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
    @sp2d_each_invariant value_bins t vb begin
        vbin = SFH.digitize(vals[t], vb)
        if 1 <= vbin <= (length(vb) - 1) && vbin <= n_val
            @inbounds sums_bl[b, t, dbin, vbin] += vals[t]
            @inbounds counts_bl[b, t, dbin, vbin] += one(CT)
        end
    end
    return nothing
end

function _bl_sp2d_shared!(
    sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4},
    x::AbstractMatrix, ub::AbstractArray{<:Any, 3}, dist_be, value_bins, geom, ::Val{D}, irange, brange,
) where {OT, CT, D}
    N = size(ub, 3); nb = size(sums_bl, 3); n_val = size(sums_bl, 4)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        Xi = _bl_pt(x, i, vW)
        for j in (i + 1):N
            Xj = _bl_pt(x, j, vW)
            ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
            dbin = SFH.digitize(dist, dist_be)
            (ok && 1 <= dbin <= nb) || continue
            rh = SFH.pair_direction(geom, frame, dist)
            for b in brange
                du = SFH.pair_delta(geom, frame, Xi, Xj, _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                vals = _sp1d_vals(LA.dot(du, rh), LA.dot(du, du))
                _bl_sp2d_write!(sums_bl, counts_bl, b - boff, dbin, vals, value_bins, n_val, CT)
            end
        end
    end
    return nothing
end

function _bl_sp2d_varying!(
    sums_bl::AbstractArray{OT, 4}, counts_bl::AbstractArray{CT, 4},
    xb::AbstractArray{<:Any, 3}, ub::AbstractArray{<:Any, 3}, dist_be, value_bins, geom, ::Val{D}, irange, brange,
) where {OT, CT, D}
    N = size(ub, 3); nb = size(sums_bl, 3); n_val = size(sums_bl, 4)
    boff = first(brange) - 1
    vW = SFH.coordinate_width(geom)
    vD = Val(D)
    @inbounds for i in irange
        for j in (i + 1):N
            for b in brange
                Xi = _bl_pt(xb, b, i, vW)
                Xj = _bl_pt(xb, b, j, vW)
                ok, dist, frame = SFH.pair_frame(geom, Xi, Xj)
                dbin = SFH.digitize(dist, dist_be)
                (ok && 1 <= dbin <= nb) || continue
                du, rh = SFH.pair_increments(geom, frame, dist, Xi, Xj,
                    _bl_vel(ub, b, i, vD), _bl_vel(ub, b, j, vD))
                vals = _sp1d_vals(LA.dot(du, rh), LA.dot(du, du))
                _bl_sp2d_write!(sums_bl, counts_bl, b - boff, dbin, vals, value_bins, n_val, CT)
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
# executor(make_accum, run_chunk!, ifull, B, accum_bytes, ws) → reduced (sums, counts), width B:
#   make_accum(bw)                     → fresh zeroed (sums_bl, counts_bl) of batch width bw
#   run_chunk!(acc, isub, brange)      → kernel over outer i ∈ isub and batch b ∈ brange
#   accum_bytes                        → bytes of one full-width accumulator, for the split model
#   ws                                 → CPUSFWorkspace to draw accumulators from, or `nothing`
#   serial  : one full-width accumulator over ifull
#   threaded: partitions (i, b); per-task accumulators are only as wide as their b-chunk
# ========================================================================================

@inline function _bl_serial_exec(make_accum, run_chunk!, ifull, B, accum_bytes, ws)
    acc = _bl_accum_pool(ws, make_accum, [B])[1]
    _bl_zero_accum!(acc)
    run_chunk!(acc, ifull, 1:B)
    return acc
end

"""
    _bl_accum_pool(ws, make_accum, widths) -> Vector

One accumulator per task, of the given batch widths, drawn from the workspace when there is one and
allocated fresh otherwise. Always built outside the parallel region, so tasks only read their slot.
"""
_bl_accum_pool(::Nothing, make_accum::F, widths::Vector{Int}) where {F} =
    [make_accum(w) for w in widths]

"""The full-width reduction accumulator, from the workspace when there is one."""
_bl_result_accum(::Nothing, make_accum::F, B::Int) where {F} = make_accum(B)

@inline function _bl_zero_accum!(acc)
    fill!(acc[1], zero(eltype(acc[1])))
    fill!(acc[2], zero(eltype(acc[2])))
    return acc
end

"""Bytes in one full-width `(sums, counts)` accumulator of shape `dims`."""
@inline _bl_accum_bytes(::Type{OT}, ::Type{CT}, dims::Vararg{Int}) where {OT, CT} =
    prod(dims) * (sizeof(OT) + sizeof(CT))

"""
    _bl_batch_chunk_count(accum_bytes, B, n_tasks) -> Int

How many chunks to split the batch axis into, given the full-width accumulator size in bytes.

Every task holds an accumulator, so the live footprint is `n_tasks * accum_bytes` — that is what
turns into GC and page-fault time, and it is what a batch chunk divides. Splitting `b` costs pair
geometry, recomputed once per batch chunk, so the right answer is the SMALLEST split that brings
the footprint under budget. Kernels whose footprint already fits (the 1D batch accumulator is tens
of KiB) keep `Bc == 1` and never pay geometry twice; the single-pass 2D accumulator at large `B`
and many threads reaches hundreds of MiB and is split until it fits.
"""
@inline function _bl_batch_chunk_count(accum_bytes::Int, B::Int, n_tasks::Int)
    footprint = accum_bytes * n_tasks
    footprint <= _BL_ACCUM_BUDGET && return 1
    return clamp(cld(footprint, _BL_ACCUM_BUDGET), 1, min(n_tasks, B))
end

# Live accumulator footprint across all tasks, above which the batch axis is split.
const _BL_ACCUM_BUDGET = 64 * 1024 * 1024

"""Contiguous batch-axis chunks; `_bl_batch_chunk_count` picks `n`."""
@inline _bl_batch_chunks(B::Int, n::Int) =
    [(((k - 1) * B) ÷ n + 1):((k * B) ÷ n) for k in 1:n]

"""
    _bl_n_tasks(backend) -> Int

How many tasks a batch call on `backend` splits into, and therefore how many accumulators a
[`CPUSFWorkspace`](@ref) must hold. A threaded backend falls back to the serial executor when
OhMyThreads is not loaded, so the count follows the executor that will actually run.
"""
_bl_n_tasks(::CB.AbstractExecutionBackend) = 1
_bl_n_tasks(::CB.AbstractThreadedBackend) = _ohmythreads_loaded() ? Threads.nthreads() : 1
_bl_n_tasks(b::CB.AbstractMPIBackend) = _bl_n_tasks(CB.local_backend(b))
_bl_n_tasks(b::CB.AbstractDistributedBackend) = _bl_n_tasks(CB.local_backend(b))

"""
    _bl_executor(backend) -> executor

The batch-leading executor a local backend runs with, as a value. Backends that compose over a
local inner backend (MPI) look it up here; extensions add methods for the backends they provide.
"""
_bl_executor(::CB.AbstractSerialBackend) = _bl_serial_exec
_bl_executor(b::CB.AbstractExecutionBackend) = throw(ArgumentError(
    "no batch-leading executor for $(nameof(typeof(b))); use SerialBackend, or load the extension \
     providing it (ThreadedBackend needs OhMyThreads)."))

"""
    _bl_add_permuted!(dest, src_bl, perm)

Add the batch-leading accumulator `src_bl` into `dest` through the permutation `perm`.

`!` entry points across the package **accumulate** into the caller's buffers; zeroing belongs to the
non-mutating wrappers. `PermutedDimsArray` is a lazy view, so this fuses the permute with the add
and allocates nothing.
"""
@inline function _bl_add_permuted!(dest, src_bl, perm)
    dest .+= PermutedDimsArray(src_bl, perm)
    return dest
end

function _bl_run_1d!(sums, counts, sf_type, x, u, dist_be, distance_metric, executor, workspace = nothing)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :sf1d, (n_bins,))
    make_accum(bw) = (zeros(OT, bw, n_bins), zeros(CT, bw, n_bins))
    run_chunk! = fixed_x ?
        ((acc, isub, br) -> _bl_shared_1d!(acc[1], acc[2], xb, ub, sf_type, dist_be, geom, vD, isub, br)) :
        ((acc, isub, br) -> _bl_varying_1d!(acc[1], acc[2], xb, ub, sf_type, dist_be, geom, vD, isub, br))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, n_bins), workspace)
    _bl_add_permuted!(reshape(sums, n_bins, B), sums_bl, (2, 1))
    _bl_add_permuted!(reshape(counts, n_bins, B), counts_bl, (2, 1))
    return nothing
end

function _bl_run_joint2d!(sums, counts, sf_type, x, u, dist_be, val_be, distance_metric, executor, workspace = nothing)
    n_dist = n_histogram_bins(dist_be); n_val = n_histogram_bins(val_be)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :joint2d, (n_dist, n_val))
    make_accum(bw) = (zeros(OT, bw, n_dist, n_val), zeros(CT, bw, n_dist, n_val))
    run_chunk! = fixed_x ?
        ((acc, isub, br) -> _bl_joint2d_shared!(acc[1], acc[2], xb, ub, sf_type, dist_be, val_be, geom, vD, isub, br)) :
        ((acc, isub, br) -> _bl_joint2d_varying!(acc[1], acc[2], xb, ub, sf_type, dist_be, val_be, geom, vD, isub, br))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, n_dist, n_val), workspace)
    _bl_add_permuted!(reshape(sums, n_dist, n_val, B), sums_bl, (2, 3, 1))
    _bl_add_permuted!(reshape(counts, n_dist, n_val, B), counts_bl, (2, 3, 1))
    return nothing
end

function _bl_run_sp1d!(sums, counts, x, u, dist_be, distance_metric, executor, workspace = nothing)
    n_bins = n_histogram_bins(dist_be)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :single_pass, (SINGLE_PASS_N, n_bins))
    make_accum(bw) = (zeros(OT, bw, SINGLE_PASS_N, n_bins), zeros(CT, bw, SINGLE_PASS_N, n_bins))
    run_chunk! = fixed_x ?
        ((acc, isub, br) -> _bl_sp1d_shared!(acc[1], acc[2], xb, ub, dist_be, geom, vD, isub, br)) :
        ((acc, isub, br) -> _bl_sp1d_varying!(acc[1], acc[2], xb, ub, dist_be, geom, vD, isub, br))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, SINGLE_PASS_N, n_bins), workspace)
    _bl_add_permuted!(reshape(sums, SINGLE_PASS_N, n_bins, B), sums_bl, (2, 3, 1))
    _bl_add_permuted!(reshape(counts, SINGLE_PASS_N, n_bins, B), counts_bl, (2, 3, 1))
    return nothing
end

function _bl_run_sp2d!(sums, counts, x, u, dist_be, value_bins, distance_metric, executor, workspace = nothing)
    n_bins = n_histogram_bins(dist_be)
    n_val = size(sums, 3)
    _validate_value_bins!(value_bins, n_val)
    OT = eltype(sums); CT = eltype(counts)
    xb, ub, B, D, W, N, fixed_x, geom = _bl_prepare(x, u, distance_metric, workspace)
    vD = Val(D)
    _validate_bl_geometry(geom, W, D)
    _validate_ws_layout(workspace, :single_pass_2d, (SINGLE_PASS_N, n_bins, n_val))
    make_accum(bw) = (zeros(OT, bw, SINGLE_PASS_N, n_bins, n_val), zeros(CT, bw, SINGLE_PASS_N, n_bins, n_val))
    run_chunk! = fixed_x ?
        ((acc, isub, br) -> _bl_sp2d_shared!(acc[1], acc[2], xb, ub, dist_be, value_bins, geom, vD, isub, br)) :
        ((acc, isub, br) -> _bl_sp2d_varying!(acc[1], acc[2], xb, ub, dist_be, value_bins, geom, vD, isub, br))
    sums_bl, counts_bl = executor(make_accum, run_chunk!, 1:(N - 1), B, _bl_accum_bytes(OT, CT, B, SINGLE_PASS_N, n_bins, n_val), workspace)
    _bl_add_permuted!(reshape(sums, SINGLE_PASS_N, n_bins, n_val, B), sums_bl, (2, 3, 4, 1))
    _bl_add_permuted!(reshape(counts, SINGLE_PASS_N, n_bins, n_val, B), counts_bl, (2, 3, 4, 1))
    return nothing
end
