# =============================================================================
# Unified parametric GPU kernel core — building blocks
#
# These are the compile-time-specialized primitives shared by the two unified
# tiled kernels (`sf_tiled_1d!`, `sf_tiled_2d!`). Everything here is `@inline`,
# allocation-free, and device-safe (validated on the KA.CPU() backend).
#
# Conventions
# -----------
# * Staged tile layout: dimension `d` of local point `k` lives at
#   `buf[(d-1)*SF_GPU_TILE + k]` in an `@localmem` buffer (matches the existing
#   tiled kernels).
# * A digitizer is the host's `digitize_plan` with its arrays on the device, binned with
#   `SFH.digitize`, so every backend bins every value alike.
# * The shared histogram uses a "lane" axis of width `L` as its fastest index:
#       hist[((m-1)*NB + (bin-1)) * L + lane]
#   For the non-fixed-x path `L = R`, the replication factor that spreads contention, and the
#   replicas are summed at flush. For the fixed-x batch path `L = W`, the batch strip: each lane is
#   a distinct velocity field, not summed but scattered to the B axis at flush. One accumulate
#   primitive, two flushes. The rotation is applied to the lane index, never to the bin index.
# =============================================================================

# -----------------------------------------------------------------------------
# Digitizers
# -----------------------------------------------------------------------------

"""
The device digitizer for `bins` in a pass of workspace kind `kind` (`Val(:value)` for value bins), or a
tuple of them for per-moment value bins: the plan the host digitizes with, its arrays on `backend`. The
distance bins of a `:sf1d` pass take logarithmic edges as an `SF.LogTableBinEdges`.
"""
_gpu_digitizer(backend, bins, kind::Val) = KA.adapt(backend, _device_plan(bins, kind))

_device_plan(bins, ::Val) = SF.digitize_plan(bins)
_device_plan(b::SF.LogBinEdges{<:Base.IEEEFloat}, ::Val{:sf1d}) = SF.LogTableBinEdges(b)
_device_plan(b::SF.InfPaddedBinEdges, kind::Val{:sf1d}) =
    (p = _device_plan(b.edges, kind); SF.InfPaddedBinEdges{eltype(p), typeof(p)}(p))
_device_plan(t::Tuple, kind::Val) = map(b -> _device_plan(b, kind), t)

"""Value bin of moment `m`. A tuple plan is indexed by an unrolled chain, so its columns may differ in type."""
@inline _sf_value_bin(plan, x, m) = SFH.digitize(x, plan)
@inline _sf_value_bin(plan::Tuple, x, m) = _sf_tuple_bin(plan, x, m)
@inline _sf_tuple_bin(plan::Tuple{Any}, x, m) = SFH.digitize(x, first(plan))
@inline _sf_tuple_bin(plan::Tuple, x, m) =
    m == 1 ? SFH.digitize(x, first(plan)) : _sf_tuple_bin(Base.tail(plan), x, m - 1)

"""Edges per value column; every column of a tuple plan has as many."""
@inline _n_value_edges(value_bins) = length(value_bins)
@inline _n_value_edges(value_bins::Tuple) = length(first(value_bins))

# A work list carries its packed tile pairs in a device array; the struct is rebuilt around the
# device-side view at launch.
KA.Adapt.adapt_structure(to, s::TilePairWorkList) =
    TilePairWorkList(KA.Adapt.adapt(to, s.pairs), s.n_tiles)

# Bin edges, digitize plans and lag schedules reach a kernel with their vectors on the device.
KA.Adapt.adapt_structure(to, b::SF.BinEdges) = SF.BinEdges(KA.Adapt.adapt(to, b.edges))
function KA.Adapt.adapt_structure(to, b::SF.BucketedBinEdges{T}) where {T}
    e = KA.Adapt.adapt(to, b.edges)
    c = KA.Adapt.adapt(to, b.cells)
    return SF.BucketedBinEdges{T, typeof(e), typeof(c)}(e, b.inv_width, b.offset, b.last_cell, b.last_edge, c)
end
function KA.Adapt.adapt_structure(to, p::SF.LogTableBinEdges{FT, T}) where {FT, T}
    e = KA.Adapt.adapt(to, p.edges)
    return SF.LogTableBinEdges{FT, T, typeof(e)}(p.a, p.c, e)
end
function KA.Adapt.adapt_structure(to, b::SF.InfPaddedBinEdges{T}) where {T}
    inner = KA.Adapt.adapt(to, b.edges)
    return SF.InfPaddedBinEdges{T, typeof(inner)}(inner)
end
function KA.Adapt.adapt_structure(to, p::SF.SquaredLogPlan{T}) where {T}
    sq = KA.Adapt.adapt(to, p.sqedges)
    return SF.SquaredLogPlan{T, typeof(sq)}(p.a, p.b, p.n_bins, sq)
end
KA.Adapt.adapt_structure(to, p::SF.SquaredBucketPlan) = SF.SquaredBucketPlan(KA.Adapt.adapt(to, p.thresholds))
KA.Adapt.adapt_structure(to, p::SF.SquaredInfPaddedPlan) = SF.SquaredInfPaddedPlan(KA.Adapt.adapt(to, p.inner))
KA.Adapt.adapt_structure(to, s::SFC.RectilinearLagSchedule) =
    SFC.RectilinearLagSchedule(s.uniform, map(v -> KA.Adapt.adapt(to, v), s.enumerated), s.axis_order)
KA.Adapt.adapt_structure(to, s::SFC.ZonalLagSchedule) =
    SFC.ZonalLagSchedule(KA.Adapt.adapt(to, s.lats), s.n_lon, s.dlon, s.radius, s.lon_periodic)
KA.Adapt.adapt_structure(to, s::SFC.ScatteredModesSchedule) =
    SFC.ScatteredModesSchedule(KA.Adapt.adapt(to, s.points), s.origin, s.box, s.modes, s.taper)

# -----------------------------------------------------------------------------
# Geometry (NDIMS-generic, via StaticArrays — unrolls for D = 2, 3)
# -----------------------------------------------------------------------------

@inline function _sf_dot(a::SA.SVector{W, T}, b::SA.SVector{W, T}) where {W, T}
    s = zero(T)
    @inbounds for d in 1:W
        s += a[d] * b[d]
    end
    return s
end

"""Load local point `k` (`W` components) from a `@localmem` tile staged as
`(d - 1) * SF_GPU_TILE + k`."""
@inline _sf_load_pt(::Val{W}, buf, k::Int) where {W} =
    SA.SVector{W}(ntuple(d -> @inbounds(buf[(d - 1) * SF_GPU_TILE + k]), Val(W)))

# -----------------------------------------------------------------------------
# Pair weights
# -----------------------------------------------------------------------------

"""
    _sf_count_type(weights, CT, n_pairs) -> Type

Element type a device count histogram accumulates in, shared and global alike. `UInt32` while the
sweep is unweighted **and** `n_pairs` fits it, since a narrower count halves the shared histogram;
the caller's count type otherwise — weights make a count a weighted pair mass, and `UInt32`
addition wraps silently past `typemax(UInt32)` pairs. Chosen host-side at the allocation, and a
kernel's shared histogram then follows `eltype` of the buffer it flushes into.
"""
@inline _sf_count_type(::SFC.NoWeights, ::Type{CT}, n_pairs::Integer) where {CT} =
    n_pairs <= typemax(UInt32) ? UInt32 : CT
@inline _sf_count_type(::AbstractVector, ::Type{CT}, ::Integer) where {CT} = CT

"""Worst-case number of pairs an `N`-point sweep can put in one bin."""
@inline _sf_worst_case_pairs(N::Integer) = (Int128(N) * (Int128(N) - 1)) ÷ 2

"""Move pair weights to the device, leaving `NoWeights()` alone."""
@inline _sf_weights_to_device(backend, w::SFC.NoWeights) = w
@inline _sf_weights_to_device(backend, w::AbstractVector) = KA.adapt(backend, w)

# -----------------------------------------------------------------------------
# Moments
# -----------------------------------------------------------------------------

"""Six single-pass invariants from a velocity difference `dU` and unit vector
`rhat`. Computes one dot product (`du_L`) and one norm (`du_norm2`); transverse
follows as `du_norm2 - du_L²` (no second projection)."""
@inline _sf_moments6(dU::SA.SVector{D,T}, rhat::SA.SVector{D,T}) where {D,T} =
    SFC.single_pass_invariants(_sf_dot(dU, rhat), _sf_dot(dU, dU))

"""Compute the moment tuple for a pair:
- `Val{6}` → the six single-pass invariants (sf_type ignored).
- `Val{1}` → the individual SF value, computed by the **authoritative** `sf_type(δu, r̂)`
  callable from `StructureFunctionTypes` — the single source of truth for the per-type
  math (basis- and dimension-dependent factors like `1/(D-1)` and the signed `mδu_t`/`n̂`
  components are handled there, and it is the same code the CPU paths use). Returned as a
  1-tuple so the accumulate path is uniform with the single-pass case."""
@inline _sf_moments(::Val{6}, sf_type, dU, rhat) = _sf_moments6(dU, rhat)
@inline _sf_moments(::Val{1}, sf_type, dU, rhat) = (sf_type(dU, rhat),)

"""
Moments needing a per-pair atomic. `T2 = S2 - L2` and `L1T2 = S3 - L3` hold for every pair, and a
histogram bin is a sum, so both are recovered exactly at flush by [`_sf_flush_moment`](@ref) and
cost no atomic on any pair.
"""
@inline _sf_accum_moments(::Val{6}) = (1, 2, 4, 5)
@inline _sf_accum_moments(::Val{1}) = (1,)

"""Value of moment `m` in bin `bin`, differencing the two moments that are never accumulated."""
@inline function _sf_flush_moment(::Val{6}, ssum, NB::Int, m::Int, bin::Int)
    @inbounds begin
        m == 3 && return ssum[bin] - ssum[NB + bin]
        m == 6 && return ssum[3 * NB + bin] - ssum[4 * NB + bin]
        return ssum[(m - 1) * NB + bin]
    end
end
@inline _sf_flush_moment(::Val{1}, ssum, ::Int, ::Int, bin::Int) = @inbounds ssum[bin]

# -----------------------------------------------------------------------------
# Shared-histogram layout (lane axis = R replicas or W batch strip)
#
# A pair's contribution to moment `m`, distance bin `bin`, lane `ℓ` lives at
#     shared_sums[(m-1)*NB*L + (bin-1)*L + ℓ]      (L = R or W, compile-time)
#     shared_cnts[(bin-1)*L + ℓ]
#
# Every looped read, write and atomic on these `@localmem` buffers is written inline in the kernel
# bodies: on the CUDA backend a `@localmem` array passed as a function argument and written in a
# loop fails to compile with a GPUCompiler MethodError. Single-element loads through a helper
# compile (`_sf_load_pt`, `_sf_load_field`).
# -----------------------------------------------------------------------------
