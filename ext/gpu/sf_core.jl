# =============================================================================
# Unified parametric GPU kernel core — building blocks
# See gpu/OPTIMAL_KERNEL_DESIGN.md for the full design rationale.
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
# * Distance digitizers are *isbits* functors (no array fields for linear/log)
#   so they can be passed directly as kernel arguments and specialize the
#   digitize at compile time. The bin-edge wrapper structs (LinearBinEdges /
#   LogBinEdges) are NOT isbits (they hold a Range), hence these lightweight
#   mirrors built host-side via `_sf_digitizer`.
# * The shared histogram uses a "lane" axis of width `L` as its fastest index:
#       hist[((m-1)*NB + (bin-1)) * L + lane]
#   For the non-fixed-x path `L = R` (replication factor → contention spreading;
#   replicas are summed at flush). For the fixed-x batch path `L = W` (batch
#   strip → each lane is a distinct velocity field; NOT summed, scattered to the
#   B axis at flush). Same accumulate primitive, different flush.
#   IMPORTANT: we rotate the *lane* (replica) index, never the *bin* index.
# =============================================================================

# -----------------------------------------------------------------------------
# Distance digitizers (isbits functors; compile-time dispatched)
# -----------------------------------------------------------------------------

"""Linear-grid distance digitizer. `n_edges` is the number of bin *edges*
(so the number of bins is `n_edges - 1`). Returns a bin in `0:n_edges`."""
struct SFLinearDigitizer{T}
    first_edge::T
    last_edge::T
    inv_step::T
    step_val::T
    n_edges::Int
end

@inline (d::SFLinearDigitizer{T})(r::T) where {T} =
    _gpu_digitize_linear(r, d.first_edge, d.last_edge, d.inv_step, d.step_val, d.n_edges)

"""Log-grid distance digitizer: `log(r)` then a linear digitize on the log grid
(matches `LogBinEdges`). Non-positive `r` falls below the first edge (→ 0)."""
struct SFLogDigitizer{T}
    first_edge::T
    last_edge::T
    inv_step::T
    step_val::T
    n_edges::Int
end

@inline (d::SFLogDigitizer{T})(r::T) where {T} =
    r <= zero(T) ? 0 :
    _gpu_digitize_linear(log(r), d.first_edge, d.last_edge, d.inv_step, d.step_val, d.n_edges)

"""General (arbitrary edges) distance digitizer: binary search over a device
edge vector. `edges` is adapted to the backend alongside the kernel arguments."""
struct SFGeneralDigitizer{E}
    edges::E
    n_edges::Int
end

# Without this the struct reaches the kernel holding a host-side `CuArray` and the launch fails
# with `KernelError: passing non-bitstype argument`.
KA.Adapt.adapt_structure(to, d::SFGeneralDigitizer) =
    SFGeneralDigitizer(KA.Adapt.adapt(to, d.edges), d.n_edges)

# A work list carries its packed tile pairs in a device array; the struct must be rebuilt around the
# device-side view at launch, exactly as the general digitizer is around its edges.
KA.Adapt.adapt_structure(to, s::TilePairWorkList) =
    TilePairWorkList(KA.Adapt.adapt(to, s.pairs), s.n_tiles)

@inline (d::SFGeneralDigitizer)(r) = _gpu_digitize_general(r, d.edges, d.n_edges)

# Number of bins (edges - 1) for a digitizer.
@inline _sf_nbins(d::SFLinearDigitizer) = d.n_edges - 1
@inline _sf_nbins(d::SFLogDigitizer) = d.n_edges - 1
@inline _sf_nbins(d::SFGeneralDigitizer) = d.n_edges - 1

# Host-side constructors from the bin-edge wrapper types.
@inline _sf_digitizer(lbe::LinearBinEdges) =
    SFLinearDigitizer(lbe.first_edge, lbe.last_edge, lbe.inv_step, lbe.step_val, length(lbe.edges))

@inline function _sf_digitizer(lbe::LogBinEdges)
    ll = lbe.log_linear
    return SFLogDigitizer(ll.first_edge, ll.last_edge, ll.inv_step, ll.step_val, length(lbe.log_edges))
end

# General edges: caller passes a device array of edges (already on backend).
@inline _sf_digitizer_general(edges_dev) = SFGeneralDigitizer(edges_dev, length(edges_dev))

# -----------------------------------------------------------------------------
# Geometry (NDIMS-generic, via StaticArrays — unrolls for D = 2, 3)
# -----------------------------------------------------------------------------

@inline _sf_dot(a::SA.SVector{2,T}, b::SA.SVector{2,T}) where {T} = a[1] * b[1] + a[2] * b[2]
@inline _sf_dot(a::SA.SVector{3,T}, b::SA.SVector{3,T}) where {T} = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

"""Load local point `k` (D components) from a staged `@localmem` tile buffer."""
@inline _sf_load_pt(::Val{2}, buf, k::Int) =
    @inbounds SA.SVector{2}(buf[k], buf[SF_GPU_TILE + k])
@inline _sf_load_pt(::Val{3}, buf, k::Int) =
    @inbounds SA.SVector{3}(buf[k], buf[SF_GPU_TILE + k], buf[2 * SF_GPU_TILE + k])

# -----------------------------------------------------------------------------
# Moments
# -----------------------------------------------------------------------------

"""Six single-pass invariants from a velocity difference `dU` and unit vector
`rhat`. Computes one dot product (`du_L`) and one norm (`du_norm2`); transverse
follows as `du_norm2 - du_L²` (no second projection)."""
@inline function _sf_moments6(dU::SA.SVector{D,T}, rhat::SA.SVector{D,T}) where {D,T}
    du_L = _sf_dot(dU, rhat)
    du_norm2 = _sf_dot(dU, dU)
    du_L2 = du_L * du_L
    du_T2 = du_norm2 - du_L2
    return (du_norm2, du_L2, du_T2, du_L * du_norm2, du_L * du_L2, du_L * du_T2)
end

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
histogram bin is a sum, so both are recovered exactly at flush by [`_sf_flush_moment`](@ref)
rather than costing an atomic on each pair.
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
# All looped reads/writes/atomics on these `@localmem` buffers are written
# INLINE in the kernel bodies, NOT via helper functions. On the CUDA backend a
# `@localmem` array passed as a function argument and written in a loop fails to
# compile (GPUCompiler MethodError); inlining matches the proven pattern used by
# the existing tiled kernels. Single-element loads through a helper are fine
# (see `_sf_load_pt` / `_sf_load_field`).
# -----------------------------------------------------------------------------
