# GPUPrototypeKernels — experimental GPU structure-function kernels (not in main ext).
# Variants explore alternatives to production `_sf_kernel_linear!` (N×N global atomics).
# Invoked via include — see gpu/README.md (cd to repo or use joinpath absolute path).

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @localmem, @synchronize
using Printf: @printf
using StaticArrays: StaticArrays as SA
using StructureFunctions:
    HelperFunctions as SFH, StructureFunctionTypes as SFT, LinearBinEdges, LogBinEdges

"""Upper bound on histogram slots compiled into prototype kernels (`@localmem` needs literals)."""
const PROTO_MAX_BINS = 64

# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

"""Optional flags for CADISHI-style tiled histogram kernels."""
struct TiledOpts
    regpriv::Bool
    nosqrt::Bool
    dim2::Bool
    bin_kind::Symbol
end

TiledOpts(regpriv::Bool, nosqrt::Bool, dim2::Bool) = TiledOpts(regpriv, nosqrt, dim2, :linear)

const TILED_OPTS_NONE = TiledOpts(false, false, false)

"""Launch parameters for a prototype kernel variant."""
struct PrototypeConfig
    name::String
    variant::Symbol
    nworkers::Int
    workgroup_size::Int
    device_resident::Bool
    tile_size::Int
    tiled_opts::Union{TiledOpts, Nothing}
end

PrototypeConfig(name, variant, nworkers, workgroup_size, device_resident) =
    PrototypeConfig(name, variant, nworkers, workgroup_size, device_resident, 0, nothing)

PrototypeConfig(name, variant, nworkers, workgroup_size, device_resident, tile_size::Int) =
    PrototypeConfig(name, variant, nworkers, workgroup_size, device_resident, tile_size, TILED_OPTS_NONE)

"""Raw sums/counts returned from a prototype run (host vectors)."""
struct PrototypeResult
    sums::Vector{Float32}
    counts::Vector{Float32}
    counts_i64::Vector{Int64}
    sum_counts_i64::Int64
    counts_eltype::Symbol
    kernel_s::Float64
    staging_s::Float64
end

"""Host-side integer histogram from a device counts buffer (exact for `UInt32`)."""
function _host_counts_i64(cnt_dev)
    c = Array(cnt_dev)
    T = eltype(c)
    if T <: Integer
        return Int64.(c)
    end
    return Int64.(round.(c))
end

"""Variants that accumulate pair counts in `UInt32` device buffers."""
const _U32_COUNT_VARIANTS = (
    :baseline_linear_u32,
    :blockshared_u32,
    :blockshared_regpriv_u32,
    :tiled_u32,
)

function linear_params(bin_edges::LinearBinEdges{FT}) where {FT}
    return (
        first_edge = bin_edges.first_edge,
        last_edge = bin_edges.last_edge,
        inv_step = bin_edges.inv_step,
        offset = bin_edges.offset,
        step_val = bin_edges.step_val,
        n_bins = length(bin_edges.edges),
    )
end

function log_params(bin_edges::LogBinEdges{FT}) where {FT}
    return (
        edges = bin_edges.edges,
        lut = bin_edges.lut,
        e_min = bin_edges.e_min,
        n_bins = length(bin_edges.edges),
    )
end

"""Host-side bin routing for prototype launches (linear / log / general)."""
struct ProtoBinLayout
    kind::Symbol
    linear
    log
    general_edges
    n_bins::Int
end

function prototype_bin_layout(bin_edges::LinearBinEdges{FT}) where {FT}
    lp = linear_params(bin_edges)
    return ProtoBinLayout(:linear, lp, nothing, nothing, lp.n_bins)
end

function prototype_bin_layout(bin_edges::LogBinEdges{FT}) where {FT}
    lp = log_params(bin_edges)
    return ProtoBinLayout(:log, nothing, lp, nothing, lp.n_bins)
end

function prototype_bin_layout(edges::AbstractVector{FT}) where {FT}
    return ProtoBinLayout(:general, nothing, nothing, edges, length(edges))
end

# ---------------------------------------------------------------------------
# Device digitize helpers (match ext/StructureFunctionsGPUExt.jl)
# ---------------------------------------------------------------------------

@inline function _gpu_digitize_linear(
    x::T,
    first_edge::T,
    last_edge::T,
    inv_step::T,
    offset::T,
    step_val::T,
    n_edges::Int,
) where {T}
    if x <= first_edge
        return 0
    end
    if x > last_edge
        return n_edges
    end
    idx = round(Int, muladd(x, inv_step, offset))
    idx = clamp(idx, 1, n_edges)
    edge_val = muladd(step_val, T(idx - 1), first_edge)
    search_idx = edge_val < x ? idx + 1 : idx
    return search_idx - 1
end

@inline function _gpu_digitize_log(
    x::T,
    edges,
    lut,
    e_min::Int,
    n_edges::Int,
) where {T}
    f = edges[1]
    if x <= f
        return 0
    end
    l = edges[n_edges]
    if x > l
        return n_edges
    end
    e = exponent(x)
    e_idx = clamp(e - e_min + 1, 1, length(lut))
    idx_start = lut[e_idx]
    idx_end = e_idx < length(lut) ? lut[e_idx + 1] : n_edges
    if idx_end - idx_start <= 8
        idx = idx_start
        while idx <= idx_end && edges[idx] < x
            idx += 1
        end
        return idx - 1
    else
        low = idx_start
        high = idx_end
        while low <= high
            mid = (low + high) >>> 1
            if edges[mid] < x
                low = mid + 1
            else
                high = mid - 1
            end
        end
        return low - 1
    end
end

@inline function _gpu_digitize_general(x::T, edges, n_edges::Int) where {T}
    low = 1
    high = n_edges
    while low <= high
        mid = (low + high) >>> 1
        if edges[mid] < x
            low = mid + 1
        else
            high = mid - 1
        end
    end
    return low - 1
end

"""Digitize linear bins using `dist²` (matches `_gpu_digitize_linear` for uniform edges)."""
@inline function _gpu_digitize_linear_sq(
    dist_sq,
    first_edge,
    last_edge,
    step_val,
    n_edges,
)
    fe2 = first_edge * first_edge
    dist_sq <= fe2 && return 0
    le2 = last_edge * last_edge
    dist_sq > le2 && return n_edges
    b = n_edges - 1
    while b >= 1
        ek = first_edge + step_val * (b - 1)
        dist_sq >= ek * ek && return b
        b -= 1
    end
    return 1
end

@inline function _row_start(i, N)
    return (i - one(i)) * N - (i - one(i)) * i ÷ 2
end

"""Map 1-based upper-triangle pair index `k ∈ 1:N(N-1)/2` to `(i, j)` with `i < j`.

Uses integer binary search only — the closed-form `sqrt` mapping loses precision in
Float32 on GPU when `N` and `k` are large (e.g. `N=20000` misses ~150k pairs).
"""
@inline function _pair_from_linear(k, N)
    lo = one(k)
    hi = N - one(k)
    while lo < hi
        mid = (lo + hi) >>> 1
        row_end = _row_start(mid, N) + (N - mid)
        if k > row_end
            lo = mid + one(k)
        else
            hi = mid
        end
    end
    i = lo
    j = i + (k - _row_start(i, N))
    return i, j
end

"""Map linear tile block id `k` to upper-triangle `(ti, tj)` with `ti ≤ tj`. Untyped for GPU index widths."""
@inline function _tile_from_linear(k, n_tiles)
    ti = one(k)
    rleft = k - one(k)
    while ti < n_tiles && rleft >= n_tiles - ti + one(k)
        rleft -= n_tiles - ti + one(k)
        ti += one(k)
    end
    tj = ti + rleft
    return ti, tj
end

@inline function _accumulate_pair_linear!(
    sums,
    cnts,
    i::Int,
    j::Int,
    x_mat,
    u_mat,
    sf_type,
    N_bins::Int,
    first_edge,
    last_edge,
    inv_step,
    offset,
    step_val,
)
    FT = eltype(sums)
    X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
    X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
    U1 = SA.SVector{3, FT}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
    U2 = SA.SVector{3, FT}(u_mat[1, j], u_mat[2, j], u_mat[3, j])

    dX = X2 - X1
    dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
    bin = _gpu_digitize_linear(
        dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
    )
    if 1 <= bin < N_bins
        r̂ = SFH.r̂(X1, X2)
        val = sf_type(U2 - U1, r̂)
        sums[bin] += val
        cnts[bin] += one(FT)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Baseline: global atomic, ndrange = (N, N)  [mirrors production kernel]
# ---------------------------------------------------------------------------

KA.@kernel function _proto_baseline_linear!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
) where {FT}
    I = @index(Global, NTuple)
    i = I[1]
    j = I[2]
    if i < j
        X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
        X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
        U1 = SA.SVector{3, FT}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
        U2 = SA.SVector{3, FT}(u_mat[1, j], u_mat[2, j], u_mat[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            r̂ = SFH.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            @atomic output[bin] += val
            @atomic counts[bin] += one(FT)
        end
    end
end

# ---------------------------------------------------------------------------
# Baseline B: global atomic, grid-stride linear pair index (canonical reference)
# Same pairs as optimized kernels; avoids (N,N) grid coverage issues at large N.
# ---------------------------------------------------------------------------

KA.@kernel function _proto_baseline_linear_pairs!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nworkers::Int,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
        X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
        U1 = SA.SVector{3, FT}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
        U2 = SA.SVector{3, FT}(u_mat[1, j], u_mat[2, j], u_mat[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            r̂ = SFH.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            @atomic output[bin] += val
            @atomic counts[bin] += one(FT)
        end
        k += nworkers
    end
end

# Same pair loop as `_proto_baseline_linear_pairs!`; counts are UInt32 (exact at large N).
KA.@kernel function _proto_baseline_linear_pairs_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nworkers::Int,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
        X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
        U1 = SA.SVector{3, FT}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
        U2 = SA.SVector{3, FT}(u_mat[1, j], u_mat[2, j], u_mat[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            r̂ = SFH.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            @atomic output[bin] += val
            @atomic counts[bin] += UInt32(1)
        end
        k += nworkers
    end
end

# ---------------------------------------------------------------------------
# Prototype A: register-private histogram per 1D worker (grid-stride pairs)
# ---------------------------------------------------------------------------

KA.@kernel function _proto_private_linear!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nworkers::Int,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2

    sums = SA.MVector{PROTO_MAX_BINS, FT}(undef)
    cnts = SA.MVector{PROTO_MAX_BINS, FT}(undef)
    @inbounds for b in 1:NB
        sums[b] = zero(FT)
        cnts[b] = zero(FT)
    end

    k = worker
    while k <= total_pairs
        i, j = _pair_from_linear(k, N_points)
        _accumulate_pair_linear!(
            sums, cnts, i, j, x_mat, u_mat, sf_type, N_bins,
            first_edge, last_edge, inv_step, offset, step_val,
        )
        k += nworkers
    end

    @inbounds for b in 1:NB
        # Always flush — partial sums can be exactly 0 while counts > 0 (signed SF values).
        @atomic output[b] += sums[b]
        if cnts[b] != zero(FT)
            @atomic counts[b] += cnts[b]
        end
    end
end

# ---------------------------------------------------------------------------
# Prototype B: block-local shared histogram (@localmem), grid-stride pairs
# ---------------------------------------------------------------------------

KA.@kernel function _proto_blockshared_linear!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nblocks::Int,
    workgroup_size::Int,
) where {FT}
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)

    shared_sums = @localmem FT (PROTO_MAX_BINS,)
    shared_cnts = @localmem FT (PROTO_MAX_BINS,)

    if lid == 1
        @inbounds for b in 1:NB
            shared_sums[b] = zero(FT)
            shared_cnts[b] = zero(FT)
        end
    end
    @synchronize

    total_pairs = N_points * (N_points - 1) ÷ 2
    pair_idx = (bid - 1) * workgroup_size + lid
    grid_stride = nblocks * workgroup_size

    while pair_idx <= total_pairs
        i, j = _pair_from_linear(pair_idx, N_points)
        X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
        X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
        U1 = SA.SVector{3, FT}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
        U2 = SA.SVector{3, FT}(u_mat[1, j], u_mat[2, j], u_mat[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            r̂ = SFH.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            @atomic shared_sums[bin] += val
            @atomic shared_cnts[bin] += one(FT)
        end
        pair_idx += grid_stride
    end
    @synchronize

    # Each thread flushes a strided subset of bins (standard block-histogram merge).
    b = lid
    while b <= NB
        # Always flush sums — block partial can be 0.0f0 while counts > 0 (L2 SF sign cancellation).
        @atomic output[b] += shared_sums[b]
        if shared_cnts[b] != zero(FT)
            @atomic counts[b] += shared_cnts[b]
        end
        b += workgroup_size
    end
end

# Register-private partials per thread, then one shared merge (cuts shared atomics ~×pairs-per-thread).
KA.@kernel function _proto_blockshared_regpriv_linear_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nblocks::Int,
    workgroup_size::Int,
) where {FT}
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)

    shared_sums = @localmem FT (PROTO_MAX_BINS,)
    shared_cnts = @localmem UInt32 (PROTO_MAX_BINS,)
    if lid == 1
        @inbounds for b in 1:NB
            shared_sums[b] = zero(FT)
            shared_cnts[b] = UInt32(0)
        end
    end
    @synchronize

    priv_sums = SA.MVector{PROTO_MAX_BINS, FT}(undef)
    priv_cnts = SA.MVector{PROTO_MAX_BINS, UInt32}(undef)
    @inbounds for b in 1:NB
        priv_sums[b] = zero(FT)
        priv_cnts[b] = UInt32(0)
    end

    total_pairs = N_points * (N_points - 1) ÷ 2
    pair_idx = (bid - 1) * workgroup_size + lid
    grid_stride = nblocks * workgroup_size

    while pair_idx <= total_pairs
        i, j = _pair_from_linear(pair_idx, N_points)
        X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
        X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
        U1 = SA.SVector{3, FT}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
        U2 = SA.SVector{3, FT}(u_mat[1, j], u_mat[2, j], u_mat[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            r̂ = SFH.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            priv_sums[bin] += val
            priv_cnts[bin] += UInt32(1)
        end
        pair_idx += grid_stride
    end

    @inbounds for b in 1:NB
        if priv_cnts[b] != UInt32(0)
            @atomic shared_sums[b] += priv_sums[b]
            @atomic shared_cnts[b] += priv_cnts[b]
        end
    end
    @synchronize

    b = lid
    while b <= NB
        @atomic output[b] += shared_sums[b]
        if shared_cnts[b] != UInt32(0)
            @atomic counts[b] += shared_cnts[b]
        end
        b += workgroup_size
    end
end

# ---------------------------------------------------------------------------
# CADISHI-style tiled kernels (generated for TILE × {regpriv,nosqrt,2d})
# ---------------------------------------------------------------------------

function _tiled_variant_name(tile::Int, opts::TiledOpts)
    parts = String[]
    push!(parts, "tiled$(tile)")
    opts.regpriv && push!(parts, "regpriv")
    opts.nosqrt && push!(parts, "nosqrt")
    opts.dim2 && push!(parts, "2d")
    opts.bin_kind !== :linear && push!(parts, string(opts.bin_kind))
    push!(parts, "u32")
    return join(parts, "_")
end

function _tiled_kernel_sym(tile::Int, opts::TiledOpts)
    return Symbol("_proto_", _tiled_variant_name(tile, opts), "_u32!")
end

const _TILED_KERNELS = Dict{Tuple{Int, TiledOpts}, Any}()

"""Build a parse-time `@kernel` expr (must not use runtime `@eval` — breaks CUDA + `MVector`)."""
function _tiled_kernel_def(TILE::Int, regpriv::Bool, nosqrt::Bool, dim2::Bool, bin_kind::Symbol = :linear)
    opts = TiledOpts(regpriv, nosqrt, dim2, bin_kind)
    sym = _tiled_kernel_sym(TILE, opts)
    bin_kind in (:linear, :log, :general) ||
        error("_tiled_kernel_def: unsupported bin_kind=$bin_kind")
    xdim = dim2 ? 2 : 3
    sh_len = TILE * xdim
    t1 = TILE
    t2 = 2 * TILE

    # CUDA: `MVector` in tile kernels spills to `gpu_gc_pool_alloc` (tile shmem + pair loop
    # exhaust registers). Same pattern works in blockshared/private. Regpriv tiled variants
    # accumulate directly into block `@localmem` histogram via shared atomics instead.
    regpriv_init = quote end
    regpriv_merge = quote end
    accum_expr = quote
        @atomic shared_sums[bin] += val
        @atomic shared_cnts[bin] += UInt32(1)
    end

    load_i = if dim2
        quote
            shared_xi[k] = x_mat[1, gi]
            shared_xi[$(t1) + k] = x_mat[2, gi]
            shared_ui[k] = u_mat[1, gi]
            shared_ui[$(t1) + k] = u_mat[2, gi]
        end
    else
        quote
            shared_xi[k] = x_mat[1, gi]
            shared_xi[$(t1) + k] = x_mat[2, gi]
            shared_xi[$(t2) + k] = x_mat[3, gi]
            shared_ui[k] = u_mat[1, gi]
            shared_ui[$(t1) + k] = u_mat[2, gi]
            shared_ui[$(t2) + k] = u_mat[3, gi]
        end
    end

    load_j = if dim2
        quote
            shared_xj[k] = x_mat[1, gj]
            shared_xj[$(t1) + k] = x_mat[2, gj]
            shared_uj[k] = u_mat[1, gj]
            shared_uj[$(t1) + k] = u_mat[2, gj]
        end
    else
        quote
            shared_xj[k] = x_mat[1, gj]
            shared_xj[$(t1) + k] = x_mat[2, gj]
            shared_xj[$(t2) + k] = x_mat[3, gj]
            shared_uj[k] = u_mat[1, gj]
            shared_uj[$(t1) + k] = u_mat[2, gj]
            shared_uj[$(t2) + k] = u_mat[3, gj]
        end
    end

    if dim2
        x1_cross = quote
            X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[$(t1) + ia])
            X2 = SA.SVector{2, FT}(shared_xj[jb], shared_xj[$(t1) + jb])
            U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[$(t1) + ia])
            U2 = SA.SVector{2, FT}(shared_uj[jb], shared_uj[$(t1) + jb])
        end
        x1_diag = quote
            X1 = SA.SVector{2, FT}(shared_xi[ia], shared_xi[$(t1) + ia])
            X2 = SA.SVector{2, FT}(shared_xi[jb], shared_xi[$(t1) + jb])
            U1 = SA.SVector{2, FT}(shared_ui[ia], shared_ui[$(t1) + ia])
            U2 = SA.SVector{2, FT}(shared_ui[jb], shared_ui[$(t1) + jb])
        end
        dist_sq_expr = quote dist_sq = dX[1]^2 + dX[2]^2 end
        rhat_expr = quote r̂ = dX / sqrt(dist_sq) end
    else
        x1_cross = quote
            X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[$(t1) + ia], shared_xi[$(t2) + ia])
            X2 = SA.SVector{3, FT}(shared_xj[jb], shared_xj[$(t1) + jb], shared_xj[$(t2) + jb])
            U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[$(t1) + ia], shared_ui[$(t2) + ia])
            U2 = SA.SVector{3, FT}(shared_uj[jb], shared_uj[$(t1) + jb], shared_uj[$(t2) + jb])
        end
        x1_diag = quote
            X1 = SA.SVector{3, FT}(shared_xi[ia], shared_xi[$(t1) + ia], shared_xi[$(t2) + ia])
            X2 = SA.SVector{3, FT}(shared_xi[jb], shared_xi[$(t1) + jb], shared_xi[$(t2) + jb])
            U1 = SA.SVector{3, FT}(shared_ui[ia], shared_ui[$(t1) + ia], shared_ui[$(t2) + ia])
            U2 = SA.SVector{3, FT}(shared_ui[jb], shared_ui[$(t1) + jb], shared_ui[$(t2) + jb])
        end
        dist_sq_expr = quote dist_sq = dX[1]^2 + dX[2]^2 + dX[3]^2 end
        rhat_expr = quote r̂ = SFH.r̂(X1, X2) end
    end

  bin_expr = if bin_kind === :linear
        if nosqrt
            quote bin = _gpu_digitize_linear_sq(dist_sq, first_edge, last_edge, step_val, N_bins) end
        else
            quote
                dist = sqrt(dist_sq)
                bin = _gpu_digitize_linear(
                    dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
                )
            end
        end
    elseif bin_kind === :log
        quote
            dist = sqrt(dist_sq)
            bin = _gpu_digitize_log(dist, edges, lut, e_min, N_bins)
        end
    else
        quote
            dist = sqrt(dist_sq)
            bin = _gpu_digitize_general(dist, distance_bins, N_bins)
        end
    end

    extra_param_exprs = if bin_kind === :linear
        (
            :(first_edge::FT),
            :(last_edge::FT),
            :(inv_step::FT),
            :(offset::FT),
            :(step_val::FT),
        )
    elseif bin_kind === :log
        (
            :(@Const(edges)),
            :(@Const(lut)),
            :(e_min::Int),
        )
    else
        (:(@Const(distance_bins)),)
    end

    return quote
        KA.@kernel function $(sym)(
            output,
            counts,
            x_mat,
            u_mat,
            sf_type,
            N_points::Int,
            N_dims::Int,
            N_bins::Int,
            NB::Int,
            $(extra_param_exprs...),
            n_tiles::Int,
            n_tile_blocks::Int,
            workgroup_size::Int,
        ) where {FT}
            lid = @index(Local, Linear)
            bid = @index(Group, Linear)
            ws = workgroup_size

            shared_xi = @localmem FT ($(sh_len),)
            shared_ui = @localmem FT ($(sh_len),)
            shared_xj = @localmem FT ($(sh_len),)
            shared_uj = @localmem FT ($(sh_len),)
            shared_sums = @localmem FT (PROTO_MAX_BINS,)
            shared_cnts = @localmem UInt32 (PROTO_MAX_BINS,)

            if lid == 1
                @inbounds for b in 1:NB
                    shared_sums[b] = zero(FT)
                    shared_cnts[b] = UInt32(0)
                end
            end
            @synchronize

            $(regpriv_init)

            if bid <= n_tile_blocks
                ti, tj = _tile_from_linear(bid, n_tiles)
                i0 = (ti - 1) * $(TILE) + 1
                j0 = (tj - 1) * $(TILE) + 1
                ni = min($(TILE), N_points - i0 + 1)
                nj = min($(TILE), N_points - j0 + 1)
                if ni > 0 && nj > 0
                    k = lid
                    while k <= ni
                        gi = i0 + k - 1
                        @inbounds $(load_i)
                        k += ws
                    end
                    if ti < tj
                        k = lid
                        while k <= nj
                            gj = j0 + k - 1
                            @inbounds $(load_j)
                            k += ws
                        end
                    end
                    @synchronize

                    n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
                    p = lid
                    while p <= n_pairs
                        if ti < tj
                            ia = (p - 1) ÷ nj + 1
                            jb = (p - 1) - (ia - 1) * nj + 1
                        else
                            ia, jb = _pair_from_linear(p, ni)
                        end
                        if ti < tj
                            $(x1_cross)
                        else
                            $(x1_diag)
                        end
                        dX = X2 - X1
                        $(dist_sq_expr)
                        $(bin_expr)
                        if 1 <= bin < N_bins
                            $(rhat_expr)
                            val = sf_type(U2 - U1, r̂)
                            $(accum_expr)
                        end
                        p += ws
                    end
                    $(regpriv_merge)
                    @synchronize

                    b = lid
                    while b <= NB
                        @atomic output[b] += shared_sums[b]
                        if shared_cnts[b] != UInt32(0)
                            @atomic counts[b] += shared_cnts[b]
                        end
                        b += ws
                    end
                end
            end
        end
    end
end

macro proto_tiled_kernel(TILE, regpriv, nosqrt, dim2)
    tile = Int(TILE)
    rp = regpriv === true
    ns = nosqrt === true
    d2 = dim2 === true
    return esc(_tiled_kernel_def(tile, rp, ns, d2, :linear))
end

function _macro_sym_arg(x, name::String)
    x isa Symbol && return x
    x isa QuoteNode && return x.value
    if x isa Expr && x.head === :quote
        return x.args[1]
    end
    error("$name: expected Symbol argument, got $(repr(x))")
end

macro proto_tiled_kernel_bins(TILE, dim2, bin_kind)
    tile = Int(TILE)
    d2 = dim2 === true
    bk = _macro_sym_arg(bin_kind, "proto_tiled_kernel_bins")
    return esc(_tiled_kernel_def(tile, false, false, d2, bk))
end

# Parse-time expansion (16 combinations); runtime `@eval` breaks CUDA regpriv `MVector`.
@proto_tiled_kernel 64 false false false
@proto_tiled_kernel 64 false false true
@proto_tiled_kernel 64 false true false
@proto_tiled_kernel 64 false true true
@proto_tiled_kernel 64 true false false
@proto_tiled_kernel 64 true false true
@proto_tiled_kernel 64 true true false
@proto_tiled_kernel 64 true true true
@proto_tiled_kernel 128 false false false
@proto_tiled_kernel 128 false false true
@proto_tiled_kernel 128 false true false
@proto_tiled_kernel 128 false true true
@proto_tiled_kernel 128 true false false
@proto_tiled_kernel 128 true false true
@proto_tiled_kernel 128 true true false
@proto_tiled_kernel 128 true true true

# Log/general tiled128 variants for SLURM timing vs legacy (N,N) paths.
@proto_tiled_kernel_bins 128 false :log
@proto_tiled_kernel_bins 128 true :log
@proto_tiled_kernel_bins 128 false :general
@proto_tiled_kernel_bins 128 true :general

function _register_tiled_kernels!()
    empty!(_TILED_KERNELS)
    mod = @__MODULE__
    for TILE in (64, 128)
        for regpriv in (false, true)
            for nosqrt in (false, true)
                for dim2 in (false, true)
                    opts = TiledOpts(regpriv, nosqrt, dim2)
                    sym = _tiled_kernel_sym(TILE, opts)
                    _TILED_KERNELS[(TILE, opts)] = getfield(mod, sym)
                end
            end
        end
    end
    return nothing
end

_register_tiled_kernels!()

function _register_tiled_bin_kernels!()
    mod = @__MODULE__
    for dim2 in (false, true)
        for bin_kind in (:log, :general)
            opts = TiledOpts(false, false, dim2, bin_kind)
            sym = _tiled_kernel_sym(128, opts)
            _TILED_KERNELS[(128, opts)] = getfield(mod, sym)
        end
    end
    return nothing
end

_register_tiled_bin_kernels!()

function _tiled_prototype_configs(nstride::Int)
    configs = PrototypeConfig[]
    for TILE in (64, 128)
        for regpriv in (false, true)
            for nosqrt in (false, true)
                for dim2 in (false, true)
                    opts = TiledOpts(regpriv, nosqrt, dim2)
                    name = _tiled_variant_name(TILE, opts) * "_w256"
                    push!(
                        configs,
                        PrototypeConfig(name, :tiled_u32, nstride, 256, false, TILE, opts),
                    )
                end
            end
        end
    end
    push!(
        configs,
        PrototypeConfig("tiled64_w128_u32", :tiled_u32, nstride, 128, false, 64, TILED_OPTS_NONE),
    )
    for dim2 in (false, true)
        tag = dim2 ? "2d" : "3d"
        for bin_kind in (:log, :general)
            opts = TiledOpts(false, false, dim2, bin_kind)
            push!(
                configs,
                PrototypeConfig(
                    "tiled128_$(tag)_$(bin_kind)_u32_w256",
                    :tiled_u32,
                    nstride,
                    256,
                    false,
                    128,
                    opts,
                ),
            )
        end
    end
    return configs
end

# Block-local histogram with UInt32 counts (full SF + exact global count merge).
KA.@kernel function _proto_blockshared_linear_u32!(
    output,
    counts,
    x_mat,
    u_mat,
    sf_type,
    N_points::Int,
    N_dims::Int,
    N_bins::Int,
    NB::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nblocks::Int,
    workgroup_size::Int,
) where {FT}
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)

    shared_sums = @localmem FT (PROTO_MAX_BINS,)
    shared_cnts = @localmem UInt32 (PROTO_MAX_BINS,)

    if lid == 1
        @inbounds for b in 1:NB
            shared_sums[b] = zero(FT)
            shared_cnts[b] = UInt32(0)
        end
    end
    @synchronize

    total_pairs = N_points * (N_points - 1) ÷ 2
    pair_idx = (bid - 1) * workgroup_size + lid
    grid_stride = nblocks * workgroup_size

    while pair_idx <= total_pairs
        i, j = _pair_from_linear(pair_idx, N_points)
        X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
        X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
        U1 = SA.SVector{3, FT}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
        U2 = SA.SVector{3, FT}(u_mat[1, j], u_mat[2, j], u_mat[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(
            dist, first_edge, last_edge, inv_step, offset, step_val, N_bins,
        )
        if 1 <= bin < N_bins
            r̂ = SFH.r̂(X1, X2)
            val = sf_type(U2 - U1, r̂)
            @atomic shared_sums[bin] += val
            @atomic shared_cnts[bin] += UInt32(1)
        end
        pair_idx += grid_stride
    end
    @synchronize

    b = lid
    while b <= NB
        @atomic output[b] += shared_sums[b]
        if shared_cnts[b] != UInt32(0)
            @atomic counts[b] += shared_cnts[b]
        end
        b += workgroup_size
    end
end

# ---------------------------------------------------------------------------
# Device-side 2D → 3D padding (for device-resident staging prototype)
# ---------------------------------------------------------------------------

KA.@kernel function _proto_pad3_kernel!(x3, u3, x_mat, u_mat, N_dims::Int, N_points::Int)
    j = @index(Global, Linear)
    FT = eltype(x3)
    if j <= N_points
        @inbounds for d in 1:3
            x3[d, j] = d <= N_dims ? x_mat[d, j] : zero(FT)
            u3[d, j] = d <= N_dims ? u_mat[d, j] : zero(FT)
        end
    end
end

# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

mutable struct DeviceBuffers
    x_dev
    u_dev
    out_dev
    cnt_dev
    N_points::Int
    N_bins::Int
    NB::Int
end

"""
    prepare_device_buffers(backend, x_mat, u_mat, n_bins)

Upload once; reuse across repeated prototype calls (device-resident path).
"""
function prepare_device_buffers(
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    n_bins::Int,
) where {FT}
    N_dims, N_points = size(x_mat)
    NB = n_bins - 1

    x_dev = KA.allocate(backend, FT, 3, N_points)
    u_dev = KA.allocate(backend, FT, 3, N_points)
    out_dev = KA.zeros(backend, FT, NB)
    cnt_dev = KA.zeros(backend, FT, NB)

    if N_dims == 3
        copyto!(x_dev, x_mat)
        copyto!(u_dev, u_mat)
    else
        x_raw = KA.allocate(backend, FT, N_dims, N_points)
        u_raw = KA.allocate(backend, FT, N_dims, N_points)
        copyto!(x_raw, x_mat)
        copyto!(u_raw, u_mat)
        pad! = _proto_pad3_kernel!(backend, 256)
        pad!(x_dev, u_dev, x_raw, u_raw, N_dims, N_points; ndrange = N_points)
    end

    return DeviceBuffers(x_dev, u_dev, out_dev, cnt_dev, N_points, n_bins, NB)
end

function _zero_device_outputs!(bufs::DeviceBuffers)
    FT = eltype(bufs.out_dev)
    z = zeros(FT, bufs.NB)
    copyto!(bufs.out_dev, z)
    copyto!(bufs.cnt_dev, z)
    return nothing
end

function _stage_host_to_device(
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    n_bins::Int;
    count_eltype::Type = FT,
) where {FT}
    N_dims, N_points = size(x_mat)
    NB = n_bins - 1

    x_host = Array(x_mat)
    u_host = Array(u_mat)
    x3 = zeros(FT, 3, N_points)
    u3 = zeros(FT, 3, N_points)
    x3[1:N_dims, :] .= x_host
    u3[1:N_dims, :] .= u_host

    x_dev = KA.allocate(backend, FT, 3, N_points)
    u_dev = KA.allocate(backend, FT, 3, N_points)
    out_dev = KA.zeros(backend, FT, NB)
    cnt_dev = KA.zeros(backend, count_eltype, NB)
    copyto!(x_dev, x3)
    copyto!(u_dev, u3)
    return x_dev, u_dev, out_dev, cnt_dev, N_points, N_dims
end

function _launch_variant!(
    cfg::PrototypeConfig,
    backend::KA.Backend,
    x_dev,
    u_dev,
    out_dev,
    cnt_dev,
    sf_type,
    layout::ProtoBinLayout,
    N_points::Int,
    N_dims::Int,
)
    N_bins = layout.n_bins
    NB = N_bins - 1
    lp = layout.linear
    fe = lp === nothing ? zero(eltype(x_dev)) : lp.first_edge
    le = lp === nothing ? zero(eltype(x_dev)) : lp.last_edge
    is_ = lp === nothing ? zero(eltype(x_dev)) : lp.inv_step
    off = lp === nothing ? zero(eltype(x_dev)) : lp.offset
    sv = lp === nothing ? zero(eltype(x_dev)) : lp.step_val

    if cfg.variant === :baseline
        kernel! = _proto_baseline_linear!(backend, cfg.workgroup_size)
        kernel!(
            out_dev, cnt_dev, x_dev, u_dev, sf_type,
            N_points, N_dims, N_bins, fe, le, is_, off, sv;
            ndrange = (N_points, N_points),
        )
    elseif cfg.variant === :baseline_linear
        kernel! = _proto_baseline_linear_pairs!(backend, cfg.workgroup_size)
        kernel!(
            out_dev, cnt_dev, x_dev, u_dev, sf_type,
            N_points, N_dims, N_bins, fe, le, is_, off, sv, cfg.nworkers;
            ndrange = cfg.nworkers,
        )
    elseif cfg.variant === :baseline_linear_u32
        kernel! = _proto_baseline_linear_pairs_u32!(backend, cfg.workgroup_size)
        kernel!(
            out_dev, cnt_dev, x_dev, u_dev, sf_type,
            N_points, N_dims, N_bins, fe, le, is_, off, sv, cfg.nworkers;
            ndrange = cfg.nworkers,
        )
    elseif cfg.variant === :private
        kernel! = _proto_private_linear!(backend, cfg.workgroup_size)
        kernel!(
            out_dev, cnt_dev, x_dev, u_dev, sf_type,
            N_points, N_dims, N_bins, NB, fe, le, is_, off, sv, cfg.nworkers;
            ndrange = cfg.nworkers,
        )
    elseif cfg.variant === :blockshared
        nblocks = cld(cfg.nworkers, cfg.workgroup_size)
        kernel! = _proto_blockshared_linear!(backend, cfg.workgroup_size)
        kernel!(
            out_dev, cnt_dev, x_dev, u_dev, sf_type,
            N_points, N_dims, N_bins, NB, fe, le, is_, off, sv, nblocks, cfg.workgroup_size;
            ndrange = nblocks * cfg.workgroup_size,
        )
    elseif cfg.variant === :blockshared_u32
        nblocks = cld(cfg.nworkers, cfg.workgroup_size)
        kernel! = _proto_blockshared_linear_u32!(backend, cfg.workgroup_size)
        kernel!(
            out_dev, cnt_dev, x_dev, u_dev, sf_type,
            N_points, N_dims, N_bins, NB, fe, le, is_, off, sv, nblocks, cfg.workgroup_size;
            ndrange = nblocks * cfg.workgroup_size,
        )
    elseif cfg.variant === :blockshared_regpriv_u32
        nblocks = cld(cfg.nworkers, cfg.workgroup_size)
        kernel! = _proto_blockshared_regpriv_linear_u32!(backend, cfg.workgroup_size)
        kernel!(
            out_dev, cnt_dev, x_dev, u_dev, sf_type,
            N_points, N_dims, N_bins, NB, fe, le, is_, off, sv, nblocks, cfg.workgroup_size;
            ndrange = nblocks * cfg.workgroup_size,
        )
    elseif cfg.variant === :tiled_u32
        TILE = cfg.tile_size
        opts = something(cfg.tiled_opts, TILED_OPTS_NONE)
        key = (TILE, opts)
        haskey(_TILED_KERNELS, key) ||
            error("No tiled kernel registered for tile=$TILE opts=$opts")
        n_tiles = cld(N_points, TILE)
        n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
        kernel! = _TILED_KERNELS[key](backend, cfg.workgroup_size)
        if opts.bin_kind === :linear
            kernel!(
                out_dev, cnt_dev, x_dev, u_dev, sf_type,
                N_points, N_dims, N_bins, NB, fe, le, is_, off, sv,
                n_tiles, n_tile_blocks, cfg.workgroup_size;
                ndrange = n_tile_blocks * cfg.workgroup_size,
            )
        elseif opts.bin_kind === :log
            lbe = layout.log
            FT = eltype(lbe.edges)
            edges_dev = KA.allocate(backend, FT, N_bins)
            lut_dev = KA.allocate(backend, Int32, length(lbe.lut))
            copyto!(edges_dev, collect(lbe.edges))
            copyto!(lut_dev, Int32.(lbe.lut))
            kernel!(
                out_dev, cnt_dev, x_dev, u_dev, sf_type,
                N_points, N_dims, N_bins, NB,
                edges_dev, lut_dev, lbe.e_min,
                n_tiles, n_tile_blocks, cfg.workgroup_size;
                ndrange = n_tile_blocks * cfg.workgroup_size,
            )
        else
            FT = eltype(layout.general_edges)
            bins_dev = KA.allocate(backend, FT, N_bins)
            copyto!(bins_dev, layout.general_edges)
            kernel!(
                out_dev, cnt_dev, x_dev, u_dev, sf_type,
                N_points, N_dims, N_bins, NB, bins_dev,
                n_tiles, n_tile_blocks, cfg.workgroup_size;
                ndrange = n_tile_blocks * cfg.workgroup_size,
            )
        end
    else
        error("Unknown prototype variant $(cfg.variant)")
    end
    return nothing
end

"""
    run_prototype!(cfg, backend, sf_type, bin_edges, x_mat, u_mat; bufs=nothing)

Run one prototype variant. Returns `PrototypeResult` with kernel and staging times.
"""
function run_prototype!(
    cfg::PrototypeConfig,
    backend::KA.Backend,
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT};
    bufs::Union{DeviceBuffers, Nothing} = nothing,
    download::Bool = true,
) where {FT}
    layout = prototype_bin_layout(bin_edges)
    lp_n_bins = layout.n_bins
    staging_s = 0.0
    use_u32 = cfg.variant in _U32_COUNT_VARIANTS

    if cfg.device_resident && bufs !== nothing
        use_u32 && error("UInt32 count variants do not support device-resident buffers yet")
        _zero_device_outputs!(bufs)
        x_dev, u_dev, out_dev, cnt_dev = bufs.x_dev, bufs.u_dev, bufs.out_dev, bufs.cnt_dev
        N_points, N_dims = bufs.N_points, size(x_mat, 1)
    else
        t_stage = @elapsed begin
            x_dev, u_dev, out_dev, cnt_dev, N_points, N_dims = _stage_host_to_device(
                backend, x_mat, u_mat, lp_n_bins; count_eltype = use_u32 ? UInt32 : FT,
            )
        end
        staging_s = t_stage
    end

    kernel_s = @elapsed begin
        _launch_variant!(
            cfg, backend, x_dev, u_dev, out_dev, cnt_dev,
            sf_type, layout, N_points, N_dims,
        )
        KA.synchronize(backend)
    end

    counts_eltype = use_u32 ? :u32 : :f32

    if download
        sums = Array(out_dev)
        counts_i64 = _host_counts_i64(cnt_dev)
        counts = Float32.(counts_i64)
    else
        sums = out_dev
        counts = cnt_dev
        counts_i64 = cnt_dev
    end

    return PrototypeResult(sums, counts, counts_i64, sum(counts_i64), counts_eltype, kernel_s, staging_s)
end

"""Default prototype sweep (linear bins). Tune via `ENV["N"]`."""
function prototype_variants(N::Int)
    total_pairs = N * (N - 1) ÷ 2
    ws = 256
    nstride = min(262_144, total_pairs)
    base = [
        # Production-like (N,N) grid — may under-count at large N; kept for comparison.
        PrototypeConfig("baseline_grid_N2", :baseline, 0, 64, false),
        # Canonical reference: same pair loop as optimized kernels, global atomics (Float32 counts).
        PrototypeConfig("baseline_linear_global", :baseline_linear, nstride, ws, false),
        # Same global path; UInt32 counts (correct at large N, same algorithm).
        PrototypeConfig("baseline_linear_global_u32", :baseline_linear_u32, nstride, ws, false),
        PrototypeConfig("private_4k_workers", :private, 4_096, ws, false),
        PrototypeConfig("private_32k_workers", :private, 32_768, ws, false),
        PrototypeConfig("private_256k_workers", :private, nstride, ws, false),
        PrototypeConfig("blockshared_32k_w256", :blockshared, 32_768, ws, false),
        PrototypeConfig("blockshared_256k_w256", :blockshared, nstride, ws, false),
        PrototypeConfig("blockshared_256k_w256_u32", :blockshared_u32, nstride, ws, false),
        PrototypeConfig("blockshared_256k_regpriv_u32", :blockshared_regpriv_u32, nstride, ws, false),
        PrototypeConfig("private_32k_device_resident", :private, 32_768, ws, true),
    ]
    return vcat(base, _tiled_prototype_configs(nstride))
end

"""Parity reference config (linear pair index + global atomic)."""
function parity_reference_config(N::Int)
    return prototype_variants(N)[2]
end

# ---------------------------------------------------------------------------
# CPU reference (gold) — same pair index, digitize, and SF logic as GPU kernels
# ---------------------------------------------------------------------------

"""Pad `x_mat`/`u_mat` to 3×N the same way `_stage_host_to_device` does."""
function _pad3_host(x_mat::AbstractMatrix{FT}, u_mat::AbstractMatrix{FT}) where {FT}
    N_dims, N_points = size(x_mat)
    x3 = zeros(FT, 3, N_points)
    u3 = zeros(FT, 3, N_points)
    x3[1:N_dims, :] .= x_mat
    u3[1:N_dims, :] .= u_mat
    return x3, u3, N_dims
end

"""Verify `_pair_from_linear` hits every `(i,j)` with `i < j` exactly once."""
function verify_pair_enumeration(N::Int)
    total_pairs = N * (N - 1) ÷ 2
    seen = falses(N, N)
    for k in 1:total_pairs
        i, j = _pair_from_linear(k, N)
        (i >= j) && error("pair index $k gave i=$i, j=$j (expected i < j)")
        seen[i, j] && error("duplicate pair (i=$i, j=$j) at k=$k")
        seen[i, j] = true
    end
    n_seen = count(seen)
    n_seen == total_pairs || error("pair enumeration incomplete: $n_seen / $total_pairs")
    return total_pairs
end

"""
    cpu_gold_histogram(x_mat, u_mat, sf_type, bin_edges; count_only=false)

Serial CPU reference using the **same** `_pair_from_linear`, `_gpu_digitize_linear`,
3D padding, and (unless `count_only`) `sf_type` as the GPU prototype kernels.
"""
function cpu_gold_histogram(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    count_only::Bool = false,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_pair_enumeration(N_points)
    total_pairs = N_points * (N_points - 1) ÷ 2
    x3, u3, N_dims = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    sums = zeros(FT, NB)
    counts = zeros(FT, NB)
    counts_i64 = zeros(Int64, NB)
    n_in = 0
    n_out = 0
    n_bin0 = 0
    n_bin21 = 0
    n_valid = 0
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
    for k in 1:total_pairs
        i, j = _pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
        X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
        if bin == 0
            n_bin0 += 1
        elseif bin >= lp.n_bins
            n_bin21 += 1
        elseif 1 <= bin < lp.n_bins
            n_valid += 1
            n_in += 1
            counts_i64[bin] += 1
            counts[bin] += one(FT)
            if !count_only
                U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
                U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
                r̂ = SFH.r̂(X1, X2)
                sums[bin] += sf_type(U2 - U1, r̂)
            end
        else
            n_out += 1
        end
    end
    sum_counts_i64 = sum(counts_i64)
    sum_counts_f32 = sum(counts)
    n_in == sum_counts_i64 || error("CPU gold: n_in ($n_in) != sum(counts_i64) ($sum_counts_i64)")
    n_in + n_out + n_bin0 + n_bin21 == total_pairs ||
        error("CPU gold bin buckets: $(n_in + n_out + n_bin0 + n_bin21) != $total_pairs")
    max_bin_count, max_bin = findmax(counts_i64)
    float_lost = n_in - Int64(round(sum_counts_f32))
    return (;
        sums,
        counts,
        counts_i64,
        sum_counts = sum_counts_i64,
        sum_counts_f32,
        float_lost,
        max_bin,
        max_bin_count,
        n_in,
        n_out,
        n_bin0,
        n_bin21,
        n_valid,
        total_pairs,
        n_in_plus_n_out = n_in + n_out + n_bin0 + n_bin21,
    )
end

"""Verify tiled pair schedule hits every `(i,j)` with `i < j` exactly once."""
function verify_tile_enumeration(N::Int, TILE::Int)
    total_pairs = N * (N - 1) ÷ 2
    n_tiles = cld(N, TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    seen = falses(N, N)
    for bid in 1:n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * TILE + 1
        j0 = (tj - 1) * TILE + 1
        ni = min(TILE, N - i0 + 1)
        nj = min(TILE, N - j0 + 1)
        (ni > 0 && nj > 0) || continue
        n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
        for p in 1:n_pairs
            if ti < tj
                ia = (p - 1) ÷ nj + 1
                jb = (p - 1) - (ia - 1) * nj + 1
                i = i0 + ia - 1
                j = j0 + jb - 1
            else
                ia, jb = _pair_from_linear(p, ni)
                i = i0 + ia - 1
                j = i0 + jb - 1
            end
            (i < j) || error("tile pair ($i,$j) at bid=$bid p=$p (ti=$ti,tj=$tj)")
            seen[i, j] && error("duplicate pair ($i,$j) at bid=$bid p=$p")
            seen[i, j] = true
        end
    end
    n_seen = count(seen)
    n_seen == total_pairs || error("tile enumeration incomplete: $n_seen / $total_pairs")
    return total_pairs
end

@inline function _proto_digitize_dist(
    dist,
    layout::ProtoBinLayout,
    n_bins::Int,
)
    if layout.kind === :linear
        lp = layout.linear
        return _gpu_digitize_linear(
            dist, lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val, n_bins,
        )
    elseif layout.kind === :log
        lp = layout.log
        return _gpu_digitize_log(dist, lp.edges, lp.lut, lp.e_min, n_bins)
    else
        return _gpu_digitize_general(dist, layout.general_edges, n_bins)
    end
end

"""
    cpu_tiled_gold_histogram(x_mat, u_mat, sf_type, bin_edges; tile=128, dim2=false, count_only=false)

CPU reference using the **same** tiled pair schedule as `_proto_tiled*_u32!` kernels.
"""
function cpu_tiled_gold_histogram(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges;
    tile::Int = 128,
    dim2::Bool = false,
    count_only::Bool = false,
) where {FT}
    layout = prototype_bin_layout(bin_edges)
    N_points = size(x_mat, 2)
    verify_tile_enumeration(N_points, tile)
    x3, u3, N_dims = _pad3_host(x_mat, u_mat)
    n_bins = layout.n_bins
    NB = n_bins - 1
    sums = zeros(FT, NB)
    counts_i64 = zeros(Int64, NB)
    n_tiles = cld(N_points, tile)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2

    for bid in 1:n_tile_blocks
        ti, tj = _tile_from_linear(bid, n_tiles)
        i0 = (ti - 1) * tile + 1
        j0 = (tj - 1) * tile + 1
        ni = min(tile, N_points - i0 + 1)
        nj = min(tile, N_points - j0 + 1)
        (ni > 0 && nj > 0) || continue
        n_pairs = ti < tj ? ni * nj : ni * (ni - 1) ÷ 2
        for p in 1:n_pairs
            if ti < tj
                ia = (p - 1) ÷ nj + 1
                jb = (p - 1) - (ia - 1) * nj + 1
                i = i0 + ia - 1
                j = j0 + jb - 1
            else
                ia, jb = _pair_from_linear(p, ni)
                i = i0 + ia - 1
                j = i0 + jb - 1
            end
            if dim2
                X1 = SA.SVector{2, FT}(x3[1, i], x3[2, i])
                X2 = SA.SVector{2, FT}(x3[1, j], x3[2, j])
                U1 = SA.SVector{2, FT}(u3[1, i], u3[2, i])
                U2 = SA.SVector{2, FT}(u3[1, j], u3[2, j])
                dX = X2 - X1
                dist_sq = dX[1]^2 + dX[2]^2
                dist = sqrt(dist_sq)
                r̂ = dist_sq > zero(FT) ? dX / dist : SA.SVector{2, FT}(zero(FT), zero(FT))
            else
                X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
                X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
                U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
                U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
                dX = X2 - X1
                dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
                r̂ = SFH.r̂(X1, X2)
            end
            bin = _proto_digitize_dist(dist, layout, n_bins)
            if 1 <= bin < n_bins
                counts_i64[bin] += 1
                if !count_only
                    sums[bin] += sf_type(U2 - U1, r̂)
                end
            end
        end
    end

    return (;
        sums,
        counts = FT.(counts_i64),
        counts_i64,
        sum_counts = sum(counts_i64),
        n_in = sum(counts_i64),
        tile,
        dim2,
        layout_kind = layout.kind,
    )
end

"""Shared pair body for CPU chunked simulators (matches GPU digitize + SF)."""
function _cpu_accumulate_pair!(
    sums::AbstractVector{FT},
    counts_i64::AbstractVector{Int64},
    i::Int,
    j::Int,
    x3,
    u3,
    sf_type::SFT.AbstractStructureFunctionType,
    lp;
    count_only::Bool = false,
) where {FT}
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
    X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
    X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
    dX = X2 - X1
    dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
    bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
    if 1 <= bin < lp.n_bins
        counts_i64[bin] += 1
        if !count_only
            U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
            U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
            r̂ = SFH.r̂(X1, X2)
            sums[bin] += sf_type(U2 - U1, r̂)
        end
    end
    return nothing
end

"""Verify grid-stride workers partition `1:total_pairs` without overlap or gaps."""
function verify_stride_partition(N::Int, nworkers::Int)
    total_pairs = N * (N - 1) ÷ 2
    seen = falses(total_pairs)
    for worker in 1:nworkers
        k = worker
        while k <= total_pairs
            seen[k] && error("duplicate pair index k=$k on worker=$worker")
            seen[k] = true
            k += nworkers
        end
    end
    n_seen = count(seen)
    n_seen == total_pairs ||
        error("stride partition incomplete: $n_seen / $total_pairs (nworkers=$nworkers)")
    return total_pairs
end

"""
    cpu_stride_global_histogram(...)

CPU simulator for `_proto_baseline_linear_pairs!`: each worker grid-strides `k += nworkers`,
accumulating directly into global histogram (serial merge, no atomics).
"""
function cpu_stride_global_histogram(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
    count_only::Bool = false,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    sums = zeros(FT, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    for worker in 1:nworkers
        k = worker
        while k <= total_pairs
            i, j = _pair_from_linear(k, N_points)
            _cpu_accumulate_pair!(sums, counts_i64, i, j, x3, u3, sf_type, lp; count_only)
            k += nworkers
        end
    end
    return (;
        sums,
        counts_i64,
        sum_counts = sum(counts_i64),
        nworkers,
        workgroup_size = 0,
        nblocks = 0,
        path = :stride_global,
    )
end

"""
    cpu_private_histogram(...)

CPU simulator for `_proto_private_linear!`: register-private partial per worker, then merge.
"""
function cpu_private_histogram(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
    count_only::Bool = false,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    sums = zeros(FT, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    worker_sums = zeros(FT, NB)
    worker_cnts = zeros(Int64, NB)
    for worker in 1:nworkers
        fill!(worker_sums, zero(FT))
        fill!(worker_cnts, 0)
        k = worker
        while k <= total_pairs
            i, j = _pair_from_linear(k, N_points)
            _cpu_accumulate_pair!(worker_sums, worker_cnts, i, j, x3, u3, sf_type, lp; count_only)
            k += nworkers
        end
        @inbounds for b in 1:NB
            sums[b] += worker_sums[b]
            counts_i64[b] += worker_cnts[b]
        end
    end
    return (;
        sums,
        counts_i64,
        sum_counts = sum(counts_i64),
        nworkers,
        workgroup_size = 0,
        nblocks = 0,
        path = :private,
    )
end

"""
    cpu_private_histogram_f64(...)

Same as `cpu_private_histogram` but worker partial sums use `Float64` (tests Float32 merge-loss hypothesis).
"""
function cpu_private_histogram_f64(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
    count_only::Bool = false,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    sums = zeros(FT, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    worker_sums = zeros(Float64, NB)
    worker_cnts = zeros(Int64, NB)
    for worker in 1:nworkers
        fill!(worker_sums, 0.0)
        fill!(worker_cnts, 0)
        k = worker
        while k <= total_pairs
            i, j = _pair_from_linear(k, N_points)
            fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
            X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
            X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
            dX = X2 - X1
            dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
            bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
            if 1 <= bin < lp.n_bins
                worker_cnts[bin] += 1
                if !count_only
                    U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
                    U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
                    r̂ = SFH.r̂(X1, X2)
                    worker_sums[bin] += Float64(sf_type(U2 - U1, r̂))
                end
            end
            k += nworkers
        end
        @inbounds for b in 1:NB
            sums[b] += FT(worker_sums[b])
            counts_i64[b] += worker_cnts[b]
        end
    end
    return (;
        sums,
        counts_i64,
        sum_counts = sum(counts_i64),
        nworkers,
        workgroup_size = 0,
        nblocks = 0,
        path = :private_f64,
    )
end

"""
    cpu_blockshared_histogram(...)

CPU simulator for `_proto_blockshared_linear!`: block-local partial histograms, then merge.
"""
function cpu_blockshared_histogram(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
    workgroup_size::Int = 256,
    count_only::Bool = false,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    nblocks = cld(nworkers, workgroup_size)
    grid_stride = nblocks * workgroup_size
    sums = zeros(FT, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    block_sums = zeros(FT, NB)
    block_cnts = zeros(Int64, NB)
    for bid in 1:nblocks
        fill!(block_sums, zero(FT))
        fill!(block_cnts, 0)
        for lid in 1:workgroup_size
            pair_idx = (bid - 1) * workgroup_size + lid
            while pair_idx <= total_pairs
                i, j = _pair_from_linear(pair_idx, N_points)
                _cpu_accumulate_pair!(
                    block_sums, block_cnts, i, j, x3, u3, sf_type, lp; count_only,
                )
                pair_idx += grid_stride
            end
        end
        @inbounds for b in 1:NB
            sums[b] += block_sums[b]
            counts_i64[b] += block_cnts[b]
        end
    end
    return (;
        sums,
        counts_i64,
        sum_counts = sum(counts_i64),
        nworkers,
        workgroup_size,
        nblocks,
        path = :blockshared,
    )
end

"""
    cpu_f64_serial_sums(...)

Serial `k = 1:total_pairs` loop with **Float64** bin sums (each SF value promoted once).
Not assumed exact — cross-check with `cpu_private_all_f64` and/or BigFloat before use as reference.
"""
function cpu_f64_serial_sums(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT},
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    sums = zeros(Float64, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
    for k in 1:total_pairs
        i, j = _pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
        X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
        if 1 <= bin < lp.n_bins
            counts_i64[bin] += 1
            U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
            U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
            r̂ = SFH.r̂(X1, X2)
            sums[bin] += Float64(sf_type(U2 - U1, r̂))
        end
    end
    return (;
        sums,
        sums_f32 = FT.(sums),
        counts_i64,
        sum_counts = sum(counts_i64),
        path = :f64_serial,
    )
end

"""
    cpu_private_all_f64(...)

Private worker partials and global merge **all in Float64** (no Float32 in sum path).
Must agree with `cpu_f64_serial_sums` to machine precision if both are valid references.
"""
function cpu_private_all_f64(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    sums = zeros(Float64, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    worker_sums = zeros(Float64, NB)
    worker_cnts = zeros(Int64, NB)
    for worker in 1:nworkers
        fill!(worker_sums, 0.0)
        fill!(worker_cnts, 0)
        k = worker
        while k <= total_pairs
            i, j = _pair_from_linear(k, N_points)
            _cpu_accumulate_pair_f64!(worker_sums, worker_cnts, i, j, x3, u3, sf_type, lp)
            k += nworkers
        end
        @inbounds for b in 1:NB
            sums[b] += worker_sums[b]
            counts_i64[b] += worker_cnts[b]
        end
    end
    return (;
        sums,
        sums_f32 = FT.(sums),
        counts_i64,
        sum_counts = sum(counts_i64),
        path = :private_all_f64,
    )
end

"""
    cpu_stride_all_f64(...)

Grid-stride workers accumulating directly into global **Float64** histogram
(same pair schedule as `cpu_stride_global_histogram`).
"""
function cpu_stride_all_f64(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    sums = zeros(Float64, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    for worker in 1:nworkers
        k = worker
        while k <= total_pairs
            i, j = _pair_from_linear(k, N_points)
            _cpu_accumulate_pair_f64!(sums, counts_i64, i, j, x3, u3, sf_type, lp)
            k += nworkers
        end
    end
    return (;
        sums,
        sums_f32 = FT.(sums),
        counts_i64,
        sum_counts = sum(counts_i64),
        path = :stride_all_f64,
    )
end

"""
    cpu_blockshared_all_f64(...)

Block-local partial histograms merged in **Float64**
(same schedule as `cpu_blockshared_histogram`).
"""
function cpu_blockshared_all_f64(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
    workgroup_size::Int = 256,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    nblocks = cld(nworkers, workgroup_size)
    grid_stride = nblocks * workgroup_size
    sums = zeros(Float64, NB)
    counts_i64 = zeros(Int64, NB)
    total_pairs = N_points * (N_points - 1) ÷ 2
    block_sums = zeros(Float64, NB)
    block_cnts = zeros(Int64, NB)
    for bid in 1:nblocks
        fill!(block_sums, 0.0)
        fill!(block_cnts, 0)
        for lid in 1:workgroup_size
            pair_idx = (bid - 1) * workgroup_size + lid
            while pair_idx <= total_pairs
                i, j = _pair_from_linear(pair_idx, N_points)
                _cpu_accumulate_pair_f64!(
                    block_sums, block_cnts, i, j, x3, u3, sf_type, lp,
                )
                pair_idx += grid_stride
            end
        end
        @inbounds for b in 1:NB
            sums[b] += block_sums[b]
            counts_i64[b] += block_cnts[b]
        end
    end
    return (;
        sums,
        sums_f32 = FT.(sums),
        counts_i64,
        sum_counts = sum(counts_i64),
        workgroup_size,
        nblocks,
        path = :blockshared_all_f64,
    )
end

"""Like `_cpu_accumulate_pair!` but partial sums stay Float64."""
function _cpu_accumulate_pair_f64!(
    sums::AbstractVector{Float64},
    counts_i64::AbstractVector{Int64},
    i::Int,
    j::Int,
    x3,
    u3,
    sf_type::SFT.AbstractStructureFunctionType,
    lp,
)
    FT = eltype(x3)
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
    X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
    X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
    dX = X2 - X1
    dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
    bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
    if 1 <= bin < lp.n_bins
        counts_i64[bin] += 1
        U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
        U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
        r̂ = SFH.r̂(X1, X2)
        sums[bin] += Float64(sf_type(U2 - U1, r̂))
    end
    return nothing
end

"""BigFloat sum for one bin (independent high-precision reference for that bin)."""
function cpu_bigfloat_bin_sum(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT},
    target_bin::Int,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    total_pairs = N_points * (N_points - 1) ÷ 2
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
    s = big(0.0)
    n = 0
    for k in 1:total_pairs
        i, j = _pair_from_linear(k, N_points)
        X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
        X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
        if bin == target_bin
            U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
            U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
            r̂ = SFH.r̂(X1, X2)
            s += big(Float64(sf_type(U2 - U1, r̂)))
            n += 1
        end
    end
    return (; sum = Float64(s), n, path = :bigfloat_bin)
end

"""
    validate_f64_references(...)

Cross-check **every** chunking schedule in Float64 against serial f64 + BigFloat:
  - `cpu_f64_serial_sums` (k = 1:total_pairs)
  - `cpu_private_all_f64`, `cpu_stride_all_f64`, `cpu_blockshared_all_f64`
  - optional BigFloat sum for `target_bin`
  - serial Float32 gold at `target_bin` (not a reference; diagnostic only)
"""
function validate_f64_references(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
    workgroup_size::Int = 256,
    target_bin::Int = 5,
    bigfloat::Bool = true,
) where {FT}
    serial = cpu_f64_serial_sums(x_mat, u_mat, sf_type, bin_edges)
    private = cpu_private_all_f64(
        x_mat, u_mat, sf_type, bin_edges; nworkers = nworkers,
    )
    stride = cpu_stride_all_f64(
        x_mat, u_mat, sf_type, bin_edges; nworkers = nworkers,
    )
    blockshared = cpu_blockshared_all_f64(
        x_mat, u_mat, sf_type, bin_edges;
        nworkers = nworkers, workgroup_size = workgroup_size,
    )
    gold_f32 = cpu_gold_histogram(x_mat, u_mat, sf_type, bin_edges)
    ref = serial.sums
    max_serial_private = maximum(abs.(ref .- private.sums))
    max_serial_stride = maximum(abs.(ref .- stride.sums))
    max_serial_blockshared = maximum(abs.(ref .- blockshared.sums))
    bf = bigfloat ?
        cpu_bigfloat_bin_sum(x_mat, u_mat, sf_type, bin_edges, target_bin) :
        (; sum = NaN, n = 0)
    bf_diff = bigfloat ?
        abs(ref[target_bin] - bf.sum) :
        NaN
    f32_serial_bin = Float64(gold_f32.sums[target_bin])
    f32_vs_f64_bin = f32_serial_bin - ref[target_bin]
    f64_tol = 1.0
    f64_ok = max_serial_private < f64_tol &&
        max_serial_stride < f64_tol &&
        max_serial_blockshared < f64_tol
    bf_ok = !bigfloat || bf_diff < f64_tol
    return (;
        serial,
        private,
        stride,
        blockshared,
        gold_f32,
        max_serial_private,
        max_serial_stride,
        max_serial_blockshared,
        bigfloat = bf,
        bf_diff,
        f32_serial_bin,
        f32_vs_f64_bin,
        f64_ok,
        bf_ok,
        reference_ok = f64_ok && bf_ok,
    )
end

"""
    reconcile_sum_paths(x_mat, u_mat, sf_type, bin_edges; nworkers)

Compare Float64 serial reference vs stride vs private. Reports per-worker partial
mismatch (first worker where private f32 partial ≠ stride f32 partial for that worker).
"""
function reconcile_sum_paths(
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    sf_type::SFT.AbstractStructureFunctionType,
    bin_edges::LinearBinEdges{FT};
    nworkers::Int,
) where {FT}
    lp = linear_params(bin_edges)
    N_points = size(x_mat, 2)
    verify_stride_partition(N_points, nworkers)
    x3, u3, _ = _pad3_host(x_mat, u_mat)
    NB = lp.n_bins - 1
    total_pairs = N_points * (N_points - 1) ÷ 2
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val

    function _pair_val(i, j)
        X1 = SA.SVector{3, FT}(x3[1, i], x3[2, i], x3[3, i])
        X2 = SA.SVector{3, FT}(x3[1, j], x3[2, j], x3[3, j])
        dX = X2 - X1
        dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
        bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, lp.n_bins)
        if !(1 <= bin < lp.n_bins)
            return 0, zero(FT)
        end
        U1 = SA.SVector{3, FT}(u3[1, i], u3[2, i], u3[3, i])
        U2 = SA.SVector{3, FT}(u3[1, j], u3[2, j], u3[3, j])
        r̂ = SFH.r̂(X1, X2)
        return bin, sf_type(U2 - U1, r̂)
    end

    ref = cpu_f64_serial_sums(x_mat, u_mat, sf_type, bin_edges)
    stride = cpu_stride_global_histogram(
        x_mat, u_mat, sf_type, bin_edges; nworkers = nworkers,
    )
    private = cpu_private_histogram(
        x_mat, u_mat, sf_type, bin_edges; nworkers = nworkers,
    )

    ref_f32 = ref.sums_f32
    max_ref_stride = maximum(abs.(ref_f32 .- stride.sums))
    max_ref_private = maximum(abs.(ref_f32 .- private.sums))
    max_stride_private = maximum(abs.(stride.sums .- private.sums))
    ratio_private_stride = sum(private.sums) / sum(stride.sums)

    # Rebuild private by merging per-worker f32 partials (must match cpu_private_histogram).
    rebuilt = zeros(FT, NB)
    worker_max = zeros(FT, nworkers)
    for worker in 1:nworkers
        wp = zeros(FT, NB)
        k = worker
        while k <= total_pairs
            i, j = _pair_from_linear(k, N_points)
            bin, val = _pair_val(i, j)
            if bin > 0
                wp[bin] += val
            end
            k += nworkers
        end
        worker_max[worker] = maximum(wp)
        @inbounds for b in 1:NB
            rebuilt[b] += wp[b]
        end
    end
    max_rebuild_private = maximum(abs.(rebuilt .- private.sums))

    return (;
        ref,
        stride,
        private,
        rebuilt,
        max_ref_stride,
        max_ref_private,
        max_stride_private,
        max_rebuild_private,
        ratio_private_stride,
    )
end

"""Per-bin table comparing histogram paths vs serial CPU gold."""
function compare_histogram_paths(
    gold,
    paths::AbstractVector;
    sum_ref = nothing,
    sum_atol::Real = 500,
    sum_rtol::Real = 1f-4,
)
    sum_reference = sum_ref === nothing ? gold.sums : sum_ref
    rows = NamedTuple[]
    for p in paths
        max_Δcnt = maximum(abs.(p.counts_i64 .- gold.counts_i64))
        max_Δsum = maximum(abs.(p.sums .- sum_reference))
        Δcnt = p.sum_counts - gold.n_in
        sums_ok = isapprox(p.sums, sum_reference; rtol = sum_rtol, atol = sum_atol)
        push!(rows, (;
            path = p.path,
            nworkers = p.nworkers,
            workgroup_size = p.workgroup_size,
            nblocks = p.nblocks,
            Δcnt,
            max_Δcnt_bin = max_Δcnt,
            max_Δsum,
            sums_ok,
            counts_ok = p.counts_i64 == gold.counts_i64,
        ))
    end
    return rows
end

"""Print per-bin diffs for the path with largest `max|Δsum|` vs gold."""
function print_worst_bin_table(gold, path; label = string(path.path))
    diffs = path.sums .- gold.sums
    cnt_diffs = path.counts_i64 .- gold.counts_i64
    order = sortperm(abs.(diffs); rev = true)
    println("\nPer-bin diffs ($label vs serial gold):")
    @printf("  %5s | %12s | %12s | %12s | %10s\n", "bin", "gold_sum", "path_sum", "Δsum", "Δcnt")
    for b in order[1:min(8, length(order))]
        @printf(
            "  %5d | %12.4g | %12.4g | %12.4g | %10d\n",
            b, gold.sums[b], path.sums[b], diffs[b], cnt_diffs[b],
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Count-only diagnostic kernels (no SF math — isolates pair loop + atomics)
# ---------------------------------------------------------------------------

@inline function _diag_dist_and_bin(i, j, x_mat, N_bins, fe, le, is_, off, sv)
    FT = eltype(x_mat)
    X1 = SA.SVector{3, FT}(x_mat[1, i], x_mat[2, i], x_mat[3, i])
    X2 = SA.SVector{3, FT}(x_mat[1, j], x_mat[2, j], x_mat[3, j])
    dX = X2 - X1
    dist = sqrt(dX[1]^2 + dX[2]^2 + dX[3]^2)
    bin = _gpu_digitize_linear(dist, fe, le, is_, off, sv, N_bins)
    return bin
end

KA.@kernel function _diag_global_float_count!(
    counts,
    x_mat,
    N_points::Int,
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nworkers::Int,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _pair_from_linear(k, N_points)
        bin = _diag_dist_and_bin(i, j, x_mat, N_bins, first_edge, last_edge, inv_step, offset, step_val)
        if 1 <= bin < N_bins
            @atomic counts[bin] += one(FT)
        end
        k += nworkers
    end
end

KA.@kernel function _diag_global_uint32_count!(
    counts,
    x_mat,
    N_points::Int,
    N_bins::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nworkers::Int,
) where {FT}
    worker = @index(Global, Linear)
    total_pairs = N_points * (N_points - 1) ÷ 2
    k = worker
    while k <= total_pairs
        i, j = _pair_from_linear(k, N_points)
        bin = _diag_dist_and_bin(i, j, x_mat, N_bins, first_edge, last_edge, inv_step, offset, step_val)
        if 1 <= bin < N_bins
            @atomic counts[bin] += UInt32(1)
        end
        k += nworkers
    end
end

KA.@kernel function _diag_blockshared_count!(
    counts,
    x_mat,
    N_points::Int,
    N_bins::Int,
    NB::Int,
    first_edge::FT,
    last_edge::FT,
    inv_step::FT,
    offset::FT,
    step_val::FT,
    nblocks::Int,
    workgroup_size::Int,
) where {FT}
    lid = @index(Local, Linear)
    bid = @index(Group, Linear)
    shared_cnts = @localmem FT (PROTO_MAX_BINS,)
    if lid == 1
        @inbounds for b in 1:NB
            shared_cnts[b] = zero(FT)
        end
    end
    @synchronize
    total_pairs = N_points * (N_points - 1) ÷ 2
    pair_idx = (bid - 1) * workgroup_size + lid
    grid_stride = nblocks * workgroup_size
    while pair_idx <= total_pairs
        i, j = _pair_from_linear(pair_idx, N_points)
        bin = _diag_dist_and_bin(i, j, x_mat, N_bins, first_edge, last_edge, inv_step, offset, step_val)
        if 1 <= bin < N_bins
            @atomic shared_cnts[bin] += one(FT)
        end
        pair_idx += grid_stride
    end
    @synchronize
    b = lid
    while b <= NB
        if shared_cnts[b] != zero(FT)
            @atomic counts[b] += shared_cnts[b]
        end
        b += workgroup_size
    end
end

"""Run count-only GPU diagnostic kernels; return `Vector{NamedTuple}` vs CPU gold."""
function run_count_diagnostics(
    backend::KA.Backend,
    x_mat::AbstractMatrix{FT},
    u_mat::AbstractMatrix{FT},
    bin_edges::LinearBinEdges{FT};
    nworkers::Int = min(262_144, size(x_mat, 2) * (size(x_mat, 2) - 1) ÷ 2),
    workgroup_size::Int = 256,
) where {FT}
    lp = linear_params(bin_edges)
    NB = lp.n_bins - 1
    N_points = size(x_mat, 2)
    x_dev, u_dev, out_dev, cnt_dev, N_points, _ = _stage_host_to_device(backend, x_mat, u_mat, lp.n_bins)
    fe, le, is_, off, sv = lp.first_edge, lp.last_edge, lp.inv_step, lp.offset, lp.step_val
    nblocks = cld(nworkers, workgroup_size)

    results = NamedTuple[]

    function _run!(name, cnt_dev, launch!)
        z = zeros(eltype(cnt_dev), NB)
        copyto!(cnt_dev, z)
        kernel_s = @elapsed begin
            launch!()
            KA.synchronize(backend)
        end
        counts = Array(cnt_dev)
        sum_counts = sum(counts)
        push!(results, (; name, sum_counts, counts, kernel_s))
    end

    cnt_f32 = cnt_dev
    _run!("diag_global_float32", cnt_f32, () -> begin
        kernel! = _diag_global_float_count!(backend, workgroup_size)
        kernel!(cnt_f32, x_dev, N_points, lp.n_bins, fe, le, is_, off, sv, nworkers; ndrange = nworkers)
    end)

    cnt_u32 = KA.zeros(backend, UInt32, NB)
    _run!("diag_global_uint32", cnt_u32, () -> begin
        kernel! = _diag_global_uint32_count!(backend, workgroup_size)
        kernel!(cnt_u32, x_dev, N_points, lp.n_bins, fe, le, is_, off, sv, nworkers; ndrange = nworkers)
    end)

    _run!("diag_blockshared_float32", cnt_f32, () -> begin
        kernel! = _diag_blockshared_count!(backend, workgroup_size)
        kernel!(cnt_f32, x_dev, N_points, lp.n_bins, NB, fe, le, is_, off, sv, nblocks, workgroup_size;
            ndrange = nblocks * workgroup_size)
    end)

    return results
end
