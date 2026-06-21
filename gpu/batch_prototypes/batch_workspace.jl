# Reusable GPU buffers — benchmarks must separate one-time upload from kernel timing.

"""
Device buffers reused across timed batch prototype calls (fixed `N`, `B`, `NB`).

`sums_dev` / `counts_dev` are always allocated. `partial_dev` is lazy (opt-in block-priv only).
`x_dev` / `u_dev` use `Union{AbstractArray{FT}, Nothing}` until `upload_batch!`.
"""
mutable struct BatchGPUWorkspace{FT, S, C, P, XS, XC}
    N::Int
    B::Int
    NB::Int
    n_tile_blocks::Int
    fixed_x::Bool
    sums_dev::S
    counts_dev::C
    partial_dev::Union{Nothing, P}
    col_sums_dev::XS
    col_counts_dev::XC
    geom_cache_dev::Union{Nothing, AbstractArray{FT, 3}}
    x_dev::Union{AbstractArray{FT, 2}, Nothing}
    u_dev::Union{AbstractArray{FT, 3}, Nothing}
end

function BatchGPUWorkspace(
    backend,
    ::Type{FT},
    N::Int,
    B::Int,
    NB::Int;
    fixed_x::Bool = true,
) where {FT}
    n_tiles = cld(N, SFGE.SF_GPU_TILE)
    n_tile_blocks = n_tiles * (n_tiles + 1) ÷ 2
    sums_dev = KA.adapt(backend, zeros(FT, NB, B))
    counts_dev = KA.adapt(backend, zeros(UInt32, NB, B))
    col_sums_dev = KA.adapt(backend, zeros(FT, NB))
    col_counts_dev = KA.adapt(backend, zeros(UInt32, NB))
    partial_placeholder = KA.adapt(backend, zeros(FT, 0, 0, 0))
    return BatchGPUWorkspace{FT, typeof(sums_dev), typeof(counts_dev), typeof(partial_placeholder),
        typeof(col_sums_dev), typeof(col_counts_dev)}(
        N, B, NB, n_tile_blocks, fixed_x,
        sums_dev, counts_dev, nothing, col_sums_dev, col_counts_dev,
        nothing,
        nothing, nothing,
    )
end

"""VRAM bytes for `(3, 128², n_tile_blocks)` geometry cache (bin + r̂ per pair slot)."""
function estimate_geom_cache_bytes(n_tile_blocks::Int, ::Type{FT}) where {FT}
    n_pairs = SFGE.SF_GPU_TILE * SFGE.SF_GPU_TILE
    return 3 * n_pairs * n_tile_blocks * sizeof(FT)
end

"""Upload host `x`, `u` once before timed kernel loops."""
function upload_batch!(ws::BatchGPUWorkspace{FT}, backend, x, u) where {FT}
    x_dev, u_dev = stage_batch_device(backend, x, u; fixed_x = ws.fixed_x)
    ws.x_dev = x_dev
    ws.u_dev = u_dev
    return ws
end

function reset_output!(ws::BatchGPUWorkspace{FT}) where {FT}
    fill!(ws.sums_dev, zero(FT))
    fill!(ws.counts_dev, zero(UInt32))
    return ws
end

"""Allocate block-private partial buffer on first use (`BATCH_BLOCK_PRIV=1`)."""
function ensure_partial_dev!(ws::BatchGPUWorkspace{FT}, backend) where {FT}
    if ws.partial_dev === nothing
        ws.partial_dev = KA.adapt(
            backend, zeros(FT, 2 * ws.NB, ws.B, ws.n_tile_blocks),
        )
    end
    return ws.partial_dev
end

function download_batch!(sums, counts, ws::BatchGPUWorkspace{FT}) where {FT}
    copy!(sums, reshape(Array(ws.sums_dev), size(sums)))
    copy!(counts, reshape(Array(ws.counts_dev), size(counts)))
    return nothing
end

"""Default VRAM cap for `(3, 128², n_tile_blocks)` geometry cache (scales as O(n_tile_blocks))."""
const BATCH_GEOM_CACHE_MAX_BYTES = 6 * 1024^3

"""True when geometry cache fits `max_bytes` (default 6 GiB)."""
function geom_cache_fits(
    n_tile_blocks::Int,
    ::Type{FT};
    max_bytes::Int = BATCH_GEOM_CACHE_MAX_BYTES,
) where {FT}
    return estimate_geom_cache_bytes(n_tile_blocks, FT) <= max_bytes
end

"""Bytes per tile block in `(3, 128², n_tile_blocks)` geometry cache."""
function geom_cache_bytes_per_tile_block(::Type{FT}) where {FT}
    n_pairs = SFGE.SF_GPU_TILE * SFGE.SF_GPU_TILE
    return 3 * n_pairs * sizeof(FT)
end

"""
Split global tile-block axis `1:n_tile_blocks` into slabs so each geometry cache
`(3, 128², slab_len)` fits `max_bytes`. Geometry is computed once per pair **within**
each slab (strip 1 save, strips 2+ load); slabs partition disjoint pair sets.
"""
function tile_block_geom_slab_ranges(
    n_tile_blocks::Int,
    max_bytes::Int,
    ::Type{FT},
) where {FT}
    n_tile_blocks <= 0 && return UnitRange{Int}[]
    per_block = geom_cache_bytes_per_tile_block(FT)
    chunk = max(1, max_bytes ÷ per_block)
    ranges = UnitRange{Int}[]
    t0 = 1
    while t0 <= n_tile_blocks
        t1 = min(n_tile_blocks, t0 + chunk - 1)
        push!(ranges, t0:t1)
        t0 = t1 + 1
    end
    return ranges
end

"""Allocate `(3, 128², slab_len)` geometry cache for one tile-block slab."""
function allocate_geom_cache_slab(backend, slab_len::Int, ::Type{FT}) where {FT}
    n_pairs = SFGE.SF_GPU_TILE * SFGE.SF_GPU_TILE
    return KA.zeros(backend, FT, 3, n_pairs, slab_len)
end

"""Lazy `(3, BATCH_GEOM_CACHE_PAIRS, n_tile_blocks)` VRAM slab (caller must check size first)."""
function ensure_geom_cache!(ws::BatchGPUWorkspace{FT}, backend, ::Type{FT}) where {FT}
    if ws.geom_cache_dev !== nothing
        return ws.geom_cache_dev
    end
    n_pairs = SFGE.SF_GPU_TILE * SFGE.SF_GPU_TILE
    cache = KA.zeros(backend, FT, 3, n_pairs, ws.n_tile_blocks)
    ws.geom_cache_dev = cache
    return cache
end
