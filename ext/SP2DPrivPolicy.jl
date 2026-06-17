# Host-side strip / privatization sizing for HTP-EJ eight-type single-pass 2D.

"""Compile-time `@localmem` strip buckets (powers of two, ≤ shared-memory budget)."""
const SP2D_STRIP_BUCKETS = (1024, 2048, 4096, 8192, 16384)

"""CUDA default per-block shared memory (bytes) when kernel does not opt in."""
const SF_GPU_SMEM_DEFAULT = 48 * 1024

"""Preferred opt-in shared memory on Ampere/A100 when `n_strips > 3` at 48 KiB."""
const SF_GPU_SMEM_PREFERRED = 96 * 1024

"""Safety margin for driver / compiler static shared outside strip histogram."""
const SF_GPU_SMEM_COMPILER_RESERVE = 2048

"""
    SP2DPrivConfig

Frozen HTP-EJ histogram strip policy for one `(n_dist, n_val, FT)` workspace.
`cells_per_strip` is a power-of-two bucket compiled into `@localmem` kernels;
`n_strips = ceil(C / cells_per_strip)` with `C = 8 * n_dist * n_val`.
"""
struct SP2DPrivConfig
    n_joint_cells::Int
    cells_per_strip::Int
    n_strips::Int
    smem_per_block::Int
    strip_bucket::Int
end

"""Total joint histogram cells `8 × n_dist × n_val`."""
@inline _sp2d_joint_cells(n_dist::Int, n_val::Int) = 8 * n_dist * n_val

"""`@localmem` width per coord buffer in `TiledSinglePass2DPrivKernels` (x/y per point)."""
const SP2D_PRIV_TILE_LOCALMEM = 256

"""`shared_block_id`, `shared_tile[4]`, `shared_strip_s` (KA `@synchronize` metadata)."""
@inline _sp2d_priv_meta_smem_bytes() = 6 * sizeof(Int)

"""Four tiled coordinate buffers (`@localmem FT (256,)` × 4)."""
@inline function _sp2d_tile_smem_overhead(::Type{FT}) where {FT}
    return 4 * SP2D_PRIV_TILE_LOCALMEM * sizeof(FT)
end

"""Static `@localmem` bytes for a compiled strip bucket (must match priv kernel)."""
@inline function _sp2d_static_smem_bytes(bucket::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return _sp2d_tile_smem_overhead(FT) +
           _sp2d_priv_meta_smem_bytes() +
           bucket * cell_bytes +
           SF_GPU_SMEM_COMPILER_RESERVE
end

@inline function _sp2d_hist_budget_bytes(smem_per_block::Int, ::Type{FT}) where {FT}
    return max(
        0,
        smem_per_block -
        _sp2d_tile_smem_overhead(FT) -
        _sp2d_priv_meta_smem_bytes() -
        SF_GPU_SMEM_COMPILER_RESERVE,
    )
end

@inline function _sp2d_raw_cells_per_strip(smem_per_block::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return _sp2d_hist_budget_bytes(smem_per_block, FT) ÷ cell_bytes
end

"""Largest power-of-two strip bucket whose static `@localmem` fits in `smem_limit`."""
function _sp2d_largest_fitting_bucket(smem_limit::Int, ::Type{FT}) where {FT}
    best = nothing
    for b in SP2D_STRIP_BUCKETS
        if _sp2d_static_smem_bytes(b, FT) <= smem_limit
            best = b
        else
            break
        end
    end
    best === nothing &&
        error("no SP2D priv strip bucket fits in smem_limit=$smem_limit for $FT")
    return best
end

"""Bucket for `C` joint cells that fits `smem_limit` (minimizes `n_strips` among fitting sizes)."""
function _sp2d_strip_bucket(C::Int, smem_limit::Int, ::Type{FT}) where {FT}
    best = _sp2d_largest_fitting_bucket(smem_limit, FT)
    for b in SP2D_STRIP_BUCKETS
        _sp2d_static_smem_bytes(b, FT) <= smem_limit || break
        b >= C && return b
        best = b
    end
    return best
end

"""
    _sp2d_priv_config(n_dist, n_val, FT) -> SP2DPrivConfig

Choose strip bucket and `smem_per_block` policy. Bucket must satisfy
`_sp2d_static_smem_bytes(bucket, FT) ≤ smem_per_block` (CUDA default 48 KiB unless
the launch path opts into 96 KiB — not wired yet, so we stay at 48 KiB).
"""
function _sp2d_priv_config(n_dist::Int, n_val::Int, ::Type{FT}) where {FT}
    C = _sp2d_joint_cells(n_dist, n_val)
    smem = SF_GPU_SMEM_DEFAULT
    bucket = _sp2d_strip_bucket(C, smem, FT)
    return SP2DPrivConfig(C, bucket, cld(C, bucket), smem, bucket)
end

"""Block-private slab bytes for `n_tile_blocks` upper-triangle tile blocks."""
@inline function _sp2d_priv_slab_bytes(config::SP2DPrivConfig, n_tile_blocks::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return n_tile_blocks * config.n_joint_cells * cell_bytes
end

"""Pick compiled kernel strip bucket (must be ≥ `config.cells_per_strip`)."""
@inline function _sp2d_kernel_strip_bucket(config::SP2DPrivConfig)
    return config.strip_bucket
end
