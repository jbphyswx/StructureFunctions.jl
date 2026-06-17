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

"""Four tiled coordinate buffers in existing sp2d kernels (`~4 KiB` for `Float32`)."""
@inline function _sp2d_tile_smem_overhead(::Type{FT}) where {FT}
    return 4 * SF_GPU_TILE * sizeof(FT)
end

@inline function _sp2d_hist_budget_bytes(smem_per_block::Int, ::Type{FT}) where {FT}
    return max(0, smem_per_block - _sp2d_tile_smem_overhead(FT) - SF_GPU_SMEM_COMPILER_RESERVE)
end

@inline function _sp2d_raw_cells_per_strip(smem_per_block::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return _sp2d_hist_budget_bytes(smem_per_block, FT) ÷ cell_bytes
end

"""Smallest power-of-two bucket holding `min(raw_cells, C)` strip cells."""
function _sp2d_strip_bucket(raw_cells::Int, C::Int)
    target = min(raw_cells, C)
    for b in SP2D_STRIP_BUCKETS
        b >= target && return b
    end
    return SP2D_STRIP_BUCKETS[end]
end

"""
    _sp2d_priv_config(n_dist, n_val, FT) -> SP2DPrivConfig

Choose strip bucket and `smem_per_block` policy: start at 48 KiB; if that yields
`n_strips > 3`, switch to 96 KiB (kernel must opt in on CUDA).
"""
function _sp2d_priv_config(n_dist::Int, n_val::Int, ::Type{FT}) where {FT}
    C = _sp2d_joint_cells(n_dist, n_val)
    raw48 = _sp2d_raw_cells_per_strip(SF_GPU_SMEM_DEFAULT, FT)
    bucket48 = _sp2d_strip_bucket(raw48, C)
    n48 = cld(C, bucket48)
    if n48 > 3
        smem = SF_GPU_SMEM_PREFERRED
        raw = _sp2d_raw_cells_per_strip(smem, FT)
        bucket = _sp2d_strip_bucket(raw, C)
        return SP2DPrivConfig(C, bucket, cld(C, bucket), smem, bucket)
    end
    return SP2DPrivConfig(C, bucket48, n48, SF_GPU_SMEM_DEFAULT, bucket48)
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
