# Host-side privatization sizing for HTP-EJ eight-type single-pass 2D.
#
# Two GPU algorithms (three `accum_mode` symbols):
#   On-chip (:shared, :typeplane) — shared histogram + joint-style flush to out_* (no merge).
#   Direct (:direct) — priv slab + merge kernel when even one n_dist×n_val plane does not fit in 48 KiB.
#
# Full design notes: gpu/SP2D_HTP_EJ.md

"""CUDA default per-block shared memory (bytes) when kernel does not opt in."""
const SF_GPU_SMEM_DEFAULT = 48 * 1024

"""Safety margin for driver / compiler static shared outside histogram."""
const SF_GPU_SMEM_COMPILER_RESERVE = 2048

"""`@localmem` width per coord buffer in tiled priv kernels (x/y per point)."""
const SP2D_PRIV_TILE_LOCALMEM = 256

"""
    SP2DPrivConfig

Frozen HTP-EJ histogram policy for one `(n_dist, n_val, FT)` workspace.
`accum_mode` is `:shared` when the full `8 × n_dist × n_val` histogram fits in
48 KiB shared memory; `:typeplane` when one or more type planes fit per pass
(`types_per_pass × n_dist × n_val` cells, `n_type_passes` pair traversals); otherwise
`:direct` uses block-partitioned global atomics (single pair pass).
`needs_priv_merge` is `true` only for `:direct` (priv slab + merge kernel).
"""
struct SP2DPrivConfig
    n_joint_cells::Int
    accum_mode::Symbol
    smem_per_block::Int
    max_shared_cells::Int
    plane_cells::Int
    types_per_pass::Int
    n_type_passes::Int
    needs_priv_merge::Bool
end

"""Total joint histogram cells `8 × n_dist × n_val`."""
@inline _sp2d_joint_cells(n_dist::Int, n_val::Int) = 8 * n_dist * n_val

"""`shared_block_id`, `shared_tile[4]` (KA `@synchronize` metadata)."""
@inline _sp2d_priv_meta_smem_bytes() = 5 * sizeof(Int)

"""Four tiled coordinate buffers (`@localmem FT (256,)` × 4)."""
@inline function _sp2d_tile_smem_overhead(::Type{FT}) where {FT}
    return 4 * SP2D_PRIV_TILE_LOCALMEM * sizeof(FT)
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

"""Largest full histogram cell count whose static `@localmem` fits in `smem_limit`."""
function _sp2d_max_shared_cells(smem_limit::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return _sp2d_hist_budget_bytes(smem_limit, FT) ÷ cell_bytes
end

"""Static `@localmem` bytes for shared-histogram kernel at `max_cells` (compile-time width)."""
@inline function _sp2d_sharedhist_static_smem_bytes(max_cells::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return _sp2d_tile_smem_overhead(FT) +
           _sp2d_priv_meta_smem_bytes() +
           max_cells * cell_bytes +
           SF_GPU_SMEM_COMPILER_RESERVE
end

"""How many SF-type planes fit in one shared-histogram pass (`1…8`)."""
@inline function _sp2d_types_per_pass(plane::Int, max_shared::Int)
    plane <= 0 && return 1
    return min(8, max(1, max_shared ÷ plane))
end

@inline function _sp2d_n_type_passes(types_per_pass::Int)
    return (8 + types_per_pass - 1) ÷ types_per_pass
end

"""
    _sp2d_priv_config(n_dist, n_val, FT) -> SP2DPrivConfig

Select `:shared`, `:typeplane`, or `:direct` from the 48 KiB shared-memory budget.
Typeplane packs `types_per_pass` SF planes per pair traversal (`n_type_passes` total).
Sets `needs_priv_merge = (mode == :direct)` for host launch/workspace routing.
"""
function _sp2d_priv_config(n_dist::Int, n_val::Int, ::Type{FT}) where {FT}
    C = _sp2d_joint_cells(n_dist, n_val)
    plane = n_dist * n_val
    smem = SF_GPU_SMEM_DEFAULT
    max_shared = _sp2d_max_shared_cells(smem, FT)
    mode = if C <= max_shared
        :shared
    elseif plane <= max_shared
        :typeplane
    else
        :direct
    end
    tpp = mode == :typeplane ? _sp2d_types_per_pass(plane, max_shared) : 8
    ntp = mode == :typeplane ? _sp2d_n_type_passes(tpp) : 1
    return SP2DPrivConfig(C, mode, smem, max_shared, plane, tpp, ntp, mode == :direct)
end

"""Block-private slab bytes for `n_tile_blocks` upper-triangle tile blocks."""
@inline function _sp2d_priv_slab_bytes(config::SP2DPrivConfig, n_tile_blocks::Int, ::Type{FT}) where {FT}
    cell_bytes = sizeof(FT) + sizeof(UInt32)
    return n_tile_blocks * config.n_joint_cells * cell_bytes
end

"""Compile-time `@localmem` histogram width for shared-histogram kernels."""
@inline function _sp2d_sharedhist_compile_cells(config::SP2DPrivConfig)
    return config.max_shared_cells
end

function _sp2d_dist_variant(::LinearBinEdges)
    return :linear
end
function _sp2d_dist_variant(::LogBinEdges)
    return :log_linear
end
