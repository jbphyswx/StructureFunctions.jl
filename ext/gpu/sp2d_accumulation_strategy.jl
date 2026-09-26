# Host-side accumulation strategy selection for HTP-EJ six-invariant single-pass 2D.
#
# Two GPU algorithms (three `accum_mode` symbols):
#   On-chip (:shared, :typeplane) — shared histogram + joint-style flush to out_*.
#   Direct (:direct) — partitioned global accumulation + merge when a type plane
#                      does not fit in the shared-memory budget.

"""How `:direct`'s block partitions merge into the output: [`SerialMerge`](@ref) or [`ParallelMerge`](@ref)."""
abstract type SP2DMerge end

"""One work item per joint cell sums that cell over every block partition."""
struct SerialMerge <: SP2DMerge end

"""One workgroup per joint cell tree-reduces that cell over the block partitions."""
struct ParallelMerge <: SP2DMerge end

"""
    SP2DAccumulationStrategy

HTP-EJ histogram accumulation strategy of one call. `accum_mode` is `:shared` when the padded
`6 × n_dist × n_val` histogram fits the shared-histogram kernel; `:typeplane` when one or more padded
type planes fit the type-plane kernel (`types_per_pass` planes per pair traversal, `n_type_passes`
traversals); otherwise `:direct` uses block-partitioned global atomics (single pair pass).
`smem_budget` is the static shared budget the choice was made against and `max_shared_cells` the
widest histogram the chosen on-chip kernel fits in it (the shared-histogram kernel's for `:direct`).
`needs_partition_merge` is `true` only for `:direct` (partition + merge kernel), whose partitions merge
by `merge`.
"""
struct SP2DAccumulationStrategy
    n_joint_cells::Int
    shared_cells::Int
    accum_mode::Symbol
    smem_budget::Int
    max_shared_cells::Int
    plane_cells::Int
    plane_shared_cells::Int
    types_per_pass::Int
    n_type_passes::Int
    needs_partition_merge::Bool
    merge::SP2DMerge
end

"""
Joint-histogram bytes above which plain global atomics beat `:direct`, so
`_launch_single_pass_2d_portable!` routes past it.

`:direct` gives up the on-chip histogram for block-private global partitions plus a merge pass, so
its cost grows with cells × tile-blocks, while plain global atomics get cheaper as more cells spread
the contention. On an A100 the two cross between 307 and 373 KiB for both element types, and this
constant is the midpoint of that band. `:shared` and `:typeplane` keep the histogram on chip and are
unaffected.
"""
const SP2D_GLOBAL_ATOMIC_HIST_BYTES = 340 * 1024

"""
    _sp2d_prefers_global_atomics(config, OT, CST) -> Bool

Whether SP2D hands a call of strategy `config`, sums of `OT` and counts of `CST` to the plain
global-atomic kernel in place of `:direct`.
"""
function _sp2d_prefers_global_atomics(config::SP2DAccumulationStrategy, ::Type{OT},
                                      ::Type{CST}) where {OT, CST}
    config.accum_mode === :direct || return false
    return config.n_joint_cells * (sizeof(OT) + sizeof(CST)) > SP2D_GLOBAL_ATOMIC_HIST_BYTES
end

"""Total joint histogram cells `6 × n_dist × n_val`."""
@inline _sp2d_joint_cells(n_dist::Int, n_val::Int) = SF_GPU_SINGLE_PASS_N * n_dist * n_val

"""How many SF-type planes fit in one shared-histogram pass (`1…6`)."""
@inline function _sp2d_types_per_pass(plane::Int, max_shared::Int)
    plane <= 0 && return 1
    return min(SF_GPU_SINGLE_PASS_N, max(1, max_shared ÷ plane))
end

@inline function _sp2d_n_type_passes(types_per_pass::Int)
    return (SF_GPU_SINGLE_PASS_N + types_per_pass - 1) ÷ types_per_pass
end

"""
    _sp2d_accumulation_strategy(caps, n_dist, n_val, W, F, FT, OT, CST) -> SP2DAccumulationStrategy

Select `:shared`, `:typeplane` or `:direct` for `W`-wide coordinates and `F`-wide fields of `FT`, sums of
`OT` and counts of `CST` on the device `caps` describes, from the static shared bytes each HTP-EJ kernel
declares. Sets `needs_partition_merge = (mode == :direct)` for host launch routing, and the serial merge.
"""
function _sp2d_accumulation_strategy(caps::SFC.GPUDeviceCaps, n_dist::Int, n_val::Int, W::Int, F::Int,
                                     ::Type{FT}, ::Type{OT}, ::Type{CST}) where {FT, OT, CST}
    C = _sp2d_joint_cells(n_dist, n_val)
    plane = n_dist * n_val
    # What must fit on chip is the bank-conflict-padded layout, which is larger than the cell count.
    Cs = _sp2d_shared_cells(n_dist, n_val)
    plane_s = _sp2d_plane_cells(n_dist, n_val)
    budget = SFC.gpu_static_smem_budget(caps)
    cell = sizeof(OT) + sizeof(CST)
    max_shared = _smem_max_cells(hc -> _sp2d_sharedhist_smem_bytes(FT, OT, CST, W, F, hc), budget, cell)
    max_plane = _smem_max_cells(hc -> _sp2d_typeplane_smem_bytes(FT, OT, CST, W, F, hc), budget, cell)
    mode, max_cells = if Cs <= max_shared
        :shared, max_shared
    elseif plane_s <= max_plane
        :typeplane, max_plane
    else
        :direct, max_shared
    end
    tpp = mode === :typeplane ? _sp2d_types_per_pass(plane_s, max_plane) : SF_GPU_SINGLE_PASS_N
    ntp = mode === :typeplane ? _sp2d_n_type_passes(tpp) : 1
    return SP2DAccumulationStrategy(
        C, Cs, mode, budget, max_cells, plane, plane_s, tpp, ntp, mode === :direct, SerialMerge(),
    )
end

"""Cell granularity for the compile-time histogram width; coarse so distinct configs share kernels."""
const SP2D_COMPILE_CELL_QUANTUM = 1024

"""
    _sp2d_sharedhist_compile_cells(config) -> Int

Compile-time `@localmem` histogram width. Sized to what the config actually needs, rounded up to
[`SP2D_COMPILE_CELL_QUANTUM`] so nearby configs reuse one compiled kernel, and never past the widest
histogram the mode's kernel fits.
"""
@inline function _sp2d_sharedhist_compile_cells(config::SP2DAccumulationStrategy)
    # Both branches must use the PADDED extents: the kernel indexes and bounds its loops with the
    # padded layout, so sizing `@localmem` from the logical cell counts runs it off the end.
    need = if config.accum_mode === :shared
        config.shared_cells
    elseif config.accum_mode === :typeplane
        config.types_per_pass * config.plane_shared_cells
    else
        0
    end
    need <= 0 && return 1
    quantized = cld(need, SP2D_COMPILE_CELL_QUANTUM) * SP2D_COMPILE_CELL_QUANTUM
    return min(quantized, config.max_shared_cells)
end
