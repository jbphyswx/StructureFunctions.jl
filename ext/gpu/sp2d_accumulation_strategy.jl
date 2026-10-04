# Host-side selection of the on-chip histogram of the HTP-EJ six-invariant single-pass 2D kernels.

"""
    SP2DAccumulationStrategy

HTP-EJ histogram accumulation strategy of one call. `accum_mode` is `:shared` when the padded
`6 × n_dist × n_val` histogram fits the shared-histogram kernel, and `:typeplane` when one or more padded
type planes fit the type-plane kernel (`types_per_pass` planes per pair traversal, `n_type_passes`
traversals). `smem_budget` is the static shared budget the choice was made against and `max_shared_cells`
the widest histogram the chosen kernel fits in it.
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
end

"""Total joint histogram cells `6 × n_dist × n_val`."""
@inline _sp2d_joint_cells(n_dist::Int, n_val::Int) = SINGLE_PASS_N * n_dist * n_val

"""How many SF-type planes fit in one shared-histogram pass (`1…6`)."""
@inline function _sp2d_types_per_pass(plane::Int, max_shared::Int)
    plane <= 0 && return 1
    return min(SINGLE_PASS_N, max(1, max_shared ÷ plane))
end

@inline function _sp2d_n_type_passes(types_per_pass::Int)
    return (SINGLE_PASS_N + types_per_pass - 1) ÷ types_per_pass
end

"""
    _sp2d_accumulation_strategy(caps, n_dist, n_val, W, F, FT, OT, CST) -> Union{SP2DAccumulationStrategy, Nothing}

Select `:shared` or `:typeplane` for `W`-wide coordinates and `F`-wide fields of `FT`, sums of `OT` and
counts of `CST` on the device `caps` describes, from the static shared bytes each HTP-EJ kernel declares;
`nothing` when not even one type plane fits.
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
    Cs <= max_shared && return SP2DAccumulationStrategy(C, Cs, :shared, budget, max_shared, plane, plane_s,
                                                        SINGLE_PASS_N, 1)
    max_plane = _smem_max_cells(hc -> _sp2d_typeplane_smem_bytes(FT, OT, CST, W, F, hc), budget, cell)
    plane_s <= max_plane || return nothing
    tpp = _sp2d_types_per_pass(plane_s, max_plane)
    return SP2DAccumulationStrategy(C, Cs, :typeplane, budget, max_plane, plane, plane_s, tpp, _sp2d_n_type_passes(tpp))
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
    # Both branches use the padded extents: the kernel indexes and bounds its loops with the padded layout.
    need = config.accum_mode === :shared ? config.shared_cells : config.types_per_pass * config.plane_shared_cells
    quantized = cld(need, SP2D_COMPILE_CELL_QUANTUM) * SP2D_COMPILE_CELL_QUANTUM
    return min(quantized, config.max_shared_cells)
end
