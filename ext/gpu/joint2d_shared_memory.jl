# Joint 2D tiled-kernel shared-memory compile width helpers.

"""
    joint2d_smem_max(backend, W, F, XT, OT, CT) -> Int

The widest joint histogram, in cells, whose tiled kernel fits `backend`'s shared memory for `W`-wide
coordinates and `F`-wide fields of `XT`, sums of `OT` and shared counts of `CT`. One kernel compiled at
this width serves every joint grid of at most that many cells, which is useful when many bin shapes are
tried in one Julia session.
"""
function SFC.joint2d_smem_max(backend, W::Int, F::Int, ::Type{XT}, ::Type{OT}, ::Type{CT}) where {XT, OT, CT}
    return _smem_max_cells(hist -> _joint2d_tiled_smem_bytes(XT, OT, CT, W, F, hist),
                           SFC.gpu_static_smem_budget(SFC.gpu_device_caps(backend)),
                           sizeof(OT) + sizeof(CT))
end

"""
    joint2d_smem_exact(n_dist, n_val)

Exact histogram cell count `n_dist × n_val` (same as omitting `joint2d_compile_cells`
on [`GPUSFWorkspace`](@ref)).
"""
function SFC.joint2d_smem_exact(n_dist::Int, n_val::Int)
    return n_dist * n_val
end

"""
    joint2d_smem_align256(n_dist, n_val)

Round `n_dist × n_val` up to a multiple of 256, so bin grids of similar size share one compiled
kernel.
"""
function SFC.joint2d_smem_align256(n_dist::Int, n_val::Int)
    return cld(n_dist * n_val, 256) * 256
end

"""The compile width a `:joint2d` workspace takes: `compile_cells`, at least `NB2`, or `NB2`."""
function _joint2d_resolve_compile_cells(NB2::Int, compile_cells::Union{Nothing, Int})
    cells = compile_cells === nothing ? NB2 : compile_cells
    cells >= NB2 ||
        throw(ArgumentError("joint2d_compile_cells=$cells is smaller than NB2=$NB2"))
    return cells
end
