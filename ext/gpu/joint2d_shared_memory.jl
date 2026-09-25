# Joint 2D tiled-kernel shared-memory compile width helpers.

"""
    joint2d_smem_max()

Compile-time `@localmem` width `SF_GPU_MAX_2D_HIST`. Reuses one GPU kernel for every joint grid
within that cap, which is useful when many bin shapes are tried in one Julia session.
"""
function SFC.joint2d_smem_max()
    return SF_GPU_MAX_2D_HIST
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

Round `n_dist × n_val` up to a multiple of 256 (capped at [`joint2d_smem_max`](@ref)),
reusing one of at most 16 kernel sizes for bin-grid sweeps.
"""
function SFC.joint2d_smem_align256(n_dist::Int, n_val::Int)
    nb2 = n_dist * n_val
    return min(SF_GPU_MAX_2D_HIST, cld(nb2, 256) * 256)
end

"""
Resolve compile-time histogram width from optional user override.
Default (`compile_cells === nothing`) is exact `NB2`.
"""
function _joint2d_resolve_compile_cells(NB2::Int, compile_cells::Union{Nothing, Int})
    cells = compile_cells === nothing ? NB2 : compile_cells
    cells >= NB2 ||
        throw(ArgumentError("joint2d_compile_cells=$cells is smaller than NB2=$NB2"))
    cells <= SF_GPU_MAX_2D_HIST ||
        throw(ArgumentError(
            "joint2d_compile_cells=$cells exceeds SF_GPU_MAX_2D_HIST=$(SF_GPU_MAX_2D_HIST)",
        ))
    return cells
end
