# Joint 2D tiled-kernel shared-memory compile width helpers and routing.

"""
    joint2d_smem_max()

Compile-time `@localmem` width `SF_GPU_MAX_2D_HIST` (4096). Reuses one GPU kernel for
any joint grid with `n_dist × n_val ≤ 4096`, which is useful when many bin shapes are
tried in one Julia session.
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

"""Internal: distance-bin route symbol for joint tiled kernel dispatch."""
function _joint2d_dist_route(dist_bins)
    dist_bins isa LinearBinEdges && return :linear
    dist_bins isa LogBinEdges && return :log
    return :general
end

"""Internal: value-bin route symbol for joint tiled kernel dispatch."""
function _joint2d_val_route(::Nothing)
    return :general
end
function _joint2d_val_route(::GPUValueLinearShared)
    return :linear
end
function _joint2d_val_route(::GPUValueInfLinearShared)
    return :inflinear
end
function _joint2d_val_route(::GPUValueLogLinearShared)
    return :log_linear
end
function _joint2d_val_route(::GPUValueVectorCols)
    return :general
end
function _joint2d_val_route(val_plan)
    throw(ArgumentError(
        "joint2d value bins: expected LinearBinEdges, LogBinEdges, InfPaddedBinEdges, or Vector (got $(typeof(val_plan)))",
    ))
end

"""
Build a frozen value digitize plan for joint 2D (single shared column).
Returns `nothing` for plain `Vector` edges → `:general` kernel route.
"""
function _joint2d_build_val_plan(backend::KA.Backend, value_bins)
    value_bins isa LinearBinEdges && return _gpu_build_value_digitize_plan(backend, value_bins)
    value_bins isa InfPaddedBinEdges && return _gpu_build_value_digitize_plan(backend, value_bins)
    value_bins isa LogBinEdges && return _gpu_build_value_digitize_plan(backend, value_bins)
    value_bins isa BinEdges && return _joint2d_build_val_plan(backend, value_bins.edges)
    value_bins isa AbstractVector && return nothing
    throw(ArgumentError("joint2d unsupported value_bins type $(typeof(value_bins))"))
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
