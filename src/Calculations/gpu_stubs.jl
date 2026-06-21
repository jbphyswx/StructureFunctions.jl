# GPU Extension and Workspace Stubs

"""
    GPUSFWorkspace(backend, distance_bins; kind=:sf1d)

Reusable GPU device histogram workspace. Requires the `GPUExt` extension
(`using KernelAbstractions`). Pass to `gpu_calculate_structure_function(!)` or
slice drivers to avoid per-call device allocation.

See also [`reset_histogram!`](@ref), [`release!`](@ref).
"""
function GPUSFWorkspace end

"""Zero device histogram buffers in a [`GPUSFWorkspace`](@ref) before the next launch."""
function reset_histogram! end

"""Release device buffers held by a [`GPUSFWorkspace`](@ref) (optional explicit free)."""
function release! end

"""Compile-time joint 2D shared-histogram width `SF_GPU_MAX_2D_HIST` (4096). Implemented in GPU extension."""
function joint2d_smem_max end

"""Exact joint histogram cell count `n_dist × n_val`. Implemented in GPU extension."""
function joint2d_smem_exact end

"""256-aligned joint compile width (capped at max). Implemented in GPU extension."""
function joint2d_smem_align256 end

"""
    flatten_grid_slices(x, u)

Reshape `(N_dims, Ny, Nx, T)` grid snapshots to `(N_dims, Ny*Nx, T)` column-major
flattened point index (matches the matrix layout expected by slice-batch APIs).
"""
function flatten_grid_slices(x::AbstractArray{T, 4}, u::AbstractArray{T, 4}) where {T}
    # This is stupid why it it asuming 4d? and is this for 2d or 3d data? here x is 4d so what is ther to flatten? seems like agent bs
    N_dims, Ny, Nx, T_len = size(x)
    size(u) == (N_dims, Ny, Nx, T_len) ||
        throw(DimensionMismatch("u must match x shape $(size(x)); got $(size(u))"))
    N_pts = Ny * Nx
    x_out = reshape(x, N_dims, N_pts, T_len)
    u_out = reshape(u, N_dims, N_pts, T_len)
    return x_out, u_out
end

"""
    gpu_calculate_structure_function(...)

GPU-accelerated structure function calculation. Requires loading `KernelAbstractions.jl`
to activate the `GPUExt` extension. The backend can be `KernelAbstractions.CPU()`
(for testing parity) or any GPU backend like `CUDABackend()` from `CUDA.jl`.

Device histogram buffers are `UInt32`; `count_eltype` (default `UInt32`) selects the
host count type after download. Pass `workspace=GPUSFWorkspace(...)` to reuse device
histogram buffers across repeated calls (see [`GPUSFWorkspace`](@ref)).

This stub exists so the extension can legally extend this function.
"""
function gpu_calculate_structure_function end

"""
    gpu_calculate_structure_function_2d(sf_type, backend, x_mat, u_mat, distance_bins, value_bins; kwargs...)

GPU 2D joint structure function (distance × SF value histogram) for one `sf_type`.
Requires loading `KernelAbstractions.jl` to activate the `GPUExt` extension.
Device counts are `UInt32`; `count_eltype` selects the host matrix type after download.
"""
function gpu_calculate_structure_function_2d end

"""
    gpu_calculate_structure_functions_single_pass_2d(backend, x, u, distance_bins, value_bins; kwargs...)

Six invariant native distance × value joint histograms on a KernelAbstractions backend.
Requires loading `KernelAbstractions.jl` to activate the `GPUExt` extension.
Device counts are `UInt32`; `count_eltype` selects the host array type after download.
"""
function gpu_calculate_structure_functions_single_pass_2d end

"""
    gpu_calculate_structure_functions_single_pass_2d!(sums, counts, backend, x, u, distance_bins, value_bins; kwargs...)

In-place GPU 2D single-pass accumulation. Requires the `GPUExt` extension.
"""
function gpu_calculate_structure_functions_single_pass_2d!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU in-place 2D single-pass is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

"""
    gpu_calculate_structure_function!(output_sums, output_counts, ...)

In-place GPU structure function reduction. Requires the `GPUExt` extension.
Accumulates into caller-owned `output_sums` and `output_counts` (same contract as
`serial_calculate_structure_function!` / `threaded_calculate_structure_function!`).
Reuses **host** buffers only; pass `workspace=GPUSFWorkspace(...)` to reuse **device**
histogram buffers across calls.
"""
function gpu_calculate_structure_function!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU in-place backend is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

"""
    gpu_calculate_structure_function_slices!(sums, counts, sf_type, backend, x, u, distance_bins; workspace=nothing, ...)

GPU slice batch over `(N_dims, N_points, T)`; host outputs `(NB, T)`. Requires `GPUExt`.
"""
function gpu_calculate_structure_function_slices!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU slice batch is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

function gpu_calculate_structure_function_batch(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU batch structure functions are unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

function gpu_calculate_structure_function_2d_batch(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU auxiliary-axis 2D joint structure functions are unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

"""GPU 2D joint slice batch; outputs `(n_dist, n_val, T)`. Requires `GPUExt`."""
function gpu_calculate_structure_function_2d_slices!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU 2D joint slice batch is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

"""GPU single-pass slice batch; outputs `(6, NB, T)`. Requires `GPUExt`."""
function gpu_calculate_structure_functions_single_pass_slices!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU single-pass slice batch is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

"""GPU single-pass 2D slice batch; outputs `(6, NB, n_val, T)`. Requires `GPUExt`."""
function gpu_calculate_structure_functions_single_pass_2d_slices!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU single-pass 2D slice batch is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end
