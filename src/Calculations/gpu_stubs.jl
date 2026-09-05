# GPU Extension and Workspace Stubs

# ---------------------------------------------------------------------------
# CUDA fast-path dispatch hooks (overridden by StructureFunctionsCUDAExt)
# ---------------------------------------------------------------------------
# The GPU (KernelAbstractions) extension calls these at its unified launch
# chokepoints. The default returns `false` ("not handled") so portable KA
# kernels run on CPU/AMD/whenever CUDA is not loaded. When both
# KernelAbstractions and CUDA are loaded, StructureFunctionsCUDAExt adds
# methods specialized on `CUDA.CUDABackend` that launch the N-body broadcast +
# (for 2D) dynamic-shared kernels and return `true`. See
# gpu/OPTIMAL_KERNEL_DESIGN.md for the settled design.

"""Try the CUDA fast 2D launch (N-body broadcast + privatized histogram, dynamic
shared for large single-pass). Returns `true` if handled, `false` to fall back to
the portable KA tiled kernel. `cull` is the active [`GPUCullMemo`](@ref) or `nothing`; the
launcher takes its tile-pair schedule from it through [`schedule_for`](@ref) at its own tile
size. Overridden by `StructureFunctionsCUDAExt`."""
gpu_fast_launch_2d_batch!(backend, out, cnt, x, u, sf_type, dist_dig, val_plan,
                          N, n_dist, n_val, B, D, nmom, fixed_x, geom, cull) = false

"""Try the CUDA fast 1D launch (N-body broadcast + privatized shared histogram).
Returns `true` if handled, `false` to fall back to the portable KA tiled kernel. `cull` as for
[`gpu_fast_launch_2d_batch!`](@ref). Overridden by `StructureFunctionsCUDAExt`."""
gpu_fast_launch_1d_batch!(backend, out, cnt, x, u, sf_type, dist_dig,
                          N, NB, B, D, nmom, fixed_x, geom, cull) = false


"""
    GPUDeviceCaps

What a GPU backend can actually offer, queried rather than assumed, so shared-memory strategy is
chosen per device instead of per hardcoded constant. Shared memory per block differs by an order of
magnitude across parts a user may run on (V100 96 KiB, L40S 100 KiB, A100 163 KiB, later parts
more), and the right accumulation strategy differs with it.

`smem_per_block` is the **opt-in** maximum, reachable only by *dynamic* shared memory that a kernel
explicitly requests; `smem_per_sm` bounds how many blocks stay resident and is what makes "use every
byte" the wrong default.

Static shared memory is capped far lower and independently — see [`GPU_SMEM_STATIC_MAX`].
KernelAbstractions' `@localmem` lowers to a *static* allocation (`CuStaticSharedArray` on CUDA) and
its launch path passes no `shmem`, so a kernel written once for every backend is bound by the static
cap. Reaching `smem_per_block` therefore takes a backend-specialized kernel that declares dynamic
shared memory and opts in at launch — which is what the vendor fast paths do, with the portable
kernel remaining as the correctness fallback for backends that have none.
"""
struct GPUDeviceCaps
    smem_per_block::Int
    smem_per_sm::Int
    n_sms::Int
    warp::Int
end

"""
Largest *static* shared allocation a block may declare. Measured, not assumed: on an A100 a static
`@localmem` of 48 KiB compiles and 64 KiB fails `ptxas`, while dynamic shared reaches the full
163 KiB opt-in. This is an architectural limit rather than a per-device one, so exceeding a device's
opt-in maximum is a separate check.
"""
const GPU_SMEM_STATIC_MAX = 48 * 1024

"""Shared memory every CUDA-class device provides without opting in."""
const GPU_SMEM_UNIVERSAL_FLOOR = 48 * 1024

"""
    gpu_static_smem_budget(caps) -> Int

Bytes a portable (static `@localmem`) kernel may use on this device: the static cap, further limited
if the device offers less than it.
"""
@inline gpu_static_smem_budget(caps::GPUDeviceCaps) =
    min(GPU_SMEM_STATIC_MAX, caps.smem_per_block)

"""
    gpu_dynamic_smem_budget(caps; target_blocks_per_sm = 2) -> Int

Bytes a dynamic-shared kernel should use per block. Expressed in device-relative terms — the opt-in
ceiling, and the per-SM pool divided by an occupancy target — so the same rule sizes correctly on any
part rather than encoding one device's byte count. `target_blocks_per_sm` is the only free parameter
and is dimensionless.
"""
@inline function gpu_dynamic_smem_budget(caps::GPUDeviceCaps; target_blocks_per_sm::Int = 2)
    per_sm_share = caps.smem_per_sm ÷ max(1, target_blocks_per_sm)
    return max(GPU_SMEM_UNIVERSAL_FLOOR, min(caps.smem_per_block, per_sm_share))
end

"""
    gpu_device_caps(backend) -> GPUDeviceCaps

Capabilities of `backend`. The default is deliberately the universal floor: a backend with no
override behaves exactly as the package did before device querying existed, so an unknown or future
backend degrades to "correct and portable" rather than to "assumes an A100". `StructureFunctionsCUDAExt`
overrides this with the real device attributes.
"""
gpu_device_caps(::Any) = GPUDeviceCaps(GPU_SMEM_UNIVERSAL_FLOOR, GPU_SMEM_UNIVERSAL_FLOOR, 1, 32)

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
    gpu_calculate_structure_function_batch!(sums, counts, sf_type, backend, x, u, distance_bins; workspace=nothing, ...)

GPU slice batch over `(N_dims, N_points, T)`; host outputs `(NB, T)`. Requires `GPUExt`.
"""
function gpu_calculate_structure_function_batch!(args...; kwargs...)
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
function gpu_calculate_structure_function_2d_batch!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU 2D joint slice batch is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

"""GPU single-pass slice batch; outputs `(6, NB, T)`. Requires `GPUExt`."""
function gpu_calculate_structure_functions_single_pass_batch!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU single-pass slice batch is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end

"""GPU single-pass 2D slice batch; outputs `(6, NB, n_val, T)`. Requires `GPUExt`."""
function gpu_calculate_structure_functions_single_pass_2d_batch!(args...; kwargs...)
    throw(
        ArgumentError(
            "GPU single-pass 2D slice batch is unavailable. Load KernelAbstractions to activate the GPUExt extension.",
        ),
    )
end
