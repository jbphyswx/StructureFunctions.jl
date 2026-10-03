# GPU Extension and Workspace Stubs

# ---------------------------------------------------------------------------
# Native kernels of a backend (supplied by StructureFunctionsCUDAExt)
# ---------------------------------------------------------------------------
# Every device launcher asks for the backend's native launch plan, launches the native kernel with it
# when there is one, and the portable KernelAbstractions kernel when there is none.

"""
    gpu_native_1d_plan(backend, XT, UT, OT, CT, weights, geom, NB, moments) -> plan or nothing

The launch plan of `backend`'s native 1-D pair kernel for coordinates of `XT`, fields of `UT`, sums of
`OT`, counts of `CT`, the weights `weights`, the coordinate and field widths of `geom`, `NB` distance
bins and the moment set `moments` (what one pair adds: an operator's value, the single-pass invariants,
a tensor's components); `nothing` when `backend` has no native kernel or that kernel does not fit the
device. A function of those types and sizes and of the device's capabilities.
`StructureFunctionsCUDAExt` supplies it for `CUDA.CUDABackend`.
"""
gpu_native_1d_plan(backend, XT, UT, OT, CT, weights, geom, NB, moments) = nothing

"""
    gpu_native_launch_1d!(plan, out, cnt, x, u, wts, moments, dist_dig, N, NB, B, fixed_x, geom, cull)

Launch the native 1-D kernel `plan` describes into `out`/`cnt` of shape `(NMOM, NB, B)`, `NMOM` the
number of moments of `moments`. `x` is
`(W, N, B)`, or `(W, N)`/`(W, N, 1)` when `fixed_x`; `u` is `(F, N, B)`; `wts` is `NoWeights()` or one
device weight per point; `cull` is the active [`GPUCullMemo`](@ref) or `nothing`, from which the launch
takes its tile-pair schedule through [`schedule_for`](@ref).
"""
function gpu_native_launch_1d! end

"""
    gpu_native_2d_plan(backend, XT, UT, OT, CT, weights, geom, moments, n_dist, n_val, val_plan) -> plan or nothing

The launch plan of `backend`'s native distance × value kernel for an `n_dist × n_val` histogram per
moment of `moments` whose value axis the device digitizer `val_plan` bins, as [`gpu_native_1d_plan`](@ref).
"""
gpu_native_2d_plan(backend, XT, UT, OT, CT, weights, geom, moments, n_dist, n_val, val_plan) = nothing

"""
    gpu_native_launch_2d!(plan, out, cnt, x, u, wts, moments, dist_dig, val_plan, N, n_dist, n_val, B,
                          fixed_x, geom, second_axis, cull, portable!)

Launch the native distance × value kernel `plan` describes into `out`/`cnt` of shape
`(NMOM, n_dist, n_val, B)`; `second_axis` is what the value axis bins; the rest as for
[`gpu_native_launch_1d!`](@ref). `portable!(out, cnt)` launches the same call on the portable kernels into buffers of
that shape, a candidate the plan may time beside its own.
"""
function gpu_native_launch_2d! end

"""Work groups of [`gpu_in_range_tally!`](@ref), one column of its tally each."""
const GPU_IN_RANGE_GROUPS = 256

"""Draws of one work group of [`gpu_in_range_tally!`](@ref)."""
const GPU_IN_RANGE_GROUP = 256

"""
    gpu_in_range_tally!(tally, backend, x, dig, NB, geom, cull, tile) -> tally

Launch on `backend` the draw of `GPU_IN_RANGE_GROUPS × GPU_IN_RANGE_GROUP` pairs, uniform over the pairs that a sweep
of the kernel coordinates `x`, `(W, N)` or `(W, N, B)`, visits over the tile pairs of `tile`-point tiles the cull memo
`cull` schedules (`nothing`: every pair) and over the slices; column `g` of the `(2, GPU_IN_RANGE_GROUPS)` `Int32`
array `tally` receives work group `g`'s draws that are pairs and its pairs in one of the `NB` bins of the device
digitizer `dig`. A function of its inputs. `StructureFunctionsKernelAbstractionsExt` supplies it.
"""
function gpu_in_range_tally! end

"""The share in range of the pairs drawn, from a host-readable tally of [`gpu_in_range_tally!`](@ref); over `n` pairs
drawn its standard error is at most `1 / (2√n)`."""
function in_range_share(tally::AbstractMatrix)
    drawn = sum(view(tally, 1, :))
    return drawn == 0 ? 0.0 : sum(view(tally, 2, :)) / drawn
end

"""
    gpu_in_range_fraction(backend, x, dig, NB, geom, cull, tile) -> Float64

[`in_range_share`](@ref) of the tally [`gpu_in_range_tally!`](@ref) draws on `backend`, read back once.
`StructureFunctionsKernelAbstractionsExt` supplies it for every backend and `StructureFunctionsCUDAExt` for
`CUDA.CUDABackend`.
"""
function gpu_in_range_fraction end

"""
    gpu_tile_worklist(grid, n_points, tile) -> TilePairWorkList

Tile pairs `(ti, tj)`, `ti ≤ tj`, of `tile`-point tiles over `grid`'s permuted order that can hold a
pair inside `grid.cutoff`, sorted, unique and built on the device holding `grid`'s vectors. A pair
inside the cutoff has its `j` cell in a stencil row of its `i` cell, and each row reaches a tile as
one shifted range of cell ids, so the list is exact; pairs beyond the cutoff that share a listed tile
pair are rejected by the bin test. The element type is `Int32` while `n_tiles^2` fits it.
`StructureFunctionsKernelAbstractionsExt` supplies it.
"""
function gpu_tile_worklist end


"""
    GPUDeviceCaps(smem_optin, smem_per_sm, n_sms, warp)

What a GPU backend offers, queried at run time: `smem_optin`, the most shared memory one block may
use when its kernel opts in to dynamic shared memory; `smem_per_sm`, the shared memory of one
multiprocessor; `n_sms`, the multiprocessor count; `warp`, the threads that execute in lockstep.

A kernel's *static* shared memory is bounded by [`GPU_SMEM_STATIC_MAX`](@ref) as well. Every
`@localmem` array is static, so every portable kernel is bound by [`gpu_static_smem_budget`](@ref);
only a backend-specialized kernel that declares dynamic shared memory reaches `smem_optin`.
"""
struct GPUDeviceCaps
    smem_optin::Int
    smem_per_sm::Int
    n_sms::Int
    warp::Int
end

"""Largest static shared allocation a block may declare, in bytes, on any device."""
const GPU_SMEM_STATIC_MAX = 48 * 1024

"""Shared memory, in bytes, every device offers a block without opting in."""
const GPU_SMEM_UNIVERSAL_FLOOR = 48 * 1024

"""Alignment, in bytes, of every static shared array a kernel declares."""
const GPU_SMEM_ALIGN = 32

"""
    gpu_localmem_bytes(T, n) -> Int

Shared bytes an `n`-element static shared array of `T` occupies.
"""
@inline gpu_localmem_bytes(::Type{T}, n::Integer) where {T} =
    cld(Int(n) * sizeof(T), GPU_SMEM_ALIGN) * GPU_SMEM_ALIGN

"""
    gpu_localmem_scalar_bytes(T, n) -> Int

Shared bytes, at most, an `n`-element static shared array of `T` occupies when every access to it
has a constant index: the compiler then declares each element as its own aligned scalar.
"""
@inline gpu_localmem_scalar_bytes(::Type{T}, n::Integer) where {T} = Int(n) * gpu_localmem_bytes(T, 1)

"""
    gpu_static_smem_budget(caps) -> Int

Static shared bytes a kernel may declare on the device `caps` describes.
"""
@inline gpu_static_smem_budget(caps::GPUDeviceCaps) = min(GPU_SMEM_STATIC_MAX, caps.smem_optin)

"""
    gpu_static_smem_fits(caps, bytes) -> Bool

Whether a kernel declaring `bytes` of static shared memory, as [`gpu_localmem_bytes`](@ref) counts
them, compiles and launches on the device `caps` describes.
"""
@inline gpu_static_smem_fits(caps::GPUDeviceCaps, bytes::Integer) = bytes <= gpu_static_smem_budget(caps)

"""
    gpu_device_caps(backend) -> GPUDeviceCaps

Capabilities of `backend`: those every device offers unless an extension supplies the device's own.
`StructureFunctionsCUDAExt` does for `CUDA.CUDABackend`.
"""
gpu_device_caps(::Any) = GPUDeviceCaps(GPU_SMEM_UNIVERSAL_FLOOR, GPU_SMEM_UNIVERSAL_FLOOR, 1, 32)

"""
    gpu_free_memory(backend) -> Int

Bytes `backend` reports free for allocation. There is deliberately **no generic method**: a staged
calculation sizes its batches against this, and a backend whose memory is unknown raises rather than
staging against a guess. `KernelAbstractions.CPU()` answers with the host's free memory and
`CUDA.CUDABackend()` with the device's.
"""
function gpu_free_memory end

"""Release device buffers held by a [`GPUSFWorkspace`](@ref) (optional explicit free)."""
function release! end

"""Invalidate prepared geometry and input caches held by a [`GPUSFWorkspace`](@ref)."""
function refresh! end

"""
    joint2d_smem_max(backend, W, F, XT, OT, CT) -> Int

The widest joint histogram, in cells, whose shared-memory kernel fits `backend` for `W`-wide
coordinates and `F`-wide fields of element type `XT`, sums of `OT` and shared counts of `CT` (`UInt32`
when the call is unweighted and its pair count fits `UInt32`, the call's count type otherwise);
supplied by the KernelAbstractions extension.
"""
function joint2d_smem_max end

"""
    joint2d_smem_exact(n_dist, n_val) -> Int

The joint histogram's cell count `n_dist * n_val`, the compile width a `:joint2d` workspace takes by
default; supplied by the KernelAbstractions extension.
"""
function joint2d_smem_exact end

"""
    joint2d_smem_align256(n_dist, n_val) -> Int

`n_dist * n_val` rounded up to a multiple of 256, a compile width one kernel serves for every
histogram of at most that many cells; supplied by the KernelAbstractions extension.
"""
function joint2d_smem_align256 end


"""
    gpu_calculate_structure_function(sf, backend, x, u, distance_bins, CT; kwargs...)

The 1-D pair histogram on the KernelAbstractions backend `backend`, with device-resident sums and
counts of element type `CT`; supplied by the KernelAbstractions extension. Pass
`workspace=GPUSFWorkspace(...)` to reuse device histogram buffers across repeated calls (see
[`GPUSFWorkspace`](@ref)).
"""
function gpu_calculate_structure_function end

"""
    gpu_calculate_structure_function_2d(sf_type, backend, x_mat, u_mat, distance_bins, value_bins, CT; kwargs...)

The distance × value joint histogram of one `sf_type` on a KernelAbstractions backend, with
device-resident counts of element type `CT`; supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_function_2d end

"""
    gpu_calculate_structure_functions_single_pass_2d(backend, x, u, distance_bins, value_bins, CT; kwargs...)

Six invariant distance × value joint histograms on a KernelAbstractions backend, with device-resident
counts of element type `CT`; supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_functions_single_pass_2d end

"""
    gpu_calculate_structure_functions_single_pass_2d!(sums, counts, backend, x, u, distance_bins, value_bins; kwargs...)

The in-place form of [`gpu_calculate_structure_functions_single_pass_2d`](@ref), accumulating into
device buffers `sums` and `counts`; supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_functions_single_pass_2d! end

"""
    gpu_calculate_structure_function!(sums, counts, sf, backend, x, u, distance_bins; kwargs...)

The in-place form of [`gpu_calculate_structure_function`](@ref), accumulating into device buffers
`sums` and `counts`; supplied by the KernelAbstractions extension. Pass `workspace=GPUSFWorkspace(...)`
to reuse device scratch across calls.
"""
function gpu_calculate_structure_function! end

"""
    gpu_calculate_structure_function_batch!(sums, counts, sf_type, backend, x, u, distance_bins; workspace=nothing, ...)

The 1-D histogram of each auxiliary slice of `(N_dims, N_points, T)` input, accumulated into device
buffers of shape `(NB, T)`; supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_function_batch! end

"""
    gpu_calculate_structure_function_batch(sf_type, backend, x, u, distance_bins, CT; kwargs...)

The allocating form of [`gpu_calculate_structure_function_batch!`](@ref); supplied by the
KernelAbstractions extension.
"""
function gpu_calculate_structure_function_batch end

"""
    gpu_calculate_structure_function_2d_batch(sf, backend, x, u, distance_bins, value_bins, CT; kwargs...)

The value-binned joint histogram of a field with auxiliary axes on a device, one histogram per
auxiliary slice; supplied by the KernelAbstractions extension.
"""
function gpu_calculate_structure_function_2d_batch end

"""The in-place form of [`gpu_calculate_structure_function_2d_batch`](@ref), outputs `(n_dist, n_val, T)`."""
function gpu_calculate_structure_function_2d_batch! end

"""Six invariant 1-D histograms of each auxiliary slice on a device, outputs `(6, NB, T)`."""
function gpu_calculate_structure_functions_single_pass_batch! end

"""Six invariant joint histograms of each auxiliary slice on a device, outputs `(6, NB, n_val, T)`."""
function gpu_calculate_structure_functions_single_pass_2d_batch! end

"""
    gpu_calculate_structure_function_2d!(sums, counts, sf, backend, x, u, distance_bins, value_bins; kwargs...)

The in-place form of [`gpu_calculate_structure_function_2d`](@ref); supplied by the
KernelAbstractions extension.
"""
function gpu_calculate_structure_function_2d! end
