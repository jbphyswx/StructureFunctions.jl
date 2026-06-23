"""
CUDA-specialized fast structure-function kernels.

Loaded automatically when **both** `KernelAbstractions` and `CUDA` are present.
Provides the settled-optimal GPU kernels (N-body broadcast + privatized /
dynamic-shared histograms) for NVIDIA GPUs, overriding the portable
KernelAbstractions tiled kernels in `StructureFunctionsGPUExt` (which remain the
CPU/AMD reference). See `gpu/OPTIMAL_KERNEL_DESIGN.md`.

These kernels use CUDA-only intrinsics not exposed by KernelAbstractions:
`CuDynamicSharedArray` (>48 KB dynamic shared via the opt-in attribute),
`CUDA.@atomic`, `@cuda launch=false`, and device shared-memory queries. The
`GPUBackend{B}` wrapper is parametric precisely so the CUDA backend can take this
specialized path while the CPU backend stays on the KA kernels.

The pure, device-callable building blocks (`_sf_moments`, `_sf_dot`,
`_gpu_digitize_value_plan`, digitizer functors, value plans) live in
`StructureFunctionsGPUExt`; this extension reuses them via `GE` so there is a
single source of truth for the per-pair math and binning.
"""
module StructureFunctionsCUDAExt

using CUDA: CUDA, CuStaticSharedArray, CuDynamicSharedArray, @cuda,
    threadIdx, blockIdx, blockDim, sync_threads
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC

# The GPU (KernelAbstractions) extension owns the shared device-callable building
# blocks and the digitizer / value-plan types. It is triggered by
# KernelAbstractions alone, so it is loaded whenever this extension's triggers
# (KernelAbstractions + CUDA) are satisfied.
const GE = let m = Base.get_extension(SF, :StructureFunctionsGPUExt)
    m === nothing &&
        error("StructureFunctionsCUDAExt: StructureFunctionsGPUExt must be loaded first " *
              "(load KernelAbstractions before / with CUDA).")
    m
end

include(joinpath(@__DIR__, "cuda", "kernels_2d.jl"))
include(joinpath(@__DIR__, "cuda", "kernels_1d.jl"))

# ---------------------------------------------------------------------------
# Dispatch hooks (override the package stubs in src/Calculations/gpu_stubs.jl).
# Specialized on CUDA.CUDABackend; the default methods return `false`.
# ---------------------------------------------------------------------------

function SFC.gpu_fast_launch_2d_batch!(
    ::CUDA.CUDABackend, out, cnt, x, u, sf_type, dist_dig, val_plan,
    N, n_dist, n_val, B, D, nmom, fixed_x,
)
    return _cuda_launch_2d!(out, cnt, x, u, sf_type, dist_dig, val_plan,
                            Int(N), Int(n_dist), Int(n_val), Int(B),
                            Int(D), Int(nmom), fixed_x)
end

function SFC.gpu_fast_launch_1d_batch!(
    ::CUDA.CUDABackend, out, cnt, x, u, sf_type, dist_dig,
    N, NB, B, D, nmom, fixed_x,
)
    return _cuda_launch_1d!(out, cnt, x, u, sf_type, dist_dig,
                            Int(N), Int(NB), Int(B), Int(D), Int(nmom), fixed_x)
end

end # module
