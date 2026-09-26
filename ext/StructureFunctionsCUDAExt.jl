"""
CUDA-specialized fast structure-function kernels.

Loaded automatically when **both** `KernelAbstractions` and `CUDA` are present.
Provides N-body broadcast kernels with privatized or dynamic-shared histograms for NVIDIA GPUs, for
weighted and unweighted calls at the geometry's coordinate and field widths, overriding the portable
KernelAbstractions tiled kernels in `StructureFunctionsKernelAbstractionsExt`, which remain the CPU
and GPU reference.

These kernels use CUDA-only intrinsics not exposed by KernelAbstractions:
`CuDynamicSharedArray` (>48 KB dynamic shared via the opt-in attribute),
`CUDA.@atomic`, `@cuda launch=false`, and device shared-memory queries. The
`GPUBackend{B}` wrapper is parametric precisely so the CUDA backend can take this
specialized path while the CPU backend stays on the KA kernels.

The pure, device-callable building blocks (`_sf_moments`, `_sf_value_bin`) live in
`StructureFunctionsKernelAbstractionsExt`; this extension reuses them via `GE`, and bins with the
host's `SFH.digitize`, so there is a single source of truth for the per-pair math and binning.
"""
module StructureFunctionsCUDAExt

using CUDA: CUDA, CuStaticSharedArray, CuDynamicSharedArray, @cuda,
    threadIdx, blockIdx, blockDim, sync_threads
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH

# The GPU (KernelAbstractions) extension owns the shared device-callable building
# blocks. It is triggered by
# KernelAbstractions alone, so it is loaded whenever this extension's triggers
# (KernelAbstractions + CUDA) are satisfied.
const GE = let m = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    isnothing(m) &&
        error("StructureFunctionsCUDAExt: StructureFunctionsKernelAbstractionsExt must be loaded first " *
              "(load KernelAbstractions before / with CUDA).")
    m
end

include(joinpath(@__DIR__, "cuda", "kernels_2d.jl"))
include(joinpath(@__DIR__, "cuda", "kernels_1d.jl"))
include(joinpath(@__DIR__, "cuda", "culling.jl"))

# ---------------------------------------------------------------------------
# Dispatch hooks (override the package stubs in src/Calculations/gpu_stubs.jl).
# Specialized on CUDA.CUDABackend; the default methods return `false`.
# ---------------------------------------------------------------------------

SFC.gpu_native_1d_plan(::CUDA.CUDABackend, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       NB::Integer, NMOM::Integer) where {XT, UT, OT, CT} =
    _cuda_1d_plan(SFC.gpu_device_caps(CUDA.CUDABackend()), XT, UT, OT, CT, wts, geom, Int(NB), Int(NMOM))

SFC.gpu_native_launch_1d!(plan::CUDA1DPlan, out, cnt, x, u, wts, sf_type, dist_dig, N, NB, B, fixed_x,
                          geom, cull) =
    _cuda_launch_1d!(plan, out, cnt, x, u, wts, sf_type, dist_dig, Int(N), Int(NB), Int(B), fixed_x, geom, cull)

SFC.gpu_native_2d_plan(::CUDA.CUDABackend, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       NMOM::Integer, n_dist::Integer, n_val::Integer) where {XT, UT, OT, CT} =
    _cuda_2d_plan(SFC.gpu_device_caps(CUDA.CUDABackend()), XT, UT, OT, CT, wts, geom, Int(NMOM),
                  Int(n_dist), Int(n_val))

SFC.gpu_native_launch_2d!(plan::CUDA2DPlan, out, cnt, x, u, wts, sf_type, dist_dig, val_plan, N, n_dist,
                          n_val, B, fixed_x, geom, second_axis, cull) =
    _cuda_launch_2d!(plan, out, cnt, x, u, wts, sf_type, dist_dig, val_plan, Int(N), Int(n_dist), Int(n_val),
                     Int(B), fixed_x, geom, second_axis, cull)

SFC.gpu_free_memory(::CUDA.CUDABackend) = Int(CUDA.free_memory())

# The real device numbers. Reached only through the CUDABackend hook, so a device exists by
# construction and a query failure is a driver fault.
function SFC.gpu_device_caps(::CUDA.CUDABackend)
    dev = CUDA.device()
    return SFC.GPUDeviceCaps(
        Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)),
        Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)),
        Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)),
        Int(CUDA.warpsize(dev)),
    )
end

end # module
