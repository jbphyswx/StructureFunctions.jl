"""
CUDA-specialized fast structure-function kernels.

Loaded automatically when **both** `KernelAbstractions` and `CUDA` are present.
Provides N-body broadcast kernels with privatized or dynamic-shared histograms for NVIDIA GPUs, for
weighted and unweighted calls at the geometry's coordinate and field widths, overriding the portable
KernelAbstractions tiled kernels in `StructureFunctionsKernelAbstractionsExt`.

These kernels use CUDA-only intrinsics not exposed by KernelAbstractions:
`CuDynamicSharedArray` (>48 KB dynamic shared via the opt-in attribute),
`CUDA.@atomic`, `@cuda launch=false`, and device shared-memory queries. The
`GPUBackend{B}` wrapper is parametric precisely so the CUDA backend can take this
specialized path.

The kernels form each pair's moments with the core moment sets (`SFC._sf_pair_moments`) and bin with the
host's `SFH.digitize`, as the portable kernels do.
"""
module StructureFunctionsCUDAExt

using CUDA: CUDA, CuStaticSharedArray, CuDynamicSharedArray, @cuda,
    threadIdx, blockIdx, blockDim, sync_threads
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    HelperFunctions as SFH, StructureFunctionTypes as SFT

"""The shared-memory share of an SM's unified L1 the native kernels ask for, in percent: all of it, so as many
blocks as their shared memory admits reside on each SM."""
const CU_CARVEOUT_MAX_SHARED = 100

include(joinpath(@__DIR__, "cuda", "plans.jl"))
include(joinpath(@__DIR__, "cuda", "kernels_2d.jl"))
include(joinpath(@__DIR__, "cuda", "kernels_1d.jl"))

# The native-kernel hooks of `SFC` for `CUDA.CUDABackend`.

"""Native plan choices, built once per device and the types and sizes that decide them."""
const CU_CHOICES = Dict{Tuple, Any}()
const CU_CHOICES_LOCK = ReentrantLock()

"""The choice `build(caps)` makes for the current device and `key`, built on the first call, or again when the kept one
is not of the type `build` returns."""
function _cuda_choice(build, key::Tuple)
    T = Base.promote_op(build, SFC.GPUDeviceCaps)
    k = (CUDA.device(), key...)
    return lock(CU_CHOICES_LOCK) do
        haskey(CU_CHOICES, k) && CU_CHOICES[k] isa T && return CU_CHOICES[k]::T
        fresh = build(SFC.gpu_device_caps(CUDA.CUDABackend()))
        CU_CHOICES[k] = fresh
        fresh
    end
end

SFC.gpu_native_1d_plan(::CUDA.CUDABackend, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       NB::Integer, moments) where {XT, UT, OT, CT} =
    _cuda_choice((:sf1d, XT, UT, OT, CT, wts isa SFC.NoWeights, typeof(geom), typeof(moments), Int(NB))) do caps
        _cuda_1d_plan(caps, XT, UT, OT, CT, wts, geom, Int(NB), moments)
    end

SFC.gpu_native_launch_1d!(plan::Union{CUDA1DPlan, CUDA1DStripPlan, CUDAChoice}, out, cnt, x, u, wts, sf_type, dist_dig,
                          N, NB, B, fixed_x, geom, cull) =
    _cuda_launch_1d!(plan, out, cnt, x, u, wts, sf_type, dist_dig, Int(N), Int(NB), Int(B), fixed_x, geom, cull)

SFC.gpu_native_2d_plan(::CUDA.CUDABackend, ::Type{XT}, ::Type{UT}, ::Type{OT}, ::Type{CT}, wts, geom,
                       moments, n_dist::Integer, n_val::Integer, val_plan) where {XT, UT, OT, CT} =
    _cuda_choice((:sf2d, XT, UT, OT, CT, wts isa SFC.NoWeights, typeof(geom), typeof(moments), Int(n_dist),
                  Int(n_val), typeof(val_plan))) do caps
        _cuda_2d_plan(caps, XT, UT, OT, CT, wts, geom, moments, Int(n_dist), Int(n_val))
    end

SFC.gpu_native_launch_2d!(plan::Union{CUDA2DPlan, CUDA2DGlobalPlan, CUDAChoice}, out, cnt, x, u, wts, sf_type,
                          dist_dig, val_plan, N, n_dist, n_val, B, fixed_x, geom, second_axis, cull, portable!) =
    _cuda_launch_2d!(plan, out, cnt, x, u, wts, sf_type, dist_dig, val_plan, Int(N), Int(n_dist), Int(n_val),
                     Int(B), fixed_x, geom, second_axis, cull, portable!)

SFC.gpu_free_memory(::CUDA.CUDABackend) = Int(CUDA.free_memory())

"""A tally of `SFC.gpu_in_range_tally!` in mapped host memory: the kernel writes it across the bus, so reading it
takes a stream synchronization and no copy."""
const CUTally = CUDA.CuArray{Int32, 2, CUDA.HostMemory}

"""Free tallies per context, each borrowed by one estimate at a time."""
const CU_FREE_TALLIES = Dict{CUDA.CuContext, Vector{CUTally}}()
const CU_TALLY_LOCK = ReentrantLock()

function SFC.gpu_in_range_fraction(backend::CUDA.CUDABackend, x, dig, NB::Int, geom, cull, tile::Int)
    ctx = CUDA.context()
    tally = lock(CU_TALLY_LOCK) do
        free = get!(Vector{CUTally}, CU_FREE_TALLIES, ctx)
        isempty(free) ? CUTally(undef, 2, SFC.GPU_IN_RANGE_GROUPS) : pop!(free)
    end
    SFC.gpu_in_range_tally!(tally, backend, x, dig, NB, geom, cull, tile)
    CUDA.synchronize()
    share = SFC.in_range_share(unsafe_wrap(Array, tally))
    lock(() -> push!(CU_FREE_TALLIES[ctx], tally), CU_TALLY_LOCK)
    return share
end

# The current device's shared-memory limits, multiprocessor count and warp size.
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
