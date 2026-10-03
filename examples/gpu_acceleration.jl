# Reuse GPU histogram scratch for repeated point-field calculations.
include("resources.jl")
ExampleResources.require_allocation(; gpu=true, cpus=Threads.nthreads())
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions
using Random: Random
using ComputationalBackends: GPUBackend
using StructureFunctions: Calculations as C, StructureFunctionTypes as T, StructureFunctionSumsAndCounts
CUDA.functional() || error("This example requires a functioning allocated CUDA device")
CUDA.allowscalar(false)

function gpu_example(; n=ExampleResources.points())
    rng = Random.MersenneTwister(15)
    x, u = CUDA.CuArray(Random.rand(rng, Float32, 3, n)), CUDA.CuArray(Random.randn(rng, Float32, 3, n))
    bins = range(0.0f0, 2.0f0; length=9)
    device = CUDA.CUDABackend()
    workspace = C.GPUSFWorkspace(device, bins)
    try
        calculate(; kwargs...) = C.calculate_structure_function(T.L2SFType(), x, u, bins,
            StructureFunctionSumsAndCounts; backend=GPUBackend(device), kwargs...)
        fresh = calculate()
        reused = calculate(; workspace)
        @assert Array(fresh.counts) == Array(reused.counts)
        @assert Array(fresh.sums) ≈ Array(reused.sums)
        println("Pairs: ", sum(reused.counts))
        return reused
    finally
        C.release!(workspace)
    end
end
result = gpu_example()
