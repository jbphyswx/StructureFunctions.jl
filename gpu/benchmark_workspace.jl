# Time of one public 1-D GPU call with and without a GPUSFWorkspace.
#
# Run on a GPU allocation:
#   ] activate gpu
#   > include("gpu/benchmark_workspace.jl")
#   > main(; N=20000)

using KernelAbstractions: KernelAbstractions as KA
using CUDA: CUDA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

function main()
    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "20000"))
    FT = Float32

    backend = CUDA.functional() ? CUDA.CUDABackend() : KA.CPU()
    println("backend: ", typeof(backend))

    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    if backend isa CUDA.CUDABackend
        x = CUDA.CuArray(x)
        u = CUDA.CuArray(u)
    end
    bins = collect(FT, range(FT(0), FT(1.5), length = 65))
    sft = SFT.L2SFType()

    ws = SFC.GPUSFWorkspace(backend, bins)
    fresh = bench_gpu_sf_fresh(backend, x, u, bins, sft)
    reused = bench_gpu_sf_with_workspace(backend, x, u, bins, sft, ws)
    for (label, t) in (("fresh_alloc", fresh), ("workspace", reused))
        println("$label: per_call=$(round(t * 1000, digits=3))ms  (N=$N)")
    end
    SFC.release!(ws)
    return nothing
end

main()
