"""
    smoke_cuda.jl

Quick CUDA + StructureFunctions smoke test. No `@test` harness.

Run from the repository root:

    julia --project=gpu gpu/smoke_cuda.jl
"""

using ComputationalBackends: ComputationalBackends as CB
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges
using Random: Random

function main()
    println("=== CUDA environment ===")
    println("CUDA.functional(): ", CUDA.functional())
    if !CUDA.functional()
        println("No functional CUDA device — nothing to do.")
        return
    end
    println("Device: ", CUDA.name(CUDA.device()))
    println("Memory (GiB): ", round(CUDA.total_memory() / 1024^3; digits = 1))
    if haskey(ENV, "CUDA_VISIBLE_DEVICES")
        println("CUDA_VISIBLE_DEVICES: ", ENV["CUDA_VISIBLE_DEVICES"])
    end

    println("\n=== Structure function on GPU ===")
    Random.seed!(42)
    N = 10_000
    FT = Float32
    x = CUDA.rand(FT, 2, N)
    u = CUDA.rand(FT, 2, N)
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    sft = SFT.L2SFType()

    res = SFC.gpu_calculate_structure_function(
        sft, CUDA.CUDABackend(), x, u, bin_edges
    )
    CUDA.synchronize()

    n_pairs = sum(res.counts)
    println("Nonzero bins: ", count(!iszero, res.counts))
    println("Total pair count: ", n_pairs)
    println("Max bin sum: ", maximum(res.sums))

    backend = CB.GPUBackend(CUDA.CUDABackend())
    x_cpu = Array(x)
    u_cpu = Array(u)
    result = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, bin_edges;
        backend = backend,
        verbose = false,
        show_progress = false,
    )
    println("GPUBackend API bins: ", length(result.distance))
    println("\nSmoke test OK.")
    return nothing
end

main()
