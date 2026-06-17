# Micro-benchmark: GPUSFWorkspace vs fresh device allocation per call.
#
# Run on a GPU node (SLURM), not in agent chat:
#   julia --project=. gpu/benchmark_workspace.jl
#
# Compares end-to-end time for K repeated 1D SF calls with N_points=20000
# (fresh alloc each call vs one GPUSFWorkspace).

using KernelAbstractions: KernelAbstractions as KA
using CUDA: CUDA
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

function main()
    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "20000"))
    K = parse(Int, get(ENV, "K", "50"))
    FT = Float32

    backend = CUDA.functional() ? CUDA.CUDABackend() : KA.CPU()
    println("backend: ", typeof(backend))

    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    if backend isa CUDA.CUDABackend
        x = CUDA.cu(x)
        u = CUDA.cu(u)
    end
    bins = collect(FT, range(FT(0), FT(1.5), length = 65))
    sft = SFT.L2SFType()

    ws = SFC.GPUSFWorkspace(backend, bins)

    fresh_t = @elapsed for _ in 1:K
        bench_gpu_sf_fresh(backend, x, u, bins, sft; warmup = 0)
    end
    ws_t = @elapsed for _ in 1:K
        bench_gpu_sf_with_workspace(backend, x, u, bins, sft, ws; warmup = 0)
    end
    for (label, t, per) in (
        ("fresh_alloc", fresh_t, fresh_t / K),
        ("workspace", ws_t, ws_t / K),
    )
        println("$label: total=$(round(t, digits=4))s  per_call=$(round(per * 1000, digits=3))ms  (K=$K, N=$N)")
    end
    CUDA.functional() && CUDA.synchronize()

    SFC.release!(ws)
    return nothing
end

main()
