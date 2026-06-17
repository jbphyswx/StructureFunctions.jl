#!/usr/bin/env julia
"""
    diagnose_reconcile.jl

Pin down the sum gap: Float64 serial reference vs stride vs private.

**SLURM only for N >= 8000.**

    include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))
    ENV["N"] = "20000"
    include_gpu("diagnose_reconcile.jl")
"""

using Printf: @printf
using Random: Random
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges

const _LOGIN_N_MAX = 8_000

function _check_n_budget!(N::Int)
    in_slurm = haskey(ENV, "SLURM_JOB_ID")
    allowed = get(ENV, "ALLOW_LARGE_N", "0") == "1"
    if N > _LOGIN_N_MAX && !in_slurm && !allowed
        npairs = N * (N - 1) ÷ 2
        error(
            "N=$N (~$(round(npairs / 1e6; digits=1))M pairs) too heavy for login node. " *
            "Run in SLURM or set ALLOW_LARGE_N=1.",
        )
    end
    return nothing
end

include(joinpath(@__DIR__, "benchmark_helpers.jl"))
include(joinpath(@__DIR__, "GPUPrototypeKernels.jl"))

function main()
    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "500"))
    _check_n_budget!(N)
    FT = Float32
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    nworkers = min(262_144, N * (N - 1) ÷ 2)

    println("=== Sum reconcile (Float64 serial reference) ===")
    println("N = $N  nworkers = $nworkers  Threads.nthreads() = $(Threads.nthreads())")
    if Threads.nthreads() > 1
        println("WARNING: use `-t 1` for deterministic CPU sims.")
    end
    println()

    r = reconcile_sum_paths(x_cpu, u_cpu, sft, bin_edges; nworkers = nworkers)

    @printf("max|f64_serial - stride|     = %.6g\n", r.max_ref_stride)
    @printf("max|f64_serial - private|    = %.6g\n", r.max_ref_private)
    @printf("max|stride - private|        = %.6g\n", r.max_stride_private)
    @printf("max|rebuild - private|       = %.6g  (private sim self-check)\n", r.max_rebuild_private)
    @printf("sum(private) / sum(stride)   = %.8f\n", r.ratio_private_stride)

    println("\nInterpretation:")
    if r.max_rebuild_private > 1f-3
        println("  private sim ≠ manual worker merge — implementation bug.")
    elseif r.max_ref_private <= 500 && r.max_ref_stride > 500
        println("  Validated f64 refs agree; private f32 ≈ f64; stride/global f32 low vs f64.")
        println("  Run prove_f32_accumulation.jl for no-op add counts on real bin values.")
    elseif r.max_stride_private <= 500
        println("  All paths agree at this N.")
    end

    return r
end

main()
