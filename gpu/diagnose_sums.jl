#!/usr/bin/env julia
"""
    diagnose_sums.jl

CPU-only sum diagnostic: serial gold vs grid-stride / private / blockshared simulators.

Load via `run.jl` (pwd-independent). Default N=500 (login-safe). N=20000 → SLURM only.

    include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))
    include_gpu("diagnose_sums.jl")
    include_gpu("diagnose_reconcile.jl")   # SLURM: pin which path is wrong
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
            "N=$N (~$(round(npairs / 1e6; digits=1))M pairs) is too heavy for the login node. " *
            "Run inside your SLURM allocation (detected via SLURM_JOB_ID), or set ALLOW_LARGE_N=1.",
        )
    end
    if N > _LOGIN_N_MAX && (in_slurm || allowed)
        @printf("Note: large-N run (N=%d, ~%.1fM pairs) — expect minutes on CPU.\n\n", N, N * (N - 1) / 2e6)
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

    total_pairs = N * (N - 1) ÷ 2
    nworkers = min(262_144, total_pairs)
    ws = 256

    println("=== Sum diagnostic (CPU simulators only) ===")
    println("N = $N  pairs = $(round(total_pairs / 1e6; digits=3))M  nworkers = $nworkers  wgs = $ws")
    println("Threads.nthreads() = $(Threads.nthreads())")
    if Threads.nthreads() > 1
        println("WARNING: use `-t 1` for deterministic CPU sims.")
    end
    println()

    gold = cpu_gold_histogram(x_cpu, u_cpu, sft, bin_edges)
    gold_f64 = cpu_f64_serial_sums(x_cpu, u_cpu, sft, bin_edges)
    sum_ref = gold_f64.sums_f32
    stride = cpu_stride_global_histogram(
        x_cpu, u_cpu, sft, bin_edges; nworkers = nworkers,
    )
    private = cpu_private_histogram(
        x_cpu, u_cpu, sft, bin_edges; nworkers = nworkers,
    )
    blockshared = cpu_blockshared_histogram(
        x_cpu, u_cpu, sft, bin_edges; nworkers = nworkers, workgroup_size = ws,
    )
    private_f64 = cpu_private_histogram_f64(
        x_cpu, u_cpu, sft, bin_edges; nworkers = nworkers,
    )

    paths = [stride, private, blockshared, private_f64]
    rows = compare_histogram_paths(gold, paths; sum_ref = sum_ref)

    println("Path            | Δcnt | max|Δcnt| | max|Δsum| vs f64 | counts | sums")
    println("-" ^ 72)
    for r in rows
        @printf(
            "%-15s | %+5d | %11d | %9.4g | %6s | %s\n",
            r.path,
            r.Δcnt,
            r.max_Δcnt_bin,
            r.max_Δsum,
            r.counts_ok ? "ok" : "FAIL",
            r.sums_ok ? "ok" : "FAIL",
        )
    end

    worst = argmax([r.max_Δsum for r in rows])
    print_worst_bin_table(gold, paths[worst])

    priv_vs_stride = maximum(abs.(private.sums .- stride.sums))
    bs_vs_private = maximum(abs.(blockshared.sums .- private.sums))
    priv_vs_f64 = maximum(abs.(private.sums .- sum_ref))
    stride_vs_f64 = maximum(abs.(stride.sums .- sum_ref))
    gold_vs_f64 = maximum(abs.(gold.sums .- sum_ref))
    println()
    @printf("vs Float64 serial: stride=%.6g  private=%.6g  f32 gold=%.6g\n",
        stride_vs_f64, priv_vs_f64, gold_vs_f64)
    @printf("stride vs private:             %.6g\n", priv_vs_stride)
    @printf("blockshared vs private:        %.6g\n", bs_vs_private)

    return rows
end

main()
