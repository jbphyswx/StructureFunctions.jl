#!/usr/bin/env julia
"""
    benchmark_scaling.jl  —  CPU thread-scaling (strong & weak) + GPU skeleton.

This script is NOT run by `runtests.jl`. It is an OPTIONAL standalone benchmark.

## Strong scaling
Fixed problem size N, increase threads. Ideal: speedup = p.

## Weak scaling
Scale N ∝ √p so each thread has constant O(N²/p) pair-wise work.
Ideal: wall-clock time stays constant as p grows.

## GPU scaling (skeleton — skipped if no CUDA GPU)
Benchmarks `gpu_calculate_structure_function` across problem sizes when a
CUDA-functional GPU is detected.

## How to run

    # From the repository root:
    julia --project=test test/benchmark_scaling.jl

    # Override max thread count (default: min(32, Sys.CPU_THREADS)):
    MAX_THREADS=16 julia --project=test test/benchmark_scaling.jl

    # Override base problem size for strong scaling (default: 4000 points):
    N_STRONG=6000 julia --project=test test/benchmark_scaling.jl

    # Override base size per thread for weak scaling (default: 1000 points/thread):
    N_WEAK_BASE=2000 julia --project=test test/benchmark_scaling.jl

See README.md § "Scaling Benchmarks" for full documentation.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────
using JSON: JSON
using Printf: Printf
using Dates: Dates

const RESULTS_DIR = joinpath(@__DIR__, "benchmark_results")
mkpath(RESULTS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
# Strong: fixed N, vary threads
const N_STRONG = parse(Int, get(ENV, "N_STRONG", "4000"))
# Weak: N ∝ √p × N_WEAK_BASE (keeps O(N²/p) = N_WEAK_BASE² constant per thread)
const N_WEAK_BASE = parse(Int, get(ENV, "N_WEAK_BASE", "1000"))
const MAX_THREADS = min(parse(Int, get(ENV, "MAX_THREADS", "32")), Sys.CPU_THREADS)
const THREAD_COUNTS = filter(t -> t <= MAX_THREADS, [1, 2, 4, 8, 16, 32])

println("="^70)
println("StructureFunctions.jl — CPU Thread Scaling Benchmark")
println("  Machine threads available : $(Sys.CPU_THREADS)")
println("  Thread counts to test     : $(THREAD_COUNTS)")
println("  Strong scaling N          : $(N_STRONG) (fixed)")
println("  Weak scaling N_base/thread: $(N_WEAK_BASE) → N = round(√p × $N_WEAK_BASE)")
println("  Results directory         : $(RESULTS_DIR)")
println("="^70)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: spawn a worker subprocess with the given thread count and N
# ─────────────────────────────────────────────────────────────────────────────
"""
    run_worker(n_threads, n_points) -> Dict

Spawn `benchmark_worker.jl` with `-t n_threads` and return the parsed JSON
result. Subprocesses are required because Julia's thread count is fixed at
startup and cannot be changed at runtime.
"""
function run_worker(n_threads::Int, n_points::Int)
    worker_script = joinpath(@__DIR__, "benchmark_worker.jl")
    julia_bin = joinpath(Sys.BINDIR, "julia")
    project_dir = joinpath(@__DIR__, "..")

    cmd = `$julia_bin --project=$project_dir/test -t $n_threads $worker_script $n_points`
    output = readchomp(cmd)
    lines = filter(!isempty, split(output, '\n'))
    return JSON.parse(last(lines))
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. STRONG SCALING: fixed N, vary threads
# ─────────────────────────────────────────────────────────────────────────────
println("\n── Strong scaling (N = $N_STRONG fixed) ────────────────────────────")
strong_results = Dict[]

for p in THREAD_COUNTS
    print("  threads = $(lpad(p, 2))  N = $(lpad(N_STRONG, 6))  …  ")
    flush(stdout)
    r = run_worker(p, N_STRONG)
    push!(strong_results, r)
    Printf.@printf("%.3f s\n", r["elapsed_s"])
end

t1_strong = strong_results[1]["elapsed_s"]
for r in strong_results
    r["speedup"] = t1_strong / r["elapsed_s"]
    r["efficiency"] = r["speedup"] / r["threads"]
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. WEAK SCALING: N ∝ √p, so N²/p ≈ constant per thread
# ─────────────────────────────────────────────────────────────────────────────
println("\n── Weak scaling (N ∝ √p, N_base = $N_WEAK_BASE per thread) ─────────")
weak_results = Dict[]

for p in THREAD_COUNTS
    N_weak = round(Int, N_WEAK_BASE * sqrt(p))
    print("  threads = $(lpad(p, 2))  N = $(lpad(N_weak, 6))  …  ")
    flush(stdout)
    r = run_worker(p, N_weak)
    r["N_weak"] = N_weak   # store for plotting
    push!(weak_results, r)
    Printf.@printf("%.3f s\n", r["elapsed_s"])
end

# Normalise: ideal weak scaling → elapsed stays constant (ratio = 1)
t1_weak = weak_results[1]["elapsed_s"]
for r in weak_results
    r["normalised_time"] = r["elapsed_s"] / t1_weak
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Save JSON
# ─────────────────────────────────────────────────────────────────────────────
json_path = joinpath(RESULTS_DIR, "scaling_results.json")
open(json_path, "w") do io
    JSON.print(io,
        Dict(
            "timestamp" => string(Dates.now()),
            "machine_threads" => Sys.CPU_THREADS,
            "N_strong" => N_STRONG,
            "N_weak_base" => N_WEAK_BASE,
            "strong_scaling" => strong_results,
            "weak_scaling" => weak_results,
        ), 2)
end
println("\nSaved results → $json_path")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Figures
# ─────────────────────────────────────────────────────────────────────────────
println("\n── Generating figures ──────────────────────────────────────────────")
try
    using CairoMakie: CairoMakie as CM

    threads_v = Float64.(THREAD_COUNTS)
    ideal_line = threads_v

    # --- Strong scaling figure ---
    fig_strong = CM.Figure(size = (1000, 700), fontsize = 14)
    CM.Label(fig_strong[0, 1:2],
        "Strong Scaling  (N = $N_STRONG points, 3D, longitudinal 2nd-order SF)",
        fontsize = 16, font = :bold)

    ax_t1 = CM.Axis(fig_strong[1, 1],
        xlabel = "Threads", ylabel = "Elapsed time (s)",
        xscale = log2, xticks = (THREAD_COUNTS, string.(THREAD_COUNTS)))
    CM.lines!(ax_t1, threads_v, [r["elapsed_s"] for r in strong_results],
        color = :steelblue, linewidth = 2.5)
    CM.scatter!(ax_t1, threads_v, [r["elapsed_s"] for r in strong_results],
        color = :steelblue, markersize = 10)

    ax_sp = CM.Axis(fig_strong[1, 2],
        xlabel = "Threads", ylabel = "Speedup  (T₁ / Tₚ)",
        xscale = log2, yscale = log2,
        xticks = (THREAD_COUNTS, string.(THREAD_COUNTS)),
        yticks = (THREAD_COUNTS, string.(THREAD_COUNTS)))
    CM.lines!(ax_sp, threads_v, ideal_line,
        color = (:gray, 0.7), linewidth = 1.5, linestyle = :dash, label = "Ideal (linear)")
    CM.lines!(ax_sp, threads_v, [r["speedup"] for r in strong_results],
        color = :crimson, linewidth = 2.5, label = "Actual")
    CM.scatter!(ax_sp, threads_v, [r["speedup"] for r in strong_results],
        color = :crimson, markersize = 10)
    CM.axislegend(ax_sp, position = :lt)

    ax_eff = CM.Axis(fig_strong[2, 1:2],
        xlabel = "Threads", ylabel = "Parallel efficiency  (speedup / p)",
        xscale = log2, xticks = (THREAD_COUNTS, string.(THREAD_COUNTS)),
        limits = (nothing, (0.0, 1.15)))
    CM.hlines!(ax_eff, [1.0], color = (:gray, 0.6), linestyle = :dash)
    CM.lines!(ax_eff, threads_v, [r["efficiency"] for r in strong_results],
        color = :seagreen, linewidth = 2.5)
    CM.scatter!(ax_eff, threads_v, [r["efficiency"] for r in strong_results],
        color = :seagreen, markersize = 10)

    CM.save(joinpath(RESULTS_DIR, "strong_scaling.png"), fig_strong, px_per_unit = 2)
    println("Saved → $(joinpath(RESULTS_DIR, "strong_scaling.png"))")

    # --- Weak scaling figure ---
    N_weak_v = Float64.([r["N_weak"] for r in weak_results])

    fig_weak = CM.Figure(size = (1000, 700), fontsize = 14)
    CM.Label(fig_weak[0, 1:2],
        "Weak Scaling  (N ∝ √p, N_base = $N_WEAK_BASE/thread, 3D, longitudinal 2nd-order SF)",
        fontsize = 16, font = :bold)

    ax_wt = CM.Axis(fig_weak[1, 1],
        xlabel = "Threads", ylabel = "Elapsed time (s)",
        xscale = log2, xticks = (THREAD_COUNTS, string.(THREAD_COUNTS)),
        title = "Wall-clock time (ideal: flat)")
    CM.hlines!(ax_wt, [weak_results[1]["elapsed_s"]], color = (:gray, 0.6),
        linestyle = :dash, label = "Ideal (constant)")
    CM.lines!(ax_wt, threads_v, [r["elapsed_s"] for r in weak_results],
        color = :darkorange, linewidth = 2.5, label = "Actual")
    CM.scatter!(ax_wt, threads_v, [r["elapsed_s"] for r in weak_results],
        color = :darkorange, markersize = 10)
    CM.axislegend(ax_wt, position = :lt)

    ax_wn = CM.Axis(fig_weak[1, 2],
        xlabel = "Threads", ylabel = "Normalised time  (Tₚ / T₁)",
        xscale = log2, xticks = (THREAD_COUNTS, string.(THREAD_COUNTS)),
        limits = (nothing, (0.0, 2.5)),
        title = "Normalised wall-clock (ideal: 1.0)")
    CM.hlines!(ax_wn, [1.0], color = (:gray, 0.6), linestyle = :dash)
    CM.lines!(ax_wn, threads_v, [r["normalised_time"] for r in weak_results],
        color = :purple, linewidth = 2.5)
    CM.scatter!(ax_wn, threads_v, [r["normalised_time"] for r in weak_results],
        color = :purple, markersize = 10)

    ax_wN = CM.Axis(fig_weak[2, 1:2],
        xlabel = "Threads", ylabel = "N (number of points)",
        xscale = log2, xticks = (THREAD_COUNTS, string.(THREAD_COUNTS)),
        title = "Problem size used at each thread count")
    CM.lines!(ax_wN, threads_v, N_weak_v, color = :teal, linewidth = 2)
    CM.scatter!(ax_wN, threads_v, N_weak_v, color = :teal, markersize = 10)

    CM.save(joinpath(RESULTS_DIR, "weak_scaling.png"), fig_weak, px_per_unit = 2)
    println("Saved → $(joinpath(RESULTS_DIR, "weak_scaling.png"))")

catch e
    @warn "CairoMakie not available — skipping figures." exception = e
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. GPU Scaling Skeleton
# ─────────────────────────────────────────────────────────────────────────────
println("\n── GPU benchmark (skeleton) ────────────────────────────────────────")
println("  Checking for GPU availability…")

gpu_available = false
gpu_results = Dict[]

try
    @eval using CUDA: CUDA
    if CUDA.functional()
        global gpu_available = true
        println("  ✓ CUDA functional: $(CUDA.name(CUDA.device()))")
    else
        println("  ✗ CUDA not functional — skipping GPU benchmark")
    end
catch
    println("  ✗ CUDA.jl not installed or not available — skipping GPU benchmark")
end

if gpu_available
    # TODO (H5): GPU strong sizing benchmark.
    #
    # Benchmark `gpu_calculate_structure_function` across N, compare to best
    # threaded CPU time.  Replace the code below with real benchmarks.
    #
    # using KernelAbstractions, CUDA
    #
    # N_gpu_sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
    # sft         = StructureFunctions.LongitudinalSecondOrderStructureFunction()
    # bin_edges   = collect(Float64, range(0.0, 1.5, length = 21))
    #
    # for N in N_gpu_sizes
    #     x_dev = CUDA.rand(Float64, 3, N)
    #     u_dev = CUDA.rand(Float64, 3, N)
    #
    #     # Warmup
    #     gpu_calculate_structure_function(CUDA.CUDABackend(), x_dev, u_dev, bin_edges, sft)
    #     CUDA.synchronize()
    #
    #     elapsed = @elapsed begin
    #         gpu_calculate_structure_function(CUDA.CUDABackend(), x_dev, u_dev, bin_edges, sft)
    #         CUDA.synchronize()
    #     end
    #     push!(gpu_results, Dict("N_points" => N, "elapsed_s" => elapsed))
    # end
    #
    # # Compare GPU vs. CPU best at each N and plot. Save to scaling_results_gpu.json
    # ...

    println("  [GPU skeleton present — fill in TODO above when on a CUDA machine]")
end

# ─────────────────────────────────────────────────────────────────────────────
println("\n── Done ────────────────────────────────────────────────────────────")
println("Results  : $RESULTS_DIR/scaling_results.json")
println("Figures  : $RESULTS_DIR/strong_scaling.png")
println("           $RESULTS_DIR/weak_scaling.png")
