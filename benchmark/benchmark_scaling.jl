#!/usr/bin/env julia
"""
    benchmark_scaling.jl  —  CPU thread scaling of the threaded point kernel, strong and weak.

Strong scaling: fixed N, increasing threads; ideal speedup p. Weak scaling: N ∝ √p, so each thread has the same
O(N²/p) pair work; ideal time constant. Each thread count runs in its own process (`benchmark_worker.jl`). Run it
where every thread gets a physical core of its own; under Slurm, `--hint=nomultithread` with `-c MAX_THREADS`.
Writes `benchmark/benchmark_results/scaling_results.json`, `strong_scaling.png` and `weak_scaling.png`; the docs
show copies of the two figures.

      pkg> activate benchmark 
    julia> include("benchmark/benchmark_scaling.jl")
    julia> main(; MAX_THREADS=16 N_STRONG=20000 N_WEAK_BASE=6000)

GPU problem-size and batch figures: `gpu/collect_benchmark_assets.jl`, then
`docs/generate_assets/generate_gpu_figures.jl`.
"""

using JSON: JSON
using Printf: Printf
using Dates: Dates
using CairoMakie: CairoMakie as CM

const RESULTS_DIR = joinpath(@__DIR__, "benchmark_results")
mkpath(RESULTS_DIR)

const N_STRONG = parse(Int, get(ENV, "N_STRONG", "4000"))
# N = √p · N_WEAK_BASE keeps the pair work per thread, N²/p, at N_WEAK_BASE².
const N_WEAK_BASE = parse(Int, get(ENV, "N_WEAK_BASE", "1000"))
const MAX_THREADS = min(parse(Int, get(ENV, "MAX_THREADS", "32")), Sys.CPU_THREADS)
const THREAD_COUNTS = filter(t -> t <= MAX_THREADS, [1, 2, 4, 8, 16, 32])

println("="^70)
println("StructureFunctions.jl — CPU Thread Scaling Benchmark")
println("  Thread counts to test     : $(THREAD_COUNTS)")
println("  Strong scaling N          : $(N_STRONG) (fixed)")
println("  Weak scaling N_base/thread: $(N_WEAK_BASE) → N = round(√p × $N_WEAK_BASE)")
println("  Results directory         : $(RESULTS_DIR)")
println("="^70)

"""
    run_worker(n_threads, n_points) -> Dict

The JSON result of `benchmark_worker.jl` run in its own process with `-t n_threads`.
"""
function run_worker(n_threads::Int, n_points::Int)
    worker_script = joinpath(@__DIR__, "benchmark_worker.jl")
    julia_bin = joinpath(Sys.BINDIR, "julia")
    project_dir = joinpath(@__DIR__, "..")
    output = readchomp(`$julia_bin --project=$project_dir/benchmark -t $n_threads $worker_script $n_points`)
    lines = filter(!isempty, split(output, '\n'))
    return JSON.parse(last(lines))
end

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

println("\n── Weak scaling (N ∝ √p, N_base = $N_WEAK_BASE per thread) ─────────")
weak_results = Dict[]
for p in THREAD_COUNTS
    N_weak = round(Int, N_WEAK_BASE * sqrt(p))
    print("  threads = $(lpad(p, 2))  N = $(lpad(N_weak, 6))  …  ")
    flush(stdout)
    r = run_worker(p, N_weak)
    r["N_weak"] = N_weak
    push!(weak_results, r)
    Printf.@printf("%.3f s\n", r["elapsed_s"])
end
t1_weak = weak_results[1]["elapsed_s"]
for r in weak_results
    r["normalised_time"] = r["elapsed_s"] / t1_weak
end

json_path = joinpath(RESULTS_DIR, "scaling_results.json")
open(json_path, "w") do io
    JSON.print(io,
        Dict(
            "timestamp" => string(Dates.now()),
            "N_strong" => N_STRONG,
            "N_weak_base" => N_WEAK_BASE,
            "strong_scaling" => strong_results,
            "weak_scaling" => weak_results,
        ), 2)
end
println("\nSaved results → $json_path")

println("\n── Generating figures ──────────────────────────────────────────────")
let threads_v = Float64.(THREAD_COUNTS), ticks = (THREAD_COUNTS, string.(THREAD_COUNTS))
    fig_strong = CM.Figure(size = (1000, 700), fontsize = 14)
    CM.Label(fig_strong[0, 1:2], "Strong scaling: N = $N_STRONG points, 3-D, longitudinal S₂", fontsize = 16,
             font = :bold)

    ax_t1 = CM.Axis(fig_strong[1, 1], xlabel = "Threads", ylabel = "Elapsed time (s)", xscale = log2, xticks = ticks)
    CM.scatterlines!(ax_t1, threads_v, [r["elapsed_s"] for r in strong_results], color = :steelblue,
                     linewidth = 2.5, markersize = 10)

    ax_sp = CM.Axis(fig_strong[1, 2], xlabel = "Threads", ylabel = "Speedup  (T₁ / Tₚ)", xscale = log2,
                    yscale = log2, xticks = ticks, yticks = ticks)
    CM.lines!(ax_sp, threads_v, threads_v, color = (:gray, 0.7), linewidth = 1.5, linestyle = :dash,
              label = "Ideal (linear)")
    CM.scatterlines!(ax_sp, threads_v, [r["speedup"] for r in strong_results], color = :crimson, linewidth = 2.5,
                     markersize = 10, label = "Measured")
    CM.axislegend(ax_sp, position = :lt)

    ax_eff = CM.Axis(fig_strong[2, 1:2], xlabel = "Threads", ylabel = "Parallel efficiency  (speedup / p)",
                     xscale = log2, xticks = ticks, limits = (nothing, (0.0, 1.15)))
    CM.hlines!(ax_eff, [1.0], color = (:gray, 0.6), linestyle = :dash)
    CM.scatterlines!(ax_eff, threads_v, [r["efficiency"] for r in strong_results], color = :seagreen,
                     linewidth = 2.5, markersize = 10)

    CM.save(joinpath(RESULTS_DIR, "strong_scaling.png"), fig_strong, px_per_unit = 2)
    println("Saved → $(joinpath(RESULTS_DIR, "strong_scaling.png"))")

    fig_weak = CM.Figure(size = (1000, 700), fontsize = 14)
    CM.Label(fig_weak[0, 1:2], "Weak scaling: N = √p · $N_WEAK_BASE points, 3-D, longitudinal S₂", fontsize = 16,
             font = :bold)

    ax_wt = CM.Axis(fig_weak[1, 1], xlabel = "Threads", ylabel = "Elapsed time (s)", xscale = log2, xticks = ticks,
                    title = "Wall-clock time (ideal: flat)")
    CM.hlines!(ax_wt, [weak_results[1]["elapsed_s"]], color = (:gray, 0.6), linestyle = :dash,
               label = "Ideal (constant)")
    CM.scatterlines!(ax_wt, threads_v, [r["elapsed_s"] for r in weak_results], color = :darkorange,
                     linewidth = 2.5, markersize = 10, label = "Measured")
    CM.axislegend(ax_wt, position = :lt)

    ax_wn = CM.Axis(fig_weak[1, 2], xlabel = "Threads", ylabel = "Normalised time  (Tₚ / T₁)", xscale = log2,
                    xticks = ticks, limits = (nothing, (0.0, 2.5)), title = "Normalised wall-clock (ideal: 1.0)")
    CM.hlines!(ax_wn, [1.0], color = (:gray, 0.6), linestyle = :dash)
    CM.scatterlines!(ax_wn, threads_v, [r["normalised_time"] for r in weak_results], color = :purple,
                     linewidth = 2.5, markersize = 10)

    ax_wN = CM.Axis(fig_weak[2, 1:2], xlabel = "Threads", ylabel = "N (number of points)", xscale = log2,
                    xticks = ticks, title = "Problem size at each thread count")
    CM.scatterlines!(ax_wN, threads_v, Float64.([r["N_weak"] for r in weak_results]), color = :teal,
                     linewidth = 2, markersize = 10)

    CM.save(joinpath(RESULTS_DIR, "weak_scaling.png"), fig_weak, px_per_unit = 2)
    println("Saved → $(joinpath(RESULTS_DIR, "weak_scaling.png"))")
end
