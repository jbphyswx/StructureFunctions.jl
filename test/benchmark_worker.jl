#!/usr/bin/env julia
"""
    benchmark_worker.jl  —  Single-thread-count timings for the scaling harness.

Called by `benchmark_scaling.jl` via `julia -t N benchmark_worker.jl`.
Outputs one JSON line to stdout:
    {"threads": N, "N_points": M, "elapsed_s": T}

This file is NOT part of the standard test suite and will NOT be run by `runtests.jl`.

Usage (internal, called automatically by benchmark_scaling.jl):
    julia --project=test -t 8 test/benchmark_worker.jl 5000
"""

using StructureFunctions: StructureFunctions, Calculations as SFC
using JSON: JSON

# Problem size (points in 3D)
N_points = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
N_threads = Threads.nthreads()

# Reproducible random data
using Random: Random
Random.seed!(42);
FT = Float64

x = (rand(FT, N_points), rand(FT, N_points), rand(FT, N_points))  # 3D positions
u = (rand(FT, N_points), rand(FT, N_points), rand(FT, N_points))  # velocity

sft = LongitudinalSecondOrderStructureFunction
n_bins = 20

# Warmup (avoid counting compile time)
SFC.calculate_structure_function(sft, x, u, n_bins; verbose = false, show_progress = false)

# Timed run
t_start = time()
SFC.calculate_structure_function(sft, x, u, n_bins; verbose = false, show_progress = false)
elapsed = time() - t_start

result = Dict("threads" => N_threads, "N_points" => N_points, "elapsed_s" => elapsed)
println(JSON.json(result))
