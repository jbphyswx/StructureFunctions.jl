#!/usr/bin/env julia
"""
    benchmark_worker.jl  —  Single-thread-count timings for the scaling harness.

Called by `benchmark_scaling.jl` via `julia -t N benchmark_worker.jl`.
Outputs one JSON line to stdout:
    {"threads": N, "N_points": M, "elapsed_s": T}

This file is NOT part of the standard test suite and will NOT be run by `runtests.jl`.

Usage (internal, called automatically by benchmark_scaling.jl):
    julia --project=benchmark -t 8 benchmark/benchmark_worker.jl 5000
"""

using StructureFunctions: StructureFunctions, Calculations as SFC
using OhMyThreads: OhMyThreads # Trigger extension
using JSON: JSON
using Random: Random

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "benchmark", "scaling_config.jl"))

# Problem size (points in 3D); optional second arg: Float32 or Float64
N_points = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
FT = length(ARGS) >= 2 && ARGS[2] == "Float32" ? Float32 : Float64
N_threads = Threads.nthreads()

Random.seed!(SCALING_SEED)
x_tup, u_tup, _, _ = scaling_synthetic_data(N_points, FT)
bins = scaling_bins(FT)
sft = SCALING_SFT

# Warmup (avoid counting compile time)
SFC.calculate_structure_function(sft, x_tup, u_tup, bins; backend = SFC.ThreadedBackend(), verbose = false, show_progress = false)

# Timed run
t_start = time()
SFC.calculate_structure_function(sft, x_tup, u_tup, bins; backend = SFC.ThreadedBackend(), verbose = false, show_progress = false)
elapsed = time() - t_start

result = Dict(
    "threads" => N_threads,
    "N_points" => N_points,
    "dtype" => string(FT),
    "elapsed_s" => elapsed,
)
println(JSON.json(result))
