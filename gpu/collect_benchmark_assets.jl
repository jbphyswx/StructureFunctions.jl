# Collect GPU benchmark JSON for docs/README figures.
#
# Run on a GPU allocation (SLURM):
#   julia --project=gpu gpu/collect_benchmark_assets.jl
#
# What this measures (honest names):
#   • problem_size_scaling — 1 GPU + serial CPU, sweep N
#   • slice_batch_scaling  — fixed N_SLICE, sweep T (time-series batch API demo)
#
# True GPU strong/weak scaling (vary GPU count 1→8) is NOT implemented; see
# gpu/collect_multi_gpu_scaling.jl (stub).
#
# Optional env:
#   N_LIST=4000,8000,16000,20000
#   N_SLICE=1000
#   T_LIST=1,2,4,8,16,32,64
#   GPU timing: warmup=2  REPEAT=3 (median). CPU always SerialBackend (1 worker).
#
# Writes: gpu/benchmark_results/assets_latest.json
# Figures: julia --project=docs/generate_assets docs/generate_assets/generate_gpu_figures.jl

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC
using JSON: JSON
using Dates: Dates
using Random: Random

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "benchmark", "scaling_config.jl"))
include(joinpath(@__DIR__, "benchmark_scaling_helpers.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "benchmark_results")
const OUTPUT_JSON = joinpath(RESULTS_DIR, "assets_latest.json")

function _device_info()
    if CUDA.functional()
        return Dict(
            "cuda_functional" => true,
            "device" => string(CUDA.name(CUDA.device())),
            "backend" => string(typeof(CUDA.CUDABackend())),
            "gpu_count" => 1,
        )
    end
    return Dict("cuda_functional" => false)
end

function _warmup_gpu_session!(backend, warmup::Int)
    N = parse(Int, get(ENV, "WARMUP_N", string(minimum(SCALING_N_LIST))))
    println("  GPU session warmup (N=$N, both dtypes, compile before timed sweep) …")
    for FT in (Float32, Float64)
        x_arr, u_arr = scaling_synthetic_data(N, FT)
        bins = scaling_bins(FT)
        x_dev, u_dev = stage_device_arrays(backend, x_arr, u_arr, FT)
        ws = SFC.GPUSFWorkspace(backend, bins)
        for _ in 1:warmup
            SFC.gpu_calculate_structure_function(
                SCALING_SFT, backend, x_dev, u_dev, bins;
                workspace = ws,
            )
        end
        gpu_sync!(backend)
        SFC.release!(ws)
    end
    return nothing
end

"""
    collect_problem_size_scaling!(backend, warmup, repeat) -> Vector{Dict}

Problem-size scaling on **one GPU**: sweep `N`, compare **serial** CPU vs GPU+workspace.
CPU threading is documented separately in `benchmark/benchmark_scaling.jl`.
"""
function collect_problem_size_scaling!(backend, warmup::Int, repeat::Int)
    rows = Dict[]
    for N in SCALING_N_LIST
        for FT in (Float32, Float64)
            x_arr, u_arr = scaling_synthetic_data(N, FT)
            bins = scaling_bins(FT)
            x_dev, u_dev = stage_device_arrays(backend, x_arr, u_arr, FT)
            ws = SFC.GPUSFWorkspace(backend, bins)
            cpu_t = bench_cpu_serial_sf(x_arr, u_arr, bins, SCALING_SFT; warmup = warmup)
            gpu_t = bench_gpu_sf_with_workspace(
                backend, x_dev, u_dev, bins, SCALING_SFT, ws;
                warmup = warmup, repeat = repeat,
            )
            SFC.release!(ws)
            push!(rows, Dict(
                "N" => N,
                "dtype" => string(FT),
                "cpu_elapsed_s" => cpu_t,
                "gpu_elapsed_s" => gpu_t,
                "speedup_cpu_over_gpu" => cpu_t / gpu_t,
                "ms_per_call_gpu" => gpu_t * 1000,
            ))
            println("  N=$N $FT  cpu=$(round(cpu_t, digits=4))s  gpu=$(round(gpu_t, digits=4))s  cpu/gpu=$(round(cpu_t/gpu_t, digits=2))×")
        end
    end
    return rows
end

"""Slice-batch scaling: fixed `N_SLICE`, sweep T; compare CPU loop vs GPU paths."""
function collect_slice_batch_scaling!(backend, warmup::Int)
    rows = Dict[]
    N = SCALING_N_SLICE
    for FT in (Float32, Float64)
        bins = scaling_bins(FT)
        NB = length(bins) - 1
        ws = SFC.GPUSFWorkspace(backend, bins)
        for T in SCALING_T_LIST
            Random.seed!(SCALING_SEED + T)
            x_batch = Array{FT}(undef, 3, N, T)
            u_batch = Array{FT}(undef, 3, N, T)
            for t in 1:T
                for d in 1:3
                    x_batch[d, :, t] = rand(FT, N)
                    u_batch[d, :, t] = rand(FT, N)
                end
            end
            x_host = copy(x_batch)
            u_host = copy(u_batch)
            x_dev, u_dev = stage_device_batch(backend, x_host, u_host, FT)

            sums_cpu = zeros(Float64, NB, T)
            counts_cpu = zeros(UInt32, NB, T)
            sums_naive = zeros(eltype(bins), NB, T)
            counts_naive = zeros(UInt32, NB, T)
            sums_slice = zeros(eltype(bins), NB, T)
            counts_slice = zeros(UInt32, NB, T)

            cpu_t = bench_cpu_serial_slice_loop!(
                x_host, u_host, bins, SCALING_SFT, sums_cpu, counts_cpu; T = T, warmup = warmup,
            )
            gpu_naive_t = bench_naive_slice_loop!(
                backend, x_host, u_host, bins, SCALING_SFT, sums_naive, counts_naive; T = T, warmup = warmup,
            )
            gpu_slice_t = bench_slice_driver!(
                backend, x_dev, u_dev, bins, SCALING_SFT, sums_slice, counts_slice, ws; warmup = warmup,
            )

            push!(rows, Dict(
                "N" => N,
                "T" => T,
                "dtype" => string(FT),
                "cpu_loop_elapsed_s" => cpu_t,
                "gpu_naive_elapsed_s" => gpu_naive_t,
                "gpu_slice_elapsed_s" => gpu_slice_t,
                "speedup_slice_vs_cpu" => cpu_t / gpu_slice_t,
                "ms_per_slice_gpu_slice" => gpu_slice_t * 1000 / T,
            ))
            println("  slice N=$N T=$T $FT  cpu=$(round(cpu_t, digits=3))s  slice=$(round(gpu_slice_t, digits=3))s")
        end
        SFC.release!(ws)
    end
    return rows
end

function main()
    if !CUDA.functional()
        error("CUDA not functional — run collect_benchmark_assets.jl on a GPU allocation")
    end
    backend = CUDA.CUDABackend()
    warmup = parse(Int, get(ENV, "WARMUP", "2"))
    repeat = parse(Int, get(ENV, "REPEAT", "3"))
    mkpath(RESULTS_DIR)

    println("Collecting GPU benchmark assets")
    println("  device: ", CUDA.name(CUDA.device()), " (1 GPU)")
    println("  CPU reference: SerialBackend (1 worker — thread scaling: benchmark_scaling.jl)")
    println("  problem-size N_LIST: ", SCALING_N_LIST)
    println("  slice batch: N_SLICE=", SCALING_N_SLICE, "  T_LIST=", SCALING_T_LIST)
    println("  GPU timing: warmup=$warmup  repeat=$repeat (median)")

    _warmup_gpu_session!(backend, warmup)

    println("\n--- Problem-size scaling (1 GPU vs serial CPU, vary N) ---")
    problem_size = collect_problem_size_scaling!(backend, warmup, repeat)

    println("\n--- Slice-batch scaling (fixed N_SLICE, vary T) ---")
    slice_batch = collect_slice_batch_scaling!(backend, warmup)

    payload = Dict(
        "generated_at" => string(Dates.now()),
        "benchmark_kind" => "problem_size_and_slice_batch",
        "device" => _device_info(),
        "config" => Dict(
            "N_anchor" => SCALING_N_ANCHOR,
            "N_slice" => SCALING_N_SLICE,
            "N_list" => SCALING_N_LIST,
            "T_list" => SCALING_T_LIST,
            "cpu_threads" => 1,
            "cpu_backend" => "SerialBackend",
            "gpu_count" => 1,
            "gpu_warmup" => warmup,
            "gpu_repeat" => repeat,
            "bins" => "range(0.0, 1.5, length=21)",
            "sf_type" => "LongitudinalSecondOrderStructureFunctionType",
            "seed" => SCALING_SEED,
            "note" => "Not HPC strong/weak GPU scaling; see gpu/collect_multi_gpu_scaling.jl for future multi-GPU plan.",
        ),
        "problem_size_scaling" => problem_size,
        "slice_batch_scaling" => slice_batch,
    )

    open(OUTPUT_JSON, "w") do io
        JSON.print(io, payload, 2)
    end
    println("\nWrote ", OUTPUT_JSON)
    return nothing
end

main()
