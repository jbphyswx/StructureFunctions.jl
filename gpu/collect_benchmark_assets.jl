# Collect GPU benchmark JSON for the docs figures: one GPU against the serial CPU over the problem size, and over the
# slice count of a batch at fixed size. Every call is a public entry; timings come from `benchmark_scaling_helpers.jl`.
#
# Run on a GPU allocation:
#   julia --project=gpu gpu/collect_benchmark_assets.jl
#
# Optional env: N_LIST=4000,8000,16000,20000  N_SLICE=1000  T_LIST=1,2,4,8,16,32,64
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
    CUDA.functional() || return Dict("cuda_functional" => false)
    return Dict(
        "cuda_functional" => true,
        "device" => string(CUDA.name(CUDA.device())),
        "backend" => string(typeof(CUDA.CUDABackend())),
        "gpu_count" => 1,
    )
end

"""
    collect_problem_size_scaling!(backend) -> Vector{Dict}

One GPU with a workspace against the serial CPU, over `N`.
"""
function collect_problem_size_scaling!(backend)
    rows = Dict[]
    for N in SCALING_N_LIST
        for FT in (Float32, Float64)
            x_arr, u_arr = scaling_synthetic_data(N, FT)
            bins = scaling_bins(FT)
            x_dev, u_dev = stage_device_arrays(backend, x_arr, u_arr, FT)
            ws = SFC.GPUSFWorkspace(backend, bins)
            cpu_t = bench_cpu_serial_sf(x_arr, u_arr, bins, SCALING_SFT)
            gpu_t = bench_gpu_sf_with_workspace(backend, x_dev, u_dev, bins, SCALING_SFT, ws)
            SFC.release!(ws)
            push!(rows, Dict("N" => N, "dtype" => string(FT), "cpu_elapsed_s" => cpu_t, "gpu_elapsed_s" => gpu_t))
            println("  N=$N $FT  cpu=$(round(cpu_t, digits=4))s  gpu=$(round(gpu_t, digits=5))s  cpu/gpu=$(round(cpu_t/gpu_t, digits=1))×")
        end
    end
    return rows
end

"""Fixed `N_SLICE`, over the slice count `T`: the serial CPU batch, one GPU call per slice, and the GPU batch."""
function collect_slice_batch_scaling!(backend)
    rows = Dict[]
    N = SCALING_N_SLICE
    for FT in (Float32, Float64)
        bins = scaling_bins(FT)
        NB = length(bins) - 1
        ws = SFC.GPUSFWorkspace(backend, bins)
        for T in SCALING_T_LIST
            Random.seed!(SCALING_SEED + T)
            x_host = rand(FT, 3, N, T)
            u_host = rand(FT, 3, N, T)
            x_dev, u_dev = stage_device_batch(backend, x_host, u_host, FT)

            sums_cpu = zeros(eltype(bins), NB, T)
            counts_cpu = zeros(UInt32, NB, T)
            sums_naive = zeros(eltype(bins), NB, T)
            counts_naive = zeros(UInt32, NB, T)
            sums_slice = KA.zeros(backend, eltype(bins), NB, T)
            counts_slice = KA.zeros(backend, UInt32, NB, T)

            cpu_t = bench_cpu_serial_batch!(x_host, u_host, bins, SCALING_SFT, sums_cpu, counts_cpu)
            gpu_naive_t = bench_naive_slice_loop!(backend, x_host, u_host, bins, SCALING_SFT, sums_naive, counts_naive;
                                                  T = T)
            gpu_slice_t = bench_slice_driver!(backend, x_dev, u_dev, bins, SCALING_SFT, sums_slice, counts_slice, ws)

            push!(rows, Dict("N" => N, "T" => T, "dtype" => string(FT), "cpu_batch_elapsed_s" => cpu_t,
                             "gpu_naive_elapsed_s" => gpu_naive_t, "gpu_batch_elapsed_s" => gpu_slice_t))
            println("  slice N=$N T=$T $FT  cpu batch=$(round(cpu_t, digits=4))s  gpu per-slice=$(round(gpu_naive_t, digits=4))s  gpu batch=$(round(gpu_slice_t, digits=4))s")
        end
        SFC.release!(ws)
    end
    return rows
end

function main()
    CUDA.functional() || error("CUDA not functional — run collect_benchmark_assets.jl on a GPU allocation")
    backend = CUDA.CUDABackend()
    mkpath(RESULTS_DIR)

    println("Collecting GPU benchmark assets")
    println("  device: ", CUDA.name(CUDA.device()), " (1 GPU)")
    println("  CPU reference: SerialBackend")
    println("  problem-size N_LIST: ", SCALING_N_LIST)
    println("  slice batch: N_SLICE=", SCALING_N_SLICE, "  T_LIST=", SCALING_T_LIST)

    println("\n--- Problem-size scaling (1 GPU vs serial CPU, vary N) ---")
    problem_size = collect_problem_size_scaling!(backend)

    println("\n--- Slice-batch scaling (fixed N_SLICE, vary T) ---")
    slice_batch = collect_slice_batch_scaling!(backend)

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
            "gpu_timing" => "one untimed call, 0.5 s of calls, then the fastest of 5 synchronized calls",
            "cpu_timing" => "one untimed call, then the fastest of 7",
            "bins" => "range(0.0, 1.5, length=21)",
            "sf_type" => "LongitudinalSecondOrderStructureFunctionType",
            "seed" => SCALING_SEED,
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
