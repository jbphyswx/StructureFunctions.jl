#!/usr/bin/env julia
"""
    gpu_full_benchmark.jl — comprehensive GPU prototype + production sweep.

Analogous to `src/BinEdges_benchmarking.jl` + `src/BinEdges_timings_and_theory.md`:
runs every prototype variant at multiple `N`, compares against production GPU ext and
optional threaded CPU, then writes JSON + markdown under `gpu/benchmark_results/`.

## How to run

Start Julia once per SLURM session. Either `cd` to repo root and `include("gpu/gpu_full_benchmark.jl")`,
or `include(joinpath("/path/to/StructureFunctions.jl", "gpu", "gpu_full_benchmark.jl"))` from any pwd.
See `gpu/README.md`.

Outputs:
  - `gpu/benchmark_results/gpu_full_<timestamp>.json`
  - `gpu/benchmark_results/gpu_full_latest.json`  (symlink/copy of latest)
  - `gpu/GPU_timings_and_theory.md`  (human-readable report; results section regenerated)
"""

using CUDA: CUDA
using Dates: Dates, now
using JSON: JSON
using Printf: Printf, @printf, @sprintf
using Random: Random
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LinearBinEdges

include(joinpath(@__DIR__, "benchmark_helpers.jl"))

ProtoGP = load_prototype_kernels()

const RESULTS_DIR = joinpath(@__DIR__, "benchmark_results")
const MD_OUT = joinpath(@__DIR__, "GPU_timings_and_theory.md")

function parse_n_list()
    if get(ENV, "QUICK", "0") == "1"
        return [parse(Int, get(ENV, "N", "4000"))]
    end
    return parse.(Int, split(get(ENV, "N_LIST", "4000,10000,20000"), ","))
end

function production_gpu_time(
    sft,
    x_gpu,
    u_gpu,
    bin_edges;
    n_warmup = 1,
    n_repeat = 3,
)
    for _ in 1:n_warmup
        SFC.gpu_calculate_structure_function(
            sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges;
            return_sums_and_counts = true,
        )
        CUDA.synchronize()
    end
    times = Float64[]
    for _ in 1:n_repeat
        t = @elapsed begin
            SFC.gpu_calculate_structure_function(
                sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges;
                return_sums_and_counts = true,
            )
            CUDA.synchronize()
        end
        push!(times, t)
    end
    return minimum(times)
end

function threaded_cpu_time(
    sft,
    x_cpu,
    u_cpu,
    bin_edges;
    n_warmup = 1,
    n_repeat = 3,
)
    backend = SFC.ThreadedBackend()
    x_tup = (x_cpu[1, :], x_cpu[2, :])
    u_tup = (u_cpu[1, :], u_cpu[2, :])
    for _ in 1:n_warmup
        SFC.calculate_structure_function(
            sft, x_tup, u_tup, bin_edges;
            backend = backend,
            verbose = false,
            show_progress = false,
        )
    end
    times = Float64[]
    for _ in 1:n_repeat
        t = @elapsed SFC.calculate_structure_function(
            sft, x_tup, u_tup, bin_edges;
            backend = backend,
            verbose = false,
            show_progress = false,
        )
        push!(times, t)
    end
    return minimum(times)
end

function benchmark_one_n(ProtoGP, N; n_repeat = 3, skip_cpu = false)
    FT = Float32
    backend = CUDA.CUDABackend()
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    n_pairs = N * (N - 1) ÷ 2

    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)

    gold = ProtoGP.cpu_gold_histogram(x_cpu, u_cpu, sft, bin_edges)
    gold_i64 = gold.n_in

    rows, _, _ = benchmark_all_variants(
        ProtoGP, backend, sft, bin_edges, x_cpu, u_cpu;
        n_repeat = n_repeat,
    )
    summary = summarize_n(rows, gold)

    prod_s = production_gpu_time(sft, x_gpu, u_gpu, bin_edges; n_repeat = n_repeat)
    cpu_s = skip_cpu ? nothing : threaded_cpu_time(sft, x_cpu, u_cpu, bin_edges; n_repeat = n_repeat)

    best_opt = nothing
    best_kernel = Inf
    for r in rows
        p = gold_parity(gold, r, r.name)
        if p.ok && r.kernel_s < best_kernel
            best_kernel = r.kernel_s
            best_opt = r
        end
    end

    return Dict{String, Any}(
        "N" => N,
        "n_pairs" => n_pairs,
        "n_bins" => length(bin_edges.edges) - 1,
        "gold_i64" => gold_i64,
        "variants" => summary.rows_json,
        "production_gpu_s" => prod_s,
        "threaded_cpu_s" => cpu_s,
        "sum_counts_timing_baseline_f32" => summary.timing_ref.sum_counts,
        "sum_counts_timing_baseline_i64" => summary.timing_ref.sum_counts_i64,
        "sum_counts_parity_baseline_i64" => summary.parity_ref.sum_counts_i64,
        "float32_undercount" => summary.float32_undercount,
        "u32_global_kernel_s" => summary.u32_global_kernel_s,
        "fastest_parity_ok" => best_opt === nothing ? nothing : Dict(
            "name" => best_opt.name,
            "kernel_s" => best_opt.kernel_s,
            "total_s" => best_opt.total_s,
            "speedup_vs_timing" => summary.timing_ref.total_s / best_opt.total_s,
        ),
    )
end

function print_n_table(result)
    N = result["N"]
    println()
    println("=" ^ 90)
    @printf("N = %d  (pairs = %.2fM)\n", N, result["n_pairs"] / 1e6)
    println("-" ^ 90)
    println("Variant                          | kernel (s) | total (s) | vs timing | cnt | Δcnt      | max|Δsum| | parity")
    println("-" ^ 105)
    for v in result["variants"]
        @printf(
            "%-32s | %8.4f | %8.4f | %8.2f× | %3s | %+10d | %9.4g | %s\n",
            v["name"], v["kernel_s"], v["total_s"], v["speedup_vs_timing"],
            v["cnt_dev"], v["Δcnt"], v["max_Δsum"], v["parity"],
        )
    end
    timing = row_by_name_from_json(result["variants"], TIMING_BASELINE)
    @printf("\nCPU gold Σcounts = %d  |  Float32 global Δcnt = %+d\n",
        result["gold_i64"], result["float32_undercount"])
    @printf("UInt32 global kernel = %.4f s  vs  Float32 global = %.4f s\n",
        result["u32_global_kernel_s"], timing["kernel_s"])
    @printf("Production gpu_calculate_structure_function: %.4f s\n", result["production_gpu_s"])
    if result["threaded_cpu_s"] !== nothing
        @printf("Threaded CPU (OhMyThreads): %.4f s\n", result["threaded_cpu_s"])
    end
    fo = result["fastest_parity_ok"]
    if fo !== nothing
        @printf("Fastest parity-ok: %s  kernel=%.4f s  (%.2f× vs timing total)\n",
            fo["name"], fo["kernel_s"], fo["speedup_vs_timing"])
    end
end

function row_by_name_from_json(rows, name)
    for v in rows
        v["name"] == name && return v
    end
    error("row $name not found")
end

function write_markdown_report(payload)
    meta = payload["meta"]
    results = payload["results"]
    ts = meta["timestamp"]

    lines = String[]
    push!(lines, "# GPU structure-function benchmarking — timings and theory")
    push!(lines, "")
    push!(lines, "This document mirrors the CPU `src/BinEdges_timings_and_theory.md` workflow:")
    push!(lines, "prototype kernel variants, correctness (parity vs block-shared histogram), and timing.")
    push!(lines, "")
    push!(lines, "## Auto-generated results")
    push!(lines, "")
    push!(lines, "_Regenerated by `gpu/gpu_full_benchmark.jl` on $(ts)._")
    push!(lines, "")
    push!(lines, "| Field | Value |")
    push!(lines, "|---|---|")
    push!(lines, "| Device | $(meta["device"]) |")
    push!(lines, "| Julia | $(meta["julia_version"]) |")
    push!(lines, "| Problem | 2D Float32, `L2SFType`, linear bins (21 edges) |")
    push!(lines, "| N values | $(join(meta["n_list"], ", ")) |")
    push!(lines, "| Timing repeats | $(meta["n_repeat"]) (min of each) |")
    push!(lines, "| Parity reference | `$(PARITY_BASELINE)` (blockshared + UInt32 counts) |")
    push!(lines, "| Timing reference | `$(TIMING_BASELINE)` (global Float32 counts — lossy at large N) |
| Count gold | Int64 CPU histogram (`cpu_gold_histogram`) |
| UInt32 variants | `baseline_linear_global_u32`, `blockshared_256k_w256_u32` |")
    push!(lines, "")

    push!(lines, "### Executive summary")
    push!(lines, "")
    push!(lines, "| N | pairs (M) | prod GPU (s) | threaded CPU (s) | best parity-ok | kernel (s) | vs timing |")
    push!(lines, "|---:|---:|---:|---:|---|---:|---:|")
    for r in results
        fo = r["fastest_parity_ok"]
        best_name = fo === nothing ? "—" : fo["name"]
        best_k = fo === nothing ? "—" : @sprintf("%.4f", fo["kernel_s"])
        best_sp = fo === nothing ? "—" : @sprintf("%.2f×", fo["speedup_vs_timing"])
        cpu_s = r["threaded_cpu_s"] === nothing ? "skipped" : @sprintf("%.4f", r["threaded_cpu_s"])
        push!(lines, @sprintf("| %d | %.2f | %.4f | %s | %s | %s | %s |",
            r["N"], r["n_pairs"] / 1e6, r["production_gpu_s"], cpu_s, best_name, best_k, best_sp))
    end
    push!(lines, "")

    push!(lines, "### Histogram totals (Int64 gold vs Float32 global)")
    push!(lines, "")
    push!(lines, "At large `N`, `baseline_linear_global` loses pair counts to Float32 saturation (2²⁴ per bin).")
    push!(lines, "`baseline_linear_global_u32` and blockshared paths match Int64 gold.")
    push!(lines, "")
    push!(lines, "| N | Int64 gold | Σcounts_i64 (f32 global) | under-count | u32 global kernel (s) |")
    push!(lines, "|---:|---:|---:|---:|---:|")
    for r in results
        push!(lines, @sprintf("| %d | %.0f | %.0f | %.0f | %.4f |",
            r["N"], r["gold_i64"], r["sum_counts_timing_baseline_i64"],
            r["float32_undercount"], r["u32_global_kernel_s"]))
    end
    push!(lines, "")

    for r in results
        push!(lines, "### N = $(r["N"]) ($(round(r["n_pairs"] / 1e6; digits=2))M pairs)")
        push!(lines, "")
        push!(lines, "| Variant | kernel (s) | total (s) | vs timing | cnt | Δcnt | max|Δsum| | parity |")
        push!(lines, "|---|---:|---:|---:|---:|---:|---:|---|")
        for v in r["variants"]
            push!(lines, @sprintf("| `%s` | %.4f | %.4f | %.2f× | %s | %+d | %.4g | %s |",
                v["name"], v["kernel_s"], v["total_s"], v["speedup_vs_timing"],
                v["cnt_dev"], v["Δcnt"], v["max_Δsum"], v["parity"]))
        end
        push!(lines, "")
        push!(lines, "- Production `gpu_calculate_structure_function`: **$(@sprintf("%.4f", r["production_gpu_s"])) s**")
        if r["threaded_cpu_s"] !== nothing
            push!(lines, "- Threaded CPU: **$(@sprintf("%.4f", r["threaded_cpu_s"])) s**")
        end
        fo = r["fastest_parity_ok"]
        if fo !== nothing
            push!(lines, "- Fastest parity-ok: **`$(fo["name"])`** — kernel $(@sprintf("%.4f", fo["kernel_s"])) s")
        end
        push!(lines, "")
    end

    push!(lines, "## Theory (static)")
    push!(lines, "")
    push!(lines, "See `gpu/README.md` for the canonical block-shared histogram design and launch config.")
    push!(lines, "Prototype implementations live in `gpu/GPUPrototypeKernels.jl`.")
    push!(lines, "")
    push!(lines, "### Variant families")
    push!(lines, "")
    push!(lines, "1. **`baseline_*`** — production-style global Float32 atomics (timing reference; lossy at large N).")
    push!(lines, "2. **`private_*`** — per-thread private histogram, global merge.")
    push!(lines, "3. **`blockshared_*`** — block-local histogram + global flush (`*_u32` = exact counts).")
    push!(lines, "4. **`baseline_linear_global_u32`** — global atomics + UInt32 counts (correct, slow).")
    push!(lines, "5. **`device_resident_*`** — reuse device buffers across calls.")
    push!(lines, "")
    push!(lines, "### How to re-run")
    push!(lines, "")
    push!(lines, "```bash")
    push!(lines, "SF_REPO = \"/path/to/StructureFunctions.jl\"")
    push!(lines, "include(joinpath(SF_REPO, \"gpu\", \"gpu_full_benchmark.jl\"))")
    push!(lines, "```")

    write(MD_OUT, join(lines, "\n") * "\n")
end

function main()
    if !CUDA.functional()
        println("CUDA not functional — skipping gpu_full_benchmark.")
        return nothing
    end

    mkpath(RESULTS_DIR)
    n_list = parse_n_list()
    n_repeat = parse(Int, get(ENV, "N_REPEAT", "3"))
    skip_cpu = get(ENV, "SKIP_CPU", "0") == "1"

    Random.seed!(42)

    println("=" ^ 70)
    println("StructureFunctions.jl — GPU full benchmark")
    println("  Device     : ", CUDA.name(CUDA.device()))
    println("  N values   : ", join(n_list, ", "))
    println("  Repeats    : ", n_repeat, " (min kernel/total)")
    println("  CPU compare: ", skip_cpu ? "skipped" : "ThreadedBackend ($(Threads.nthreads()) threads)")
    println("  Results dir: ", RESULTS_DIR)
    println("=" ^ 70)

    results = Dict{String, Any}[]
    for N in n_list
        println("\n>>> Benchmarking N = $N ...")
        r = benchmark_one_n(ProtoGP, N; n_repeat = n_repeat, skip_cpu = skip_cpu)
        push!(results, r)
        print_n_table(r)
    end

    payload = Dict{String, Any}(
        "meta" => Dict(
            "timestamp" => string(now()),
            "device" => CUDA.name(CUDA.device()),
            "julia_version" => string(VERSION),
            "n_list" => n_list,
            "n_repeat" => n_repeat,
            "skip_cpu" => skip_cpu,
            "parity_baseline" => PARITY_BASELINE,
            "timing_baseline" => TIMING_BASELINE,
        ),
        "results" => results,
    )

    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    json_path = joinpath(RESULTS_DIR, "gpu_full_$(stamp).json")
    latest_path = joinpath(RESULTS_DIR, "gpu_full_latest.json")
    open(json_path, "w") do io
        JSON.print(io, payload, 2)
    end
    cp(json_path, latest_path; force = true)
    write_markdown_report(payload)

    println()
    println("Wrote JSON : ", json_path)
    println("Wrote JSON : ", latest_path)
    println("Wrote report: ", MD_OUT)
    return payload
end

Base.invokelatest(main)
