#!/usr/bin/env julia
"""
    benchmark_prototypes.jl

Compare GPU prototype kernels vs serial CPU gold (exact per-bin counts + SF sums).

Run once per SLURM session (do not restart `julia` between includes):

    include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))
    include_gpu("benchmark_prototypes.jl")
"""

using CUDA: CUDA
using Printf: @printf
using StructureFunctions: Calculations as SFC, StructureFunctionTypes as SFT, LinearBinEdges
using Random: Random

include(joinpath(@__DIR__, "benchmark_helpers.jl"))

"""Load prototype kernels into a fresh module (Julia skips re-evaluating `module` files on re-include)."""
function _load_gpup()
    m = Module()
    Base.include(m, joinpath(@__DIR__, "GPUPrototypeKernels.jl"))
    return m
end

ProtoGP = _load_gpup()

const _TIMING_BASELINE = TIMING_BASELINE
const _PARITY_BASELINE = PARITY_BASELINE

function _production_time(
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

function main()
    if !CUDA.functional()
        println("CUDA not functional — skipping prototype benchmark.")
        return
    end

    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "20000"))
    FT = Float32
    backend = CUDA.CUDABackend()
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))

    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)

    gold = ProtoGP.cpu_gold_histogram(x_cpu, u_cpu, sft, bin_edges)
    gold_f64 = ProtoGP.cpu_f64_serial_sums(x_cpu, u_cpu, sft, bin_edges)
    f64_check = ProtoGP.validate_f64_references(
        x_cpu, u_cpu, sft, bin_edges; nworkers = min(262_144, N * (N - 1) ÷ 2),
    )
    if !f64_check.reference_ok
        error("Float64 reference failed validation (serial vs private_all_f64 / BigFloat); aborting benchmark")
    end
    sum_ref = gold_f64.sums_f32

    println("=== GPU prototype benchmark ===")
    println("Device: ", CUDA.name(CUDA.device()))
    n_pairs = N * (N - 1) ÷ 2
    println("N = $N  (pairs = $(round(n_pairs / 1e6; digits=2))M, 2D Float32, L2SFType, linear bins)")
    @printf("CPU gold: Σcounts = %d  (Float32 serial loop Σ = %.0f, lost %.0f)\n",
        gold.n_in, gold.sum_counts_f32, gold.float_lost)
    @printf(
        "Sum parity ref: validated f64 serial (max|serial-private_all_f64|=%.4g, bf Δ bin5=%.4g)\n",
        f64_check.max_serial_private, f64_check.bf_diff,
    )
    println("Parity: exact counts vs gold; sums vs Float64 serial (rtol=1e-4, atol=500)")
    println()

    variants = ProtoGP.prototype_variants(N)
    ref_cfg = reference_config(variants)
    bufs = ProtoGP.prepare_device_buffers(backend, x_cpu, u_cpu, length(bin_edges.edges))

    for _ in 1:2
        ProtoGP.run_prototype!(ref_cfg, backend, sft, bin_edges, x_cpu, u_cpu)
    end
    ref = ProtoGP.run_prototype!(ref_cfg, backend, sft, bin_edges, x_cpu, u_cpu)

    raw = Tuple{String, Any}[]
    for cfg in variants
        ProtoGP.run_prototype!(
            cfg, backend, sft, bin_edges, x_cpu, u_cpu;
            bufs = cfg.device_resident ? bufs : nothing,
        )
        res = ProtoGP.run_prototype!(
            cfg, backend, sft, bin_edges, x_cpu, u_cpu;
            bufs = cfg.device_resident ? bufs : nothing,
        )
        push!(raw, (cfg.name, res))
    end

    println("Variant                          | kernel (s) | total (s) | vs base | cnt | Δcnt      | max|Δsum| | parity")
    println("-" ^ 120)

    results = Vector{Tuple{String, Any, Bool, NamedTuple}}()

    for (name, res) in raw
        total = res.kernel_s + res.staging_s
        base_total = ref.kernel_s + ref.staging_s
        speedup = base_total / total
        p = gold_parity(gold, res, name; sum_ref = sum_ref)

        @printf(
            "%-32s | %8.4f | %8.4f | %6.2f× | %3s | %+10d | %9.4g | %s\n",
            name, res.kernel_s, total, speedup, p.cnt_tag, p.Δcnt, p.max_Δsum, p.label,
        )
        push!(results, (name, res, p.ok, p))
    end

    println()
    passed = filter(r -> r[3], results)
    failed = filter(r -> !r[3] && r[1] != _TIMING_BASELINE, results)
    @printf("Strict gold parity: %d / %d variants pass (exact counts + sums vs Float64 serial).\n",
        length(passed), length(results))

    u32_idx = findfirst(v -> v.name == "baseline_linear_global_u32", variants)
    f32_global_idx = findfirst(v -> v.name == "baseline_linear_global", variants)
    if u32_idx !== nothing && f32_global_idx !== nothing
        u32_res = raw[u32_idx][2]
        f32_global = raw[f32_global_idx][2]
        @printf("\nUInt32 global vs Float32 global kernel: %.4f s vs %.4f s (%.2f×)\n",
            u32_res.kernel_s, f32_global.kernel_s, f32_global.kernel_s / u32_res.kernel_s)
    end

    bs_u32_idx = findfirst(v -> v.name == "blockshared_256k_w256_u32", variants)
    bs_idx = findfirst(v -> v.name == "blockshared_256k_w256", variants)
    if bs_u32_idx !== nothing && bs_idx !== nothing
        bs_u32 = raw[bs_u32_idx][2]
        bs_f32 = raw[bs_idx][2]
        @printf("UInt32 blockshared vs Float32 blockshared kernel: %.4f s vs %.4f s (%.2f×)\n",
            bs_u32.kernel_s, bs_f32.kernel_s, bs_f32.kernel_s / bs_u32.kernel_s)
    end

    if bs_u32_idx !== nothing && u32_idx !== nothing
        bs_u32 = raw[bs_u32_idx][2]
        global_u32 = raw[u32_idx][2]
        @printf("\nSum cross-check (max per-bin |Δsum|):\n")
        @printf("  vs f32 serial gold:\n")
        @printf("    global_u32:      %.6g\n", maximum(abs.(global_u32.sums .- gold.sums)))
        @printf("    blockshared_u32: %.6g\n", maximum(abs.(bs_u32.sums .- gold.sums)))
        @printf("  vs Float64 serial ref:\n")
        @printf("    global_u32:      %.6g\n", maximum(abs.(global_u32.sums .- sum_ref)))
        @printf("    blockshared_u32: %.6g\n", maximum(abs.(bs_u32.sums .- sum_ref)))
        cross = maximum(abs.(global_u32.sums .- bs_u32.sums))
        @printf("  global_u32 vs blockshared_u32: %.6g\n", cross)
        worst = argmax(abs.(bs_u32.sums .- sum_ref))
        @printf("  worst bin (blockshared vs f64 ref): %d  ref=%.6g  got=%.6g\n",
            worst, sum_ref[worst], bs_u32.sums[worst])
    end

    if !isempty(failed)
        println("\nFailed vs CPU gold (counts and/or sums):")
        for (name, res, _, p) in failed
            @printf("  %-32s  cnt_dev=%s  Δcnt=%+d  max_bin=%d  max|Δsum|=%.6g  sums_ok=%s\n",
                name, p.cnt_tag, p.Δcnt, p.max_Δcnt_bin, p.max_Δsum, p.sums_ok)
        end
    end

    println()
    println("Production ext (host staging each call):")
    try
        t_prod = _production_time(sft, x_gpu, u_gpu, bin_edges)
        base_total = ref.kernel_s + ref.staging_s
        @printf("  gpu_calculate_structure_function: %.4f s  (%.2f× vs prototype baseline)\n",
            t_prod, base_total / t_prod)
    catch err
        @printf("  skipped (%s)\n", sprint(showerror, err))
    end

    gold_ok = filter(r -> r[3], results)
    if !isempty(gold_ok)
        _, best_res, _, _ = findmin(gold_ok, by = r -> r[2].kernel_s)
        @printf("\nFastest gold-ok: %s  kernel=%.4f s  (%.2f× vs Float32 global baseline)\n",
            best_res[1], best_res[2].kernel_s, ref.kernel_s / best_res[2].kernel_s)
    end

    return nothing
end

Base.invokelatest(main)
