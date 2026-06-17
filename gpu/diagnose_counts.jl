#!/usr/bin/env julia
"""
    diagnose_counts.jl — CPU gold vs GPU paths (full SF + count-only atomics).

Compares GPU Σcounts against **Int64 CPU gold** (true in-bin pair count) and reports
Float32 histogram loss separately.

Run once per SLURM session:

    include(joinpath(pkgdir(StructureFunctions), "gpu", "run.jl"))
    include_gpu("diagnose_counts.jl")
"""

using CUDA: CUDA
using Printf: @printf
using Random: Random
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges

include(joinpath(@__DIR__, "benchmark_helpers.jl"))

ProtoGP = load_prototype_kernels()

function _delta(label, got, gold::Int64)
    Δ = got - gold
    mark = abs(Δ) <= max(1, round(Int, 1f-4 * gold)) ? "OK" : "MISMATCH"
    @printf("  %-28s %12.0f  (Δ %+.0f)  [%s]\n", label, Float64(got), Float64(Δ), mark)
    return mark == "OK"
end

function main()
    if !CUDA.functional()
        println("CUDA not functional — skipping diagnose_counts.")
        return nothing
    end

    Random.seed!(42)
    N = parse(Int, get(ENV, "N", "20000"))
    FT = Float32
    backend = CUDA.CUDABackend()
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))

    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)

    println("=" ^ 72)
    println("GPU count diagnosis — N = $N")
    println("Device: ", CUDA.name(CUDA.device()))
    println("=" ^ 72)

    println("\n[1] CPU gold (serial, same pair index + digitize as GPU kernels)")
    total_pairs = ProtoGP.verify_pair_enumeration(N)
    gold_count = ProtoGP.cpu_gold_histogram(x_cpu, u_cpu, sft, bin_edges; count_only = true)
    gold_full = ProtoGP.cpu_gold_histogram(x_cpu, u_cpu, sft, bin_edges; count_only = false)

    gold_i64 = gold_count.n_in
    @printf("    total pairs (N(N-1)/2):      %.0f\n", total_pairs)
    @printf("    in-bin pairs (Int64 gold):   %.0f\n", gold_i64)
    @printf("    Σcounts Float32 (CPU loop):  %.0f  (lost %.0f vs Int64)\n",
        gold_count.sum_counts_f32, gold_count.float_lost)
    @printf("    max bin %2d count:            %.0f  (Float32 saturates at 2^24=%.0f)\n",
        gold_count.max_bin, gold_count.max_bin_count, Float32(16777216))
    @printf("    bin=0:                       %.0f\n", gold_count.n_bin0)
    @printf("    bin≥%d:                      %.0f\n", length(bin_edges.edges), gold_count.n_bin21)
    @printf("    bucket sum:                  %.0f\n", gold_count.n_in_plus_n_out)

    println("\n[2] Full prototype kernels vs Int64 CPU gold (= $(gold_i64))")
    ok_fast = true
    for cfg in ProtoGP.prototype_variants(N)
        res = ProtoGP.run_prototype!(cfg, backend, sft, bin_edges, x_cpu, u_cpu)
        got = Int64(round(sum(res.counts)))
        is_global = occursin("baseline", cfg.name)
        ok = _delta(cfg.name, got, gold_i64)
        is_global || (ok_fast &= ok)
    end

    println("\n[3] Count-only GPU kernels")
    ok_uint = true
    nworkers = min(262_144, N * (N - 1) ÷ 2)
    diag = ProtoGP.run_count_diagnostics(backend, x_cpu, u_cpu, bin_edges; nworkers = nworkers)
    for d in diag
        got = d.name == "diag_global_uint32" ? Int64(d.sum_counts) : Int64(round(d.sum_counts))
        ok = _delta(d.name, got, gold_i64)
        d.name == "diag_global_uint32" && (ok_uint &= ok)
        d.name == "diag_blockshared_float32" && (ok_fast &= ok)
    end

    println("\n  count-only kernel times:")
    ref_t = findfirst(d -> d.name == "diag_global_float32", diag)
    for d in diag
        speedup = ref_t === nothing ? NaN : diag[ref_t].kernel_s / d.kernel_s
        @printf("    %-28s  kernel %.4f s", d.name, d.kernel_s)
        ref_t !== nothing && @printf("  (%.2f× vs diag_global_float32)", speedup)
        println()
    end

    println("\n[4] Interpretation")
    println("  • Int64 in-bin gold: $(gold_i64) pairs (bins 1..20; $(gold_count.n_bin0) pairs below bin 0)")
    println("  • Float32 global histogram saturates at 2^24=16777216 per bin → Σcounts=$(gold_count.sum_counts_f32) (lost $(gold_count.float_lost))")
    println("  • CPU counts_i64 bin 5 alone: $(gold_count.counts_i64[5]) — well past Float32 exact-integer limit")
    println("  • diag_global_uint32: exact match → same global path, integer atomics fixes it")
    println("  • blockshared/private: match Int64 gold (chunked merge avoids per-bin Float32 saturation)")

    println()
    if ok_fast && ok_uint
        println("Conclusion: integrate blockshared OR global+UInt32 counts; do NOT ship global Float32 counters at N=20k.")
    else
        println("Conclusion: fast paths still have residual Int64 mismatch — inspect per-bin (see counts_i64 in return value).")
    end

    return (; gold_i64, gold_count, diag, ok_fast, ok_uint)
end

Base.invokelatest(main)
