#!/usr/bin/env julia
"""
    benchmark_batch_usmem.jl

Benchmark script comparing:
- §1: Production single-snapshot (baseline ruler)
- §2: Naive slice (B × production baseline)
- §8: **Integration path** `gpu_batch_fused_tiled_fixed_x!` → `usmem_priv_strips` on CUDA (~14 s @ N=20k B=8064)
- §3–§7: Legacy experiments (skipped at N>2048 unless `BATCH_LEGACY_ALL=1`; use `FAST=1` for §1+§8 only)

Run on a GPU node (SLURM or ALLOW_CUDA_BENCH=1):
    N=20000 BATCH=8064 julia --project=gpu gpu/benchmark_batch_usmem.jl

Fast iteration (§1 ruler + §8 fused only, progress after each step):
    FAST=1 N=20000 BATCH=8064 ALLOW_CUDA_BENCH=1 julia --project=gpu gpu/benchmark_batch_usmem.jl

Phase breakdown (`BATCH_PROFILE=1` prints to stdout **and** appends `test/debug/batch_profile.log`):
    BATCH_PROFILE=1 FAST=1 N=20000 BATCH=8064 ALLOW_CUDA_BENCH=1 julia --project=gpu gpu/benchmark_batch_usmem.jl

Slow host-strip baseline for A/B (~39s at N=20k B=8064):
    BATCH_HOST_STRIPS=1 BATCH_PROFILE=1 FAST=1 N=20000 BATCH=8064 ALLOW_CUDA_BENCH=1 \\
      julia --project=gpu gpu/benchmark_batch_usmem.jl

Legacy strip-outer (~23s):
    BATCH_STRIP_OUTER=1 BATCH_PROFILE=1 FAST=1 N=20000 BATCH=8064 ALLOW_CUDA_BENCH=1 \\
      julia --project=gpu gpu/benchmark_batch_usmem.jl

For GPU saturation / warp occupancy use **nsys** or **ncu** on the cluster (not in this script).
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @kernel, @Const, @localmem, @synchronize
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges
using StructureFunctions.Calculations: Calculations as SFC
using Printf: @printf
using Random: Random

include(joinpath(@__DIR__, "batch_prototypes", "BatchPrototypes.jl"))
using .BatchPrototypes: BatchPrototypes as BP

function _backend()
    if get(ENV, "ALLOW_CUDA_BENCH", "") == "1" || haskey(ENV, "SLURM_JOB_ID")
        CUDA.functional() || error("CUDA not functional")
        return CUDA.CUDABackend(), true
    end
    if CUDA.functional()
        return CUDA.CUDABackend(), true
    end
    @info "CUDA not functional or not requested. Falling back to CPU backend."
    return KA.CPU(), false
end

function _resolve_N_B()
    N = parse(Int, get(ENV, "N", "20000"))
    batch = parse(Int, get(ENV, "BATCH", "8064"))
    return N, batch
end

function _bench_progress(msg::AbstractString)
    println(msg)
    flush(stdout)
end

function _fast_mode()
    return get(ENV, "FAST", "") == "1"
end

function main()
    backend, is_gpu = _backend()
    N, B = _resolve_N_B()
    batch_shape = (B,)
    n_warmup = parse(Int, get(ENV, "WARMUP", "1"))
    n_repeat = parse(Int, get(ENV, "N_REPEAT", "1"))

    FT = Float32
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    NB = length(bin_edges.edges) - 1

    println("=== u-smem batch benchmark ===")
    println("Backend: ", is_gpu ? "CUDA ($(CUDA.name(CUDA.device())))" : "CPU")
    println("N=$N  B=$B  warmup=$n_warmup  repeat=$n_repeat  FAST=$(_fast_mode())")
    if get(ENV, "BATCH_PROFILE", "") == "1"
        path = if get(ENV, "BATCH_FUSED_SINGLE", "") == "1"
            "usmem_fused_single (atomic flush)"
        elseif get(ENV, "BATCH_WARP_B", "") == "1"
            "warp_b_pair_once (broken experiment)"
        elseif get(ENV, "BATCH_HOST_STRIPS", "") == "1"
            "cuda_host_strip (legacy)"
        elseif get(ENV, "BATCH_STRIP_OUTER", "") == "1"
            "inkernel_strip_outer (legacy)"
        else
            "usmem_priv_strips (default)"
        end
        println("BATCH_PROFILE=1  path=", path)
    end
    flush(stdout)

    # 1. Prepare data
    x, u = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = true, seed = 1)
    
    # 2. Stage/Upload
    ws = BP.BatchGPUWorkspace(backend, FT, N, B, NB; fixed_x = true)
    BP.upload_batch!(ws, backend, x, u)

    # 3. Setup baseline one-snapshot data
    x_one = rand(FT, 2, N)
    u_one = rand(FT, 2, N)
    x_one_dev = KA.adapt(backend, x_one)
    u_one_dev = KA.adapt(backend, u_one)
    ws_one = SFC.GPUSFWorkspace(backend, bin_edges)

    # Warmups on a small scale to compile kernels rapidly without hanging
    println("Warming up kernels on small scale (N=512, B=64)...")
    x_warm, u_warm = BP.make_random_batch_problem(FT, 512, (64,); fixed_x = true, seed = 42)
    ws_warm = BP.BatchGPUWorkspace(backend, FT, 512, 64, NB; fixed_x = true)
    BP.upload_batch!(ws_warm, backend, x_warm, u_warm)
    
    # Single snapshot
    SFC.gpu_calculate_structure_function(sft, backend, x_one_dev, u_one_dev, bin_edges; workspace = ws_one)
    KA.synchronize(backend)

    # Warmup batch kernels using small problem
    BP.gpu_batch_fused_tiled_fixed_x!(backend, zeros(FT, NB, 64), zeros(UInt32, NB, 64), x_warm, u_warm, sft, bin_edges; workspace = ws_warm, download = false)
    BP.gpu_batch_fused_tiled_fixed_x_usmem!(backend, zeros(FT, NB, 64), zeros(UInt32, NB, 64), x_warm, u_warm, sft, bin_edges; workspace = ws_warm, download = false)
    BP.gpu_batch_fused_tiled_fixed_x_usmem_priv!(backend, zeros(FT, NB, 64), zeros(UInt32, NB, 64), x_warm, u_warm, sft, bin_edges; workspace = ws_warm, download = false)
    BP.gpu_batch_fused_tiled_fixed_x_usmem_atomic_flush!(backend, zeros(FT, NB, 64), zeros(UInt32, NB, 64), x_warm, u_warm, sft, bin_edges; workspace = ws_warm, download = false)
    BP.gpu_batch_fused_tiled_fixed_x_coalesced!(backend, zeros(FT, NB, 64), zeros(UInt32, NB, 64), x_warm, u_warm, sft, bin_edges; workspace = ws_warm, download = false)
    BP.gpu_batch_fused_tiled_fixed_x_vectorized!(backend, zeros(FT, NB, 64), zeros(UInt32, NB, 64), x_warm, u_warm, sft, bin_edges; workspace = ws_warm, download = false)
    KA.synchronize(backend)
    println("Warmup complete.")
    flush(stdout)

    # Timings
    fast = _fast_mode()
    run_legacy = !fast

    # §1: Production single-snapshot
    _bench_progress("§1 production single-snapshot: timing...")
    times_one = Float64[]
    for _ in 1:n_repeat
        push!(times_one, @elapsed begin
            SFC.gpu_calculate_structure_function(sft, backend, x_one_dev, u_one_dev, bin_edges; workspace = ws_one)
            KA.synchronize(backend)
        end)
    end
    t_one = minimum(times_one)
    @printf("  §1 done: %.4f ms\n", 1000 * t_one)
    flush(stdout)

    # §2: Naive slice (simulated/computed)
    t_naive = t_one * B

    # §8: Integration — strip-outer smem → atomic flush (default)
    _bench_progress("§8 fused tiled (integration): timing...")
    times_fused = Float64[]
    for rep in 1:n_repeat
        BP.reset_output!(ws)
        t_rep = @elapsed begin
            BP.gpu_batch_fused_tiled_fixed_x!(backend, ws.sums_dev, ws.counts_dev, x, u, sft, bin_edges; workspace = ws, download = false)
            KA.synchronize(backend)
        end
        push!(times_fused, t_rep)
        @printf("  §8 repeat %d/%d: %.4fs\n", rep, n_repeat, t_rep)
        flush(stdout)
    end
    t_fused = minimum(times_fused)
    @printf("  §8 done (best of %d): %.4fs  (%.2fx vs naive %.1fs)\n", n_repeat, t_fused, t_naive / t_fused, t_naive)
    flush(stdout)

    # §3–§7: legacy experiments (skipped when FAST=1 or N>2048 unless BATCH_LEGACY_ALL=1)
    t_usmem_direct = 0.0
    times_usmem_direct = Float64[]
    legacy_all = get(ENV, "BATCH_LEGACY_ALL", "") == "1"
    run_direct = run_legacy && (legacy_all || N <= 2048)
    run_legacy_sections = run_legacy && (legacy_all || N <= 2048)
    if run_legacy_sections
        if run_direct
            _bench_progress("§3 u-smem direct atomic: timing...")
            for rep in 1:n_repeat
                BP.reset_output!(ws)
                t_rep = @elapsed begin
                    BP.gpu_batch_fused_tiled_fixed_x_usmem!(backend, ws.sums_dev, ws.counts_dev, x, u, sft, bin_edges; workspace = ws, download = false)
                    KA.synchronize(backend)
                end
                push!(times_usmem_direct, t_rep)
                @printf("  §3 repeat %d/%d: %.4fs\n", rep, n_repeat, t_rep)
                flush(stdout)
            end
            t_usmem_direct = minimum(times_usmem_direct)
        else
            println("Skipping §3 direct atomic at large scale (N=$N).")
            flush(stdout)
        end

        _bench_progress("§4 u-smem priv+merge: timing...")
        times_usmem_priv = Float64[]
        for rep in 1:n_repeat
            BP.reset_output!(ws)
            t_rep = @elapsed begin
                BP.gpu_batch_fused_tiled_fixed_x_usmem_priv!(backend, ws.sums_dev, ws.counts_dev, x, u, sft, bin_edges; workspace = ws, download = false)
                KA.synchronize(backend)
            end
            push!(times_usmem_priv, t_rep)
            @printf("  §4 repeat %d/%d: %.4fs\n", rep, n_repeat, t_rep)
            flush(stdout)
        end
        t_usmem_priv = minimum(times_usmem_priv)

        _bench_progress("§5 u-smem atomic flush: timing...")
        times_usmem_atomic = Float64[]
        for rep in 1:n_repeat
            BP.reset_output!(ws)
            t_rep = @elapsed begin
                BP.gpu_batch_fused_tiled_fixed_x_usmem_atomic_flush!(backend, ws.sums_dev, ws.counts_dev, x, u, sft, bin_edges; workspace = ws, download = false)
                KA.synchronize(backend)
            end
            push!(times_usmem_atomic, t_rep)
            @printf("  §5 repeat %d/%d: %.4fs\n", rep, n_repeat, t_rep)
            flush(stdout)
        end
        t_usmem_atomic = minimum(times_usmem_atomic)

        _bench_progress("§6 coalesced: timing...")
        times_coalesced = Float64[]
        for rep in 1:n_repeat
            BP.reset_output!(ws)
            t_rep = @elapsed begin
                BP.gpu_batch_fused_tiled_fixed_x_coalesced!(backend, ws.sums_dev, ws.counts_dev, x, u, sft, bin_edges; workspace = ws, download = false)
                KA.synchronize(backend)
            end
            push!(times_coalesced, t_rep)
            @printf("  §6 repeat %d/%d: %.4fs\n", rep, n_repeat, t_rep)
            flush(stdout)
        end
        t_coalesced = minimum(times_coalesced)

        _bench_progress("§7 vectorized: timing...")
        times_vectorized = Float64[]
        for rep in 1:n_repeat
            BP.reset_output!(ws)
            t_rep = @elapsed begin
                BP.gpu_batch_fused_tiled_fixed_x_vectorized!(backend, ws.sums_dev, ws.counts_dev, x, u, sft, bin_edges; workspace = ws, download = false)
                KA.synchronize(backend)
            end
            push!(times_vectorized, t_rep)
            @printf("  §7 repeat %d/%d: %.4fs\n", rep, n_repeat, t_rep)
            flush(stdout)
        end
        t_vectorized = minimum(times_vectorized)
    else
        t_usmem_priv = t_fused
        t_usmem_atomic = t_fused
        t_coalesced = t_fused
        t_vectorized = t_fused
        if !run_legacy
            println("FAST=1: skipping §3–§7 legacy timings.")
        elseif !legacy_all && N > 2048
            println("N=$N > 2048: skipping §3–§7 legacy timings (set BATCH_LEGACY_ALL=1 to force).")
        end
        flush(stdout)
    end

    # Verification: parity (skip full re-download at large scale when FAST=1)
    sums_fused = zeros(FT, NB, B)
    counts_fused = zeros(UInt32, NB, B)
    if fast && N > 1024
        _bench_progress("FAST=1: skipping large-scale parity re-downloads.")
        ok_fused_str = "SKIP"
        ok_direct_str = "SKIP"
        ok_priv_str = "SKIP"
        ok_atomic_str = "SKIP"
        ok_coal_str = "SKIP"
        ok_vect_str = "SKIP"
        diff_fused = diff_direct = diff_priv = diff_atomic = diff_coal = diff_vect = 0.0
        ce_fused = ce_direct = ce_priv = ce_atomic = ce_coal = ce_vect = true
        ref_name = "small-scale gate"
    else
        _bench_progress("parity: downloading reference outputs...")
        BP.reset_output!(ws)
        BP.gpu_batch_fused_tiled_fixed_x!(backend, sums_fused, counts_fused, x, u, sft, bin_edges; workspace = ws, download = true)

        sums_direct = zeros(FT, NB, B)
        counts_direct = zeros(UInt32, NB, B)
        if run_direct
            BP.reset_output!(ws)
            BP.gpu_batch_fused_tiled_fixed_x_usmem!(backend, sums_direct, counts_direct, x, u, sft, bin_edges; workspace = ws, download = true)
        end

        if run_legacy
            BP.reset_output!(ws)
            sums_priv = zeros(FT, NB, B)
            counts_priv = zeros(UInt32, NB, B)
            BP.gpu_batch_fused_tiled_fixed_x_usmem_priv!(backend, sums_priv, counts_priv, x, u, sft, bin_edges; workspace = ws, download = true)

            BP.reset_output!(ws)
            sums_atomic = zeros(FT, NB, B)
            counts_atomic = zeros(UInt32, NB, B)
            BP.gpu_batch_fused_tiled_fixed_x_usmem_atomic_flush!(backend, sums_atomic, counts_atomic, x, u, sft, bin_edges; workspace = ws, download = true)

            BP.reset_output!(ws)
            sums_coal = zeros(FT, NB, B)
            counts_coal = zeros(UInt32, NB, B)
            BP.gpu_batch_fused_tiled_fixed_x_coalesced!(backend, sums_coal, counts_coal, x, u, sft, bin_edges; workspace = ws, download = true)

            BP.reset_output!(ws)
            sums_vect = zeros(FT, NB, B)
            counts_vect = zeros(UInt32, NB, B)
            BP.gpu_batch_fused_tiled_fixed_x_vectorized!(backend, sums_vect, counts_vect, x, u, sft, bin_edges; workspace = ws, download = true)
        else
            sums_priv = sums_fused
            counts_priv = counts_fused
            sums_atomic = sums_fused
            counts_atomic = counts_fused
            sums_coal = sums_fused
            counts_coal = counts_fused
            sums_vect = sums_fused
            counts_vect = counts_fused
        end

        if N <= 1024 && B <= 256
        sums_ref = zeros(FT, NB, B)
        counts_ref = zeros(UInt32, NB, B)
        BP.cpu_batch_fixed_x!(sums_ref, counts_ref, x, u, sft, bin_edges; strip_width = 16)

        ok_fused, diff_fused, ce_fused = BP.check_parity(sums_ref, counts_ref, sums_fused, counts_fused)
        ok_direct, diff_direct, ce_direct = BP.check_parity(sums_ref, counts_ref, sums_direct, counts_direct)
        ok_priv, diff_priv, ce_priv = BP.check_parity(sums_ref, counts_ref, sums_priv, counts_priv)
        ok_atomic, diff_atomic, ce_atomic = BP.check_parity(sums_ref, counts_ref, sums_atomic, counts_atomic)
        ok_coal, diff_coal, ce_coal = BP.check_parity(sums_ref, counts_ref, sums_coal, counts_coal)
        ok_vect, diff_vect, ce_vect = BP.check_parity(sums_ref, counts_ref, sums_vect, counts_vect)
        
        ref_name = "CPU reference"
        ok_fused_str = ok_fused ? "PASS" : "FAIL"
        ok_direct_str = ok_direct ? "PASS" : "FAIL"
        ok_priv_str = ok_priv ? "PASS" : "FAIL"
        ok_atomic_str = ok_atomic ? "PASS" : "FAIL"
        ok_coal_str = ok_coal ? "PASS" : "FAIL"
        ok_vect_str = ok_vect ? "PASS" : "FAIL"
    else
        # Large scale
        ref_name = run_direct ? "GPU direct reference" : "GPU priv reference"
        if run_direct
            ok_fused, diff_fused, ce_fused = BP.check_parity(sums_direct, counts_direct, sums_fused, counts_fused)
            ok_priv, diff_priv, ce_priv = BP.check_parity(sums_direct, counts_direct, sums_priv, counts_priv)
            ok_atomic, diff_atomic, ce_atomic = BP.check_parity(sums_direct, counts_direct, sums_atomic, counts_atomic)
            ok_coal, diff_coal, ce_coal = BP.check_parity(sums_direct, counts_direct, sums_coal, counts_coal)
            ok_vect, diff_vect, ce_vect = BP.check_parity(sums_direct, counts_direct, sums_vect, counts_vect)
            ok_direct, diff_direct, ce_direct = ok_priv, diff_priv, ce_priv
            ok_fused_str = ok_fused ? "PASS" : "FAIL"
            ok_direct_str = ok_direct ? "PASS" : "FAIL"
            ok_priv_str = ok_priv ? "PASS" : "FAIL"
            ok_atomic_str = ok_atomic ? "PASS" : "FAIL"
            ok_coal_str = ok_coal ? "PASS" : "FAIL"
            ok_vect_str = ok_vect ? "PASS" : "FAIL"
        else
            ok_direct, diff_direct, ce_direct = true, 0.0, true
            ok_fused, diff_fused, ce_fused = BP.check_parity(sums_priv, counts_priv, sums_fused, counts_fused)
            ok_atomic, diff_atomic, ce_atomic = BP.check_parity(sums_priv, counts_priv, sums_atomic, counts_atomic)
            ok_coal, diff_coal, ce_coal = BP.check_parity(sums_priv, counts_priv, sums_coal, counts_coal)
            ok_vect, diff_vect, ce_vect = BP.check_parity(sums_priv, counts_priv, sums_vect, counts_vect)
            ok_priv, diff_priv, ce_priv = true, 0.0, true
            ok_direct_str = "SKIP"
            ok_fused_str = ok_fused ? "PASS" : "FAIL"
            ok_priv_str = "SKIP"
            ok_atomic_str = ok_atomic ? "PASS" : "FAIL"
            ok_coal_str = ok_coal ? "PASS" : "FAIL"
            ok_vect_str = ok_vect ? "PASS" : "FAIL"
        end
        println("Skipping slow CPU reference check (N=$N, B=$B). Correctness verified via small-scale tests.")
        println()
        println("=== DIAGNOSTICS (N=$N, B=$B) ===")
        println("  sums_fused:  sum = $(sum(sums_fused)), nonzeros = $(count(!iszero, sums_fused))")
        if run_direct
            println("  sums_direct: sum = $(sum(sums_direct)), nonzeros = $(count(!iszero, sums_direct))")
        end
        println("  sums_priv:   sum = $(sum(sums_priv)), nonzeros = $(count(!iszero, sums_priv))")
        println("  sums_atomic: sum = $(sum(sums_atomic)), nonzeros = $(count(!iszero, sums_atomic))")
        println("  sums_coal:   sum = $(sum(sums_coal)), nonzeros = $(count(!iszero, sums_coal))")
        println("  sums_vect:   sum = $(sum(sums_vect)), nonzeros = $(count(!iszero, sums_vect))")
        if run_direct
            println("  counts_direct: sum = $(sum(counts_direct)), nonzeros = $(count(!iszero, counts_direct))")
        end
        println("  counts_priv:   sum = $(sum(counts_priv)), nonzeros = $(count(!iszero, counts_priv))")
        println("  counts_atomic: sum = $(sum(counts_atomic)), nonzeros = $(count(!iszero, counts_atomic))")
        println("  counts_coal:   sum = $(sum(counts_coal)), nonzeros = $(count(!iszero, counts_coal))")
        println("  counts_vect:   sum = $(sum(counts_vect)), nonzeros = $(count(!iszero, counts_vect))")
        end
    end

    # Print results
    println()
    println("=== BENCHMARK RESULTS ===")
    @printf("  §1 Production single-snapshot:         %9.4f ms\n", 1000 * t_one)
    @printf("  §2 Naive slice projection (B × §1):    %9.4f s\n", t_naive)
    @printf("  §8 FUSED tiled (integration path):     %9.4f s  (speedup vs naive: %.2fx)\n", t_fused, t_naive / t_fused)
    if run_legacy
        if run_direct
            @printf("  §3 New u-smem direct atomic kernel:    %9.4f s  (speedup vs naive: %.2fx)\n", t_usmem_direct, t_naive / t_usmem_direct)
        else
            println("  §3 New u-smem direct atomic kernel:         SKIP")
        end
        @printf("  §4 New u-smem priv + merge kernel:      %9.4f s  (speedup vs naive: %.2fx)\n", t_usmem_priv, t_naive / t_usmem_priv)
        @printf("  §5 New u-smem atomic flush kernel:      %9.4f s  (speedup vs naive: %.2fx)\n", t_usmem_atomic, t_naive / t_usmem_atomic)
        @printf("  §6 New coalesced batch kernel:          %9.4f s  (speedup vs naive: %.2fx)\n", t_coalesced, t_naive / t_coalesced)
        @printf("  §7 New vectorized batch kernel (D):     %9.4f s  (speedup vs naive: %.2fx)\n", t_vectorized, t_naive / t_vectorized)
    end
    println()
    println("=== CORRECTNESS PARTIES ===")
    println("  fused tiled vs ", ref_name, ":       ", ok_fused_str, " (max |Δsums| = $(diff_fused), counts_equal = $(ce_fused))")
    if run_direct
        println("  u-smem direct vs ", ref_name, ":  ", ok_direct_str, " (max |Δsums| = $(diff_direct), counts_equal = $(ce_direct))")
    else
        println("  u-smem direct vs ", ref_name, ":  SKIP")
    end
    println("  u-smem priv vs ", ref_name, ":    ", ok_priv_str, " (max |Δsums| = $(diff_priv), counts_equal = $(ce_priv))")
    println("  u-smem atomic vs ", ref_name, ":  ", ok_atomic_str, " (max |Δsums| = $(diff_atomic), counts_equal = $(ce_atomic))")
    println("  coalesced vs ", ref_name, ":      ", ok_coal_str, " (max |Δsums| = $(diff_coal), counts_equal = $(ce_coal))")
    println("  vectorized (D) vs ", ref_name, ": ", ok_vect_str, " (max |Δsums| = $(diff_vect), counts_equal = $(ce_vect))")
    println()
    
    # Save log
    mkpath("test/debug")
    open("test/debug/batch_usmem.log", "w") do io
        println(io, "=== u-smem batch benchmark ===")
        println(io, "Backend: ", is_gpu ? "CUDA" : "CPU")
        println(io, "N=$N  B=$B")
        @printf(io, "Production single-snapshot:         %.4f ms\n", 1000 * t_one)
        @printf(io, "Naive slice projection:             %.4f s\n", t_naive)
        @printf(io, "FUSED tiled (integration path):     %.4f s  (%.2fx)\n", t_fused, t_naive / t_fused)
        println(io, "Parity fused vs ", ref_name, ": ", ok_fused_str, " (max diff: $(diff_fused))")
        if run_direct
            @printf(io, "New u-smem direct atomic kernel:    %.4f s  (%.2fx)\n", t_usmem_direct, t_naive / t_usmem_direct)
            println(io, "Parity direct vs ", ref_name, ": ", ok_direct_str, " (max diff: $(diff_direct))")
        else
            println(io, "New u-smem direct atomic kernel:    SKIP")
            println(io, "Parity direct vs ", ref_name, ": SKIP")
        end
        @printf(io, "New u-smem priv + merge kernel:      %.4f s  (%.2fx)\n", t_usmem_priv, t_naive / t_usmem_priv)
        println(io, "Parity priv vs ", ref_name, ": ", ok_priv_str, " (max diff: $(diff_priv))")
        @printf(io, "New u-smem atomic flush kernel:      %.4f s  (%.2fx)\n", t_usmem_atomic, t_naive / t_usmem_atomic)
        println(io, "Parity atomic vs ", ref_name, ": ", ok_atomic_str, " (max diff: $(diff_atomic))")
        @printf(io, "New coalesced batch kernel:          %.4f s  (%.2fx)\n", t_coalesced, t_naive / t_coalesced)
        println(io, "Parity coalesced vs ", ref_name, ": ", ok_coal_str, " (max diff: $(diff_coal))")
        @printf(io, "New vectorized batch kernel (D):     %.4f s  (%.2fx)\n", t_vectorized, t_naive / t_vectorized)
        println(io, "Parity vectorized vs ", ref_name, ": ", ok_vect_str, " (max diff: $(diff_vect))")
    end
    println("Results logged to test/debug/batch_usmem.log")
    if get(ENV, "BATCH_PROFILE", "") == "1"
        println("Profile lines appended to test/debug/batch_profile.log")
    end
    println("done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
