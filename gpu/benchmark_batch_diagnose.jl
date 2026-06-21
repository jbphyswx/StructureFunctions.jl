#!/usr/bin/env julia
"""
    benchmark_batch_diagnose.jl

Isolated GPU benchmarks to **prove or falsify** batch-accum hypotheses.
Run on a GPU node (SLURM or ALLOW_CUDA_BENCH=1):

    # quick smoke (~30 s)
    julia --project=gpu gpu/benchmark_batch_diagnose.jl

    # production scale
    N=20000 BATCH=8064 julia --project=gpu gpu/benchmark_batch_diagnose.jl

    # skip sections (comma-separated: layout,atomics,toy,real,baseline)
    SKIP=real N=20000 BATCH=8064 julia --project=gpu gpu/benchmark_batch_diagnose.jl

Sections
--------
1. **layout** — inner-`b` load+add only (no atomics, no SF): `(2,N,B)` vs `(B,N,2)`.
2. **atomics** — synthetic 1× pair pass: plain vs global `@atomic` inner-B (same work).
3. **toy** — simplified pair grid accum kernels (checksums must PASS).
4. **real** — fused one launch vs host smem strip vs legacy direct vs VRAM private.
5. **baseline** — production L2SF one snapshot at same N (kernel + sync, no D2H).

Read the **INTERPRETATION** block printed at the end.
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @kernel, @Const, @localmem, @synchronize
using StructureFunctions: StructureFunctionTypes as SFT, LinearBinEdges
using StructureFunctions.Calculations: Calculations as SFC
using Printf: @printf
using Random: Random

include(joinpath(@__DIR__, "batch_prototypes", "BatchPrototypes.jl"))
using .BatchPrototypes: BatchPrototypes as BP

# Toy accum kernels (top-level include — avoids Julia world-age error in §3).
include(joinpath(@__DIR__, "benchmark_batch_accum_toy.jl"))

const TOY_NB = 20

function _backend()
    if get(ENV, "ALLOW_CUDA_BENCH", "") == "1" || haskey(ENV, "SLURM_JOB_ID")
        CUDA.functional() || error("CUDA not functional")
        return CUDA.CUDABackend(), true
    end
    CUDA.functional() || error("CUDA required; set ALLOW_CUDA_BENCH=1 or run under SLURM")
    return CUDA.CUDABackend(), true
end

function _resolve_N_B()
    N = parse(Int, get(ENV, "N", "2048"))
    batch = parse(Int, get(ENV, "BATCH", "512"))
    return N, batch
end

function _skip_set()
    raw = strip(get(ENV, "SKIP", ""))
    isempty(raw) && return Set{String}()
    return Set(split(raw, ",") |> collect .|> strip)
end

function _min_time(f, n_warmup::Int, n_repeat::Int)
    for _ in 1:n_warmup
        f()
    end
    times = Float64[]
    for _ in 1:n_repeat
        push!(times, @elapsed f())
    end
    return minimum(times)
end

# ---------------------------------------------------------------------------
# §1 Layout microbench — load + add inner b, NO atomics
# ---------------------------------------------------------------------------

@kernel function _diag_inner_b_trailing!(
    out,
    @Const(u),
    gi::Int,
    gj::Int,
    B::Int,
    n_pairs::Int,
    nworkers::Int,
)
    k = @index(Global, Linear)
    while k <= n_pairs
        acc = 0.0f0
        @inbounds for col in 1:B
            acc += u[1, gi, col] + u[1, gj, col]
        end
        out[k] = acc
        k += nworkers
    end
end

@kernel function _diag_inner_b_batchmajor!(
    out,
    @Const(u),
    gi::Int,
    gj::Int,
    B::Int,
    n_pairs::Int,
    nworkers::Int,
)
    k = @index(Global, Linear)
    while k <= n_pairs
        acc = 0.0f0
        @inbounds for col in 1:B
            acc += u[col, gi, 1] + u[col, gj, 1]
        end
        out[k] = acc
        k += nworkers
    end
end

function _run_layout_section!(backend, N, B, n_warmup, n_repeat, wg, nworkers)
    FT = Float32
    gi = N ÷ 2
    gj = gi ÷ 2
    n_pairs = min(nworkers, 65_536)

    # Production layout: (2, N, B) — inner col strides 2*N elements
    u_trail = rand(FT, 2, N, B)
    u_trail_dev = KA.adapt(backend, u_trail)
    out_t = KA.adapt(backend, zeros(FT, n_pairs))

    # Batch-contiguous at fixed point: (B, N, 2) — inner col stride 1
    u_batch = permutedims(u_trail, (3, 2, 1))  # (B, N, 2)
    u_batch_dev = KA.adapt(backend, u_batch)
    out_b = KA.adapt(backend, zeros(FT, n_pairs))

    kt! = _diag_inner_b_trailing!(backend, wg)
    kb! = _diag_inner_b_batchmajor!(backend, wg)

    t_trail = _min_time(n_warmup, n_repeat) do
        kt!(out_t, u_trail_dev, gi, gj, B, n_pairs, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
    end
    t_batch = _min_time(n_warmup, n_repeat) do
        kb!(out_b, u_batch_dev, gi, gj, B, n_pairs, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
    end

    bytes = 4 * 2 * B * n_pairs  # two u loads per inner-b iter × pairs
    println()
    println("=== §1 LAYOUT (load+add inner b, no atomics, no SF) ===")
    @printf("  u shape (2,N,B) trailing batch   %.4fs  (stride 2N=%d elems between b)\n",
        t_trail, 2 * N)
    @printf("  u shape (B,N,2) batch-major      %.4fs  (stride 1 between b)\n", t_batch)
    @printf("  batch-major / trailing           = %.3fx  (>1 => layout hurts trailing)\n",
        t_trail / t_batch)
    @printf("  effective read bandwidth trailing  %.2f GB/s\n", bytes / t_trail / 1e9)
    @printf("  effective read bandwidth batchmaj  %.2f GB/s\n", bytes / t_batch / 1e9)
    flush(stdout)
    return t_trail, t_batch
end

# ---------------------------------------------------------------------------
# §2 Atomics microbench — same (2,N,B) layout, inner b
# ---------------------------------------------------------------------------

@kernel function _diag_inner_b_plain!(
    out,
    @Const(u),
    gi::Int,
    gj::Int,
    B::Int,
    NB::Int,
    n_pairs::Int,
    nworkers::Int,
)
    k = @index(Global, Linear)
    while k <= n_pairs
        bin = (k % NB) + 1
        acc = 0.0f0
        @inbounds for col in 1:B
            acc += u[1, gi, col] - u[1, gj, col]
        end
        out[bin, k % B + 1] = acc  # non-atomic write (one thread per k — not a real histogram)
        k += nworkers
    end
end

@kernel function _diag_inner_b_global_atomic!(
    output,
    @Const(u),
    gi::Int,
    gj::Int,
    B::Int,
    NB::Int,
    n_pairs::Int,
    nworkers::Int,
)
    k = @index(Global, Linear)
    while k <= n_pairs
        bin = (k % NB) + 1
        @inbounds for col in 1:B
            val = u[1, gi, col] - u[1, gj, col]
            @atomic output[bin, col] += val
        end
        k += nworkers
    end
end

function _run_atomics_section!(backend, N, B, n_warmup, n_repeat, wg, nworkers)
    FT = Float32
    gi = N ÷ 2
    gj = gi ÷ 2
    NB = TOY_NB
    n_pairs = min(nworkers, 32_768)

    u_dev = KA.adapt(backend, rand(FT, 2, N, B))
    out_plain = KA.adapt(backend, zeros(FT, NB, B))
    out_atom = KA.adapt(backend, zeros(FT, NB, B))

    kp! = _diag_inner_b_plain!(backend, wg)
    ka! = _diag_inner_b_global_atomic!(backend, wg)

    t_plain = _min_time(n_warmup, n_repeat) do
        fill!(out_plain, zero(FT))
        kp!(out_plain, u_dev, gi, gj, B, NB, n_pairs, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
    end
    t_atom = _min_time(n_warmup, n_repeat) do
        fill!(out_atom, zero(FT))
        ka!(out_atom, u_dev, gi, gj, B, NB, n_pairs, nworkers; ndrange = nworkers, workgroupsize = wg)
        KA.synchronize(backend)
    end

    println()
    println("=== §2 ATOMICS (same (2,N,B) u, synthetic n_pairs=$n_pairs, 1× pair pass each) ===")
    @printf("  plain inner-B (no histogram atomics)     %.4fs\n", t_plain)
    @printf("  global @atomic inner-B (like fused)       %.4fs  (atom/plain=%.3fx)\n",
        t_atom, t_atom / t_plain)
    flush(stdout)
    return t_plain, t_atom
end

# ---------------------------------------------------------------------------
# §3 Toy full accum (include toy kernels)
# ---------------------------------------------------------------------------

function _run_toy_section!(backend, N, B, n_warmup, n_repeat, wg, nworkers)
    P = parse(Int, get(ENV, "TOY_P", "4"))
    W = parse(Int, get(ENV, "STRIP_W", "32"))
    u_dev = KA.adapt(backend, rand(Float32, N, B))
    partial_dev = _make_partial!(backend, B, cld(nworkers, wg))
    n_strip_chunks = P * cld(B, W)

    for _ in 1:n_warmup
        _launch_fused_vram!(backend, KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B)), u_dev, N, B, P, nworkers, wg)
        _launch_fused_vram_private!(backend, KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B)), partial_dev, u_dev, N, B, P, nworkers, wg)
        _launch_fused_block_smem!(backend, KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B)), u_dev, N, B, P, W, nworkers, wg)
    end

    times_v = Float64[]
    times_p = Float64[]
    times_s = Float64[]
    for _ in 1:max(n_repeat, 1)
        push!(times_v, @elapsed _launch_fused_vram!(backend, KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B)), u_dev, N, B, P, nworkers, wg))
        push!(times_p, @elapsed _launch_fused_vram_private!(backend, KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B)), partial_dev, u_dev, N, B, P, nworkers, wg))
        push!(times_s, @elapsed _launch_fused_block_smem!(backend, KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B)), u_dev, N, B, P, W, nworkers, wg))
    end
    t_v, t_p, t_s = minimum(times_v), minimum(times_p), minimum(times_s)

    out_v = KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B))
    _launch_fused_vram!(backend, out_v, u_dev, N, B, P, nworkers, wg)
    out_p = KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B))
    _launch_fused_vram_private!(backend, out_p, partial_dev, u_dev, N, B, P, nworkers, wg)
    out_s = KA.adapt(backend, zeros(Float32, 2 * TOY_NB, B))
    _launch_fused_block_smem!(backend, out_s, u_dev, N, B, P, W, nworkers, wg)
    parity_p = isapprox(Array(out_v), Array(out_p); rtol = 1e-4, atol = 64.0)
    parity_s = isapprox(Array(out_v), Array(out_s); rtol = 1e-4, atol = 64.0)

    println()
    println("=== §3 TOY (pair grid N=$N B=$B P=$P W=$W) ===")
    @printf("  fused_vram (1× pairs, inner-B global)     %.4fs\n", t_v)
    @printf("  fused_vram_private (1× pairs, priv+merge)  %.4fs  (private/vram=%.3fx)\n", t_p, t_p / t_v)
    @printf("  fused_block_smem (%d× pair replays)        %.4fs  (smem/vram=%.3fx)\n",
        n_strip_chunks, t_s, t_s / t_v)
    @printf("  parity vram vs private=%s  vram vs smem=%s\n",
        parity_p ? "PASS" : "FAIL", parity_s ? "PASS" : "FAIL")
    flush(stdout)
    return t_v, t_p, t_s, parity_p, parity_s
end

# ---------------------------------------------------------------------------
# §4 Real tiled128 SF kernels
# ---------------------------------------------------------------------------

function _run_real_section!(backend, N, batch_shape, n_warmup, n_repeat)
    FT = Float32
    B = prod(batch_shape)
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    NB = length(bin_edges.edges) - 1

    x, u = BP.make_random_batch_problem(FT, N, batch_shape; fixed_x = true, seed = 1)
    sums_h = zeros(FT, NB, batch_shape...)
    counts_h = zeros(UInt32, NB, batch_shape...)
    ws = BP.BatchGPUWorkspace(backend, FT, N, B, NB; fixed_x = true)
    BP.upload_batch!(ws, backend, x, u)
    x_dev, u_dev = ws.x_dev, ws.u_dev
    n_strips = cld(B, BP.BATCH_TILED_STRIP_W)

    for _ in 1:n_warmup
        BP.gpu_batch_tiled_fixed_x!(backend, sums_h, counts_h, x, u, sft, bin_edges; workspace = ws, download = false)
        BP._launch_fused_fixed_x_in_kernel_timed!(backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges)
        BP._launch_fused_fixed_x_direct_timed!(backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges)
        BP._launch_fused_fixed_x_priv_timed!(backend, ws.sums_dev, ws.counts_dev, BP.ensure_partial_dev!(ws, backend), x_dev, u_dev, sft, N, B, bin_edges)
    end

    strip_t = Float64[]
    fused_t = Float64[]
    direct_t = Float64[]
    vfill = Float64[]
    vacc = Float64[]
    vmerge = Float64[]
    for _ in 1:n_repeat
        BP.reset_output!(ws)
        push!(strip_t, @elapsed BP.gpu_batch_tiled_fixed_x!(backend, sums_h, counts_h, x, u, sft, bin_edges; workspace = ws, download = false))
        BP.reset_output!(ws)
        ph = BP._launch_fused_fixed_x_in_kernel_timed!(backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges)
        push!(fused_t, ph.accum_s)
        BP.reset_output!(ws)
        ph = BP._launch_fused_fixed_x_direct_timed!(backend, ws.sums_dev, ws.counts_dev, x_dev, u_dev, sft, N, B, bin_edges)
        push!(direct_t, ph.accum_s)
        BP.reset_output!(ws)
        ph = BP._launch_fused_fixed_x_priv_timed!(backend, ws.sums_dev, ws.counts_dev, BP.ensure_partial_dev!(ws, backend), x_dev, u_dev, sft, N, B, bin_edges)
        push!(vfill, ph.fill_s)
        push!(vacc, ph.accum_s)
        push!(vmerge, ph.merge_s)
    end

    t_strip = minimum(strip_t)
    t_fused = minimum(fused_t)
    t_direct = minimum(direct_t)
    t_vfill = minimum(vfill)
    t_vacc = minimum(vacc)
    t_vmerge = minimum(vmerge)

    println()
    println("=== §4 REAL tiled128 SF (N=$N B=$B NB=$NB) ===")
    @printf("  fused_1xlaunch (one pair loop)       %.4fs\n", t_fused)
    @printf("  strip_host_%dxlaunch (pair replay)   %.4fs\n", n_strips, t_strip)
    @printf("  direct_global inner-B 1x (legacy)  %.4fs\n", t_direct)
    @printf("  vram_private fill/accum/merge        %.4f / %.4f / %.4f s\n", t_vfill, t_vacc, t_vmerge)
    @printf("  fused / strip                      = %.3fx\n", t_fused / t_strip)
    @printf("  fused / direct                     = %.3fx\n", t_fused / t_direct)
    @printf("  strip / vram_accum                 = %.3fx  fill+merge= %.2f%% of vram\n",
        t_strip / t_vacc, 100 * (t_vfill + t_vmerge) / (t_vfill + t_vacc + t_vmerge))
    flush(stdout)
    return t_fused, t_strip, t_direct, t_vacc
end

function _run_baseline_section!(backend, N, n_warmup, n_repeat)
    FT = Float32
    sft = SFT.L2SFType()
    bin_edges = LinearBinEdges(range(FT(0.1), FT(2.0); length = 21))
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_dev = KA.adapt(backend, x_cpu)
    u_dev = KA.adapt(backend, u_cpu)
    ws = SFC.GPUSFWorkspace(backend, bin_edges)

    run = () -> begin
        SFC.gpu_calculate_structure_function(sft, backend, x_dev, u_dev, bin_edges; workspace = ws)
        KA.synchronize(backend)
    end
    for _ in 1:n_warmup
        run()
    end
    times = Float64[]
    for _ in 1:n_repeat
        push!(times, @elapsed run())
    end
    t = minimum(times)

    println()
    println("=== §5 PRODUCTION BASELINE (N=$N, one L2SF snapshot, kernel+sync) ===")
    @printf("  gpu_calculate_structure_function     %.4fs  (%.2f ms)\n", t, 1000 * t)
    flush(stdout)
    return t
end

function _print_interpretation()
    println()
    println("=== INTERPRETATION ===")
    println("§1 layout: batch-major much faster than trailing → permute u to (B,N,2) on upload.")
    println("§2 atomics: global inner-B >> plain at same 1× pair pass (atom/plain ratio).")
    println("§3 toy: vram_private ≈ vram → block-private partial does not fix inner-B atomics.")
    println("§4 real: fused = one pair loop; strip = 252× pair replay; compare §5 for ruler.")
    println("§5 baseline: production one-snapshot ms × batch factor is the honest target band.")
    println("Log path suggestion: test/debug/batch_diagnose.log")
end

function main()
    backend, _ = _backend()
    N, B = _resolve_N_B()
    batch_shape = (B,)
    n_warmup = parse(Int, get(ENV, "WARMUP", "1"))
    n_repeat = parse(Int, get(ENV, "N_REPEAT", "3"))
    wg = parse(Int, get(ENV, "WORKGROUP", "256"))
    nworkers = parse(Int, get(ENV, "NWORKERS", string(min(262_144, max(4096, N * (N - 1) ÷ 2 ÷ 64)))))
    skip = _skip_set()

    println("=== batch diagnose ===")
    println("device: ", CUDA.name(CUDA.device()))
    println("N=$N  B=$B  nworkers=$nworkers  wg=$wg  warmup=$n_warmup  repeat=$n_repeat")
    println("SKIP=$(join(sort(collect(skip)), ","))")
    flush(stdout)

    if "layout" ∉ skip
        _run_layout_section!(backend, N, B, n_warmup, n_repeat, wg, nworkers)
    end
    if "atomics" ∉ skip
        _run_atomics_section!(backend, N, B, n_warmup, n_repeat, wg, nworkers)
    end
    if "toy" ∉ skip
        toy_N = parse(Int, get(ENV, "TOY_N", string(min(N, 512))))
        toy_B = parse(Int, get(ENV, "TOY_B", string(min(B, 8192))))
        _run_toy_section!(backend, toy_N, toy_B, n_warmup, n_repeat, wg, min(nworkers, 65_536))
    end
    if "real" ∉ skip
        _run_real_section!(backend, N, batch_shape, n_warmup, n_repeat)
    end
    if "baseline" ∉ skip
        _run_baseline_section!(backend, N, n_warmup, n_repeat)
    end

    _print_interpretation()
    println("done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
