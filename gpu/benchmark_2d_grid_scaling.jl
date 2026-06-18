#!/usr/bin/env julia
"""
    benchmark_2d_grid_scaling.jl

Compare single-type joint 2D vs eight-type single-pass 2D on GPU.

**Single-type joint 2D** (`gpu_calculate_structure_function_2d`):
  tiled block-local when `n_dist * n_val ≤ 4096`; default exact `@localmem` width
  (`joint2d_compile_cells = n_dist × n_val`). A/B vs max smem via two workspaces.

**Eight-type single-pass 2D** (`gpu_calculate_structure_functions_single_pass_2d!`):
  HTP-EJ path when distance bins are typed and `n_dist ≤ 64`:
  - `:shared` / `:typeplane` — on-chip shared histogram + direct flush to output (no merge)
  - `:direct` — priv slab + serial merge (`ENV["SP2D_MERGE"]` for experiments)

Gate: e2e SP2D < `8 × joint_2d`. Logs `output=on-chip-flush` vs `priv+merge`.

Full design: `gpu/SP2D_HTP_EJ.md`

Run on GPU:

    julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
    N_DIST=20 N_VAL=20 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
    N_DIST=50 N_VAL=50 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using Printf: @printf
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using StructureFunctions: joint2d_smem_max
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT

function _bench(f, warmup::Int, repeat_::Int)
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()
    elapsed = 0.0
    for _ in 1:repeat_
        elapsed += @elapsed begin
            f()
            CUDA.synchronize()
        end
    end
    return elapsed / repeat_
end

function _dist_bins(n_dist::Int, ::Type{FT}) where {FT}
    return LogBinEdges(Vector{FT}(exp.(range(log(FT(1000)), log(FT(50000)); length = n_dist + 1))))
end

function _value_shared(n_val_inner::Int, ::Type{FT}) where {FT}
    return InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_val_inner + 1)))
end

using Dates: Dates

const _GPUExt = Base.get_extension(SF, :StructureFunctionsGPUExt)
_GPUExt === nothing && error("StructureFunctionsGPUExt not loaded — activate gpu project with CUDA")

function main()
    CUDA.functional() || error("CUDA not functional")

    N = parse(Int, get(ENV, "N", "20000"))
    n_dist = parse(Int, get(ENV, "N_DIST", "20"))
    n_val_inner = parse(Int, get(ENV, "N_VAL", "20"))
    warmup = parse(Int, get(ENV, "WARMUP", "2"))
    repeat_ = parse(Int, get(ENV, "REPEAT", "5"))
    FT = Float32
    backend = CUDA.CUDABackend()
    gpu = SF.GPUBackend(backend)

    dist = _dist_bins(n_dist, FT)
    value_bins = _value_shared(n_val_inner, FT)
    n_val = length(value_bins) - 1
    NB2 = n_dist * n_val
    C = 8 * NB2
    joint_eligible = _GPUExt._gpu_joint_2d_tiled_eligible(n_dist, n_val)
    dist_route = _GPUExt._joint2d_dist_route(_GPUExt._gpu_normalize_bins(dist))

    log_dir = joinpath(@__DIR__, "..", "test", "debug")
    mkpath(log_dir)
    log_path = joinpath(log_dir, "sp2d_phase_profile.log")

    println("=" ^ 72)
    println("2D grid scaling — block-local vs global-atomic paths")
    println("Device: ", CUDA.name(CUDA.device()))
    @printf(
        "N=%d  n_dist=%d  n_val=%d  NB2=%d  C=%d  joint_2d tiled eligible=%s\n",
        N, n_dist, n_val, NB2, C, joint_eligible,
    )

    Random.seed!(42)
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)
    sft = SFT.L2SFType()

    # --- single-type joint 2D: exact smem (default); typed InfPadded value bins ---
    ws_j_exact = SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :joint2d)
    val_route = _GPUExt._joint2d_val_route(ws_j_exact.val_plan)
    @printf("joint dist_route=%s  value_route=%s\n", dist_route, val_route)
    println("=" ^ 72)

    j_exact_run = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, value_bins; workspace = ws_j_exact,
    )
    t_joint_exact = _bench(j_exact_run, warmup, repeat_)
    compile_exact = ws_j_exact.joint2d_compile_cells
    @printf(
        "joint 2D exact smem       %8.3f ms  [compile_cells=%d NB2=%d]\n",
        1_000t_joint_exact, compile_exact, NB2,
    )

    # --- single-type joint 2D: max smem (legacy 4096) ---
    ws_j_max = SFC.GPUSFWorkspace(
        backend, dist, value_bins;
        kind = :joint2d, joint2d_compile_cells = joint2d_smem_max(),
    )
    j_max_run = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, value_bins; workspace = ws_j_max,
    )
    t_joint_max = _bench(j_max_run, warmup, repeat_)
    compile_max = ws_j_max.joint2d_compile_cells
    saved_pct = 100 * (t_joint_max - t_joint_exact) / t_joint_max
    @printf(
        "joint 2D max smem         %8.3f ms  [compile_cells=%d; %.1f%% vs exact]\n",
        1_000t_joint_max, compile_max, saved_pct,
    )

    t_joint = t_joint_exact
    t_joint8 = 8 * t_joint
    @printf("8 × joint 2D (exact)      %8.3f ms  [reference column]\n", 1_000t_joint8)

    # --- eight-type sp2d (HTP-EJ privatized) ---
    ws_sp = SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :single_pass_2d)
    cfg = ws_sp.sp2d_priv_config
    mode_label = if cfg.accum_mode == :typeplane
        "typeplane ($(cfg.types_per_pass)×$(cfg.n_type_passes) passes)"
    else
        string(cfg.accum_mode)
    end
    output_path = cfg.needs_priv_merge ? "priv+merge" : "on-chip-flush"
    @printf("sp2d accum_mode          %s  (max_shared=%d, output=%s)\n",
        mode_label, cfg.max_shared_cells, output_path)

    sums = zeros(FT, 8, n_dist, n_val)
    counts = zeros(UInt32, 8, n_dist, n_val)
    sp_run = () -> SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums, counts, backend, x, u, dist, value_bins; workspace = ws_sp,
    )
    t_sp2d = _bench(sp_run, warmup, repeat_)

    # Phase split: pair kernel vs merge (reuse workspace priv slabs)
    x_dev = KA.allocate(backend, FT, 2, N)
    u_dev = KA.allocate(backend, FT, 2, N)
    copyto!(x_dev, x)
    copyto!(u_dev, u)
    val_plan = ws_sp.val_plan
    n_dist_edges = _GPUExt._gpu_n_edges(dist)
    n_val_edges = _GPUExt._sp2d_n_val_edges(value_bins)

    pair_run = if cfg.needs_priv_merge
        () -> begin
            _GPUExt._sp2d_priv_pair_bufs_and_launch!(
                backend, sums, x_dev, u_dev, ws_sp.dist_bins, val_plan,
                N, n_dist_edges, n_val_edges, n_dist, cfg; workspace = ws_sp,
            )
        end
    else
        () -> begin
            _GPUExt._launch_sp2d_onchip!(
                backend, ws_sp.out_sums_dev, ws_sp.out_cnts_dev, x_dev, u_dev,
                ws_sp.dist_bins, val_plan, N, n_dist_edges, n_val_edges, n_dist, cfg;
                workspace = ws_sp,
            )
        end
    end
    if cfg.needs_priv_merge
        priv_sums, priv_cnts, n_tb = pair_run()
        CUDA.synchronize()
    else
        pair_run()
        CUDA.synchronize()
        priv_sums, priv_cnts, n_tb = nothing, nothing, 0
    end
    t_pair = _bench(pair_run, warmup, repeat_)
    if cfg.needs_priv_merge
        priv_sums, priv_cnts, n_tb = pair_run()
        CUDA.synchronize()
        merge_serial = () -> _GPUExt._launch_merge_sp2d_priv!(
            backend, ws_sp.out_sums_dev, ws_sp.out_cnts_dev, priv_sums, priv_cnts,
            n_dist, n_val, n_tb; merge_mode = :serial,
        )
        merge_parallel = () -> _GPUExt._launch_merge_sp2d_priv!(
            backend, ws_sp.out_sums_dev, ws_sp.out_cnts_dev, priv_sums, priv_cnts,
            n_dist, n_val, n_tb; merge_mode = :parallel,
        )
        t_merge_serial = _bench(merge_serial, warmup, repeat_)
        priv_sums, priv_cnts, n_tb = pair_run()
        CUDA.synchronize()
        t_merge_parallel = _bench(merge_parallel, warmup, repeat_)
        t_merge_prod = t_merge_serial
    else
        t_merge_serial = 0.0
        t_merge_parallel = 0.0
        t_merge_prod = 0.0
    end

    @printf("sp2d pair kernel         %8.3f ms  [%s; %s]\n", 1_000t_pair, mode_label, output_path)
    if cfg.needs_priv_merge
        @printf("sp2d merge (serial)     %8.3f ms  [SP2D_MERGE=serial default]\n", 1_000t_merge_serial)
        @printf("sp2d merge (parallel)   %8.3f ms  [comparison only; slow when C large]\n", 1_000t_merge_parallel)
        @printf("sp2d total (end-to-end)  %8.3f ms  [pair + serial merge + host]\n", 1_000t_sp2d)
    else
        @printf("sp2d merge (serial)     %8.3f ms  [skipped — on-chip direct flush]\n", 0.0)
        @printf("sp2d merge (parallel)   %8.3f ms  [skipped — on-chip direct flush]\n", 0.0)
        @printf("sp2d total (end-to-end)  %8.3f ms  [pair flush + host]\n", 1_000t_sp2d)
    end

    # --- reference: 8-type sp1d same distance bins ---
    ws_sp1 = SFC.GPUSFWorkspace(backend, dist; kind = :single_pass)
    sums1 = zeros(FT, 8, n_dist)
    counts1 = zeros(UInt32, 8, n_dist)
    sp1_run = () -> SFC.calculate_structure_functions_single_pass!(
        sums1, counts1, x, u, dist; backend = gpu, workspace = ws_sp1,
    )
    t_sp1 = _bench(sp1_run, warmup, repeat_)
    @printf("sp1d (8 SF types)        %8.3f ms  [block-local (8, NB)]\n", 1_000t_sp1)

    gate_ok = t_sp2d < t_joint8
    @printf(
        "\nsp2d / joint_2d = %.1f×   sp2d / sp1d = %.1f×   sp2d < 8×joint = %s\n",
        t_sp2d / t_joint, t_sp2d / t_sp1, gate_ok ? "PASS" : "FAIL",
    )
    @printf("sp2d pair+merge ≈ %.1f ms (%.0f%% pair; production merge=serial)\n",
        1_000(t_pair + t_merge_prod), 100t_pair / (t_pair + t_merge_prod))

    open(log_path, "a") do io
        println(io, "--- $(Dates.now()) ---")
        @printf(io,
            "device=%s N=%d n_dist=%d n_val=%d NB2=%d compile_exact=%d compile_max=%d dist_route=%s val_route=%s C=%d mode=%s output=%s tpp=%d ntp=%d\n",
            CUDA.name(CUDA.device()), N, n_dist, n_val, NB2, compile_exact, compile_max,
            dist_route, val_route, C, cfg.accum_mode, output_path, cfg.types_per_pass, cfg.n_type_passes)
        @printf(io,
            "joint_exact=%.6f joint_max=%.6f joint8=%.6f sp2d=%.6f pair=%.6f merge_s=%.6f merge_p=%.6f sp1d=%.6f gate=%s\n",
            t_joint_exact, t_joint_max, t_joint8, t_sp2d, t_pair, t_merge_serial,
            t_merge_parallel, t_sp1, gate_ok ? "PASS" : "FAIL")
    end
    println("\nLogged to ", log_path)
    println("Re-run production grid: N_DIST=50 N_VAL=50 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl")
    println("=" ^ 72)

    gate_ok || error("sp2d gate failed: sp2d ($(round(1_000t_sp2d; digits=1)) ms) >= 8×joint ($(round(1_000t_joint8; digits=1)) ms)")
end

main()
