#!/usr/bin/env julia
"""
    benchmark_2d_grid_scaling.jl

Compare single-type joint 2D vs six-type single-pass 2D on GPU.

**Single-type joint 2D** (`gpu_calculate_structure_function_2d`):
  tiled block-local when its histogram fits the device's shared memory; default exact `@localmem`
  width (`joint2d_compile_cells = n_dist × n_val`). A/B vs the widest fitting width via two workspaces.

**Six-type single-pass 2D** (`gpu_calculate_structure_functions_single_pass_2d!`), and its portable kernel alone:
  the HTP-EJ on-chip histogram (`:shared` / `:typeplane`) flushed into the output, or the global-atomic kernel for a
  histogram neither mode holds.

Gate: e2e SP2D < `6 × joint_2d`.

Run on GPU:

    julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
    N_DIST=20 N_VAL=20 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
    N_DIST=50 N_VAL=50 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
"""

using ComputationalBackends: ComputationalBackends as CB
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using Printf: Printf
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using StructureFunctions.Calculations: joint2d_smem_max
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.HelperFunctions: HelperFunctions as SFH

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
    return LogBinEdges(FT(1000), FT(50000), n_dist + 1)
end

function _value_shared(n_val_inner::Int, ::Type{FT}) where {FT}
    return InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_val_inner + 1)))
end

using Dates: Dates

const _GPUExt = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
_GPUExt === nothing && error("StructureFunctionsKernelAbstractionsExt not loaded — activate gpu project with CUDA")

function main()
    CUDA.functional() || error("CUDA not functional")

    N = parse(Int, get(ENV, "N", "20000"))
    n_dist = parse(Int, get(ENV, "N_DIST", "20"))
    n_val_inner = parse(Int, get(ENV, "N_VAL", "20"))
    warmup = parse(Int, get(ENV, "WARMUP", "2"))
    repeat_ = parse(Int, get(ENV, "REPEAT", "5"))
    FT = Float32
    backend = CUDA.CUDABackend()
    gpu = CB.GPUBackend(backend)

    dist = _dist_bins(n_dist, FT)
    value_bins = _value_shared(n_val_inner, FT)
    n_val = length(value_bins) - 1
    NB2 = n_dist * n_val
    C = 8 * NB2
    caps = SFC.gpu_device_caps(backend)
    joint_eligible = _GPUExt._gpu_joint_2d_tiled_eligible(caps, 2, 2, FT, FT, UInt32, NB2)

    log_dir = joinpath(@__DIR__, "..", "test", "debug")
    mkpath(log_dir)
    log_path = joinpath(log_dir, "sp2d_phase_profile.log")

    println("=" ^ 72)
    println("2D grid scaling — block-local vs global-atomic paths")
    println("Device: ", CUDA.name(CUDA.device()))
    Printf.@printf(
        "N=%d  n_dist=%d  n_val=%d  NB2=%d  C=%d  joint_2d tiled eligible=%s\n",
        N, n_dist, n_val, NB2, C, joint_eligible,
    )

    Random.seed!(42)
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)
    sft = SFT.L2SFType()

    # --- single-type joint 2D: exact smem (default); typed InfPadded value bins ---
    ws_j_exact = SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :joint2d)
    dist_route = nameof(typeof(ws_j_exact.dist_digitizer))
    val_route = nameof(typeof(ws_j_exact.val_plan))
    Printf.@printf("joint dist digitizer=%s  value digitizer=%s\n", dist_route, val_route)
    println("=" ^ 72)

    j_exact_run = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, value_bins, UInt32; workspace = ws_j_exact, geometry = SFH.FlatGeometry{2}(),
    )
    t_joint_exact = _bench(j_exact_run, warmup, repeat_)
    compile_exact = ws_j_exact.joint2d_compile_cells
    Printf.@printf(
        "joint 2D exact smem       %8.3f ms  [compile_cells=%d NB2=%d]\n",
        1_000t_joint_exact, compile_exact, NB2,
    )

    # --- single-type joint 2D: max smem compile width ---
    ws_j_max = SFC.GPUSFWorkspace(
        backend, dist, value_bins;
        kind = :joint2d, joint2d_compile_cells = joint2d_smem_max(backend, 2, 2, FT, FT, UInt32),
    )
    j_max_run = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, value_bins, UInt32; workspace = ws_j_max, geometry = SFH.FlatGeometry{2}(),
    )
    t_joint_max = _bench(j_max_run, warmup, repeat_)
    compile_max = ws_j_max.joint2d_compile_cells
    saved_pct = 100 * (t_joint_max - t_joint_exact) / t_joint_max
    Printf.@printf(
        "joint 2D max smem         %8.3f ms  [compile_cells=%d; %.1f%% vs exact]\n",
        1_000t_joint_max, compile_max, saved_pct,
    )

    t_joint = t_joint_exact
    t_joint6 = 6 * t_joint
    Printf.@printf("6 × joint 2D (exact)      %8.3f ms  [reference column]\n", 1_000t_joint6)

    # --- six-type sp2d (HTP-EJ privatized) ---
    ws_sp = SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :single_pass_2d)
    cfg = _GPUExt._sp2d_accumulation_strategy(caps, n_dist, n_val, 2, 2, FT, FT, UInt32)
    mode_label = if cfg === nothing
        "global atomics"
    elseif cfg.accum_mode == :typeplane
        "typeplane ($(cfg.types_per_pass)×$(cfg.n_type_passes) passes)"
    else
        string(cfg.accum_mode)
    end
    Printf.@printf("sp2d portable kernel     %s\n", mode_label)

    sums = CUDA.zeros(FT, 6, n_dist, n_val)
    counts = CUDA.zeros(UInt32, 6, n_dist, n_val)
    sp_run = () -> SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums, counts, backend, x, u, dist, value_bins; workspace = ws_sp, geometry = SFH.FlatGeometry{2}(),
    )
    t_sp2d = _bench(sp_run, warmup, repeat_)

    x_dev = KA.allocate(backend, FT, 2, N)
    u_dev = KA.allocate(backend, FT, 2, N)
    copyto!(x_dev, x)
    copyto!(u_dev, u)
    ddig = ws_sp.dist_digitizer
    vplan = ws_sp.val_plan
    n_dist_edges = length(dist)
    n_val_edges = _GPUExt._n_value_edges(value_bins)
    geom = SF.HelperFunctions.FlatGeometry{2}()
    portable_run = () -> _GPUExt._launch_single_pass_2d_portable!(
        backend, 64, sums, counts, x_dev, u_dev, ddig, vplan, N, n_dist_edges, n_val_edges, geom,
    )
    t_portable = _bench(portable_run, warmup, repeat_)
    Printf.@printf("sp2d portable kernel     %8.3f ms  [%s]\n", 1_000t_portable, mode_label)
    Printf.@printf("sp2d total (end-to-end)  %8.3f ms\n", 1_000t_sp2d)

    # --- reference: six-type sp1d same distance bins ---
    ws_sp1 = SFC.GPUSFWorkspace(backend, dist; kind = :single_pass)
    sums1 = CUDA.zeros(FT, 6, n_dist)
    counts1 = CUDA.zeros(UInt32, 6, n_dist)
    sp1_run = () -> SFC.calculate_structure_functions_single_pass!(
        sums1, counts1, x, u, dist; backend = gpu, workspace = ws_sp1,
    )
    t_sp1 = _bench(sp1_run, warmup, repeat_)
    Printf.@printf("sp1d (6 SF types)        %8.3f ms  [block-local (6, NB)]\n", 1_000t_sp1)

    gate_ok = t_sp2d < t_joint6
    Printf.@printf(
        "\nsp2d / joint_2d = %.1f×   sp2d / sp1d = %.1f×   sp2d < 6×joint = %s\n",
        t_sp2d / t_joint, t_sp2d / t_sp1, gate_ok ? "PASS" : "FAIL",
    )

    open(log_path, "a") do io
        println(io, "--- $(Dates.now()) ---")
        Printf.@printf(io,
            "device=%s N=%d n_dist=%d n_val=%d NB2=%d compile_exact=%d compile_max=%d dist_route=%s val_route=%s C=%d portable=%s\n",
            CUDA.name(CUDA.device()), N, n_dist, n_val, NB2, compile_exact, compile_max,
            dist_route, val_route, C, mode_label)
        Printf.@printf(io,
            "joint_exact=%.6f joint_max=%.6f joint6=%.6f sp2d=%.6f portable=%.6f sp1d=%.6f gate=%s\n",
            t_joint_exact, t_joint_max, t_joint6, t_sp2d, t_portable, t_sp1, gate_ok ? "PASS" : "FAIL")
    end
    println("\nLogged to ", log_path)
    println("Re-run production grid: N_DIST=50 N_VAL=50 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl")
    println("=" ^ 72)

    gate_ok || error("sp2d gate failed: sp2d ($(round(1_000t_sp2d; digits=1)) ms) >= 6×joint ($(round(1_000t_joint6; digits=1)) ms)")
end

main()
