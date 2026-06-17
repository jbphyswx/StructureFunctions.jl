#!/usr/bin/env julia
"""
    benchmark_2d_grid_scaling.jl

Clarifies which 2D GPU paths are block-local vs global-atomic.

**Single-type joint 2D** (`gpu_calculate_structure_function_2d`):
  block-local when `n_dist * n_val ≤ 4096` (`_gpu_joint_2d_tiled_eligible`).

**Eight-type single-pass 2D** (`gpu_calculate_structure_functions_single_pass_2d!`):
  always global atomics into `(8, n_dist, n_val)` when tiled — bin grid size
  does NOT switch to block-local (see `TiledSinglePass2DKernels.jl` header).

Run on GPU:

    julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
    N_DIST=20 N_VAL=20 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl
"""

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using Printf: @printf
using Random: Random
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
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
    joint_eligible = n_dist * n_val <= 4096

    println("=" ^ 72)
    println("2D grid scaling — block-local vs global-atomic paths")
    println("Device: ", CUDA.name(CUDA.device()))
    @printf(
        "N=%d  n_dist=%d  n_val=%d (cells=%d)  joint_2d block-local eligible=%s\n",
        N, n_dist, n_val, n_dist * n_val, joint_eligible,
    )
    println("=" ^ 72)

    Random.seed!(42)
    x = rand(FT, 2, N) .* FT(50000)
    u = randn(FT, 2, N) .* FT(0.5)
    sft = SFT.L2SFType()
    val_edges = collect(FT, range(-1, 2; length = n_val + 1))

    # --- single-type joint 2D (has block-local path) ---
    ws_j = SFC.GPUSFWorkspace(backend, dist, val_edges; kind = :joint2d)
    j_run = () -> SFC.gpu_calculate_structure_function_2d(
        sft, backend, x, u, dist, val_edges; workspace = ws_j,
    )
    t_joint = _bench(j_run, warmup, repeat_)
    @printf("joint 2D (1 SF type)     %8.3f ms  [block-local when cells≤4096]\n", 1_000t_joint)

    # --- eight-type sp2d (global atomics always, when tiled) ---
    ws_sp = SFC.GPUSFWorkspace(backend, dist, value_bins; kind = :single_pass_2d)
    sums = zeros(FT, 8, n_dist, n_val)
    counts = zeros(UInt32, 8, n_dist, n_val)
    sp_run = () -> SFC.gpu_calculate_structure_functions_single_pass_2d!(
        sums, counts, backend, x, u, dist, value_bins; workspace = ws_sp,
    )
    t_sp2d = _bench(sp_run, warmup, repeat_)
    @printf("sp2d (8 SF types)         %8.3f ms  [global atomics — NOT block-local]\n", 1_000t_sp2d)

    # --- reference: 8-type sp1d same distance bins ---
    ws_sp1 = SFC.GPUSFWorkspace(backend, dist; kind = :single_pass)
    sums1 = zeros(FT, 8, n_dist)
    counts1 = zeros(UInt32, 8, n_dist)
    sp1_run = () -> SFC.calculate_structure_functions_single_pass!(
        sums1, counts1, x, u, dist; backend = gpu, workspace = ws_sp1,
    )
    t_sp1 = _bench(sp1_run, warmup, repeat_)
    @printf("sp1d (8 SF types)        %8.3f ms  [block-local (8, NB)]\n", 1_000t_sp1)

    @printf("\nsp2d / joint_2d = %.1f×   sp2d / sp1d = %.1f×\n", t_sp2d / t_joint, t_sp2d / t_sp1)
    println("\nRe-run production grid: N_DIST=50 N_VAL=50 julia --project=gpu gpu/benchmark_2d_grid_scaling.jl")
    println("=" ^ 72)
end

main()
