#!/usr/bin/env julia
"""
    benchmark_single_pass_2d_scaling.jl

One-hour (T=1) GPU timings for six-type single-pass 2D vs reference kernels.

Run inside a GPU allocation only:

    N=20000 julia --project=gpu gpu/benchmark_single_pass_2d_scaling.jl

Loads GPUExt the same way as ``gpu/benchmark_cuda.jl`` (CUDA + KernelAbstractions +
OhMyThreads + StructureFunctions).
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

function _synthetic_value_bins_ntuple(n_bins::Int, ::Type{FT}) where {FT <: AbstractFloat}
    template = InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_bins + 1)))
    return ntuple(_ -> template, 6)
end

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

function main()
    CUDA.functional() || error("CUDA not functional — run inside srun --gres=gpu:1")

    N = parse(Int, get(ENV, "N", "20000"))
    warmup = parse(Int, get(ENV, "WARMUP", "1"))
    repeat_ = parse(Int, get(ENV, "REPEAT", "3"))
    FT = get(ENV, "FT", "Float32") == "Float64" ? Float64 : Float32
    ka_backend = CUDA.CUDABackend()
    gpu_backend = SF.GPUBackend(ka_backend)

    println("=" ^ 72)
    println("Six-type single_pass_2d vs GPU reference kernels (T=1 hour each)")
    println("Device: ", CUDA.name(CUDA.device()))
    @printf("N=%d  dtype=%s  warmup=%d  repeat=%d\n", N, FT, warmup, repeat_)
    println("=" ^ 72)

    Random.seed!(42)
    dist_vec = LogBinEdges(Vector{FT}(exp.(range(log(FT(1000)), log(FT(50000)); length = 51))))
    value_bins = _synthetic_value_bins_ntuple(50, FT)
    n_dist = length(dist_vec) - 1
    n_val = length(value_bins[1]) - 1

    x2 = rand(FT, 2, N) .* FT(50000)
    u1 = randn(FT, 2, N) .* FT(0.5)
    x_batch = repeat(x2, 1, 1, 1)
    u_batch = reshape(u1, 2, N, 1)

    # --- Production-style path: in-place slice batch, kernel only ---
    ws_sp2d = SFC.GPUSFWorkspace(ka_backend, dist_vec, value_bins; kind = :single_pass_2d)
    sums_sp2d = zeros(FT, 6, n_dist, n_val, 1)
    counts_sp2d = zeros(Int64, 6, n_dist, n_val, 1)
    sp2d_run = () -> SFC.calculate_structure_functions_single_pass_2d_batch!(
        sums_sp2d, counts_sp2d, x_batch, u_batch, dist_vec, value_bins;
        backend = gpu_backend, workspace = ws_sp2d,
    )
    t_sp2d = _bench(sp2d_run, warmup, repeat_)
    @printf("\n[prod]  single_pass_2d_slices!     %8.3f s   (%8.1f ms/hour)\n", t_sp2d, t_sp2d * 1000)

    # --- 1D L2SF tiled reference ---
    bins_1d = collect(range(FT(0), FT(1.5); length = 21))
    sft = SFT.L2SFType()
    ws_l2 = SFC.GPUSFWorkspace(ka_backend, bins_1d)
    l2_run = () -> SFC.gpu_calculate_structure_function(
        sft, ka_backend, x2, u1, bins_1d; workspace = ws_l2,
    )
    t_l2 = _bench(l2_run, warmup, repeat_)
    @printf("[ref]   1D L2SF tiled kernel        %8.4f s   (%8.2f ms)\n", t_l2, t_l2 * 1000)

    # --- Six-type single-pass 1D: kernel-only (in-place) ---
    ws_sp1 = SFC.GPUSFWorkspace(ka_backend, dist_vec; kind = :single_pass)
    sums_sp1 = zeros(FT, 6, n_dist)
    counts_sp1 = zeros(Int64, 6, n_dist)
    sp1_kernel_run = () -> SFC.calculate_structure_functions_single_pass!(
        sums_sp1, counts_sp1, x2, u1, dist_vec;
        backend = gpu_backend, workspace = ws_sp1,
    )
    t_sp1_kernel = _bench(sp1_kernel_run, warmup, repeat_)
    @printf("[ref]   6-type sp1d kernel (!)      %8.3f s   (%8.1f ms)\n", t_sp1_kernel, t_sp1_kernel * 1000)

    # --- Six-type single-pass 1D: full allocating API (+ Helmholtz append) ---
    sp1_full_run = () -> SFC.calculate_structure_functions_single_pass(
        x2, u1, dist_vec; backend = gpu_backend, workspace = ws_sp1,
    )
    t_sp1_full = _bench(sp1_full_run, warmup, repeat_)
    @printf("[ref]   6-type sp1d full API        %8.3f s   (%8.1f ms)\n", t_sp1_full, t_sp1_full * 1000)

    @printf(
        "\nRatios at N=%d:\n  sp2d / L2SF tiled:        %.0f×\n  sp2d / sp1d kernel:       %.1f×\n  sp1d full / sp1d kernel:  %.1f×\n",
        N, t_sp2d / t_l2, t_sp2d / t_sp1_kernel, t_sp1_full / t_sp1_kernel,
    )
    println(
        "\nNote: single-pass 1D/2D use tiled128 pair traversal when NB ≤ 64. ",
        "2D uses global atomics into (6, n_dist, n_val); 1D uses block-local (6, NB).",
    )

    SFC.release!(ws_sp2d)
    SFC.release!(ws_l2)
    SFC.release!(ws_sp1)
    println()
    return nothing
end

main()
