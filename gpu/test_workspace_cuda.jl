"""
    test_workspace_cuda.jl

CUDA workspace + slice-batch parity (run on a GPU node / SLURM allocation).

    julia --project=. gpu/test_workspace_cuda.jl

Compares fresh-alloc vs `GPUSFWorkspace` and slice drivers on `CUDA.CUDABackend()`.
CPU reference uses serial `calculate_structure_function` on host arrays.
"""

using Test: Test
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

Random.seed!(42)

Test.@testset "CUDA GPUSFWorkspace & slices" begin
    Test.@test CUDA.functional()

    backend = CUDA.CUDABackend()
    N = 500
    T = 4
    FT = Float32
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)
    bins = collect(FT, range(0.0f0, 1.4f0; length = 11))
    NB = length(bins) - 1
    sft = SFT.L2SFType()

    ref = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, bins;
        return_sums_and_counts = true, verbose = false, show_progress = false,
    )

    res_fresh = SFC.gpu_calculate_structure_function(
        sft, backend, x_gpu, u_gpu, bins; return_sums_and_counts = true,
    )
    CUDA.synchronize()
    Test.@test res_fresh.counts ≈ ref.counts atol = 0.0
    max_Δ_fresh = maximum(abs, res_fresh.sums .- ref.sums)
    # Float32 GPU atomics vs serial CPU — same tolerance as test_cuda_parity.jl
    Test.@test max_Δ_fresh < 0.05f0

    ws = SFC.GPUSFWorkspace(backend, bins)
    res_ws = SFC.gpu_calculate_structure_function(
        sft, backend, x_gpu, u_gpu, bins; return_sums_and_counts = true, workspace = ws,
    )
    CUDA.synchronize()
    Test.@test res_ws.counts ≈ ref.counts atol = 0.0
    max_Δ_ws = maximum(abs, res_ws.sums .- ref.sums)
    Test.@test max_Δ_ws < 0.05f0

    # repeated-call accumulation with workspace
    sums_acc = zeros(Float64, NB)
    counts_acc = zeros(UInt32, NB)
    for _ in 1:3
        SFC.gpu_calculate_structure_function!(
            sums_acc, counts_acc, sft, backend, x_gpu, u_gpu, bins; workspace = ws,
        )
    end
    CUDA.synchronize()
    Test.@test counts_acc ≈ 3 .* ref.counts
    max_Δ_acc = maximum(abs, sums_acc .- 3 .* ref.sums)
    Test.@test max_Δ_acc < 0.15f0

    # slice batch on device-resident (N_dims, N_points, T)
    x_batch_cpu = rand(FT, 2, N, T)
    u_batch_cpu = rand(FT, 2, N, T)
    x_batch = CUDA.cu(x_batch_cpu)
    u_batch = CUDA.cu(u_batch_cpu)

    sums_ref = zeros(Float64, NB, T)
    counts_ref = zeros(UInt32, NB, T)
    for t in 1:T
        ref_t = SFC.gpu_calculate_structure_function(
            sft, backend, x_batch_cpu[:, :, t], u_batch_cpu[:, :, t], bins;
            return_sums_and_counts = true,
        )
        CUDA.synchronize()
        sums_ref[:, t] .= ref_t.sums
        counts_ref[:, t] .= ref_t.counts
    end

    sums_drv = zeros(Float64, NB, T)
    counts_drv = zeros(UInt32, NB, T)
    ws_slice = SFC.GPUSFWorkspace(backend, bins)
    SFC.gpu_calculate_structure_function_slices!(
        sums_drv, counts_drv, sft, backend, x_batch, u_batch, bins;
        workspace = ws_slice,
    )
    CUDA.synchronize()
    max_Δ_slice = maximum(abs, sums_drv .- sums_ref)
    Test.@test max_Δ_slice < 0.05f0
    Test.@test counts_drv ≈ counts_ref

    SFC.release!(ws)
    SFC.release!(ws_slice)
    println("CUDA workspace tests passed on ", CUDA.name(CUDA.device()))
end
