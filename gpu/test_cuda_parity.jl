"""
    test_cuda_parity.jl

CUDA device parity: compare `gpu_calculate_structure_function` on `CUDABackend`
against the serial CPU reference.

Included by `runtests.jl`; not run from the main `test/` suite.
"""

using Test: Test
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges, LogBinEdges
using Random: Random
using StaticArrays: StaticArrays as SA

Random.seed!(42)

Test.@testset "CUDA structure-function parity" begin
    Test.@test CUDA.functional()

    N = 500
    FT = Float32
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)

    bin_edges = collect(FT, range(0.0, 1.4; length = 11))
    bin_tuples = SA.SVector{10, Tuple{FT, FT}}(
        [(bin_edges[i], bin_edges[i + 1]) for i in 1:(length(bin_edges) - 1)]...,
    )
    sft = SFT.L2SFType()
    x_tup = (x_cpu[1, :], x_cpu[2, :])
    u_tup = (u_cpu[1, :], u_cpu[2, :])

    res_ref = SFC.calculate_structure_function(
        sft, x_tup, u_tup, bin_tuples;
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )

    res_cuda = SFC.gpu_calculate_structure_function(
        sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges;
        return_sums_and_counts = true,
    )
    CUDA.synchronize()

    Test.@test res_cuda.counts ≈ res_ref.counts atol = 0.0

    max_Δ = maximum(abs, res_cuda.sums .- res_ref.sums)
    # Float32 GPU path uses atomic adds; order differs from serial CPU → small drift.
    Test.@test max_Δ < 0.05f0

    println("CUDA linear parity OK  max |Δ sums| = ", max_Δ)
end

Test.@testset "CUDA log-spaced bin parity" begin
    Test.@test CUDA.functional()

    N = 500
    FT = Float32
    x_cpu = rand(FT, 2, N) .+ FT(0.01)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)

    log_edge_vec = exp.(range(log(FT(0.05)), log(FT(1.4)); length = 11))
    bin_edges = LogBinEdges(log_edge_vec)
    bin_tuples = SA.SVector{10, Tuple{FT, FT}}(
        [(bin_edges[i], bin_edges[i + 1]) for i in 1:(length(log_edge_vec) - 1)]...,
    )
    sft = SFT.L2SFType()
    x_tup = (x_cpu[1, :], x_cpu[2, :])
    u_tup = (u_cpu[1, :], u_cpu[2, :])

    res_ref = SFC.calculate_structure_function(
        sft, x_tup, u_tup, bin_tuples;
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )

    res_cuda = SFC.gpu_calculate_structure_function(
        sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges;
        return_sums_and_counts = true,
    )
    CUDA.synchronize()

    Test.@test res_cuda.counts ≈ res_ref.counts atol = 0.0
    max_Δ = maximum(abs, res_cuda.sums .- res_ref.sums)
    Test.@test max_Δ < 0.05f0
    println("CUDA log-bin parity OK  max |Δ sums| = ", max_Δ)
end
