using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

Random.seed!(42)

Test.@testset "GPU Kernel Parity (KA.CPU)" begin
    # Small dataset suitable for N² kernel
    N = 50
    FT = Float64
    x = rand(FT, 2, N)    # (N_dims, N_points) layout
    u = rand(FT, 2, N)

    # Bin edges (monotone)
    bin_edges = collect(FT, range(0.0, 1.4, length = 11))   # 10 bins

    # CPU reference requires Tuple-pair format: [(lo, hi), ...]
    bin_tuples = SA.SVector{10, Tuple{FT, FT}}(
        [(bin_edges[i], bin_edges[i + 1]) for i in 1:(length(bin_edges) - 1)]...,
    )

    sft = SFT.L2SFType()

    # Create Tuple format for CPU reference
    x_tup = (x[1, :], x[2, :])
    u_tup = (u[1, :], u[2, :])

    # --- Reference: existing CPU implementation ---
    res_ref = SFC.calculate_structure_function(
        sft, x_tup, u_tup, bin_tuples;
        verbose = false, show_progress = false, return_sums_and_counts = true,
    )
    ref_vals = res_ref.sums
    ref_counts = res_ref.counts

    # --- GPU extension (CPU backend for parity test) ---
    res_gpu = SFC.gpu_calculate_structure_function(
        sft, KA.CPU(), x, u, bin_edges;
        return_sums_and_counts = true,
    )
    gpu_vals = res_gpu.sums
    gpu_counts = res_gpu.counts

    Test.@test gpu_counts ≈ ref_counts atol = 0.0
    Test.@test gpu_vals ≈ ref_vals atol = 1e-12

    println("Parity check passed!  max Δ = $(maximum(abs.(gpu_vals .- ref_vals)))")
end
