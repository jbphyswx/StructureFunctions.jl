using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random
using LinearAlgebra: LinearAlgebra as LA

Random.seed!(42)

Test.@testset "GPU Kernel Parity (KA.CPU)" begin
    # Small dataset suitable for N² kernel
    N   = 50
    FT  = Float64
    x   = rand(FT, 2, N)    # (N_dims, N_points) layout
    u   = rand(FT, 2, N)

    # Bin edges (monotone)
    bin_edges  = collect(FT, range(0.0, 1.4, length = 11))   # 10 bins

    # CPU reference requires Tuple-pair format: [(lo, hi), ...]
    bin_tuples = [(bin_edges[i], bin_edges[i+1]) for i in 1:length(bin_edges)-1]

    sft = SF.StructureFunctionTypes.LongitudinalSecondOrderStructureFunction()

    # --- Reference: existing CPU implementation (tuple-of-vectors API) ---
    res_ref = SFC.calculate_structure_function(
        sft, x_tup, u_tup, bin_tuples;
        verbose = false, show_progress = false, return_sums_and_counts = true,
    )
    ref_vals = res_ref.values
    ref_counts = res_ref.counts

    # --- GPU extension (CPU backend for parity test) ---
    res_gpu = SFC.gpu_calculate_structure_function(
        sft, KA.CPU(), x, u, bin_edges;
        return_sums_and_counts = true,
    )
    gpu_vals = res_gpu.values
    gpu_counts = res_gpu.counts

    Test.@test gpu_counts ≈ ref_counts  atol=0.0
    Test.@test gpu_vals   ≈ ref_vals    atol=1e-12

    println("Parity check passed!  max Δ = $(maximum(abs.(gpu_vals .- ref_vals)))")
end
