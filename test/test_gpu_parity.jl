using Test
using KernelAbstractions: CPU
using StructureFunctions
using StructureFunctions.Calculations: gpu_calculate_structure_function
using Random
using LinearAlgebra

Random.seed!(42)

@testset "GPU Kernel Parity (KA.CPU)" begin
    # Small dataset suitable for N² kernel
    N   = 50
    FT  = Float64
    x   = rand(FT, 2, N)    # (N_dims, N_points) layout
    u   = rand(FT, 2, N)

    # Bin edges (monotone)
    bin_edges  = collect(FT, range(0.0, 1.4, length = 11))   # 10 bins

    # CPU reference requires Tuple-pair format: [(lo, hi), ...]
    bin_tuples = [(bin_edges[i], bin_edges[i+1]) for i in 1:length(bin_edges)-1]

    sft = StructureFunctions.LongitudinalSecondOrderStructureFunction()

    # --- Reference: existing CPU implementation (tuple-of-vectors API) ---
    # return_sums_and_counts=true → ((output, counts), distance_bins)
    x_tup = Tuple(x[i, :] for i in 1:2)
    u_tup = Tuple(u[i, :] for i in 1:2)

    (ref_vals, ref_counts), _ = calculate_structure_function(
        x_tup, u_tup, bin_tuples, sft;
        verbose = false, show_progress = false, return_sums_and_counts = true,
    )

    # --- GPU extension (CPU backend for parity test) ---
    gpu_vals, gpu_counts = gpu_calculate_structure_function(
        CPU(), x, u, bin_edges, sft,
    )

    @test gpu_counts ≈ ref_counts  atol=0.0
    @test gpu_vals   ≈ ref_vals    atol=1e-12

    println("Parity check passed!  max Δ = $(maximum(abs.(gpu_vals .- ref_vals)))")
end
