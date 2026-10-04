using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random

Random.seed!(42)

Test.@testset "GPU Kernel Parity (KA.CPU)" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)    # (N_dims, N_points) layout
    u = rand(FT, 2, N)

    bin_edges = collect(FT, range(0.0, 1.4, length = 11))   # 10 bins

    sft = SFT.L2SFType()

    res_ref = SFC.calculate_structure_function(
        sft, x, u, bin_edges, SF.StructureFunctionSumsAndCounts,
    )
    ref_vals = res_ref.sums
    ref_counts = res_ref.counts

    res_gpu = SFC.calculate_structure_function(
        sft, x, u, bin_edges, SF.StructureFunctionSumsAndCounts;
        backend = CB.GPUBackend(KA.CPU()),
    )
    gpu_vals = res_gpu.sums
    gpu_counts = res_gpu.counts

    Test.@test gpu_counts == ref_counts
    Test.@test gpu_vals ≈ ref_vals atol = 1e-12
end
