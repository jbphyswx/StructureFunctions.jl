using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions, SpectralAnalysis as SFSA
using Random: Random
using LinearAlgebra: LinearAlgebra

Random.seed!(42)

Test.@testset "Spectral GPU Parity (KA.CPU)" begin
    # Small 2D dataset
    N = 100
    FT = Float64
    x = (rand(FT, N), rand(FT, N))
    u = (rand(FT, N), rand(FT, N))

    ms = (8, 8)
    iflag = 1

    # --- Reference: existing CPU implementation ---
    ref_coeffs, ref_ks = SFSA.calculate_spectrum(
        x, u, ms;
        backend = SFSA.DirectSumBackend(),
        iflag = iflag,
    )

    # --- GPU extension (CPU backend) ---
    gpu_coeffs, gpu_ks = SFSA.gpu_calculate_spectrum(
        KA.CPU(), x, u, ms;
        iflag = iflag,
    )

    Test.@test gpu_coeffs ≈ ref_coeffs atol = 1e-12
    Test.@test gpu_ks[1] == ref_ks[1]
    Test.@test gpu_ks[2] == ref_ks[2]

    println("Spectral GPU Parity check passed!")
end
