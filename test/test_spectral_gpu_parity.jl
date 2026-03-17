using Test
using KernelAbstractions: CPU
using StructureFunctions
using StructureFunctions.SpectralAnalysis
using Random
using LinearAlgebra

Random.seed!(42)

@testset "Spectral GPU Parity (KA.CPU)" begin
    # Small 2D dataset
    N = 100
    FT = Float64
    x = (rand(FT, N), rand(FT, N))
    u = (rand(FT, N), rand(FT, N))
    
    ms = (8, 8)
    iflag = 1
    
    # --- Reference: existing CPU implementation ---
    ref_coeffs, ref_ks = calculate_spectrum(
        x, u, ms;
        backend = DirectSumBackend(),
        iflag = iflag
    )
    
    # --- GPU extension (CPU backend) ---
    gpu_coeffs, gpu_ks = gpu_calculate_spectrum(
        CPU(), x, u, ms;
        iflag = iflag
    )
    
    @test gpu_coeffs ≈ ref_coeffs atol=1e-12
    @test gpu_ks[1] == ref_ks[1]
    @test gpu_ks[2] == ref_ks[2]
    
    println("Spectral GPU Parity check passed!")
end
