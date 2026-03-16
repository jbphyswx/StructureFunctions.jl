using Test
using Random
using StructureFunctions
using StructureFunctions.SpectralAnalysis
using StaticArrays
using FINUFFT, FFTW

include("test_synthetic_data.jl")
using .SyntheticData

@testset "Phase 6: Exhaustive Input Matrix Tests" begin
    # Base Settings
    N = 100
    ms = (16, 16)
    L = 10.0
    dk = 2π / L
    peaks = [(dk * 2, dk * 1, 1.0)]
    
    # 1. Standard Vectors (Ground Truth)
    # generate_nonuniform_domain(N_side) returns N_side^2 points if no mask
    N_side = 10
    N_total = N_side^2
    xs_std, ys_std, _ = generate_nonuniform_domain(N_side; R_mask=0.0, lat_range=(-5.0, 5.0), lon_range=(0.0, 10.0))
    u_std = generate_spectral_field(xs_std, ys_std; peaks=peaks, noise_level=0.0)
    # DirectSum is our ground truth
    r_ref = calculate_spectrum(DirectSumBackend(), (xs_std, ys_std), (real(u_std),), ms; domain_size=(L,L))
    c_ref, k_ref = r_ref

    @testset "Coordinate Types" begin
        # Test 1: Tuple of StaticArrays
        # Use exact length N_total
        xs_s = SVector{N_total, Float64}(xs_std)
        ys_s = SVector{N_total, Float64}(ys_std)
        c, k = calculate_spectrum(DirectSumBackend(), (xs_s, ys_s), (real(u_std),), ms; domain_size=(L,L))
        @test c ≈ c_ref
        
        # Test 2: Views and SubArrays
        xs_v = view(xs_std, 1:N_total)
        ys_v = view(ys_std, 1:N_total)
        c, k = calculate_spectrum(DirectSumBackend(), (xs_v, ys_v), (real(u_std),), ms; domain_size=(L,L))
        @test c ≈ c_ref
    end

    @testset "Field Types" begin
        # Test 1: Multiple fields (NU > 1)
        u2 = generate_spectral_field(xs_std, ys_std; peaks=[(dk*3, 0.0, 1.0)], noise_level=0.0)
        c, k = calculate_spectrum(DirectSumBackend(), (xs_std, ys_std), (real(u_std), real(u2)), ms; domain_size=(L,L))
        @test size(c, 3) == 2
        @test c[:,:,1] ≈ c_ref[:,:,1]

        # exp(im * k * x) with forward transform e^(-im * k * x) peaks at +k
        c, k = calculate_spectrum(DirectSumBackend(), (xs_std, ys_std), (u_std,), ms; domain_size=(L,L))
        @test size(c) == (ms..., 1)
        
        # Look for peak at k_std = (2*dk, 1*dk)
        idx_pos = argmin(abs.(k[1] .- 2*dk))
        idy_pos = argmin(abs.(k[2] .- 1*dk))
        @test abs(c[idx_pos, idy_pos, 1]) > 0.9
        
        # Check that -k has low power (leakage expected on scattered dots)
        idx_neg = argmin(abs.(k[1] .+ 2*dk))
        idy_neg = argmin(abs.(k[2] .+ 1*dk))
        @test abs(c[idx_neg, idy_neg, 1]) < 0.02
    end

    @testset "Backend Robustness with Views" begin
        # Check if FINUFFT handles views
        xs_v = view(xs_std, 1:N_total)
        ys_v = @view ys_std[1:N_total]
        u_v  = view(real(u_std), 1:N_total)
        
        c_n, k_n = calculate_spectrum(FINUFFTBackend(), (xs_v, ys_v), (u_v,), ms; domain_size=(L,L), eps=1e-12)
        @test isapprox(c_n, c_ref, atol=1e-10)
        
        # Check if FFTW handles views
        N_u = 16
        # Match domain size exactly
        xs_u = range(0.0, stop=L * (N_u-1)/N_u, length=N_u)
        ys_u = range(0.0, stop=L * (N_u-1)/N_u, length=N_u)
        xv = vec([x for x in xs_u, y in ys_u])
        yv = vec([y for x in xs_u, y in ys_u])
        uv = real.(generate_spectral_field(xv, yv; peaks=peaks, noise_level=0.0))
        
        c_f, k_f = calculate_spectrum(FFTBackend(), (view(xv,:), view(yv,:)), (view(uv,:),), (N_u, N_u); domain_size=(L,L))
        @test size(c_f) == (N_u, N_u, 1)
    end
end
