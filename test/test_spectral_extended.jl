using Test: Test
using Random: Random
using Statistics: Statistics
using StructureFunctions: StructureFunctions as SF, SpectralAnalysis as SFSA
using FINUFFT: FINUFFT
using FFTW: FFTW
using CairoMakie: CairoMakie as CM



Test.@testset "Spectral Recovery: Tiered Verification" begin
    # Settings
    N_side_gen = 64 # High spatial density to avoid aliasing artifacts
    ms = (32, 32)     # Moderate spectral resolution for large, visible pixels
    L = 10.0
    # Use frequencies commensurate with L=10.0 to avoid FFT leakage for parity check
    dk = 2π / L
    peaks = [(2*dk, 1*dk, 1.0), (4*dk, -3*dk, 1.0)] # Equal amplitudes for clarity
    FT = Float64

    Test.@testset "Uniform Grid Bit-Parity" begin
        # 1. Perfectly uniform grid (N points covering [0, L-dx])
        dx = L / ms[1]
        xs_u = range(0.0, stop=L-dx, length=ms[1])
        ys_u = range(0.0, stop=L-dx, length=ms[2])
        
        # StructureFunctions API expects (x_vec, y_vec, ...)
        xv = vec([x for x in xs_u, y in ys_u])
        yv = vec([y for x in xs_u, y in ys_u])
        
        # Analytic signal
        u = SyntheticData.generate_spectral_field(xv, yv; peaks=peaks, noise_level=0.0)
        
        # Backends (ms is (N_side, N_side))
        L = 10.0
        r_direct = SFSA.calculate_spectrum(SFSA.DirectSumBackend(), (xv, yv), (real(u),), ms; domain_size=(L, L))
        r_nufft  = SFSA.calculate_spectrum(SFSA.FINUFFTBackend(), (xv, yv), (real(u),), ms; domain_size=(L, L), eps=1e-13)
        r_fft    = SFSA.calculate_spectrum(SFSA.FFTBackend(), (xv, yv), (real(u),), ms; domain_size=(L, L))
        
        c_direct, k_direct = r_direct
        c_nufft, k_nufft   = r_nufft
        c_fft, k_fft       = r_fft

        # Bit-parity check
        Test.@test isapprox(c_direct, c_nufft, atol=1e-12)
        Test.@test isapprox(c_direct, c_fft, atol=1e-12)
        Test.@test all(isapprox(k_direct[i], k_nufft[i], rtol=1e-12) for i in 1:2)
        Test.@test all(isapprox(k_direct[i], k_fft[i], rtol=1e-12) for i in 1:2)
        
        # Consolidate Diagnostic Plot
        sym_peaks = vcat(peaks, [(-p[1], -p[2], p[3]) for p in peaks])
        fig = SFSA.compare_spectral_analysis(((xv, yv), (real(u),)), [
            "Direct Sum" => r_direct,
            "FINUFFT"    => r_nufft,
            "FFTW (FFT)" => r_fft
        ]; peaks=sym_peaks, title="Uniform Grid Parity Comparison (u=real, noise=0.0)")
        CM.save("test/plots/verify_spectral_parity_uniform_grid.png", fig)
    end

    Test.@testset "Scattered Grid Parity (Noisy)" begin
        # 2. Non-uniform domain with side-loaded mask
        # Force domain to match L=10.0 analytic signal
        xs, ys, mask = SyntheticData.generate_nonuniform_domain(N_side_gen; R_mask=5.0, lat_range=(-5.0, 5.0), lon_range=(0.0, 10.0))
        u = SyntheticData.generate_spectral_field(xs, ys; peaks=peaks, noise_level=0.05)
        
        # Backends
        r_direct = SFSA.calculate_spectrum(SFSA.DirectSumBackend(), (xs, ys), (real(u),), ms; domain_size=(L, L))
        r_nufft  = SFSA.calculate_spectrum(SFSA.FINUFFTBackend(), (xs, ys), (real(u),), ms; eps=1e-12, domain_size=(L, L))
        
        c_direct, k_direct = r_direct
        c_nufft, k_nufft   = r_nufft
        
        # Verify peaks exist at the right locations in k-space
        idx1 = argmin(abs.(k_direct[1] .- 2*dk))
        idx2 = argmin(abs.(k_direct[2] .- 1*dk))
        
        Test.@test abs(c_nufft[idx1, idx2, 1]) > Statistics.mean(abs.(c_nufft)) * 5.0
        Test.@test isapprox(c_direct, c_nufft, atol=1e-10)
        
        # Mark both k and -k peaks for real(u)
        sym_peaks = vcat(peaks, [(-p[1], -p[2], p[3]) for p in peaks])
        fig = SFSA.compare_spectral_analysis(((xs, ys), (real(u),)), [
            "Direct Sum (Scattered)" => r_direct,
            "FINUFFT (Scattered)"    => r_nufft
        ]; peaks=sym_peaks, title="Scattered Grid Parity (u=real, noise=0.05)")
        CM.save("test/plots/verify_spectral_parity_scattered_grid.png", fig)
    end
    
    Test.@testset "Sampling Validation (High-res FFT vs Scattered NUFFT)" begin
        # Generate perfect high-res signal
        N_high = 128
        xh, yh, uh = SyntheticData.generate_highres_uniform(N_high; lat_range=(-5.0, 5.0), lon_range=(0.0, 10.0), peaks=peaks)
        
        # Sample it non-uniformly with higher density to avoid aliasing artifacts in plots
        # Aliasing at k_samp = N * dk. For N=20, k_samp ~ 12.5 (visible). For N=64, k_samp ~ 40 (off-plot).
        N_dense = 64
        xs, ys, _ = SyntheticData.generate_nonuniform_domain(N_dense; R_mask=0.0, lat_range=(-5.0, 5.0), lon_range=(0.0, 10.0))
        us = SyntheticData.generate_spectral_field(xs, ys; peaks=peaks, noise_level=0.0)
        
        # NUFFT of scattered samples
        r_nufft  = SFSA.calculate_spectrum(SFSA.FINUFFTBackend(), (xs, ys), (real(us),), ms; eps=1e-12, domain_size=(L,L))
        # Direct Sum ground truth
        r_direct = SFSA.calculate_spectrum(SFSA.DirectSumBackend(), (xs, ys), (real(us),), ms; domain_size=(L,L))
        
        c_nufft, k_nufft   = r_nufft
        c_direct, k_direct = r_direct
        
        Test.@test isapprox(c_nufft, c_direct, atol=1e-10)
        # Peak recovery check
        idx1 = argmin(abs.(k_nufft[1] .- 2*dk))
        idx2 = argmin(abs.(k_nufft[2] .- 1*dk))
        Test.@test abs(c_nufft[idx1, idx2, 1]) > 0.45 # Dominant peak
        
        sym_peaks = vcat(peaks, [(-p[1], -p[2], p[3]) for p in peaks])
        fig = SFSA.compare_spectral_analysis(((xs, ys), (real(us),)), [
            "NUFFT (Scattered Samples)" => r_nufft,
            "Direct Sum (Ground Truth)" => r_direct
        ]; peaks=sym_peaks, title="Sampling Validation (Scattered vs Truth)")
        CM.save("test/plots/verify_spectral_sampling_validation.png", fig)
    end
end
