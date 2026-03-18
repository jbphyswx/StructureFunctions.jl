using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, SpectralAnalysis as SFSA
using StaticArrays: StaticArrays as SA
using FINUFFT: FINUFFT
using FFTW: FFTW

Test.@testset "Inputs: Exhaustive Input Matrix Tests" begin
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
    xs_std, ys_std, _ = SyntheticData.generate_nonuniform_domain(
        N_side;
        R_mask = 0.0,
        lat_range = (-5.0, 5.0),
        lon_range = (0.0, 10.0),
    )
    u_std = SyntheticData.generate_spectral_field(
        xs_std,
        ys_std;
        peaks = peaks,
        noise_level = 0.0,
    )
    # DirectSum is our ground truth
    r_ref = SFSA.calculate_spectrum(
        SFSA.DirectSumBackend(),
        (xs_std, ys_std),
        (real(u_std),),
        ms;
        domain_size = (L, L),
    )
    c_ref, k_ref = r_ref

    Test.@testset "Coordinate Types" begin
        # Test 1: Tuple of StaticArrays
        # Use exact length N_total
        xs_s = SA.SVector{N_total, Float64}(xs_std)
        ys_s = SA.SVector{N_total, Float64}(ys_std)
        c, k = SFSA.calculate_spectrum(
            SFSA.DirectSumBackend(),
            (xs_s, ys_s),
            (real(u_std),),
            ms;
            domain_size = (L, L),
        )
        Test.@test c ≈ c_ref

        # Test 2: Views and SubArrays
        xs_v = view(xs_std, 1:N_total)
        ys_v = view(ys_std, 1:N_total)
        c, k = SFSA.calculate_spectrum(
            SFSA.DirectSumBackend(),
            (xs_v, ys_v),
            (real(u_std),),
            ms;
            domain_size = (L, L),
        )
        Test.@test c ≈ c_ref
    end

    Test.@testset "Field Types" begin
        # Test 1: Multiple fields (NU > 1)
        u2 = SyntheticData.generate_spectral_field(
            xs_std,
            ys_std;
            peaks = [(dk * 3, 0.0, 1.0)],
            noise_level = 0.0,
        )
        c, k = SFSA.calculate_spectrum(
            SFSA.DirectSumBackend(),
            (xs_std, ys_std),
            (real(u_std), real(u2)),
            ms;
            domain_size = (L, L),
        )
        Test.@test size(c, 3) == 2
        Test.@test c[:, :, 1] ≈ c_ref[:, :, 1]

        # exp(im * k * x) with forward transform e^(-im * k * x) peaks at +k
        c, k = SFSA.calculate_spectrum(
            SFSA.DirectSumBackend(),
            (xs_std, ys_std),
            (u_std,),
            ms;
            domain_size = (L, L),
        )
        Test.@test size(c) == (ms..., 1)

        # Look for peak at k_std = (2*dk, 1*dk)
        idx_pos = argmin(abs.(k[1] .- 2 * dk))
        idy_pos = argmin(abs.(k[2] .- 1 * dk))
        Test.@test abs(c[idx_pos, idy_pos, 1]) > 0.9

        # Check that -k has low power (leakage expected on scattered dots)
        idx_neg = argmin(abs.(k[1] .+ 2 * dk))
        idy_neg = argmin(abs.(k[2] .+ 1 * dk))
        Test.@test abs(c[idx_neg, idy_neg, 1]) < 0.02
    end

    Test.@testset "Backend Robustness with Views" begin
        # Check if FINUFFT handles views
        xs_v = view(xs_std, 1:N_total)
        ys_v = @view ys_std[1:N_total]
        u_v = view(real(u_std), 1:N_total)

        c_n, k_n = SFSA.calculate_spectrum(
            SFSA.FINUFFTBackend(),
            (xs_v, ys_v),
            (u_v,),
            ms;
            domain_size = (L, L),
            eps = 1e-12,
        )
        Test.@test isapprox(c_n, c_ref, atol = 1e-10)

        # Check if FFTW handles views
        N_u = 16
        # Match domain size exactly
        xs_u = range(0.0, stop = L * (N_u - 1) / N_u, length = N_u)
        ys_u = range(0.0, stop = L * (N_u - 1) / N_u, length = N_u)
        xv = vec([x for x in xs_u, y in ys_u])
        yv = vec([y for x in xs_u, y in ys_u])
        uv =
            real.(
                SyntheticData.generate_spectral_field(
                    xv,
                    yv;
                    peaks = peaks,
                    noise_level = 0.0,
                )
            )

        c_f, k_f = SFSA.calculate_spectrum(
            SFSA.FFTBackend(),
            (view(xv, :), view(yv, :)),
            (view(uv, :),),
            (N_u, N_u);
            domain_size = (L, L),
        )
        Test.@test size(c_f) == (N_u, N_u, 1)
    end
end
