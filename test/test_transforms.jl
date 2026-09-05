using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using Bessels: Bessels
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using LinearAlgebra: LinearAlgebra
using Random: Random
using Test: Test

# A periodic field of a few random modes, with its mean removed.
function _modal_field(dims, dx, D; seed = 90, nmodes = 6)
    Random.seed!(seed)
    f = zeros(dims...)
    for _ in 1:nmodes
        mv = ntuple(_ -> rand(1:5), D)
        a = randn()
        ph = 2π * rand()
        for I in CartesianIndices(dims)
            f[I] += a * cos(sum(mv[d] * (I[d] - 1) * dx for d in 1:D) + ph)
        end
    end
    f .-= sum(f) / prod(dims)
    return f
end

Test.@testset "the isotropic kernel is the angular average of cos(k·r)" begin
    for x in (0.0, 0.3, 0.7, 2.5, 11.0)
        Test.@test SFC.isotropic_kernel(Val(1), x) == cos(x)
        Test.@test SFC.isotropic_kernel(Val(2), x) == Bessels.besselj0(x)
    end
    for x in (0.3, 0.7, 2.5, 11.0)
        Test.@test SFC.isotropic_kernel(Val(3), x) ≈ sin(x) / x
    end
    # every kernel is 1 at zero separation, so S₂(0) = 0 for any mode
    for D in (1, 2, 3)
        Test.@test SFC.isotropic_kernel(Val(D), 0.0) ≈ 1.0
        Test.@test SFC.isotropic_kernel(Val(D), 1e-10) ≈ 1.0
    end
    Test.@test_throws ArgumentError SFC.isotropic_kernel(Val(5), 0.7)

    Test.@test SFC.solid_angle(Val(1)) == 2
    Test.@test SFC.solid_angle(Val(2)) ≈ 2π
    Test.@test SFC.solid_angle(Val(3)) ≈ 4π
end

Test.@testset "a single mode has the kernel as its structure function" begin
    # S₂(r) = A²[1 − Λ_D(k₀r)] after angular averaging, which is what makes Λ_D the transform kernel
    A, k0 = 1.3, 2.7
    for D in (1, 2, 3)
        for r in (0.4, 1.1, 3.0)
            s2 = A^2 * (1 - SFC.isotropic_kernel(Val(D), k0 * r))
            Test.@test s2 >= 0
            Test.@test s2 <= 2 * A^2 + 1e-12
        end
    end
end

Test.@testset "only the second-order trace inverts to a spectrum" begin
    Test.@test SFC.assert_invertible(SFT.S2SFType()) === nothing

    # a single projection carries about half the trace, so inverting it would silently halve E(k)
    for op in (SFT.L2SFType(), SFT.T2SFType())
        err = Test.@test_throws ArgumentError SFC.assert_invertible(op)
        Test.@test occursin("trace", err.value.msg)
        Test.@test occursin("isotropy", err.value.msg)
    end

    for op in (SFT.L3SFType(), SFT.L1T2SFType())
        err = Test.@test_throws ArgumentError SFC.assert_invertible(op)
        Test.@test occursin("flux", err.value.msg)
    end
end

Test.@testset "the transform recovers prescribed spectral lines" begin
    # C(r) = Σ (A²/2)cos(kr) gives S₂(r) = Σ A²(1 − cos(kr)), whose spectrum is a line at each k
    # with weight proportional to A².
    ks = (3.0, 7.0, 12.0)
    As = (1.0, 0.6, 0.3)
    asym = sum(a^2 for a in As)
    r = collect(range(0.0, 60.0; length = 6000))
    s2 = [sum(As[m]^2 * (1 - cos(ks[m] * rr)) for m in eachindex(ks)) for rr in r]
    kq = collect(range(0.5, 20.0; length = 400))

    P = SFC.isotropic_spectrum(SFT.S2SFType(), r, s2, kq, Val(1); asymptote = asym)
    Test.@test length(P) == length(kq)
    Test.@test all(isfinite, P)

    # each line lands on its own wavenumber
    for k in ks
        window = abs.(kq .- k) .< 0.4
        Test.@test kq[argmax(P .* window)] ≈ k atol = 0.15
    end

    # and their relative strengths follow A², to the accuracy a finite r range allows
    peaks = [maximum(P[abs.(kq .- k) .< 0.4]) for k in ks]
    expected = [a^2 for a in As]
    Test.@test peaks ./ peaks[1] ≈ expected ./ expected[1] rtol = 0.15

    # the asymptote sets the k = 0 content, which the transform cannot report anyway, so getting it
    # wrong may rescale the lines but must not move them
    P_off = SFC.isotropic_spectrum(SFT.S2SFType(), r, s2, kq, Val(1); asymptote = 1.4 * asym)
    Test.@test P_off != P
    for k in ks
        window = abs.(kq .- k) .< 0.4
        Test.@test kq[argmax(P_off .* window)] ≈ k atol = 0.15
    end
end

Test.@testset "the transform reproduces an analytic spectrum" begin
    # A Gaussian correlation C(r) = σ² exp(-r²/2ℓ²) has S₂(r) = 2σ²[1 - exp(-r²/2ℓ²)] and the
    # closed-form density σ² ℓ^D exp(-k²ℓ²/2) / (2π)^(D/2) under the convention that the density
    # integrates over d^D k to the variance. Unlike a discrete line it decays, so the transform is
    # not truncation-limited and the comparison is pointwise.
    # Resolution is set by quadrature error, which falls as 1/length(r): these sizes give ~1e-3
    # pointwise in a quarter of a second, while a wrong kernel or a missing (2π)^D is wrong by
    # orders of magnitude, so the assertions still discriminate with room to spare.
    σ2, ℓ = 1.7, 0.8
    for D in (1, 2, 3)
        r = collect(range(0.0, 20ℓ; length = 8_000))
        s2 = @. 2σ2 * (1 - exp(-r^2 / (2ℓ^2)))
        kq = collect(range(1e-4, 40 / ℓ; length = 800))

        P = SFC.isotropic_spectrum(SFT.S2SFType(), r, s2, kq, Val(D); asymptote = 2σ2)
        exact = @. σ2 * ℓ^D * exp(-kq^2 * ℓ^2 / 2) / (2π)^(D / 2)
        Test.@test maximum(abs, P .- exact) / maximum(exact) < 3e-3
        Test.@test all(>(-1e-3 * maximum(exact)), P)          # a density is non-negative

        # the convention: the shell spectrum integrates over k to the variance
        E = SFC.shell_spectrum(P, kq, Val(D))
        Test.@test sum(E) * (kq[2] - kq[1]) ≈ σ2 rtol = 0.10
    end

    # the width scales inversely, so a wider correlation is a narrower spectrum
    r = collect(range(0.0, 30.0; length = 6_000))
    kq = collect(range(1e-3, 20.0; length = 800))
    widths = Float64[]
    for ℓ in (0.5, 1.0, 2.0)
        s2 = @. 2 * (1 - exp(-r^2 / (2ℓ^2)))
        P = SFC.isotropic_spectrum(SFT.S2SFType(), r, s2, kq, Val(3); asymptote = 2.0)
        half = findlast(>(maximum(P) / 2), P)
        push!(widths, kq[half])
    end
    Test.@test issorted(widths; rev = true)
    Test.@test widths[1] / widths[3] ≈ 4.0 rtol = 0.05
end

Test.@testset "the transform refuses what it cannot answer" begin
    r = collect(range(0.0, 10.0; length = 100))
    s2 = @. 1 - cos(2.0 * r)
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(
        SFT.S2SFType(), r, s2, [0.0, 1.0], Val(1))
    Test.@test_throws DimensionMismatch SFC.isotropic_spectrum(
        SFT.S2SFType(), r, s2[1:end-1], [1.0], Val(1))
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(
        SFT.S2SFType(), reverse(r), s2, [1.0], Val(1))
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(
        SFT.L2SFType(), r, s2, [1.0], Val(1))
end

Test.@testset "a result object transforms back to the wavenumber it was built from" begin
    # End to end: scattered points -> the package's own S₂ -> the transform. A single mode of
    # wavenumber k₀ must put the spectral peak at k₀.
    A = 1.0
    for (D, mv) in ((2, (5, 2)), (3, (4, 2, 1)))
        k0 = sqrt(sum(abs2, mv))
        Random.seed!(500 + D)
        N = 8000
        x = 2π .* rand(D, N)
        u = zeros(D, N)
        for p in 1:N
            u[1, p] = A * cos(sum(mv[d] * x[d, p] for d in 1:D) + 0.4)
        end
        bins = collect(range(0.0, 3.0; length = 61))
        kq = collect(range(0.3, 12.0; length = 500))

        raw = SFC.calculate_structure_function(
            SFT.S2SFType(), x, u, bins; backend = CB.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false,
            show_progress = false)
        P = SFC.isotropic_spectrum(raw, kq, Val(D); asymptote = A^2)
        Test.@test all(isfinite, P)
        Test.@test kq[argmax(P)] ≈ k0 rtol = 0.05

        # the averaged object carries the same information, so it must give the same answer
        avg = SFC.calculate_structure_function(
            SFT.S2SFType(), x, u, bins; backend = CB.SerialBackend(), verbose = false,
            show_progress = false)
        Test.@test SFC.isotropic_spectrum(avg, kq, Val(D); asymptote = A^2) ≈ P
    end
end

Test.@testset "empty bins are dropped, not carried as NaN" begin
    # A bin holding no pair averages to NaN, which would otherwise propagate through the quadrature
    # into every wavenumber.
    edges = collect(range(0.0, 10.0; length = 21))
    sums = [Float64(i) for i in 1:20]
    counts = fill(UInt32(5), 20)
    counts[3] = 0
    counts[11] = 0
    sums[3] = 0.0
    sums[11] = 0.0
    raw = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, sums, counts)
    kq = collect(range(0.5, 5.0; length = 50))
    P = SFC.isotropic_spectrum(raw, kq, Val(3); asymptote = 20.0)
    Test.@test all(isfinite, P)

    allempty = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, zeros(20),
                                                 zeros(UInt32, 20))
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(allempty, kq, Val(3))
end

Test.@testset "the gridded transform is exact against the field's own spectrum" begin
    # Over the whole lag space nothing is angularly averaged and nothing is radially binned, so this
    # must agree with the field's own transform to round-off — unlike the isotropic route, which
    # assumes a uniform sampling of direction that a rectilinear grid does not provide.
    for (D, n) in ((1, 64), (2, 32), (3, 16))
        dx = 2π / n
        dims = ntuple(_ -> n, D)
        scal = _modal_field(dims, dx, D; seed = 90 + D)
        variance = sum(abs2, scal) / prod(dims)
        u = zeros(D, dims...)
        u[1, ntuple(_ -> Colon(), D)...] = scal

        sched = SFC.UniformLagSchedule(dims, ntuple(_ -> dx, D), ntuple(_ -> true, D))
        kaxes, density = SFC.gridded_spectrum(
            u, sched, Val(D), SB.FastFourierTransformSpectralBackend())

        Test.@test length(kaxes) == D
        dk = prod(ntuple(d -> kaxes[d][2] - kaxes[d][1], D))
        modes = density .* dk

        direct = abs2.(FFTW.fft(scal)) ./ prod(dims)^2
        direct[ntuple(_ -> 1, D)...] = 0.0
        Test.@test maximum(abs, modes .- direct) / maximum(direct) < 1e-12
        # and the convention holds: the density integrates to the variance
        Test.@test sum(modes) ≈ variance rtol = 1e-10
    end
end

Test.@testset "the gridded transform survives missing data" begin
    # The reason to reach a spectrum through a structure function at all: with cells missing the
    # field's own transform is meaningless, while the pair average is still unbiased.
    D, n = 2, 32
    dx = 2π / n
    dims = (n, n)
    scal = _modal_field(dims, dx, D; seed = 4242)
    u = zeros(D, dims...)
    u[1, :, :] = scal
    sched = SFC.UniformLagSchedule(dims, (dx, dx), (true, true))
    kaxes, full = SFC.gridded_spectrum(u, sched, Val(D), SB.FastFourierTransformSpectralBackend())

    # The absolute error depends on how many pairs each lag retains, hence on the grid size; the
    # assertion that carries the claim is the comparison against zero-filling the gaps.
    for frac in (0.1, 0.3, 0.5)
        Random.seed!(7)
        valid = rand(prod(dims)) .> frac
        _, masked = SFC.gridded_spectrum(u, sched, Val(D),
                                         SB.FastFourierTransformSpectralBackend(); valid = valid)
        err = maximum(abs, masked .- full) / maximum(full)

        naive = copy(scal)
        naive[.!reshape(valid, dims)] .= 0.0
        nspec = abs2.(FFTW.fft(naive)) ./ prod(dims)^2
        nspec[1, 1] = 0.0
        dk = (kaxes[1][2] - kaxes[1][1]) * (kaxes[2][2] - kaxes[2][1])
        naive_err = maximum(abs, nspec .- full .* dk) / maximum(full .* dk)

        Test.@test err < 0.2
        Test.@test err < naive_err / 4       # the whole point: far better than zero-filling
    end
end

Test.@testset "the gridded transform refuses a bounded direction" begin
    # A bounded direction has no natural Fourier basis, and picking one is a windowing choice.
    Test.@test_throws ArgumentError SFC.gridded_spectrum(
        zeros(1, 8), SFC.UniformLagSchedule((8,), (0.1,), (false,)), Val(1),
        SB.FastFourierTransformSpectralBackend())
end

Test.@testset "shell averaging conserves the spectrum" begin
    D, n = 2, 32
    dx = 2π / n
    dims = (n, n)
    scal = _modal_field(dims, dx, D; seed = 11)
    u = zeros(D, dims...)
    u[1, :, :] = scal
    sched = SFC.UniformLagSchedule(dims, (dx, dx), (true, true))
    kaxes, density = SFC.gridded_spectrum(u, sched, Val(D),
                                          SB.FastFourierTransformSpectralBackend())

    kmax = maximum(abs, kaxes[1]) * sqrt(D) + 1
    edges = collect(range(0.0, kmax; length = 40))
    mids, E = SFC.shell_average(kaxes, density, edges)
    Test.@test length(mids) == length(E) == length(edges) - 1

    dk = (kaxes[1][2] - kaxes[1][1]) * (kaxes[2][2] - kaxes[2][1])
    Test.@test sum(E .* (edges[2] - edges[1])) ≈ sum(density) * dk rtol = 1e-10
end

Test.@testset "the flux quadrature matches a closed-form integral" begin
    # ∫₀^R J₁(Kr) dr = (1 − J₀(KR))/K, so a constant advective structure function `c` gives
    # Π_K = −(K/2)·c·(1 − J₀(KR))/K = −(c/2)(1 − J₀(KR)) exactly — which pins the prefactor.
    c, R = 0.8, 60.0
    r = collect(range(0.0, R; length = 200_000))
    vals = fill(c, length(r))
    Ks = [0.5, 1.0, 2.0, 5.0, 20.0]

    got = SFC.spectral_flux(SFT.VectorDotSFType(1, 2), r, vals, Ks)
    expected = [-(c / 2) * (1 - Bessels.besselj0(K * R)) for K in Ks]
    Test.@test got ≈ expected rtol = 1e-4
    # and it settles on −c/2, the whole-line value, once J₀(KR) has decayed
    Test.@test all(abs.(got .+ c / 2) .< 0.05)

    # linear in the structure function it is given
    Test.@test SFC.spectral_flux(SFT.VectorDotSFType(1, 2), r, 3 .* vals, Ks) ≈ 3 .* got
    # a positive advective structure function gives a negative flux, per the relation's sign
    Test.@test all(<(0), got)
end

Test.@testset "a flux needs a cross-channel moment" begin
    # ⟨δφ δ𝓐_φ⟩ is a moment across two channels; the diagonal is a variance and carries no flux.
    for op in (SFT.VectorDotSFType(1, 1), SFT.ScalarDotSFType(2, 2))
        err = Test.@test_throws ArgumentError SFC.assert_advective(op)
        Test.@test occursin("diagonal", err.value.msg)
    end
    for op in (SFT.S2SFType(), SFT.L2SFType(), SFT.L3SFType())
        err = Test.@test_throws ArgumentError SFC.assert_advective(op)
        Test.@test occursin("cross-channel", err.value.msg)
    end
    Test.@test SFC.assert_advective(SFT.VectorDotSFType(1, 2)) === nothing
    Test.@test SFC.assert_advective(SFT.ScalarDotSFType(1, 2)) === nothing

    r = collect(range(0.0, 5.0; length = 100))
    Test.@test_throws ArgumentError SFC.spectral_flux(SFT.S2SFType(), r, fill(1.0, 100), [1.0])
end

Test.@testset "a result object carries into the flux relation" begin
    edges = collect(range(0.0, 10.0; length = 21))
    sums = fill(2.0, 20)
    counts = fill(UInt32(4), 20)
    counts[5] = 0
    sums[5] = 0.0
    raw = SF.StructureFunctionSumsAndCounts(SFT.VectorDotSFType(1, 2), edges, sums, counts)
    Ks = [0.5, 1.5, 3.0]
    fromobj = SFC.spectral_flux(raw, Ks)
    Test.@test all(isfinite, fromobj)

    keep = [i for i in 1:20 if counts[i] > 0]
    mids = SF.midpoints(edges)
    direct = SFC.spectral_flux(SFT.VectorDotSFType(1, 2), collect(mids)[keep],
                               [sums[i] / counts[i] for i in keep], Ks)
    Test.@test fromobj ≈ direct
end

Test.@testset "the covariance is the variance less half the structure function" begin
    # D(r) = 2[C(0) - C(r)] exactly, so a Gaussian correlation must come back as it went in.
    σ2, ℓ = 1.7, 0.8
    edges = collect(range(0.0, 6.0; length = 41))
    mids = SF.midpoints(edges)
    d = [2σ2 * (1 - exp(-r^2 / (2ℓ^2))) for r in mids]
    res = SF.StructureFunction(SFT.S2SFType(), edges, d)

    r, C = SFC.covariance(res, σ2)
    Test.@test r ≈ collect(mids)
    Test.@test C ≈ [σ2 * exp(-rr^2 / (2ℓ^2)) for rr in mids] rtol = 1e-12
    # C(0) is the variance and the structure function cannot supply it
    Test.@test SFC.covariance(res, 2σ2)[2] ≈ C .+ σ2

    # end to end: a single mode has C(r) = (A²/2) Λ_D(k₀ r), which the package's own S₂ must give back
    A = 1.1
    for (D, mv) in ((2, (4, 3)), (3, (3, 2, 1)))
        k0 = sqrt(sum(abs2, mv))
        Random.seed!(800 + D)
        N = 8000
        x = 2π .* rand(D, N)
        u = zeros(D, N)
        for p in 1:N
            u[1, p] = A * cos(sum(mv[dd] * x[dd, p] for dd in 1:D) + 0.9)
        end
        bins = collect(range(0.0, 2.0; length = 41))
        raw = SFC.calculate_structure_function(
            SFT.S2SFType(), x, u, bins; backend = CB.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false,
            show_progress = false)
        rr, CC = SFC.covariance(raw, A^2 / 2)
        expect = [(A^2 / 2) * SFC.isotropic_kernel(Val(D), k0 * q) for q in rr]
        Test.@test maximum(abs, CC .- expect) < 0.05 * A^2
    end
end

Test.@testset "a covariance needs a second-order moment" begin
    edges = collect(range(0.0, 4.0; length = 11))
    vals = fill(1.0, 10)
    for op in (SFT.L3SFType(), SFT.L1T2SFType(), SFT.VectorDotSFType(1, 2))
        bad = SF.StructureFunction(op, edges, vals)
        Test.@test_throws ArgumentError SFC.covariance(bad, 1.0)
    end
    for op in (SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType())
        Test.@test SFC.covariance(SF.StructureFunction(op, edges, vals), 1.0) isa Tuple
    end
end

Test.@testset "a covariance matrix is checked, not assumed, positive semi-definite" begin
    Random.seed!(31)
    pts = 3.0 .* rand(2, 60)
    gauss(s) = 1.4 * exp(-s^2 / (2 * 0.9^2))

    # A Gaussian kernel is positive definite, but interpolating it is not: the error falls as the
    # square of the separation spacing, so it must be resolved before the matrix is valid.
    fine = collect(range(0.0, 6.0; length = 5_000))
    Σ = SFC.covariance_matrix(pts, fine, gauss.(fine))
    Test.@test size(Σ) == (60, 60)
    Test.@test Σ ≈ transpose(Σ)
    Test.@test minimum(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(Σ))) > -1e-6 * maximum(abs, Σ)
    # every point is at zero separation from itself, so the diagonal is the variance
    Test.@test all(≈(gauss(0.0)), LinearAlgebra.diag(Σ))
    # and it matches the kernel evaluated directly, which needs no interpolation at all
    exact = [gauss(sqrt(sum(abs2, pts[:, i] .- pts[:, j]))) for i in 1:60, j in 1:60]
    Test.@test maximum(abs, Σ .- exact) < 1e-5 * maximum(abs, exact)

    # too coarse to be a valid kernel, and saying so is the useful behaviour
    coarse = collect(range(0.0, 6.0; length = 60))
    Test.@test_throws ArgumentError SFC.covariance_matrix(pts, coarse, gauss.(coarse))
    Test.@test SFC.covariance_matrix(pts, coarse, gauss.(coarse); posdef_rtol = 1e-2) isa Matrix

    # an oscillating "covariance" is no kernel at all — off by the matrix scale itself, not by a
    # discretisation error, so no tolerance rescues it
    bad = [cos(6s) for s in fine]
    Test.@test_throws ArgumentError SFC.covariance_matrix(pts, fine, bad)
    Test.@test_throws ArgumentError SFC.covariance_matrix(pts, fine, bad; posdef_rtol = 1e-2)
    Test.@test SFC.covariance_matrix(pts, fine, bad; check_posdef = false) isa Matrix
end

Test.@testset "the Helmholtz components transform to rotational and divergent spectra" begin
    n_bins = 60
    edges = collect(10 .^ range(-2, 0.5; length = n_bins + 1))
    mids = SF.midpoints(edges)
    counts = ones(UInt32, n_bins)

    # a purely rotational 2-D field: D_TT = (5/3) D_LL for D_LL ∝ r^(2/3), so D_div ≡ 0
    D_LL = [r^(2 / 3) for r in mids]
    D_TT = (5 / 3) .* D_LL
    h = SFC.helmholtz_decompose_2d(edges, D_LL, counts, D_TT, counts)

    kq = collect(range(0.5, 40.0; length = 300))
    rot_asym = maximum(h.rotational_sums)
    div_asym = maximum(h.divergent_sums)
    spec = SFC.helmholtz_spectra(h, kq; rotational_asymptote = rot_asym,
                                 divergent_asymptote = div_asym)
    Test.@test haskey(spec, :rotational) && haskey(spec, :divergent)
    Test.@test all(isfinite, spec.rotational) && all(isfinite, spec.divergent)

    # D_rot + D_div = D_LL + D_TT exactly, and the transform is linear, so the two spectra must sum
    # to the trace's — an identity that holds whatever the field is
    trace = SFC.isotropic_spectrum(SFT.S2SFType(), collect(mids), D_LL .+ D_TT, kq, Val(2);
                                   asymptote = rot_asym + div_asym)
    Test.@test spec.rotational .+ spec.divergent ≈ trace rtol = 1e-10

    # the field has no divergent part, so its divergent spectrum is small beside the rotational one
    Test.@test maximum(abs, spec.divergent) < 0.1 * maximum(abs, spec.rotational)

    # the mirror case: swapping the roles makes the field irrotational
    h2 = SFC.helmholtz_decompose_2d(edges, D_TT, counts, D_LL, counts)
    spec2 = SFC.helmholtz_spectra(h2, kq)
    Test.@test maximum(abs, spec2.rotational) < 0.1 * maximum(abs, spec2.divergent)

    # and each component is accepted by the invertibility gate on its own
    Test.@test SFC.assert_invertible(SFT.RotationalSecondOrderStructureFunctionType()) === nothing
    Test.@test SFC.assert_invertible(SFT.DivergentSecondOrderStructureFunctionType()) === nothing
end

Test.@testset "the shell spectrum carries the dimensional weight" begin
    kq = [0.5, 1.0, 2.0, 4.0]
    P = [3.0, 2.0, 1.0, 0.5]
    for D in (1, 2, 3)
        E = SFC.shell_spectrum(P, kq, Val(D))
        Test.@test E ≈ [SFC.solid_angle(Val(D)) * k^(D - 1) * p for (k, p) in zip(kq, P)]
    end
    # in one dimension the weight is the two directions along the line
    Test.@test SFC.shell_spectrum(P, kq, Val(1)) ≈ 2 .* P
end
