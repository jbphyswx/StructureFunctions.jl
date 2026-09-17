using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using Bessels: Bessels
using LsqFit: LsqFit
using LinearAlgebra: LinearAlgebra as LA
using Random: Random

J0(x) = Bessels.besselj0(x)
J1(x) = Bessels.besselj1(x)
J2(x) = Bessels.besselj(2, x)

# ∫ f dk over [a, b] by the trapezoid rule on a fine grid.
function _trapz(f, a, b; n = 200_001)
    k = range(a, b; length = n)
    v = f.(k)
    return sum((v[1:(end - 1)] .+ v[2:end]) ./ 2 .* step(k))
end

# ∫ f dk over [a, b] by Simpson's rule on a fine grid.
function _simpson(f, a, b; n = 200_001)
    n = isodd(n) ? n : n + 1
    k = range(a, b; length = n)
    v = f.(k)
    return step(k) / 3 * (v[1] + v[end] + 4 * sum(v[2:2:(end - 1)]) + 2 * sum(v[3:2:(end - 2)]))
end

_centres(edges) = [(edges[j] + edges[j + 1]) / 2 for j in 1:(length(edges) - 1)]
_widths(edges) = diff(edges)
_logedges(a, b, n) = exp.(range(log(a), log(b); length = n + 1))

Test.@testset "the forward models are the relations they name" begin
    r = collect(range(0.05, 15.0; length = 120))
    k_edges = _logedges(0.2, 20.0, 12)
    # one dimension: ∫ (1 − cos kr) dk is elementary
    m1 = SFC.SpectrumForwardModel(Val(1), r, k_edges)
    H1 = [2 * ((k_edges[j + 1] - k_edges[j]) - (sin(k_edges[j + 1] * ri) - sin(k_edges[j] * ri)) / ri)
          for ri in r, j in 1:(length(k_edges) - 1)]
    Test.@test SFC.forward_matrix(m1) ≈ H1 rtol = 1e-12
    # two dimensions against a fine quadrature of the J₀ kernel
    m2 = SFC.SpectrumForwardModel(Val(2), r[1:20:end], k_edges)
    for (i, ri) in enumerate(r[1:20:end]), j in 1:3
        Test.@test m2.H[i, j] ≈ _trapz(k -> 2 * (1 - J0(k * ri)), k_edges[j], k_edges[j + 1]) rtol = 1e-8
    end
    # the flux model is exact for a piecewise-constant flux: eq. 8 written out, and eq. 6 integrated
    Random.seed!(4310)
    mf = SFC.FluxForwardModel(r, k_edges)
    kc, dk = _centres(k_edges), _widths(k_edges)
    ε = 0.3
    ξ = rand(length(kc))
    x = vcat(ε, ξ)
    S3 = SFC.forward_matrix(mf) * x
    Test.@test S3 ≈ [2ε * ri - sum(4 * ξ[j] * dk[j] / kc[j] * J1(kc[j] * ri) for j in eachindex(kc)) for ri in r] rtol = 1e-12
    F(k) = -ε + sum(ξ[j] * dk[j] for j in eachindex(kc) if kc[j] <= k; init = 0.0)
    Kmax = 60.0
    pieces = vcat(1e-6, kc, Kmax)          # F is constant between its jumps at the bin centres
    for ri in r[1:30:end]
        integral = sum(F(pieces[p]) * _simpson(k -> J2(k * ri) / k, pieces[p], pieces[p + 1]; n = 200_001)
                       for p in 1:(length(pieces) - 1))
        integral += F(Kmax) * J1(Kmax * ri) / (Kmax * ri)      # ∫_K^∞ J₂(kr)/k dk = J₁(Kr)/(Kr)
        Test.@test -4 * ri * integral ≈ S3[findfirst(==(ri), r)] rtol = 1e-8
    end
    Fc = SFC.flux_matrix(mf) * x
    Test.@test Fc ≈ F.(kc)
    # the Helmholtz model's sum and difference lines
    mh = SFC.HelmholtzForwardModel(r[1:20:end], k_edges)
    nr, nk = length(r[1:20:end]), length(kc)
    for (i, ri) in enumerate(r[1:20:end]), j in 1:2
        plus = _trapz(k -> 1 - J0(k * ri), k_edges[j], k_edges[j + 1])
        minus = _trapz(k -> J2(k * ri), k_edges[j], k_edges[j + 1])
        Test.@test mh.H[i, j] + mh.H[nr + i, j] ≈ 2plus rtol = 1e-8
        Test.@test mh.H[i, j] - mh.H[nr + i, j] ≈ 2minus rtol = 1e-8
        Test.@test mh.H[i, nk + j] - mh.H[nr + i, nk + j] ≈ -2minus rtol = 1e-8
    end
end

Test.@testset "round trips through the inversions" begin
    Random.seed!(4320)
    r = collect(range(0.05, 15.0; length = 150))
    k_edges = _logedges(0.2, 20.0, 12)
    mf = SFC.FluxForwardModel(r, k_edges)
    x = vcat(0.4, rand(length(k_edges) - 1))
    y = SFC.forward_matrix(mf) * x
    W = fill(1e-20, length(y))
    xr, C = SFC.solve(SFC.RegularizedLeastSquares(nothing), mf.H, y, W)
    Test.@test xr ≈ x rtol = 1e-8
    Test.@test size(C) == (length(x), length(x))
    xn, Cn = SFC.solve(SFC.NonNegativeLeastSquares(), mf.H, y, nothing)
    Test.@test all(>=(0), xn)
    Test.@test xn ≈ x rtol = 1e-8
    Test.@test Cn === nothing
    # a flux with a sink cannot be represented by the monotone model: the constrained fit stays non-negative
    xs = copy(x)
    xs[5] = -0.5
    xneg, _ = SFC.solve(SFC.NonNegativeLeastSquares(), mf.H, mf.H * xs, nothing)
    Test.@test all(>=(0), xneg)
    Test.@test xneg[5] == 0
    # through the result objects
    edges = collect(range(0.0, 15.0; length = 151))
    mids = SF.midpoints(edges)
    S3 = SFC.forward_matrix(SFC.FluxForwardModel(collect(mids), k_edges)) * x
    res = SF.StructureFunction(SFT.S3SFType(), edges, S3)
    fit = SFC.fit_flux(res, k_edges, SFC.RegularizedLeastSquares(nothing); W = fill(1e-20, length(S3)))
    Test.@test fit.ε ≈ x[1] rtol = 1e-8
    Test.@test fit.ξ ≈ x[2:end] rtol = 1e-8
    Test.@test fit.F ≈ SFC.flux_matrix(SFC.FluxForwardModel(collect(mids), k_edges)) * x rtol = 1e-8
    Test.@test fit.k == _centres(k_edges)
    Test.@test size(fit.flux_covariance) == (length(k_edges) - 1, length(k_edges) - 1)
    # a spectrum on bins, in one dimension: exact data on the model's own bins come back
    E = 0.5 .+ rand(12)
    m1 = SFC.SpectrumForwardModel(Val(1), r, k_edges)
    y1 = m1.H * E
    res1 = SF.StructureFunction(SFT.S2SFType(), edges, SFC.SpectrumForwardModel(Val(1), collect(mids), k_edges).H * E)
    f1 = SFC.fit_spectrum(res1, k_edges, SFC.RegularizedLeastSquares(nothing), Val(1); W = fill(1e-20, 150))
    Test.@test f1.E ≈ E rtol = 1e-7
    f1n = SFC.fit_spectrum(res1, k_edges, SFC.NonNegativeLeastSquares(), Val(1))
    Test.@test f1n.E ≈ E rtol = 1e-7
    # sums-and-counts input with an empty bin dropped
    counts = fill(UInt32(5), 150)
    counts[40] = 0
    sums = 5 .* res1.values .* (counts .> 0)
    raw = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, sums, counts)
    f1r = SFC.fit_spectrum(raw, k_edges, SFC.RegularizedLeastSquares(nothing), Val(1); W = fill(1e-20, 149))
    Test.@test f1r.E ≈ E rtol = 1e-7
end

Test.@testset "the Helmholtz fit separates gradient from curl" begin
    ℓ = 1.0
    r = collect(range(0.05, 6.0; length = 80))
    C_LL(r) = (1 / ℓ^2 - r^2 / ℓ^4) * exp(-r^2 / (2ℓ^2))
    C_TT(r) = exp(-r^2 / (2ℓ^2)) / ℓ^2
    DLL = 2 .* (C_LL(0.0) .- C_LL.(r))
    DTT = 2 .* (C_TT(0.0) .- C_TT.(r))
    E_E(k) = k^3 * ℓ^2 * exp(-k^2 * ℓ^2 / 2)
    k_edges = collect(range(0.02, 8.0; length = 41))
    nk = length(k_edges) - 1
    Ebar = [_trapz(E_E, k_edges[j], k_edges[j + 1]; n = 2001) / (k_edges[j + 1] - k_edges[j]) for j in 1:nk]
    m = SFC.HelmholtzForwardModel(r, k_edges)
    # the model on the bin-averaged gradient spectrum reproduces the closed forms to the binning error
    pred = m.H * vcat(Ebar, zeros(nk))
    Test.@test pred[1:80] ≈ DLL rtol = 2e-2
    Test.@test pred[81:end] ≈ DTT rtol = 2e-2
    # and the inversion of the model's own data returns the spectra, with no curl, on bins the separations
    # can resolve (a bin narrower than ~π/r_max is not)
    kr_edges = collect(range(0.5, 8.0; length = 9))
    Er = [_trapz(E_E, kr_edges[j], kr_edges[j + 1]; n = 2001) / (kr_edges[j + 1] - kr_edges[j]) for j in 1:8]
    edges = collect(range(0.0, 6.0; length = 81))
    mids = collect(SF.midpoints(edges))
    mm = SFC.HelmholtzForwardModel(mids, kr_edges)
    y = mm.H * vcat(Er, zeros(8))
    L2 = SF.StructureFunction(SFT.L2SFType(), edges, y[1:80])
    T2 = SF.StructureFunction(SFT.T2SFType(), edges, y[81:end])
    fit = SFC.fit_helmholtz_spectra(L2, T2, kr_edges, SFC.RegularizedLeastSquares(nothing); W = fill(1e-20, 160))
    Test.@test fit.E ≈ Er rtol = 1e-6
    Test.@test maximum(abs, fit.B) < 1e-6 * maximum(Er)
    Test.@test fit.k == _centres(kr_edges)
    fitn = SFC.fit_helmholtz_spectra(L2, T2, kr_edges, SFC.NonNegativeLeastSquares())
    Test.@test fitn.E ≈ Er rtol = 1e-6
    Test.@test all(>=(0), fitn.B)
    Test.@test_throws ArgumentError SFC.fit_helmholtz_spectra(T2, L2, kr_edges, SFC.NonNegativeLeastSquares())
end

Test.@testset "the segmented power law recovers slopes and breaks" begin
    k_lo, k_hi = 0.3, 30.0
    r = collect(range(0.05, 12.0; length = 100))
    edges = collect(range(0.0, 12.0; length = 101))
    mids = collect(SF.midpoints(edges))
    for D in (1, 2)
        # one segment: a power law comes back
        e1 = SFC._segment_edges(k_lo, k_hi, 1)
        kq, _, K = SFC._segmented_design(Val(D), mids, e1)
        p_true = [0.7, -5 / 3]
        y = K * SFC.segmented_spectrum(p_true, kq, e1)
        res = SF.StructureFunction(SFT.S2SFType(), edges, y)
        f = SFC.fit_spectrum(res, [k_lo, k_hi], SFC.SegmentedPowerLaw(1), Val(D))
        Test.@test f.converged
        Test.@test f.parameters ≈ p_true rtol = 1e-6
        Test.@test f.E ≈ SFC.segmented_spectrum(p_true, f.k, e1) rtol = 1e-6
        Test.@test size(f.covariance) == (2, 2)
        # two segments with the break at the log-uniform midpoint
        e2 = SFC._segment_edges(k_lo, k_hi, 2)
        kq2, _, K2 = SFC._segmented_design(Val(D), mids, e2)
        p2 = [0.7, -1.0, -3.0]
        y2 = K2 * SFC.segmented_spectrum(p2, kq2, e2)
        res2 = SF.StructureFunction(SFT.S2SFType(), edges, y2)
        f2 = SFC.fit_spectrum(res2, [k_lo, k_hi], SFC.SegmentedPowerLaw(2), Val(D))
        Test.@test f2.converged
        Test.@test f2.parameters ≈ p2 rtol = 1e-5
        Test.@test f2.breakpoints ≈ e2
        sel = SFC.select_segments(res2, [k_lo, k_hi], 1:3, Val(D))
        Test.@test sel.segments == 2
        Test.@test sel.misfits[2] < 1e-6
        Test.@test sel.misfits[1] > 1e-2
        Test.@test issorted(sel.misfits[1:2]; rev = true)
        # a data covariance weighs the residual, where the default weighs the relative misfit
        fw = SFC.fit_spectrum(res2, [k_lo, k_hi], SFC.SegmentedPowerLaw(2), Val(D); W = fill(1e-6, 100))
        Test.@test fw.parameters ≈ p2 rtol = 1e-5
    end
    # continuity at the breakpoints
    e3 = SFC._segment_edges(k_lo, k_hi, 3)
    p3 = [1.0, -1.0, -2.0, -3.5]
    for b in e3[2:3]
        Test.@test SFC.segmented_spectrum(p3, [prevfloat(b)], e3) ≈ SFC.segmented_spectrum(p3, [nextfloat(b)], e3) rtol = 1e-10
    end
end

Test.@testset "the fitted flux is the flux the transform recovers" begin
    Random.seed!(4340)
    R = 200.0
    r = collect(range(0.02, R; length = 20_000))
    k_edges = _logedges(0.2, 20.0, 12)
    kc = _centres(k_edges)
    x = vcat(0.2, 0.1 .+ rand(12))
    m = SFC.FluxForwardModel(r, k_edges)
    S3 = SFC.forward_matrix(m) * x
    F = SFC.flux_matrix(m) * x
    # between the jumps at the bin centres the flux is F[j]; at a jump the transform gives the midpoint value
    K = k_edges[5:11]
    Π = SFC.spectral_flux(SFT.S3SFType(), r, S3, K)
    Test.@test Π ≈ F[4:10] rtol = 5e-2
    # and at twice the truncation radius, where the truncation error has moved through its oscillation
    r2 = collect(range(0.02, 2R; length = 40_000))
    S3b = SFC.forward_matrix(SFC.FluxForwardModel(r2, k_edges)) * x
    Π2 = SFC.spectral_flux(SFT.S3SFType(), r2, S3b, K)
    Test.@test Π2 ≈ F[4:10] rtol = 5e-2
end

Test.@testset "posterior covariance and the trade-off curve" begin
    Random.seed!(4350)
    r = collect(range(0.05, 15.0; length = 150))
    k_edges = _logedges(0.2, 20.0, 12)
    m = SFC.FluxForwardModel(r, k_edges)
    H = SFC.forward_matrix(m)
    x = vcat(0.4, rand(12))
    σ2 = 0.03
    y = H * x .+ sqrt(σ2) .* randn(length(r))
    xhat, C = SFC.solve(SFC.RegularizedLeastSquares(nothing), H, y, fill(σ2, length(y)))
    Test.@test Matrix(C) ≈ σ2 .* inv(H' * H) rtol = 1e-10
    Test.@test xhat ≈ (H' * H) \ (H' * y) rtol = 1e-10
    edges = collect(range(0.0, 15.0; length = 151))
    fit = SFC.fit_flux(SF.StructureFunction(SFT.S3SFType(), edges, H * x), k_edges,
                       SFC.RegularizedLeastSquares(fill(1e-2, 13)); W = fill(σ2, 150))
    Test.@test Matrix(fit.flux_covariance) ≈ SFC.flux_matrix(m) * Matrix(fit.covariance) * SFC.flux_matrix(m)' rtol = 1e-12
    # a full data covariance equals its diagonal form
    xd, Cd = SFC.solve(SFC.RegularizedLeastSquares(fill(1e-2, 13)), H, y, fill(σ2, length(y)))
    xm, Cm = SFC.solve(SFC.RegularizedLeastSquares(fill(1e-2, 13)), H, y, LA.Diagonal(fill(σ2, length(y))) |> Matrix)
    Test.@test xd ≈ xm rtol = 1e-10
    Test.@test Matrix(Cd) ≈ Matrix(Cm) rtol = 1e-10
    xp, _ = SFC.solve(SFC.RegularizedLeastSquares(1e-2 .* LA.I(13) |> Matrix), H, y, fill(σ2, length(y)))
    Test.@test xp ≈ xd rtol = 1e-10
    # the trade-off curve: a weaker prior lowers the misfit and raises the norm
    priors = 10.0 .^ range(-6, 2; length = 17)
    curve = SFC.tradeoff_curve(m, y, fill(σ2, length(y)), priors)
    Test.@test issorted(curve.misfit; rev = true)
    Test.@test issorted(curve.norm)
    Test.@test length(curve.misfit) == 17
end

Test.@testset "the data covariance from a value-binned joint histogram" begin
    Random.seed!(4360)
    N = 200
    x = rand(2, N) .* 10.0
    u = randn(2, N)
    bins = [0.0, 2.0, 4.0, 7.0]
    nb = length(bins) - 1
    vals = [Float64[] for _ in 1:nb]
    for i in 1:(N - 1), j in (i + 1):N
        dx = x[:, j] .- x[:, i]
        rr = LA.norm(dx)
        b = searchsortedfirst(bins, rr) - 1
        1 <= b <= nb || continue
        push!(vals[b], SFT.L2SFType()(u[:, j] .- u[:, i], dx ./ rr))
    end
    truth = [sum(abs2, v .- sum(v) / length(v)) / length(v) / length(v) for v in vals]
    vmax = maximum(maximum, vals)
    fine = collect(range(-1e-9, vmax * (1 + 1e-9); length = 4001))
    joint = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, fine; backend = CB.SerialBackend(),
                                             verbose = false, show_progress = false)
    est = SFC.independent_pair_variance(joint)
    Test.@test est ≈ truth rtol = 2e-3
    coarse = collect(range(-1e-9, vmax * (1 + 1e-9); length = 6))
    jc = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, coarse; backend = CB.SerialBackend(),
                                          verbose = false, show_progress = false)
    estc = SFC.independent_pair_variance(jc)
    Test.@test all(estc .<= truth .* (1 + 1e-12))
    Test.@test all(estc .> 0)
end

Test.@testset "the fits refuse what they cannot mean" begin
    r = collect(range(0.05, 15.0; length = 60))
    edges = collect(range(0.0, 15.0; length = 61))
    k_edges = _logedges(0.2, 20.0, 8)
    y = rand(60)
    L2 = SF.StructureFunction(SFT.L2SFType(), edges, y)
    S2 = SF.StructureFunction(SFT.S2SFType(), edges, y)
    S3 = SF.StructureFunction(SFT.S3SFType(), edges, y)
    Test.@test_throws ArgumentError SFC.fit_flux(L2, k_edges, SFC.NonNegativeLeastSquares())
    Test.@test_throws ArgumentError SFC.fit_spectrum(L2, k_edges, SFC.NonNegativeLeastSquares(), Val(2))
    Test.@test_throws ArgumentError SFC.fit_spectrum(S2, k_edges, SFC.RegularizedLeastSquares(nothing), Val(2))
    Test.@test_throws ArgumentError SFC.fit_flux(S3, reverse(k_edges), SFC.NonNegativeLeastSquares())
    Test.@test_throws ArgumentError SFC.fit_flux(S3, vcat(0.0, k_edges), SFC.NonNegativeLeastSquares())
    Test.@test_throws ArgumentError SFC.SegmentedPowerLaw(0)
    Test.@test_throws ArgumentError SFC.SegmentedPowerLaw(2; slope_bounds = (1.0, -1.0))
    Test.@test_throws ArgumentError SFC.RegularizedLeastSquares([1.0, -1.0])
    Test.@test_throws DimensionMismatch SFC.solve(SFC.RegularizedLeastSquares(fill(1.0, 3)), rand(60, 9), y, fill(1.0, 60))
    Test.@test_throws DimensionMismatch SFC.fit_flux(S3, k_edges, SFC.RegularizedLeastSquares(nothing); W = fill(1.0, 7))
    Test.@test_throws ArgumentError SFC.fit_flux(S3, k_edges, SFC.RegularizedLeastSquares(nothing); W = fill(-1.0, 60))
    Test.@test_throws ArgumentError SFC.SpectrumForwardModel(Val(1), reverse(r), k_edges)
    Test.@test_throws DimensionMismatch SFC.segmented_spectrum([1.0, 2.0], [1.0], [0.1, 1.0, 10.0])
end
