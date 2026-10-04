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

# ∫ f dk over [a, b] by the trapezoid rule on `n` points.
function _trapz(f, a, b; n = 101)
    k = range(a, b; length = n)
    v = f.(k)
    return sum((v[1:(end - 1)] .+ v[2:end]) ./ 2 .* step(k))
end

# ∫ f dk over [a, b] by Simpson's rule on `n` points.
function _simpson(f, a, b; n = 801)
    n = isodd(n) ? n : n + 1
    k = range(a, b; length = n)
    v = f.(k)
    return step(k) / 3 * (v[1] + v[end] + 4 * sum(v[2:2:(end - 1)]) + 2 * sum(v[3:2:(end - 2)]))
end

_centres(edges) = [(edges[j] + edges[j + 1]) / 2 for j in 1:(length(edges) - 1)]
_widths(edges) = diff(edges)
_logedges(a, b, n) = exp.(range(log(a), log(b); length = n + 1))

# Each forward model against its defining integral, in closed form in one dimension and by Simpson's rule beyond.
Test.@testset "the forward models are the relations they name" begin
    r = collect(range(0.05, 15.0; length = 120))
    rs = r[1:20:end]
    k_edges = _logedges(0.2, 20.0, 12)
    nk = length(k_edges) - 1
    H1 = [2 * ((k_edges[j + 1] - k_edges[j]) - (sin(k_edges[j + 1] * ri) - sin(k_edges[j] * ri)) / ri)
          for ri in r, j in 1:nk]
    Test.@test SFC.forward_matrix(SFC.SpectrumForwardModel(Val(1), r, k_edges)) ≈ H1 rtol = 1e-12
    for (D, kernel) in ((2, J0), (3, x -> sin(x) / x))
        H = [_simpson(k -> 2 * (1 - kernel(k * ri)), k_edges[j], k_edges[j + 1]) for ri in rs, j in 1:nk]
        Test.@test all(isapprox.(SFC.forward_matrix(SFC.SpectrumForwardModel(Val(D), rs, k_edges)), H; rtol = 1e-9))
    end

    mh = SFC.forward_matrix(SFC.HelmholtzForwardModel(rs, k_edges))
    nr = length(rs)
    plus = [_simpson(k -> 1 - J0(k * ri), k_edges[j], k_edges[j + 1]) for ri in rs, j in 1:nk]
    minus = [_simpson(k -> J2(k * ri), k_edges[j], k_edges[j + 1]) for ri in rs, j in 1:nk]
    Test.@test mh[1:nr, :] .+ mh[(nr + 1):end, :] ≈ hcat(2plus, 2plus) rtol = 1e-8
    Test.@test mh[1:nr, :] .- mh[(nr + 1):end, :] ≈ hcat(2minus, -2minus) rtol = 1e-8

    Random.seed!(4310)
    mf = SFC.FluxForwardModel(r, k_edges)
    kc, dk = _centres(k_edges), _widths(k_edges)
    ε = 0.3
    ξ = rand(nk)
    x = vcat(ε, ξ)
    F(k) = -ε + sum(ξ[j] * dk[j] for j in eachindex(kc) if kc[j] <= k; init = 0.0)
    Kmax = 60.0
    pieces = vcat(1e-6, kc, Kmax)
    S3 = map(r[1:30:end]) do ri
        integral = sum(F(pieces[p]) * _simpson(k -> J2(k * ri) / k, pieces[p], pieces[p + 1];
                                                n = 101 + 20 * ceil(Int, ri * (pieces[p + 1] - pieces[p])))
                       for p in 1:(length(pieces) - 1))
        -4 * ri * (integral + F(Kmax) * J1(Kmax * ri) / (Kmax * ri))
    end
    Test.@test all(isapprox.((SFC.forward_matrix(mf) * x)[1:30:end], S3; rtol = 1e-8))
    Test.@test SFC.flux_matrix(mf) * x ≈ F.(kc)
end

# Noise-free values from each forward model, binned as a result, fit back to the values they came from.
Test.@testset "round trips through the inversions" begin
    Random.seed!(4320)
    k_edges = _logedges(0.2, 20.0, 12)
    x = vcat(0.4, rand(length(k_edges) - 1))
    edges = collect(range(0.0, 15.0; length = 151))
    mids = collect(SF.midpoints(edges))
    mf = SFC.FluxForwardModel(mids, k_edges)
    res = SF.StructureFunction(SFT.S3SFType(), edges, SFC.forward_matrix(mf) * x)
    fit = SFC.fit_flux(res, k_edges, SFC.RegularizedLeastSquares(nothing); W = fill(1e-20, 150))
    Test.@test fit.ε ≈ x[1] rtol = 1e-8
    Test.@test fit.ξ ≈ x[2:end] rtol = 1e-8
    Test.@test fit.F ≈ SFC.flux_matrix(mf) * x rtol = 1e-8
    Test.@test fit.k == _centres(k_edges)

    E = 0.5 .+ rand(12)
    res1 = SF.StructureFunction(SFT.S2SFType(), edges, SFC.SpectrumForwardModel(Val(1), mids, k_edges).H * E)
    f1 = SFC.fit_spectrum(res1, k_edges, SFC.RegularizedLeastSquares(nothing), Val(1); W = fill(1e-20, 150))
    Test.@test f1.E ≈ E rtol = 1e-7
    f1n = SFC.fit_spectrum(res1, k_edges, SFC.NonNegativeLeastSquares(), Val(1))
    Test.@test f1n.E ≈ E rtol = 1e-7
    Test.@test f1n.covariance === nothing
    counts = fill(UInt32(5), 150)
    counts[40] = 0
    raw = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, 5 .* res1.values .* (counts .> 0), counts)
    f1r = SFC.fit_spectrum(raw, k_edges, SFC.RegularizedLeastSquares(nothing), Val(1); W = fill(1e-20, 149))
    Test.@test f1r.E ≈ E rtol = 1e-7
end

# A batch with a NaN bin in one slice, under one or per-slice data covariances, fits as its slices do one by one.
Test.@testset "a batch is fitted slice by slice" begin
    Random.seed!(4370)
    k_edges = _logedges(0.2, 20.0, 10)
    edges = collect(range(0.0, 15.0; length = 61))
    mids = collect(SF.midpoints(edges))
    H = SFC.SpectrumForwardModel(Val(2), mids, k_edges).H
    vals = H * (0.5 .+ rand(10, 3))
    vals[7, 3] = NaN
    batch = SF.StructureFunction(SFT.S2SFType(), edges, vals)
    slice(j) = SF.StructureFunction(SFT.S2SFType(), edges, vals[:, j])
    method = SFC.RegularizedLeastSquares(fill(10.0, 10))
    W = fill(1e-6, 60)
    fits = SFC.fit_spectrum(batch, k_edges, method, Val(2); W)
    single = [SFC.fit_spectrum(slice(j), k_edges, method, Val(2); W) for j in 1:3]
    Test.@test size(fits) == (3,)
    Test.@test all(j -> isapprox(fits[j].E, single[j].E; rtol = 1e-12), 1:3)
    Test.@test all(j -> isapprox(Matrix(fits[j].covariance), Matrix(single[j].covariance); rtol = 1e-12), 1:3)
    Ws = [fill(1e-6, 60), fill(4e-6, 60), fill(9e-6, 60)]
    fits_w = SFC.fit_spectrum(batch, k_edges, method, Val(2); W = Ws)
    Test.@test all(j -> isapprox(fits_w[j].E, SFC.fit_spectrum(slice(j), k_edges, method, Val(2); W = Ws[j]).E;
                                 rtol = 1e-12), 1:3)
    Test.@test_throws DimensionMismatch SFC.fit_spectrum(batch, k_edges, method, Val(2); W = Ws[1:2])
    nn = SFC.fit_spectrum(batch, k_edges, SFC.NonNegativeLeastSquares(), Val(2))
    Test.@test nn[2].E ≈ SFC.fit_spectrum(slice(2), k_edges, SFC.NonNegativeLeastSquares(), Val(2)).E rtol = 1e-12
    F = SFC.forward_matrix(SFC.FluxForwardModel(mids, k_edges)) * randn(11, 2)
    flux = SFC.RegularizedLeastSquares(fill(10.0, 11))
    fb = SFC.fit_flux(SF.StructureFunction(SFT.S3SFType(), edges, F), k_edges, flux; W)
    Test.@test fb[2].F ≈ SFC.fit_flux(SF.StructureFunction(SFT.S3SFType(), edges, F[:, 2]), k_edges, flux; W).F rtol = 1e-12
end

# A pure gradient spectrum gives the closed-form D_LL and D_TT, and its L2 and T2 fit back to it with no curl.
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
    Ebar = [_trapz(E_E, k_edges[j], k_edges[j + 1]) / (k_edges[j + 1] - k_edges[j]) for j in 1:nk]
    pred = SFC.HelmholtzForwardModel(r, k_edges).H * vcat(Ebar, zeros(nk))
    Test.@test pred[1:80] ≈ DLL rtol = 4e-3
    Test.@test pred[81:end] ≈ DTT rtol = 4e-3

    kr_edges = collect(range(0.5, 8.0; length = 9))
    Er = [_trapz(E_E, kr_edges[j], kr_edges[j + 1]) / (kr_edges[j + 1] - kr_edges[j]) for j in 1:8]
    edges = collect(range(0.0, 6.0; length = 81))
    mids = collect(SF.midpoints(edges))
    y = SFC.HelmholtzForwardModel(mids, kr_edges).H * vcat(Er, zeros(8))
    L2 = SF.StructureFunction(SFT.L2SFType(), edges, y[1:80])
    T2 = SF.StructureFunction(SFT.T2SFType(), edges, y[81:end])
    fit = SFC.fit_helmholtz_spectra(L2, T2, kr_edges, SFC.RegularizedLeastSquares(nothing); W = fill(1e-20, 160))
    Test.@test fit.E ≈ Er rtol = 1e-6
    Test.@test maximum(abs, fit.B) < 1e-6 * maximum(Er)
    Test.@test fit.k == _centres(kr_edges)
    Test.@test SFC.fit_helmholtz_spectra(L2, T2, kr_edges, SFC.NonNegativeLeastSquares()).E ≈ Er rtol = 1e-6
    Test.@test_throws ArgumentError SFC.fit_helmholtz_spectra(T2, L2, kr_edges, SFC.NonNegativeLeastSquares())
end

# (dimension, fit): every fit once, each dimension's kernel twice.
const SEGMENT_CASES = ((1, :one_segment), (1, :selection), (2, :two_segments), (2, :weighted))

# Values of a known segmented power law fit back to its slopes, its log-uniform breakpoints and its segment count.
Test.@testset "the segmented power law recovers slopes and breaks" begin
    k_lo, k_hi = 0.3, 30.0
    breaks(S) = exp.(range(log(k_lo), log(k_hi); length = S + 1))
    edges = collect(range(0.0, 12.0; length = 101))
    mids = collect(SF.midpoints(edges))
    p1, p2 = [0.7, -5 / 3], [0.7, -1.0, -3.0]
    function power_law(D, p)
        e = breaks(length(p) - 1)
        kq, _, K = SFC._segmented_design(Val(D), mids, e)
        return SF.StructureFunction(SFT.S2SFType(), edges, K * SFC.segmented_spectrum(p, kq, e))
    end
    for (D, fit) in SEGMENT_CASES
        if fit === :one_segment
            f = SFC.fit_spectrum(power_law(D, p1), [k_lo, k_hi], SFC.SegmentedPowerLaw(1), Val(D))
            Test.@test f.converged
            Test.@test f.parameters ≈ p1 rtol = 1e-6
            Test.@test size(f.covariance) == (2, 2)
        elseif fit === :two_segments
            f2 = SFC.fit_spectrum(power_law(D, p2), [k_lo, k_hi], SFC.SegmentedPowerLaw(2), Val(D))
            Test.@test f2.converged
            Test.@test f2.parameters ≈ p2 rtol = 1e-5
            Test.@test f2.breakpoints ≈ breaks(2)
        elseif fit === :selection
            sel = SFC.select_segments(power_law(D, p2), [k_lo, k_hi], 1:3, Val(D))
            Test.@test sel.segments == 2
            Test.@test sel.misfits[2] < 1e-6
        else
            fw = SFC.fit_spectrum(power_law(D, p2), [k_lo, k_hi], SFC.SegmentedPowerLaw(2), Val(D); W = fill(1e-6, 100))
            Test.@test fw.parameters ≈ p2 rtol = 1e-5
        end
    end
    e3 = breaks(3)
    p3 = [1.0, -1.0, -2.0, -3.5]
    Test.@test all(b -> isapprox(SFC.segmented_spectrum(p3, [prevfloat(b)], e3), SFC.segmented_spectrum(p3, [nextfloat(b)], e3);
                                 rtol = 1e-10), e3[2:3])
end

# The S3 of a flux model, transformed back by `spectral_flux` at the bin edges, is the model's flux there.
Test.@testset "the fitted flux is the flux the transform recovers" begin
    Random.seed!(4340)
    r = collect(range(0.02, 50.0; length = 1000))
    k_edges = _logedges(0.2, 20.0, 12)
    x = vcat(0.2, 0.1 .+ rand(12))
    m = SFC.FluxForwardModel(r, k_edges)
    Π = SFC.spectral_flux(SFT.S3SFType(), r, SFC.forward_matrix(m) * x, k_edges[5:11])
    Test.@test Π ≈ (SFC.flux_matrix(m) * x)[4:10] rtol = 2e-2
end

Test.@testset "the flux covariance is the parameter covariance carried through the flux matrix" begin
    Random.seed!(4350)
    k_edges = _logedges(0.2, 20.0, 12)
    edges = collect(range(0.0, 15.0; length = 151))
    m = SFC.FluxForwardModel(collect(SF.midpoints(edges)), k_edges)
    y = SFC.forward_matrix(m) * vcat(0.4, rand(12))
    fit = SFC.fit_flux(SF.StructureFunction(SFT.S3SFType(), edges, y), k_edges,
                       SFC.RegularizedLeastSquares(fill(1e-2, 13)); W = fill(0.03, 150))
    G = SFC.flux_matrix(m)
    Test.@test Matrix(fit.flux_covariance) ≈ G * Matrix(fit.covariance) * G' rtol = 1e-12
end

# Variances given as a vector are the diagonal covariance matrix they name, for the data and for the prior.
Test.@testset "a vector of variances is the diagonal covariance" begin
    Random.seed!(4350)
    r = collect(range(0.05, 15.0; length = 150))
    H = SFC.forward_matrix(SFC.FluxForwardModel(r, _logedges(0.2, 20.0, 12)))
    σ2 = 0.03
    y = H * vcat(0.4, rand(12)) .+ sqrt(σ2) .* randn(length(r))
    prior = SFC.RegularizedLeastSquares(fill(1e-2, 13))
    xd, Cd = SFC.solve(prior, H, y, fill(σ2, length(y)))
    xm, Cm = SFC.solve(prior, H, y, Matrix(LA.Diagonal(fill(σ2, length(y)))))
    Test.@test xd ≈ xm rtol = 1e-10
    Test.@test Matrix(Cd) ≈ Matrix(Cm) rtol = 1e-10
    xp, _ = SFC.solve(SFC.RegularizedLeastSquares(Matrix(1e-2 .* LA.I(13))), H, y, fill(σ2, length(y)))
    Test.@test xp ≈ xd rtol = 1e-10
end

# A looser prior lowers the misfit and raises the norm of the fit.
Test.@testset "the trade-off curve" begin
    Random.seed!(4350)
    r = collect(range(0.05, 15.0; length = 150))
    m = SFC.FluxForwardModel(r, _logedges(0.2, 20.0, 12))
    y = SFC.forward_matrix(m) * vcat(0.4, rand(12)) .+ sqrt(0.03) .* randn(length(r))
    curve = SFC.tradeoff_curve(m, y, fill(0.03, length(y)), 10.0 .^ range(-6, 2; length = 17))
    Test.@test length(curve.misfit) == 17
    Test.@test issorted(curve.misfit; rev = true)
    Test.@test issorted(curve.norm)
end

# The pair variance read from a value-binned histogram, per batch slice; angle cells and pair masses are refused.
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
    fine = collect(range(-1e-9, vmax * (1 + 1e-9); length = 101))
    joint = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, fine; backend = CB.SerialBackend())
    Test.@test SFC.independent_pair_variance(joint) ≈ truth rtol = 2e-3
    coarse = collect(range(-1e-9, vmax * (1 + 1e-9); length = 6))
    jc = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, coarse; backend = CB.SerialBackend())
    estc = SFC.independent_pair_variance(jc)
    Test.@test all(estc .<= truth .* (1 + 1e-12))
    Test.@test all(estc .> 0)
    fine4 = collect(range(-1e-9, 4 * vmax * (1 + 1e-9); length = 401))
    jb = SFC.calculate_structure_function(SFT.L2SFType(), x, cat(u, 2 .* u; dims = 3), bins, fine4;
                                          backend = CB.SerialBackend())
    Test.@test SFC.independent_pair_variance(jb) ≈ hcat(truth, 16 .* truth) rtol = 2e-3
    ja = SF.StructureFunction2DSumsAndCounts(SFT.L2SFType(), bins, collect(range(prevfloat(0.0), π; length = 5)),
                                             zeros(nb, 4), zeros(nb, 4), SFC.SeparationAngleAxis([1.0, 0.0]))
    Test.@test_throws ArgumentError SFC.independent_pair_variance(ja)
    jw = SF.StructureFunction2DSumsAndCounts(SFT.L2SFType(), bins, fine, joint.sums, Float64.(joint.counts),
                                             SFC.InvariantValueAxis())
    Test.@test_throws ArgumentError SFC.independent_pair_variance(jw)
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
