using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using Bessels: Bessels
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using ComputationalBackends: ComputationalBackends as CB
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using Random: Random
using Statistics: mean, var

const FFT_TAG = SB.FastFourierTransformSpectralBackend()

# The gradient of a Gaussian-correlated potential on the plane, in the package's convention that a
# density integrates over d²k to the variance: C_LL(0) = C_TT(0) = 1/ℓ², P_E = k² P_Φ / (2π)².
function _gradient_gaussian(ℓ)
    CLL(r) = (1 / ℓ^2 - r^2 / ℓ^4) * exp(-r^2 / (2ℓ^2))
    CTT(r) = exp(-r^2 / (2ℓ^2)) / ℓ^2
    DLL(r) = 2 * (1 / ℓ^2 - CLL(r))
    DTT(r) = 2 * (1 / ℓ^2 - CTT(r))
    PE(k) = k^2 * 2π * ℓ^2 * exp(-k^2 * ℓ^2 / 2) / (2π)^2
    return DLL, DTT, PE
end

Test.@testset "the two-line Helmholtz inversion recovers gradient and curl spectra" begin
    ℓ = 0.7
    DLL, DTT, PE = _gradient_gaussian(ℓ)
    edges = collect(range(0.0, 40ℓ; length = 20_001))
    mids = collect(SF.midpoints(edges))
    kq = collect(range(0.3, 6.0; length = 60))
    exact = PE.(kq)
    scale = maximum(exact)
    asym = 4 / ℓ^2

    L2 = SF.StructureFunction(SFT.L2SFType(), edges, DLL.(mids))
    T2 = SF.StructureFunction(SFT.T2SFType(), edges, DTT.(mids))
    spec = SFC.helmholtz_spectra(L2, T2, kq; asymptote = asym)
    Test.@test maximum(abs, spec.divergent .- exact) < 2e-3 * scale
    Test.@test maximum(abs, spec.rotational) < 2e-3 * scale

    # the curl of the same potential swaps the two projections and the two spectra
    L2c = SF.StructureFunction(SFT.L2SFType(), edges, DTT.(mids))
    T2c = SF.StructureFunction(SFT.T2SFType(), edges, DLL.(mids))
    specc = SFC.helmholtz_spectra(L2c, T2c, kq; asymptote = asym)
    Test.@test maximum(abs, specc.rotational .- exact) < 2e-3 * scale
    Test.@test maximum(abs, specc.divergent) < 2e-3 * scale

    # the sum of the two is the trace's spectrum, by construction of the first line
    trace = SFC.isotropic_spectrum(SFT.S2SFType(), mids, DLL.(mids) .+ DTT.(mids), kq, Val(2); asymptote = asym)
    Test.@test spec.rotational .+ spec.divergent ≈ trace rtol = 1e-12

    # sums-and-counts inputs give the same answer, and empty bins are dropped from both
    counts = ones(UInt32, length(mids))
    counts[7] = 0
    rawL = SF.StructureFunctionSumsAndCounts(SFT.L2SFType(), edges, DLL.(mids) .* counts, counts)
    rawT = SF.StructureFunctionSumsAndCounts(SFT.T2SFType(), edges, DTT.(mids) .* counts, counts)
    spec2 = SFC.helmholtz_spectra(rawL, rawT, kq; asymptote = asym)
    Test.@test maximum(abs, spec2.divergent .- exact) < 2e-3 * scale

    # the arguments are the two projections, in that order, on one set of bins
    Test.@test_throws ArgumentError SFC.helmholtz_spectra(T2, L2, kq)
    Test.@test_throws ArgumentError SFC.helmholtz_spectra(
        L2, SF.StructureFunction(SFT.T2SFType(), 2 .* edges, DTT.(mids)), kq)
    Test.@test_throws ArgumentError SFC.helmholtz_spectra(
        SF.StructureFunction(SFT.S2SFType(), edges, DLL.(mids)), T2, kq)
end

# The Blackman–Tukey estimator written out: the unbiased autocovariance from the pairs a lag names on
# the grid as given, weighted by the taper, transformed on the padded grid `P`.
function _bt_reference(u, dims, spacing, periodic, P, weight)
    D = size(u, 1)
    N = prod(dims)
    uf = reshape(u, D, N)
    σ2 = sum(var(uf[c, :]; corrected = false) for c in 1:D)
    Dg = length(dims)
    cov = zeros(P)
    cart = CartesianIndices(dims)
    lin = LinearIndices(dims)
    for I in CartesianIndices(P)
        h = ntuple(d -> (m = I[d] - 1; m > P[d] ÷ 2 ? m - P[d] : m), Dg)
        acc = 0.0
        n = 0
        for c in cart
            j = ntuple(d -> c[d] + h[d], Dg)
            ok = true
            j = ntuple(Dg) do d
                if periodic[d]
                    mod(j[d] - 1, dims[d]) + 1
                else
                    (1 <= j[d] <= dims[d]) || (ok = false)
                    j[d]
                end
            end
            ok || continue
            acc += sum(abs2, uf[:, lin[j...]] .- uf[:, lin[c]])
            n += 1
        end
        n == 0 && continue
        r = sqrt(sum(d -> (h[d] * spacing[d])^2, 1:Dg))
        cov[I] = weight(r) * (σ2 - acc / (2n))
    end
    dens = real.(FFTW.fft(cov)) .* (prod(abs, spacing) / (2π)^Dg)
    return σ2, dens
end

Test.@testset "the bounded-domain estimator is the transform of the unbiased autocovariance" begin
    Random.seed!(8100)
    for (dims, spacing, periodic) in (((14,), (0.3,), (false,)), ((9, 7), (0.2, 0.25), (false, false)),
                                      ((8, 7), (0.2, 0.25), (true, false)))
        Dg = length(dims)
        u = randn(Dg, dims...)
        s = SFC.UniformLagSchedule(dims, spacing, periodic)
        r_max = sqrt(sum(d -> (periodic[d] ? (dims[d] ÷ 2) * spacing[d] : (dims[d] - 1) * spacing[d])^2, 1:Dg))
        for (taper, weight) in ((SFC.NoTaper(), r -> 1.0), (SFC.Bartlett(), r -> max(0.0, 1 - r / r_max)),
                                (SFC.GaussianTaper(0.5), r -> exp(-r^2 / 0.5)))
            kax, dens = SFC.gridded_spectrum(u, s, Val(Dg), FFT_TAG; taper)
            P = size(dens)
            Test.@test all(d -> periodic[d] ? P[d] == dims[d] : P[d] >= 2dims[d] - 1, 1:Dg)
            Test.@test all(d -> length(kax[d]) == P[d], 1:Dg)
            σ2, ref = _bt_reference(u, dims, spacing, periodic, P, weight)
            Test.@test maximum(abs, dens .- ref) < 1e-12 * maximum(abs, ref)
            # the density integrates to the variance of the cells, whatever the taper: w(0) = 1
            dk = prod(d -> kax[d][2] - kax[d][1], 1:Dg)
            Test.@test sum(dens) * dk ≈ σ2 rtol = 1e-10
        end
    end
end

Test.@testset "a bounded window of a periodic field converges to its spectrum as the window grows" begin
    # A single line seen through a window: the line's power spreads over the window's spectral kernel,
    # whose width in wavenumber is the resolution 2π/(P dx). At a fixed physical distance from the line
    # the leaked power therefore falls with the window, as 1/n for the boxcar of lags |h| < n. The
    # unbiased autocovariance is not positive-definite, so the leaked power carries a sign.
    A, dx = 1.3, 0.1
    δ = 3 * 2π / (32 * dx)                                # three bins of the coarsest window
    outside = Float64[]
    for n in (32, 64, 128)
        M = n ÷ 16                                       # periods in the window
        k0 = 2π * M / (n * dx)
        x = (0:(n - 1)) .* dx
        u = reshape(A .* cos.(k0 .* x .+ 0.4), 1, n)
        s = SFC.UniformLagSchedule((n,), (dx,), (false,))
        kax, dens = SFC.gridded_spectrum(u, s, Val(1), FFT_TAG)
        dk = kax[1][2] - kax[1][1]
        near = abs.(abs.(kax[1]) .- k0) .<= δ
        push!(outside, abs(sum(dens[.!near]) * dk) / (A^2 / 2))
        Test.@test sum(dens) * dk ≈ var(vec(u); corrected = false) rtol = 1e-10
    end
    Test.@test issorted(outside; rev = true)
    Test.@test outside[1] > 1.5 * outside[end]
    Test.@test outside[end] < 0.05
end

Test.@testset "the Bartlett taper trims the estimator's variance" begin
    # A Gaussian-correlated periodic field, windowed: the far lags are averaged over few pairs, and
    # weighting them down lowers the scatter of the estimate across realisations.
    Random.seed!(8200)
    n_big, n, dx, ℓ = 512, 64, 0.1, 0.4
    kfull = 2π .* FFTW.fftfreq(n_big, 1 / dx)
    s = SFC.UniformLagSchedule((n,), (dx,), (false,))
    plain = Vector{Vector{Float64}}()
    tapered = Vector{Vector{Float64}}()
    kax = nothing
    for _ in 1:40
        amp = exp.(-(kfull .* ℓ) .^ 2 ./ 4)
        f = real.(FFTW.ifft(amp .* FFTW.fft(randn(n_big))))
        u = reshape(f[1:n], 1, n)
        kax, d0 = SFC.gridded_spectrum(u, s, Val(1), FFT_TAG)
        _, d1 = SFC.gridded_spectrum(u, s, Val(1), FFT_TAG; taper = SFC.Bartlett())
        push!(plain, d0)
        push!(tapered, d1)
    end
    band = findall(k -> 0.5 <= k <= 4.0, kax[1])
    v0 = mean(var([p[j] for p in plain]) for j in band)
    v1 = mean(var([p[j] for p in tapered]) for j in band)
    Test.@test v1 < 0.85 * v0
    # and both keep the mean level of the band
    m0 = mean(mean(p[j] for p in plain) for j in band)
    m1 = mean(mean(p[j] for p in tapered) for j in band)
    Test.@test isapprox(m0, m1; rtol = 0.25)
end

Test.@testset "a lag no held pair names is refused by default and zeroed on request" begin
    n = 16
    u = reshape(randn(n), 1, n)
    valid = falses(n)
    valid[1:3] .= true                                   # lags 4…8 join no two held cells
    s = SFC.UniformLagSchedule((n,), (0.1,), (true,))
    Test.@test_throws ArgumentError SFC.gridded_spectrum(u, s, Val(1), FFT_TAG; valid)
    kax, dens = SFC.gridded_spectrum(u, s, Val(1), FFT_TAG; valid, missing_lags = SFC.ZeroDeviationAtMissingLags())
    Test.@test all(isfinite, dens)
    dk = kax[1][2] - kax[1][1]
    Test.@test sum(dens) * dk ≈ var(u[1, 1:3]; corrected = false) rtol = 1e-10
end

# P_0 … P_L and their derivatives at x, by the recurrences, for the test's own series.
function _legendre_and_derivative(x, L)
    P = zeros(L + 1)
    dP = zeros(L + 1)
    P[1] = 1.0
    L >= 1 && (P[2] = x; dP[2] = 1.0)
    for l in 1:(L - 1)
        P[l + 2] = ((2l + 1) * x * P[l + 1] - l * P[l]) / (l + 1)
        dP[l + 2] = dP[l] + (2l + 1) * P[l + 1]
    end
    return P, dP
end

Test.@testset "the spherical inversion recovers a prescribed angular spectrum" begin
    R = 6.371e6
    g = SFH.SphericalGeometry{2}(SFH.SphericalDistance(R), R)
    L = 10
    Cl = [0.0; [1.7 / (l + 1)^2 for l in 1:L]]
    edges_σ = collect(range(0.0, π; length = 4001))
    mids_σ = collect(SF.midpoints(edges_σ))
    edges = R .* edges_σ

    # a scalar: C(σ) = Σ (2l+1)/(4π) C_l P_l(cos σ), D = 2[C(0) − C]
    C(σ) = sum((2l + 1) / (4π) * Cl[l + 1] * _legendre_and_derivative(cos(σ), L)[1][l + 1] for l in 0:L)
    D = [2 * (C(0.0) - C(σ)) for σ in mids_σ]
    for op in (SFT.ScalarSFType{2}(), SFT.S2SFType())
        sf = SF.StructureFunction(op, edges, D)
        out = SFC.isotropic_spectrum(sf, g, L + 2)
        Test.@test out.l == 1:(L + 2)
        Test.@test maximum(abs, out.C[1:L] .- Cl[2:end]) < 1e-5 * maximum(Cl)
        Test.@test maximum(abs, out.C[(L + 1):end]) < 1e-5 * maximum(Cl)
    end
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(SF.StructureFunction(SFT.L2SFType(), edges, D), g, L)
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(
        SF.StructureFunction(SFT.S2SFType(), 2 .* edges, D), g, L)          # beyond a half turn
    raw = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, D .* 3, fill(UInt32(3), length(D)))
    Test.@test SFC.isotropic_spectrum(raw, g, L).C ≈ SFC.isotropic_spectrum(SF.StructureFunction(SFT.S2SFType(), edges, D), g, L).C
    raw.counts[5] = 0
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(raw, g, L)

    # a tangent vector field: ξ₊ from d^l_{11}, ξ₋ from d^l_{1,−1}, both in closed form from P_l, P_l'
    CE = [0.0; [0.9 / l^2 for l in 1:L]]
    CB = [0.0; [0.4 / (l + 2)^2 for l in 1:L]]
    variance = sum((2l + 1) / (4π) * (CE[l + 1] + CB[l + 1]) for l in 1:L)
    function ξ(σ)
        x = cos(σ)
        P, dP = _legendre_and_derivative(x, L)
        plus = 0.0
        minus = 0.0
        for l in 1:L
            d11 = (l * (l + 1) * P[l + 1] + (1 - x) * dP[l + 1]) / (l * (l + 1))
            d1m1 = (-l * (l + 1) * P[l + 1] + (1 + x) * dP[l + 1]) / (l * (l + 1))
            plus += (2l + 1) / (4π) * (CE[l + 1] + CB[l + 1]) * d11
            minus -= (2l + 1) / (4π) * (CE[l + 1] - CB[l + 1]) * d1m1
        end
        return plus, minus
    end
    DLL = Float64[]
    DTT = Float64[]
    for σ in mids_σ
        p, m = ξ(σ)
        push!(DLL, 2 * (variance / 2 - (p + m) / 2))
        push!(DTT, 2 * (variance / 2 - (p - m) / 2))
    end
    L2 = SF.StructureFunction(SFT.L2SFType(), edges, DLL)
    T2 = SF.StructureFunction(SFT.T2SFType(), edges, DTT)
    out = SFC.helmholtz_spectra(L2, T2, g, L + 2; variance)
    Test.@test maximum(abs, out.E[1:L] .- CE[2:end]) < 1e-5 * maximum(CE)
    Test.@test maximum(abs, out.B[1:L] .- CB[2:end]) < 1e-5 * maximum(CE)
    Test.@test maximum(abs, out.E[(L + 1):end]) < 1e-5 * maximum(CE)
    Test.@test maximum(abs, out.B[(L + 1):end]) < 1e-5 * maximum(CE)
    Test.@test_throws UndefKeywordError SFC.helmholtz_spectra(L2, T2, g, L)
end

# Fibonacci points and a degree-2 harmonic Φ = xy with its surface gradient in (east, north).
function _sphere_harmonic_field(N)
    φg = (1 + sqrt(5)) / 2
    x = Matrix{Float64}(undef, 2, N)
    Φ = Vector{Float64}(undef, N)
    u = Matrix{Float64}(undef, 2, N)
    for i in 1:N
        lat = asin(1 - 2 * (i - 0.5) / N)
        lon = mod(2π * i / φg, 2π)
        x[1, i] = lon
        x[2, i] = lat
        p = SA.SVector(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
        Φ[i] = p[1] * p[2]
        grad = SA.SVector(p[2], p[1], 0.0)
        tang = grad - (grad ⋅ p) * p
        E = SA.SVector(-sin(lon), cos(lon), 0.0)
        Nn = SA.SVector(-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))
        u[1, i] = tang ⋅ E
        u[2, i] = tang ⋅ Nn
    end
    return x, Φ, u
end
using LinearAlgebra: ⋅

Test.@testset "the spherical inversion of the package's own pair sums" begin
    # Φ = xy is a pure l = 2 harmonic: C_2 = 4π⟨Φ²⟩/5 and every other C_l vanishes; its gradient has
    # C^E_2 = l(l+1) C^Φ_2 and no curl. Pair sums over a point set carry Monte-Carlo scatter.
    N = 4000
    x, Φ, u = _sphere_harmonic_field(N)
    g = SFH.SphericalGeometry{2}(SFH.SphericalDistance(1.0), 1.0)
    edges = collect(range(0.0, π; length = 41))
    meanΦ2 = sum(abs2, Φ) / N
    l = 2
    C2 = 4π * meanΦ2 / 5

    sfΦ = SFC.calculate_structure_function(SFT.ScalarSFType{2}(), x, Fields(scalars = (Φ,)), edges;
        distance_metric = SFH.SphericalDistance(1.0), backend = CB.SerialBackend(),
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test all(>(0), sfΦ.counts)
    outΦ = SFC.isotropic_spectrum(sfΦ, g, 6)
    Test.@test abs(outΦ.C[2] - C2) < 0.03 * C2
    Test.@test all(abs.(outΦ.C[[1, 3, 4, 5, 6]]) .< 0.03 * C2)

    nb = length(edges) - 1
    sL = zeros(nb); cL = zeros(Int, nb)
    SFC.calculate_structure_function!(sL, cL, SFT.L2SFType(), x, u, edges; distance_metric = SFH.SphericalDistance(1.0))
    sT = zeros(nb); cT = zeros(Int, nb)
    SFC.calculate_structure_function!(sT, cT, SFT.T2SFType(), x, u, edges; distance_metric = SFH.SphericalDistance(1.0))
    L2 = SF.StructureFunctionSumsAndCounts(SFT.L2SFType(), edges, sL, cL)
    T2 = SF.StructureFunctionSumsAndCounts(SFT.T2SFType(), edges, sT, cT)
    variance = sum(abs2, u) / N
    Test.@test variance ≈ l * (l + 1) * meanΦ2 rtol = 0.02
    out = SFC.helmholtz_spectra(L2, T2, g, 6; variance)
    E2 = l * (l + 1) * C2
    Test.@test abs(out.E[2] - E2) < 0.05 * E2
    Test.@test all(abs.(out.E[[1, 3, 4, 5, 6]]) .< 0.05 * E2)
    Test.@test all(abs.(out.B) .< 0.05 * E2)
end

Test.@testset "helmholtz_decompose_2d carries each component's own counts" begin
    edges = collect(range(0.0, 2.0; length = 11))
    mids = SF.midpoints(edges)
    DLL = [r^(2 / 3) for r in mids]
    DTT = (5 / 3) .* DLL
    cL = fill(UInt32(4), 10)
    cT = fill(UInt32(7), 10)
    h = SFC.helmholtz_decompose_2d(edges, DLL .* cL, cL, DTT .* cT, cT)
    Test.@test h.rotational_counts == cT
    Test.@test h.divergent_counts == cL
    Test.@test h.rotational_sums ./ h.rotational_counts .+ h.divergent_sums ./ h.divergent_counts ≈ DLL .+ DTT
    Test.@test eltype(h.rotational_sums) == Float64
end
