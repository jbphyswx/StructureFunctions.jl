using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH, Fields, HarmonicNodes
using SpectralBackends: SpectralBackends as SB
using NUFSHT: NUFSHT
using NonuniformFFTs: NonuniformFFTs
using FlowGeometries: FlowGeometries as FG
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using LinearAlgebra: dot, cross
using Random: Random

const DS = SB.DirectSumSpectralBackend()
const NU = SB.NUFSHTSpectralBackend()

# Wigner's explicit sum, independent of the recurrence under test.
function _wigner_explicit(j, mp, m, β)
    (abs(mp) > j || abs(m) > j) && return 0.0
    c, s = cos(β / 2), sin(β / 2)
    f(n) = factorial(big(n))
    pref = sqrt(f(j + mp) * f(j - mp) * f(j + m) * f(j - m))
    tot = big(0.0)
    for k in max(0, m - mp):min(j + m, j - mp)
        tot += (-1)^(mp - m + k) * big(c)^(2j + m - mp - 2k) * big(s)^(mp - m + 2k) /
               (f(j + m - k) * f(k) * f(mp - m + k) * f(j - mp - k))
    end
    return Float64(pref * tot)
end

# A band-limited test field: the surface gradient of the degree-2 potential Φ = 0.7xy + 0.3xz + 0.5x +
# 0.6(3z² − 1)/2, and its curl p̂ × ∇Φ, in (east, north) components; and a scalar of degree 2.
function _field_at(p::SA.SVector{3})
    grad = SA.SVector(0.7p[2] + 0.3p[3] + 0.5, 0.7p[1], 0.3p[1] + 1.8p[3])
    return grad - dot(grad, p) * p
end
_scalar_at(p::SA.SVector{3}) = 0.9p[1] * p[2] - 0.4p[3] + 0.5 * (3p[3]^2 - 1) / 2
_east(lon, lat) = SA.SVector(-sin(lon), cos(lon), 0.0)
_north(lon, lat) = SA.SVector(-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))
_position(lon, lat) = SA.SVector(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))

# Gauss–Legendre grid with exact quadrature weights, and the test fields on it.
function _gl_grid(nlat, nlon)
    μ, wμ = SF.gauss_legendre(nlat)
    lats = asin.(μ)
    lons = (0:(nlon - 1)) .* (2π / nlon)
    N = nlat * nlon
    x = Matrix{Float64}(undef, 2, N)
    w = Vector{Float64}(undef, N)
    ug = Matrix{Float64}(undef, 2, N)
    uc = Matrix{Float64}(undef, 2, N)
    Φ = Vector{Float64}(undef, N)
    k = 0
    for j in 1:nlat, i in 1:nlon
        k += 1
        lon, lat = lons[i], lats[j]
        x[1, k] = lon
        x[2, k] = lat
        w[k] = wμ[j] * 2π / nlon
        p = _position(lon, lat)
        t = _field_at(p)
        E, Nn = _east(lon, lat), _north(lon, lat)
        ug[1, k], ug[2, k] = dot(t, E), dot(t, Nn)
        c = cross(p, t)
        uc[1, k], uc[2, k] = dot(c, E), dot(c, Nn)
        Φ[k] = _scalar_at(p)
    end
    return x, w, ug, uc, Φ
end

# Fibonacci points with the test fields.
function _fibonacci(N)
    φg = (1 + sqrt(5)) / 2
    x = Matrix{Float64}(undef, 2, N)
    u = Matrix{Float64}(undef, 2, N)
    Φ = Vector{Float64}(undef, N)
    for i in 1:N
        lat = asin(1 - 2 * (i - 0.5) / N)
        lon = mod(2π * i / φg, 2π)
        x[1, i], x[2, i] = lon, lat
        p = _position(lon, lat)
        t = _field_at(p)
        u[1, i], u[2, i] = dot(t, _east(lon, lat)), dot(t, _north(lon, lat))
        Φ[i] = _scalar_at(p)
    end
    return x, u, Φ
end

# The continuous rotation average of `sf(δu)` over every pair at separation β, by a quadrature over
# the first point (Gauss–Legendre in latitude, uniform in longitude) and over the bearing of the
# geodesic to the second: exact for a band-limited field, since every integrand is a polynomial in
# the coordinates and a trigonometric polynomial in the angles.
function _rotation_average(sf, β; nlat = 40, nlon = 81, nα = 64, field = _field_at, scalar = false)
    μ, wμ = SF.gauss_legendre(nlat)
    acc = 0.0
    total = 0.0
    for j in 1:nlat, i in 1:nlon, a in 1:nα
        lat = asin(μ[j])
        lon = (i - 1) * 2π / nlon
        α = (a - 1) * 2π / nα
        p = _position(lon, lat)
        θ̂ = -_north(lon, lat)
        φ̂ = _east(lon, lat)
        tA = cos(α) * θ̂ + sin(α) * φ̂                   # geodesic tangent at p, toward q
        q = cos(β) * p + sin(β) * tA
        tB = -sin(β) * p + cos(β) * tA                    # the geodesic's tangent at q, continuing
        m̂ = cross(p, tA)
        w = wμ[j]
        if scalar
            inc = SF.Channels.ChannelIncrement{0, 0, 1, Float64}((), (_scalar_at(q) - _scalar_at(p),))
            acc += w * sf(inc, SA.SVector(1.0, 0.0))
        else
            uA, uB = field(p), field(q)
            δu = SA.SVector(dot(uB, tB) - dot(uA, tA), dot(uB - uA, m̂))
            acc += w * sf(δu, SA.SVector(1.0, 0.0))
        end
        total += w
    end
    return acc / total
end

# `_rotation_average` for the mixed operator needs the scalar with the vector.
function _rotation_average(sf::SFT.MixedStructureFunctionType, β; nlat = 40, nlon = 81, nα = 64)
    μ, wμ = SF.gauss_legendre(nlat)
    acc = 0.0
    total = 0.0
    for j in 1:nlat, i in 1:nlon, a in 1:nα
        lat = asin(μ[j])
        lon = (i - 1) * 2π / nlon
        α = (a - 1) * 2π / nα
        p = _position(lon, lat)
        tA = cos(α) * (-_north(lon, lat)) + sin(α) * _east(lon, lat)
        q = cos(β) * p + sin(β) * tA
        tB = -sin(β) * p + cos(β) * tA
        m̂ = cross(p, tA)
        uA, uB = _field_at(p), _field_at(q)
        δu = SA.SVector(dot(uB, tB) - dot(uA, tA), dot(uB - uA, m̂))
        inc = SF.Channels.ChannelIncrement{2, 1, 1, Float64}((δu,), (_scalar_at(q) - _scalar_at(p),))
        acc += wμ[j] * sf(inc, SA.SVector(1.0, 0.0))
        total += wμ[j]
    end
    return acc / total
end

Test.@testset "the Wigner-d recurrence agrees with the explicit formula and the tables" begin
    worst = 0.0
    for β in (0.0, 0.3, 1.2, 2.5, π), m in -3:3, n in -3:3
        col = SFC.wigner_d_column(m, n, β, 12)
        for l in 0:12
            worst = max(worst, abs(col[l + 1] - _wigner_explicit(l, m, n, β)))
        end
    end
    Test.@test worst < 1e-13
    β = 0.7
    c, s = cos(β), sin(β)
    Test.@test SFC.wigner_d_column(1, 1, β, 2)[2:3] ≈ [(1 + c) / 2, (2c^2 + c - 1) / 2]
    Test.@test SFC.wigner_d_column(1, -1, β, 2)[2:3] ≈ [(1 - c) / 2, (-2c^2 + c + 1) / 2]
    Test.@test SFC.wigner_d_column(2, 0, β, 2)[3] ≈ sqrt(3 / 8) * s^2
    Test.@test SFC.wigner_d_column(1, 0, β, 2)[2:3] ≈ [-s / sqrt(2), -sqrt(3 / 8) * sin(2β)]
    # spin zero is Legendre
    Test.@test SFC.wigner_d_column(0, 0, β, 3)[4] ≈ (5c^3 - 3c) / 2
    # the closed forms the spherical inversion of G4 rests on
    x = c
    P, dP = SFC._legendre_values(x, 7), nothing
    for l in 1:6
        dPl = l * (x * P[l + 1] - P[l]) / (x^2 - 1)
        Test.@test SFC.wigner_d_column(1, 1, β, 6)[l + 1] ≈ (1 - x) * dPl / (l * (l + 1)) + P[l + 1]
        Test.@test SFC.wigner_d_column(1, -1, β, 6)[l + 1] ≈ (1 + x) * dPl / (l * (l + 1)) - P[l + 1]
    end
end

Test.@testset "the harmonic nodes and their weights" begin
    n = HarmonicNodes(12, 20)
    Test.@test length(n) == 12
    Test.@test issorted(n.separations)
    Test.@test sum(n.weights) ≈ 2.0
    Test.@test SF.midpoints(n) === n.separations
    Test.@test SF.n_histogram_bins(n) == 12
    μ, w = SF.gauss_legendre(8)
    Test.@test sum(w .* μ .^ 6) ≈ 2 / 7            # exact for degree 15
    m = HarmonicNodes([0.2, 0.9, 2.0], 10; taper = SF.Bartlett())
    Test.@test sum(m.weights) ≈ 2.0
    Test.@test_throws ArgumentError HarmonicNodes([0.9, 0.2], 10)
    Test.@test_throws ArgumentError HarmonicNodes([0.2, 3.5], 10)
    res = SF.StructureFunctionSumsAndCounts(SFT.L2SFType(), m, zeros(3), zeros(3))
    Test.@test length(res.sums) == 3
    Test.@test_throws DimensionMismatch SF.StructureFunctionSumsAndCounts(SFT.L2SFType(), m, zeros(4), zeros(4))
end

Test.@testset "a scalar's kernel-binned moments are the pseudo-spectral series exactly" begin
    # The identity, with arbitrary weights and a mask: Σ_ij w_i w_j (θ_j − θ_i)^p K_L(γ_ij, β) equals the
    # series of the masked monomials' cross pseudo-spectra, term for term, to round-off.
    Random.seed!(3100)
    N, L = 110, 8
    x = Matrix(vcat((2π .* rand(N))', (asin.(2 .* rand(N) .- 1))'))
    f = randn(N)
    w = 0.5 .+ rand(N)
    valid = rand(N) .> 0.3
    nodes = HarmonicNodes([0.3, 1.0, 2.0, 2.9], L)
    pts = [_position(x[1, i], x[2, i]) for i in 1:N]
    Pβ = [SFC.wigner_d_column(0, 0, β, L) for β in nodes.separations]
    Pγ = [SFC.wigner_d_column(0, 0, acos(clamp(dot(pts[i], pts[j]), -1, 1)), L) for i in 1:N, j in 1:N]
    for sf in (SFT.ScalarSFType{2}(), SFT.ScalarSFType{3}(), SFT.ScalarSFType{4}())
        P = SFT.order(sf)
        if isodd(P)
            Test.@test_throws ArgumentError SFC.calculate_structure_function(sf, x, Fields(scalars = (f,)), nodes, DS;
                weights = w, valid, verbose = false)
            continue
        end
        res = SFC.calculate_structure_function(sf, x, Fields(scalars = (f,)), nodes, DS; weights = w, valid,
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
        for (k, β) in enumerate(nodes.separations)
            ss = 0.0
            cc = 0.0
            for i in 1:N, j in 1:N
                (valid[i] && valid[j]) || continue
                Kl = sum((2l + 1) * Pγ[i, j][l + 1] * Pβ[k][l + 1] for l in 0:L) / (16π^2)
                cc += w[i] * w[j] * Kl
                ss += w[i] * w[j] * (f[j] - f[i])^P * Kl
            end
            Test.@test isapprox(res.sums[k], ss; rtol = 1e-11)
            Test.@test isapprox(res.counts[k], cc; rtol = 1e-11)
        end
    end
end

Test.@testset "second-order vector statistics equal the spin-1 closed forms on a Gauss–Legendre grid" begin
    # A pure degree-2 potential: C^E_2 = l(l+1)·4π⟨Φ²⟩/5, ξ₊ = (5/4π)C^E_2 d²_{11}, ξ₋ = −(5/4π)C^E_2 d²_{1,−1};
    # with exact quadrature the pseudo-coefficients are the coefficients and the identity is exact.
    μ, wμ = SF.gauss_legendre(12)
    nlon = 25
    N = 12 * nlon
    x = Matrix{Float64}(undef, 2, N)
    w = Vector{Float64}(undef, N)
    ug = Matrix{Float64}(undef, 2, N)
    uc = Matrix{Float64}(undef, 2, N)
    Φ = Vector{Float64}(undef, N)
    k = 0
    for j in 1:12, i in 1:nlon
        k += 1
        lon, lat = (i - 1) * 2π / nlon, asin(μ[j])
        x[1, k], x[2, k], w[k] = lon, lat, wμ[j] * 2π / nlon
        p = _position(lon, lat)
        Φ[k] = p[1] * p[2]
        grad = SA.SVector(p[2], p[1], 0.0)
        t = grad - dot(grad, p) * p
        ug[1, k], ug[2, k] = dot(t, _east(lon, lat)), dot(t, _north(lon, lat))
        c = cross(p, t)
        uc[1, k], uc[2, k] = dot(c, _east(lon, lat)), dot(c, _north(lon, lat))
    end
    meanΦ2 = sum(w .* Φ .^ 2) / (4π)
    CE2 = 6 * 4π * meanΦ2 / 5
    L = 8
    nodes = HarmonicNodes(9, L)
    d11 = [SFC.wigner_d_column(1, 1, β, 2)[3] for β in nodes.separations]
    d1m1 = [SFC.wigner_d_column(1, -1, β, 2)[3] for β in nodes.separations]
    ξp = (5 / (4π)) * CE2 .* d11
    ξm = -(5 / (4π)) * CE2 .* d1m1
    CLL, CTT = (ξp .+ ξm) ./ 2, (ξp .- ξm) ./ 2
    C0 = (5 / (4π)) * CE2 / 2
    DLL, DTT = 2 .* (C0 .- CLL), 2 .* (C0 .- CTT)
    for (sf, ref) in ((SFT.L2SFType(), DLL), (SFT.T2SFType(), DTT), (SFT.S2SFType(), DLL .+ DTT),
                      (SFT.T2ComponentSFType(), DTT))
        r = SFC.calculate_structure_function(sf, x, ug, nodes, DS; weights = w,
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
        Test.@test maximum(abs, r.counts .- 1) < 1e-12               # ∫∫ K dΩ dΩ′ = 1 on the full sphere
        Test.@test maximum(abs, r.sums ./ r.counts .- ref) < 1e-12 * maximum(abs, ref)
    end
    for (sf, ref) in ((SFT.L2SFType(), DTT), (SFT.T2SFType(), DLL))
        r = SFC.calculate_structure_function(sf, x, uc, nodes, DS; weights = w, verbose = false)
        Test.@test maximum(abs, r.values .- ref) < 1e-12 * maximum(abs, ref)
    end
    # the same numbers through a Fields bundle, and the scalar's own closed form 2⟨Φ²⟩(1 − P₂)
    fb = Fields(vectors = (ug,), scalars = (Φ,))
    rL = SFC.calculate_structure_function(SFT.L2SFType(), x, fb, nodes, DS; weights = w, verbose = false)
    Test.@test maximum(abs, rL.values .- DLL) < 1e-12 * maximum(abs, DLL)
    rΦ = SFC.calculate_structure_function(SFT.ScalarSFType{2}(), x, fb, nodes, DS; weights = w, verbose = false)
    P2 = [SFC.wigner_d_column(0, 0, β, 2)[3] for β in nodes.separations]
    Test.@test maximum(abs, rΦ.values .- 2 .* meanΦ2 .* (1 .- P2)) < 1e-12 * meanΦ2
    # gradient is E, curl is B, and the scalar's spectrum is its own
    sg = SFC.harmonic_spectra(x, ug, L, DS; weights = w)
    sc = SFC.harmonic_spectra(x, uc, L, DS; weights = w)
    Test.@test sg.EE[3] ≈ CE2 rtol = 1e-12
    Test.@test maximum(abs, sg.BB) < 1e-12 * CE2
    Test.@test maximum(abs, sg.EE[[1, 2, 4, 5, 6, 7, 8, 9]]) < 1e-12 * CE2
    Test.@test sc.BB[3] ≈ CE2 rtol = 1e-12
    Test.@test maximum(abs, sc.EE) < 1e-12 * CE2
    Test.@test sum((2l + 1) / (4π) * (sg.EE[l + 1] + sg.BB[l + 1]) for l in 0:L) ≈ sum(w .* vec(sum(abs2, ug; dims = 1))) / (4π) rtol = 1e-12
    Test.@test SFC.harmonic_spectra(x, reshape(Φ, 1, :), L, DS; weights = w).C[3] ≈ 4π * meanΦ2 / 5 rtol = 1e-12
end

Test.@testset "higher-order statistics on a band-limited field equal the exact rotation average" begin
    # A degree-2 field's cubic and quartic products have degree ≤ 6 and 8, so the series terminate: with
    # lmax above that and exact quadrature, the kernel-binned statistic IS the continuous rotation average,
    # which an independent quadrature over pairs computes.
    x, w, ug, uc, Φ = _gl_grid(14, 29)
    nodes = HarmonicNodes([0.4, 1.1, 1.9, 2.6], 10)
    for sf in (SFT.L3SFType(), SFT.S3SFType(), SFT.L1T2SFType(), SFT.L2T1SFType(), SFT.T3SFType(),
               SFT.ProjectedStructureFunctionType{4, 0}(), SFT.L2SFType(), SFT.T2SFType())
        r = SFC.calculate_structure_function(sf, x, ug, nodes, DS; weights = w,
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
        Test.@test maximum(abs, r.counts .- 1) < 1e-12
        ref = [_rotation_average(sf, β) for β in nodes.separations]
        scale = max(maximum(abs, ref), 1e-3)
        Test.@test maximum(abs, r.sums .- ref) < 1e-9 * scale
    end
    # a mixed bundle: the scalar with the vector, at third order
    fb = Fields(vectors = (ug,), scalars = (Φ,))
    r = SFC.calculate_structure_function(SFT.MixedSFType{1, 0, 2}(), x, fb, nodes, DS; weights = w,
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
    ref = [_rotation_average(SFT.MixedSFType{1, 0, 2}(), β) for β in nodes.separations]
    Test.@test maximum(abs, r.sums .- ref) < 1e-9 * maximum(abs, ref)
    # the scalar alone at fourth order
    r4 = SFC.calculate_structure_function(SFT.ScalarSFType{4}(), x, Fields(scalars = (Φ,)), nodes, DS; weights = w,
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
    ref4 = [_rotation_average(SFT.ScalarSFType{4}(), β; scalar = true) for β in nodes.separations]
    Test.@test maximum(abs, r4.sums .- ref4) < 1e-9 * maximum(abs, ref4)
end

Test.@testset "on a point set the kernel-binned statistic converges to the rotation average as lmax grows" begin
    N = 1500                                              # equal-area points ≈ 0.09 rad apart
    x, u, Φ = _fibonacci(N)
    βs = [0.6, 1.2, 1.8, 2.4]
    for (sf, field, scalar) in ((SFT.L2SFType(), u, false), (SFT.T2SFType(), u, false), (SFT.L3SFType(), u, false),
                                (SFT.ScalarSFType{2}(), Fields(scalars = (Φ,)), true))
        exact = [_rotation_average(sf, β; scalar) for β in βs]
        scale = maximum(abs, exact)
        # kernels wider than the point spacing: the error falls with the kernel width
        errs = [maximum(abs.(SFC.calculate_structure_function(sf, x, field, HarmonicNodes(βs, L; taper = SF.GaussianTaper(1.0 / L)),
                                                              DS; verbose = false).values .- exact)) / scale for L in (6, 12, 24, 48)]
        Test.@test issorted(errs; rev = true)
        Test.@test errs[end] < 2e-3
    end
end

Test.@testset "the fast transform gives the direct sum's answer" begin
    Random.seed!(3200)
    N = 400
    x, u, Φ = _fibonacci(N)
    valid = rand(N) .> 0.25
    w = 0.5 .+ rand(N)
    nodes = HarmonicNodes(6, 24; taper = SF.Bartlett())
    fb = Fields(vectors = (u,), scalars = (Φ,))
    for sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.L1T2SFType(), SFT.ScalarSFType{2}(), SFT.MixedSFType{1, 0, 2}())
        a = SFC.calculate_structure_function(sf, x, fb, nodes, DS; weights = w, valid,
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
        b = SFC.calculate_structure_function(sf, x, fb, nodes, NU; weights = w, valid,
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
        Test.@test maximum(abs, a.sums .- b.sums) < 1e-8 * maximum(abs, a.sums)
        Test.@test maximum(abs, a.counts .- b.counts) < 1e-8 * maximum(abs, a.counts)
    end
    sa = SFC.harmonic_spectra(x, u, 24, DS; weights = w, valid)
    sb = SFC.harmonic_spectra(x, u, 24, NU; weights = w, valid)
    Test.@test maximum(abs, sa.EE .- sb.EE) < 1e-8 * maximum(sa.EE)
    Test.@test maximum(abs, sa.BB .- sb.BB) < 1e-8 * maximum(sa.EE)
end

Test.@testset "the harmonic route refuses what it cannot mean" begin
    x, u, Φ = _fibonacci(60)
    nodes = HarmonicNodes(4, 6)
    # odd in a scalar: both readings of every pair are summed, so the moment is identically zero
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.ScalarSFType{3}(), x, Fields(scalars = (Φ,)), nodes, DS; verbose = false)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.MixedSFType{1, 0, 1}(), x, Fields(vectors = (u,), scalars = (Φ,)), nodes, DS; verbose = false)
    # a norm is no polynomial
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.FullVectorStructureFunctionType{3}(), x, u, nodes, DS; verbose = false)
    # the sphere is the only geometry
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.L2SFType(), x, u, nodes, DS; distance_metric = DI.Euclidean(), verbose = false)
    # no provider without its package
    Test.@test_throws ArgumentError SFC.harmonic_sweep!(zeros(4), zeros(4), SFT.L2SFType(),
        SFH.SphericalGeometry{2}(DI.SphericalAngle(), 1.0), x, ones(60), u, nodes, Val(2), Val(1), Val(0), :none)
    # a channel the bundle lacks
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.VectorDotSFType(1, 2), x, Fields(vectors = (u,)), nodes, DS; verbose = false)
end

Test.@testset "a spherical grid reaches the harmonic route with its cell measure as weights" begin
    geo = FG.Geometry.SphericalGeometry(1.0)
    grid = FG.Connectivity.structured_grid(FG.SphericalSampling.GaussLegendreSampling(), 12; geometry = geo, nlon = 25)
    coords = FG.Grids.materialize(grid)
    N = length(coords[1])
    u = Matrix{Float64}(undef, 2, N)
    for i in 1:N
        lon, lat = coords[1][i], coords[2][i]
        t = _field_at(_position(lon, lat))
        u[1, i], u[2, i] = dot(t, _east(lon, lat)), dot(t, _north(lon, lat))
    end
    nodes = HarmonicNodes(6, 10)
    ug = reshape(u, 2, size(FG.Grids.mask(grid))...)
    got = SFC.calculate_structure_function(SFT.L2SFType(), grid, ug, nodes, DS; verbose = false,
        output_type = SF.StructureFunctionSumsAndCounts)
    x = Matrix(hcat(coords[1], coords[2])')
    ref = SFC.calculate_structure_function(SFT.L2SFType(), x, u, nodes, DS; weights = vec(FG.Grids.measure_array(grid)),
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false)
    Test.@test got.sums ≈ ref.sums rtol = 1e-12
    Test.@test got.counts ≈ ref.counts rtol = 1e-12
    Test.@test got.distance === nodes
end

Test.@testset "kernel-binned results invert to the spectra they came from" begin
    g = SFH.SphericalGeometry{2}(SFH.SphericalDistance(1.0), 1.0)
    # exact on a Gauss–Legendre grid: the nodes' quadrature integrates a band-limited D(β) exactly
    x, w, ug, _, Φ = _gl_grid(16, 33)
    nodes = HarmonicNodes(10, 8)
    rs = SFC.calculate_structure_function(SFT.ScalarSFType{2}(), x, Fields(scalars = (Φ,)), nodes, DS; weights = w, verbose = false)
    C = SFC.isotropic_spectrum(rs, g, 6)
    Test.@test C.l == 1:6
    Test.@test C.C ≈ [0.16 * 4π / 9, (0.2π + 3.24π / 15) / 5, 0, 0, 0, 0] atol = 1e-12       # from the scalar's coefficients
    L2 = SFC.calculate_structure_function(SFT.L2SFType(), x, ug, nodes, DS; weights = w, verbose = false)
    T2 = SFC.calculate_structure_function(SFT.T2SFType(), x, ug, nodes, DS; weights = w, verbose = false)
    variance = sum(w .* vec(sum(abs2, ug; dims = 1))) / (4π)
    eb = SFC.helmholtz_spectra(L2, T2, g, 6; variance)
    Test.@test eb.E ≈ [2π / 9, 6 * (1.96π / 15 + 0.36π / 15 + 0.288π) / 5, 0, 0, 0, 0] atol = 1e-12   # l(l+1) C^Φ_l
    Test.@test maximum(abs, eb.B) < 1e-12
    Test.@test_throws ArgumentError SFC.helmholtz_spectra(L2, SFC.calculate_structure_function(SFT.T2SFType(), x, ug,
        HarmonicNodes(10, 9), DS; weights = w, verbose = false), g, 6; variance)
    # a masked ensemble: the ratio of kernel-binned sums to kernel-binned counts, inverted, follows the
    # exact spectrum of each realisation where the pseudo-spectrum of the masked field does not
    Random.seed!(3)
    xg, wg, _, _, _ = _gl_grid(24, 48)
    N = size(xg, 2)
    Lf = 6
    Ct = [1 / (l + 1)^2 for l in 1:Lf]
    plan = NUFSHT.make_spin_plan(ComplexF64, π / 2 .- xg[2, :], xg[1, :], Lf, 0)
    mask = rand(N) .> 0.4
    L = 24
    nodes = HarmonicNodes(32, L; taper = SF.GaussianTaper(1.0 / L))
    lmax = 8
    ratio = zeros(lmax)
    pseudo = zeros(lmax + 1)
    full = zeros(lmax + 1)
    R = 60
    for _ in 1:R
        a = zeros(ComplexF64, Lf + 1, 2Lf + 1)
        for l in 1:Lf, m in -l:l
            a[l + 1, m + Lf + 1] = sqrt(Ct[l]) * randn(ComplexF64)
        end
        f = real.(NUFSHT.nusht_type2_spin!(zeros(ComplexF64, N), a, plan))
        res = SFC.calculate_structure_function(SFT.ScalarSFType{2}(), xg, Fields(scalars = (f,)), nodes, DS;
                                               weights = wg, valid = mask, verbose = false)
        ratio .+= SFC.isotropic_spectrum(res, g, lmax).C
        pseudo .+= SFC.harmonic_spectra(xg, reshape(f, 1, :), lmax, DS; weights = wg, valid = mask).C
        full .+= SFC.harmonic_spectra(xg, reshape(f, 1, :), lmax, DS; weights = wg).C
    end
    truth = Ct ./ 2                                        # the real part of a circular field carries half the power
    Test.@test maximum(abs.(full[2:(Lf + 1)] .- R .* truth) ./ (R .* truth)) < 0.1
    Test.@test maximum(abs.(ratio[1:Lf] .- full[2:(Lf + 1)]) ./ (R .* truth)) < 0.1
    Test.@test all(pseudo[2:(Lf + 1)] ./ full[2:(Lf + 1)] .< 0.6)
    Test.@test maximum(abs, ratio[(Lf + 1):lmax]) < 0.05 * R * truth[end]
end
