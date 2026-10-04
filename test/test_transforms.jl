using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using Bessels: Bessels
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using Random: Random
using Statistics: var
using Test: Test
using JLArrays: JLArrays, JLArray
using GPUArraysCore: GPUArraysCore

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

# S₂ = 2σ²[1 − exp(−r²/2ℓ²)] transforms pointwise to σ² ℓ^D exp(−k²ℓ²/2)/(2π)^(D/2), on linear and log separations.
Test.@testset "the transform reproduces an analytic spectrum" begin
    σ2, ℓ = 1.7, 0.8
    s2(x) = 2σ2 * (1 - exp(-x^2 / (2ℓ^2)))
    r = collect(range(0.0, 20ℓ; length = 401))
    rl = SF.midpoints(SF.LogBinEdges(1e-3, 20ℓ, 401))
    kq = collect(range(1e-4, 40 / ℓ; length = 200))
    for (D, tol) in ((1, 1e-13), (2, 4e-4), (3, 1e-13))
        exact = @. σ2 * ℓ^D * exp(-kq^2 * ℓ^2 / 2) / (2π)^(D / 2)
        P = SFC.isotropic_spectrum(SFT.S2SFType(), r, s2.(r), kq, Val(D); variance = σ2)
        Test.@test maximum(abs, P .- exact) < tol * maximum(exact)
        Pl = SFC.isotropic_spectrum(SFT.S2SFType(), rl, s2.(rl), kq, Val(D); variance = σ2)
        Test.@test maximum(abs, Pl .- exact) < 2e-4 * maximum(exact)
    end
end

Test.@testset "the transform refuses what it cannot answer" begin
    r = collect(range(0.0, 10.0; length = 100))
    s2 = @. 1 - cos(2.0 * r)
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(
        SFT.S2SFType(), r, s2, [0.0, 1.0], Val(1); variance = 0.5)
    Test.@test_throws DimensionMismatch SFC.isotropic_spectrum(
        SFT.S2SFType(), r, s2[1:end-1], [1.0], Val(1); variance = 0.5)
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(
        SFT.S2SFType(), reverse(r), s2, [1.0], Val(1); variance = 0.5)
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(SFT.S2SFType(), r, s2, [1.0], Val(5); variance = 0.5)
    for op in (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.L1T2SFType())
        Test.@test_throws ArgumentError SFC.isotropic_spectrum(op, r, s2, [1.0], Val(1); variance = 0.5)
    end
    Test.@test_throws UndefKeywordError SFC.isotropic_spectrum(SFT.S2SFType(), r, s2, [1.0], Val(1))
end

# One mode of wavenumber k₀ on scattered points, through the package's own pair sums, peaks at k₀.
Test.@testset "a result object transforms back to the wavenumber it was built from" begin
    mv = (5, 2)
    k0 = sqrt(sum(abs2, mv))
    Random.seed!(502)
    N = 300
    x = 2π .* rand(2, N)
    u = zeros(2, N)
    for p in 1:N
        u[1, p] = cos(mv[1] * x[1, p] + mv[2] * x[2, p] + 0.4)
    end
    bins = collect(range(0.0, 3.0; length = 61))
    kq = collect(range(0.3, 12.0; length = 500))
    raw = SFC.calculate_structure_function(
        SFT.S2SFType(), x, u, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend())
    P = SFC.isotropic_spectrum(raw, kq, Val(2); variance = var(u[1, :]; corrected = false))
    Test.@test kq[argmax(P)] ≈ k0 rtol = 0.05
end

# Both result types transform as their bins holding a value do; a result with none is refused.
Test.@testset "empty bins are dropped, not carried as NaN" begin
    edges = collect(range(0.0, 10.0; length = 21))
    mids = collect(SF.midpoints(edges))
    counts = fill(UInt32(5), 20)
    counts[[3, 11]] .= 0
    sums = [Float64(i) for i in 1:20] .* (counts .> 0)
    keep = findall(>(0), counts)
    kq = collect(range(0.5, 5.0; length = 50))
    direct = SFC.isotropic_spectrum(SFT.S2SFType(), mids[keep], sums[keep] ./ counts[keep], kq, Val(3); variance = 10.0)
    raw = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, sums, counts)
    avg = SF.StructureFunction(SFT.S2SFType(), edges, ifelse.(counts .> 0, sums ./ counts, NaN))
    Test.@test SFC.isotropic_spectrum(raw, kq, Val(3); variance = 10.0) ≈ direct
    Test.@test SFC.isotropic_spectrum(avg, kq, Val(3); variance = 10.0) ≈ direct

    allempty = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, zeros(20), zeros(UInt32, 20))
    Test.@test_throws ArgumentError SFC.isotropic_spectrum(allempty, kq, Val(3); variance = 10.0)
end

# On a complete periodic grid the lag-space transform is the field's own periodogram, to round-off.
Test.@testset "the gridded transform is exact against the field's own spectrum" begin
    for (D, n) in ((1, 32), (2, 16), (3, 8))
        dx = 2π / n
        dims = ntuple(_ -> n, D)
        scal = _modal_field(dims, dx, D; seed = 90 + D)
        u = zeros(D, dims...)
        u[1, ntuple(_ -> Colon(), D)...] = scal
        sched = SFC.UniformLagSchedule(dims, ntuple(_ -> dx, D), ntuple(_ -> true, D))
        kaxes, density = SFC.gridded_spectrum(u, sched, Val(D), SB.FastFourierTransformSpectralBackend())
        modes = density .* prod(ntuple(d -> kaxes[d][2] - kaxes[d][1], D))
        direct = abs2.(FFTW.fft(scal)) ./ prod(dims)^2
        direct[ntuple(_ -> 1, D)...] = 0.0
        Test.@test maximum(abs, modes .- direct) / maximum(direct) < 1e-12
    end
end

# With cells missing the lag-space spectrum stays close to the complete one, far closer than zero-filling.
Test.@testset "the gridded transform survives missing data" begin
    D, n = 2, 32
    dx = 2π / n
    dims = (n, n)
    scal = _modal_field(dims, dx, D; seed = 4242)
    u = zeros(D, dims...)
    u[1, :, :] = scal
    sched = SFC.UniformLagSchedule(dims, (dx, dx), (true, true))
    kaxes, full = SFC.gridded_spectrum(u, sched, Val(D), SB.FastFourierTransformSpectralBackend())

    Random.seed!(7)
    valid = rand(prod(dims)) .> 0.3
    _, masked = SFC.gridded_spectrum(u, sched, Val(D), SB.FastFourierTransformSpectralBackend(); valid = valid)
    err = maximum(abs, masked .- full) / maximum(full)

    naive = copy(scal)
    naive[.!reshape(valid, dims)] .= 0.0
    nspec = abs2.(FFTW.fft(naive)) ./ prod(dims)^2
    nspec[1, 1] = 0.0
    dk = (kaxes[1][2] - kaxes[1][1]) * (kaxes[2][2] - kaxes[2][1])
    naive_err = maximum(abs, nspec .- full .* dk) / maximum(full .* dk)

    Test.@test err < 0.2
    Test.@test err < naive_err / 4
end

# Every wavenumber cell lands in exactly one shell of unequal widths.
Test.@testset "shell averaging conserves the spectrum" begin
    Random.seed!(11)
    n = 8
    kaxes = (2π .* collect(FFTW.fftfreq(n, 1.0)), 2π .* collect(FFTW.fftfreq(n, 2.0)))
    density = rand(n, n)
    kmax = sqrt(maximum(abs2, kaxes[1]) + maximum(abs2, kaxes[2]))
    edges = kmax .* [0.0, 0.1, 0.25, 0.5, 0.7, 1.01]
    mids, E = SFC.shell_average(kaxes, density, edges)
    Test.@test length(mids) == length(E) == length(edges) - 1
    Test.@test sum(E .* diff(edges)) ≈ sum(density) rtol = 1e-12
end

# S₃ = c r, S = c r, L₃ = c₃ r with S₃ = 4c₃r/3, a constant SF_A and SF_Au = a r² each give −(c/2)(1 − J₀(KR)) up to scale.
Test.@testset "each flux relation matches its closed form" begin
    R = 3.0
    r = collect(range(0.0, R; length = 1001))
    Ks = [0.7, 2.0, 5.0]
    c, c3, a = 1.3, 0.8, 0.6
    closed(c) = [-(c / 2) * (1 - Bessels.besselj0(K * R)) for K in Ks]
    Test.@test SFC.spectral_flux(SFT.VectorDotSFType(1, 2), r, fill(c, length(r)), Ks) ≈ closed(c) rtol = 2e-5
    Test.@test SFC.spectral_flux(SFT.ScalarDotSFType(1, 2), r, fill(c, length(r)), Ks) ≈ closed(c) rtol = 2e-5
    Test.@test SFC.spectral_flux(SFT.S3SFType(), r, c .* r, Ks) ≈ closed(c) rtol = 5e-5
    Test.@test SFC.spectral_flux(SFT.MixedSFType{1, 0, 2}(), r, c .* r, Ks) ≈ closed(c) rtol = 5e-5
    Test.@test SFC.spectral_flux(SFT.L3SFType(), r, c3 .* r, (4c3 / 3) .* r, Ks) ≈ closed(4c3 / 3) rtol = 5e-5
    Test.@test SFC.enstrophy_flux(SFT.VectorDotSFType(1, 2), r, a .* r .^ 2, Ks) ≈ closed(-4a) rtol = 5e-5
end

# Each result-object method is its vector method over the bins holding a value.
Test.@testset "a result object carries into the flux relation" begin
    edges = collect(range(0.0, 3.0; length = 41))
    mids = collect(SF.midpoints(edges))
    cnt = fill(UInt32(3), 40)
    cnt[7] = 0
    keep = findall(>(0), cnt)
    Ks = [0.7, 2.0, 5.0]
    L(r) = 0.8r * exp(-r^2 / 2)
    S(r) = r * exp(-r)
    A(r) = cos(r)
    raw(op, f) = SF.StructureFunctionSumsAndCounts(op, edges, 3 .* f.(mids) .* (cnt .> 0), cnt)
    L3o, S3o = raw(SFT.L3SFType(), L), raw(SFT.S3SFType(), S)
    Ao = SF.StructureFunction(SFT.VectorDotSFType(1, 2), edges, ifelse.(cnt .> 0, A.(mids), NaN))
    rk = mids[keep]
    Test.@test SFC.spectral_flux(Ao, Ks) ≈ SFC.spectral_flux(SFT.VectorDotSFType(1, 2), rk, A.(rk), Ks)
    Test.@test SFC.spectral_flux(S3o, Ks) ≈ SFC.spectral_flux(SFT.S3SFType(), rk, S.(rk), Ks)
    Test.@test SFC.spectral_flux(L3o, S3o, Ks) ≈ SFC.spectral_flux(SFT.L3SFType(), rk, L.(rk), S.(rk), Ks)
    Test.@test SFC.enstrophy_flux(Ao, Ks) ≈ SFC.enstrophy_flux(SFT.VectorDotSFType(1, 2), rk, A.(rk), Ks)
    Test.@test_throws ArgumentError SFC.spectral_flux(S3o, L3o, Ks)
end

Test.@testset "a flux refuses a moment that carries none" begin
    r = collect(range(0.0, 3.0; length = 31))
    v = fill(1.0, 31)
    Ks = [1.0]
    for op in (SFT.VectorDotSFType(1, 1), SFT.ScalarDotSFType(2, 2), SFT.S2SFType(), SFT.L2SFType(), SFT.L3SFType())
        Test.@test_throws ArgumentError SFC.spectral_flux(op, r, v, Ks)
    end
    Test.@test_throws ArgumentError SFC.enstrophy_flux(SFT.VectorDotSFType(1, 1), r, v, Ks)
    Test.@test_throws ArgumentError SFC.enstrophy_flux(SFT.VectorDotSFType(1, 2), [1.0], [1.0], Ks)
    Test.@test_throws DimensionMismatch SFC.spectral_flux(SFT.L3SFType(), r, v, v[1:10], Ks)
    Test.@test_throws ArgumentError SFC.spectral_flux(SFT.S3SFType(), reverse(r), v, Ks)
end

# C = σ² − D/2 for a Gaussian correlation, from each second-order moment, and linear in the variance given.
Test.@testset "the covariance is the variance less half the structure function" begin
    σ2, ℓ = 1.7, 0.8
    edges = collect(range(0.0, 6.0; length = 41))
    mids = collect(SF.midpoints(edges))
    d = [2σ2 * (1 - exp(-r^2 / (2ℓ^2))) for r in mids]
    exact = [σ2 * exp(-r^2 / (2ℓ^2)) for r in mids]
    for op in (SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType())
        r, C = SFC.covariance(SF.StructureFunction(op, edges, d), σ2)
        Test.@test r ≈ mids
        Test.@test C ≈ exact rtol = 1e-12
    end
    Test.@test SFC.covariance(SF.StructureFunction(SFT.S2SFType(), edges, d), 2σ2)[2] ≈ exact .+ σ2
end

Test.@testset "a covariance needs a second-order moment" begin
    edges = collect(range(0.0, 4.0; length = 11))
    vals = fill(1.0, 10)
    for op in (SFT.L3SFType(), SFT.L1T2SFType(), SFT.VectorDotSFType(1, 2))
        Test.@test_throws ArgumentError SFC.covariance(SF.StructureFunction(op, edges, vals), 1.0)
    end
end

# An interpolated kernel is accepted when resolved, refused when too coarse, and an oscillating one always refused.
Test.@testset "a covariance matrix is checked, not assumed, positive semi-definite" begin
    Random.seed!(31)
    pts = 3.0 .* rand(2, 60)
    gauss(s) = 1.4 * exp(-s^2 / (2 * 0.9^2))

    fine = collect(range(0.0, 6.0; length = 1000))
    Σ = SFC.covariance_matrix(pts, fine, gauss.(fine))
    exact = [gauss(sqrt(sum(abs2, pts[:, i] .- pts[:, j]))) for i in 1:60, j in 1:60]
    Test.@test maximum(abs, Σ .- exact) < 1e-5 * maximum(abs, exact)

    coarse = collect(range(0.0, 6.0; length = 60))
    Test.@test_throws ArgumentError SFC.covariance_matrix(pts, coarse, gauss.(coarse))
    Test.@test SFC.covariance_matrix(pts, coarse, gauss.(coarse); posdef_rtol = 1e-2) isa Matrix

    bad = [cos(6s) for s in fine]
    Test.@test_throws ArgumentError SFC.covariance_matrix(pts, fine, bad)
    Test.@test_throws ArgumentError SFC.covariance_matrix(pts, fine, bad; posdef_rtol = 1e-2)
    Test.@test SFC.covariance_matrix(pts, fine, bad; check_posdef = false) isa Matrix
end

# A decomposition's spectra are those of its two projections, and each Helmholtz component inverts as the trace does.
Test.@testset "a Helmholtz decomposition transforms as its projections do" begin
    edges = collect(10 .^ range(-2, log10(8.0); length = 33))
    mids = collect(SF.midpoints(edges))
    counts = ones(UInt32, length(mids))
    D_LL = @. 2 * (1 - exp(-mids^2 / 2))
    D_TT = @. 2 * (1 - (1 - mids^2 / 2) * exp(-mids^2 / 2))
    kq = collect(range(0.05, 6.0; length = 25))
    h = SFC.helmholtz_decompose_2d(edges, D_LL, counts, D_TT, counts)
    fromh = SFC.helmholtz_spectra(h, kq; variance = 2.0)
    pair = SFC.helmholtz_spectra(SF.StructureFunction(SFT.L2SFType(), edges, D_LL),
                                 SF.StructureFunction(SFT.T2SFType(), edges, D_TT), kq; variance = 2.0)
    Test.@test fromh.rotational ≈ pair.rotational rtol = 1e-12
    Test.@test fromh.divergent ≈ pair.divergent rtol = 1e-12

    trace = SFC.isotropic_spectrum(SFT.S2SFType(), mids, D_LL .+ D_TT, kq, Val(2); variance = 2.0)
    for op in (SFT.RotationalSecondOrderStructureFunctionType(), SFT.DivergentSecondOrderStructureFunctionType())
        Test.@test SFC.isotropic_spectrum(op, mids, D_LL .+ D_TT, kq, Val(2); variance = 2.0) == trace
    end
end

Test.@testset "the shell spectrum carries the dimensional weight" begin
    kq = [0.5, 1.0, 2.0, 4.0]
    P = [3.0, 2.0, 1.0, 0.5]
    for (D, Ω) in ((1, 2.0), (2, 2π), (3, 4π))
        Test.@test SFC.shell_spectrum(P, kq, Val(D)) ≈ Ω .* kq .^ (D - 1) .* P
    end
end

# Spectra and fluxes of a device-array result stay in its array family, without scalar reads, equal to the host's.
Test.@testset "post-processing a result held in a device array family stays in it" begin
    GPUArraysCore.allowscalar(false)
    Random.seed!(4400)
    edges = collect(range(0.0, 10.0; length = 41))
    mids = collect(SF.midpoints(edges))
    counts = UInt32.(rand(5:40, 40))
    counts[7] = 0
    Ks = [0.3, 0.8, 1.7, 3.1]
    on_device(r) = SF.StructureFunctionSumsAndCounts(r.operator, r.distance, JLArray(r.sums), JLArray(r.counts))
    same(d, h) = d isa JLArray && isapprox(Array(d), h; rtol = 1e-12)
    host_s2 = SF.StructureFunctionSumsAndCounts(SFT.S2SFType(), edges, (2 .- exp.(-mids ./ 2)) .* counts, counts)
    dev_s2 = on_device(host_s2)
    Test.@test same(SFC.isotropic_spectrum(dev_s2, Ks, Val(2); variance = 1.0),
                    SFC.isotropic_spectrum(host_s2, Ks, Val(2); variance = 1.0))
    Test.@test same(last(SFC.covariance(dev_s2, 1.0)), last(SFC.covariance(host_s2, 1.0)))
    Test.@test SFC.shell_spectrum(JLArray([3.0, 2.0, 1.0, 0.5]), Ks, Val(2)) isa JLArray
    host_s3 = SF.StructureFunctionSumsAndCounts(SFT.S3SFType(), edges, -0.1 .* mids .* counts, counts)
    host_l3 = SF.StructureFunctionSumsAndCounts(SFT.L3SFType(), edges, -0.04 .* mids .* counts, counts)
    host_adv = SF.StructureFunctionSumsAndCounts(SFT.VectorDotSFType(1, 2), edges, sin.(mids) .* counts, counts)
    Test.@test same(SFC.spectral_flux(on_device(host_s3), Ks), SFC.spectral_flux(host_s3, Ks))
    Test.@test same(SFC.spectral_flux(on_device(host_l3), on_device(host_s3), Ks), SFC.spectral_flux(host_l3, host_s3, Ks))
    Test.@test same(SFC.spectral_flux(on_device(host_adv), Ks), SFC.spectral_flux(host_adv, Ks))
    Test.@test same(SFC.enstrophy_flux(on_device(host_adv), Ks), SFC.enstrophy_flux(host_adv, Ks))
    host_l2 = SF.StructureFunctionSumsAndCounts(SFT.L2SFType(), edges, (1 .- exp.(-mids)) .* counts, counts)
    host_t2 = SF.StructureFunctionSumsAndCounts(SFT.T2SFType(), edges, (1 .- exp.(-mids ./ 3)) .* counts, counts)
    h = SFC.helmholtz_spectra(host_l2, host_t2, Ks; variance = 1.0)
    d = SFC.helmholtz_spectra(on_device(host_l2), on_device(host_t2), Ks; variance = 1.0)
    Test.@test same(d.rotational, h.rotational) && same(d.divergent, h.divergent)
end

# A power law A k^-β has S₂ = 2A I_D(β) r^(β-1), with I_1 and I_3 in closed form by its Mellin transform.
Test.@testset "the equivalent spectrum of a power law, debiased, is the power law" begin
    g = Bessels.gamma
    I1(β) = π / (2 * g(β) * sin(π * (β - 1) / 2))
    I3(β) = g(3 - β) * sin(π * β / 2) / (-β * (1 - β) * (2 - β))
    r = exp.(range(log(1e-2), log(10.0); length = 100))
    inner = 3:(length(r) - 2)
    A = 0.7
    for (D, I, b, β) in ((1, I1, 1.0, 5 / 3), (3, I3, 2.0, 2.5))
        e = SFC.equivalent_spectrum(SFT.S2SFType(), r, 2A * I(β) .* r .^ (β - 1), Val(D))
        Test.@test e.wavenumber ≈ reverse(b ./ r)
        Test.@test maximum(abs, e.debiased[inner] ./ (A .* e.wavenumber[inner] .^ -β) .- 1) < 1e-3
    end
    S2 = 2 * I3(2.5) .* r .^ 1.5
    S2[50] = -1.0
    e = SFC.equivalent_spectrum(SFT.S2SFType(), r, S2, Val(3))
    Test.@test all(x -> isfinite(x) && x > 0, e.spectrum)
    Test.@test_throws ArgumentError SFC.equivalent_spectrum(SFT.L2SFType(), r, S2, Val(3))
    Test.@test_throws ArgumentError SFC.equivalent_spectrum(SFT.S2SFType(), reverse(r), S2, Val(3))
end
