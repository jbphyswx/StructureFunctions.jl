using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using Distances: Distances as DI
using Random: Random
using Test: Test

# Results known in closed form ahead of the calculation.

Test.@testset "the Helmholtz split does not depend on the unit of length" begin
    FT = Float64
    n_bins = 40
    counts = ones(UInt32, n_bins)
    edges = collect(FT, 10 .^ range(-3, 0; length = n_bins + 1))
    mids = FT[sqrt(edges[k] * edges[k + 1]) for k in 1:n_bins]
    # D_TT = (5/3) D_LL for D_LL ∝ r^(2/3) is 2-D solenoidal, so D_div ≡ 0
    D_LL = mids .^ (2 / 3)
    D_TT = (5 / 3) .* D_LL

    h = SFC.helmholtz_decompose_2d(edges, D_LL, counts, D_TT, counts)
    Test.@test maximum(abs, h.divergent_sums) < 0.05 * maximum(D_LL)

    # metres to millimetres: the same field, so the split is unchanged
    h_mm = SFC.helmholtz_decompose_2d(edges .* 1000, D_LL, counts, D_TT, counts)
    Test.@test isapprox(h_mm.divergent_sums, h.divergent_sums; rtol = 1e-10, atol = 1e-12)
    Test.@test isapprox(h_mm.rotational_sums, h.rotational_sums; rtol = 1e-10, atol = 1e-12)

    # D_LL ∝ r^(2/3) with D_LL = (5/3) D_TT is irrotational, so D_rot ≡ 0
    h_irrot = SFC.helmholtz_decompose_2d(edges, D_TT, counts, D_LL, counts)
    Test.@test maximum(abs, h_irrot.rotational_sums) < 0.05 * maximum(D_TT)

    Test.@test isapprox(h.rotational_sums .+ h.divergent_sums, D_LL .+ D_TT;
                        rtol = 1e-12, atol = 1e-14)
end

Test.@testset "the tensor trace is the second-order structure function" begin
    Random.seed!(2600)
    N = 60
    serial = CB.SerialBackend()

    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 1.4; length = 6))
    t = SFC.calculate_structure_function_tensor(
        Val(2), x, u, bins; backend = serial,
        output_type = SF.StructureFunctionTensorSumsAndCounts)
    s2 = SFC.calculate_structure_function(
        SFT.S2SFType(), x, u, bins; backend = serial,
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test t.counts == s2.counts
    Test.@test isapprox([sum(t.sums[d, d, b] for d in 1:2) for b in 1:(length(bins) - 1)], s2.sums;
                        rtol = 1e-10, atol = 1e-12)

    # (λ, φ) in radians, so the metric is SphericalAngle. The trace is frame-independent only if
    # the tensor's components share one frame per pair.
    xs = vcat(reshape(2π .* rand(N), 1, N), reshape((rand(N) .- 0.5) .* 1.4, 1, N))
    us = randn(2, N)
    sbins = collect(range(0.0, 2.4; length = 6))
    ts = SFC.calculate_structure_function_tensor(
        Val(2), xs, us, sbins; backend = serial, distance_metric = DI.SphericalAngle(),
        output_type = SF.StructureFunctionTensorSumsAndCounts)
    s2s = SFC.calculate_structure_function(
        SFT.S2SFType(), xs, us, sbins; backend = serial, distance_metric = DI.SphericalAngle(),
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test sum(ts.counts) > 0
    Test.@test ts.counts == s2s.counts
    Test.@test isapprox([sum(ts.sums[d, d, b] for d in 1:2) for b in 1:(length(sbins) - 1)],
                        s2s.sums; rtol = 1e-10, atol = 1e-12)

    # on a sphere the pair frame is the basis, so the longitudinal direction is ê₁
    l2s = SFC.calculate_structure_function(
        SFT.L2SFType(), xs, us, sbins; backend = serial, distance_metric = DI.SphericalAngle(),
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test isapprox(ts.sums[1, 1, :], l2s.sums; rtol = 1e-10, atol = 1e-12)
end

function _lag_sweep(sf, u, dims, spacing, periodic, bins, D)
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    sums = zeros(Float64, nb)
    counts = zeros(Int, nb)
    sched = SFC.UniformLagSchedule(dims, spacing, periodic)
    SFC.gridded_lag_sweep!(sums, counts, sf, u, sched, bins, Val(D))
    return sums, counts
end

# For u(x) = A ê cos(k·x + φ) averaged over a full period, D_ab(r) = A² ê_a ê_b (1 − cos(k·r)) and
# every odd moment vanishes. On a periodic grid whose cell count is a multiple of the mode's period
# the average is exact, so these hold to round-off rather than to a sampling tolerance.
Test.@testset "a single Fourier mode gives its closed-form structure function" begin
    A = 1.7
    φ = 0.41
    m = 3

    n, d = 32, 0.25
    k = 2π * m / (n * d)
    u = reshape([A * cos(k * (j - 1) * d + φ) for j in 1:n], 1, n)
    # each bin holds exactly one lag, so a binned average is a per-lag average
    bins = [(h - 0.5) * d for h in 1:16]
    s2, c2 = _lag_sweep(SFT.S2SFType(), u, (n,), (d,), (true,), bins, 1)
    Test.@test all(==(n), c2)
    Test.@test isapprox(s2 ./ c2, [A^2 * (1 - cos(2π * m * h / n)) for h in 1:15];
                        rtol = 1e-13, atol = 1e-13)

    l3, c3 = _lag_sweep(SFT.L3SFType(), u, (n,), (d,), (true,), bins, 1)
    Test.@test c3 == c2
    Test.@test maximum(abs, l3 ./ c3) < 1e-13 * A^3

    # anisotropic spacing separates the lag classes, so a bin isolates one direction. The mode runs
    # along x, so a lag along y joins two cells of equal phase.
    nx, ny, dx, dy = 16, 16, 1.0, 0.37
    kx = 2π * m / (nx * dx)
    u2 = zeros(2, nx, ny)
    for i in 1:nx, j in 1:ny
        u2[1, i, j] = A * cos(kx * (i - 1) * dx + φ)
    end
    dims, spacing, periodic = (nx, ny), (dx, dy), (true, true)
    edges = [0.3, 0.5, 0.9, 1.05]
    expected = A^2 * (1 - cos(kx * dx))

    s2g, c2g = _lag_sweep(SFT.S2SFType(), u2, dims, spacing, periodic, edges, 2)
    Test.@test c2g[1] > 0 && c2g[3] > 0
    Test.@test abs(s2g[1]) < 1e-24 * expected * c2g[1]
    Test.@test isapprox(s2g[3] / c2g[3], expected; rtol = 1e-13)

    l2g, _ = _lag_sweep(SFT.L2SFType(), u2, dims, spacing, periodic, edges, 2)
    t2g, _ = _lag_sweep(SFT.T2SFType(), u2, dims, spacing, periodic, edges, 2)
    Test.@test isapprox(l2g[3] / c2g[3], expected; rtol = 1e-13)
    Test.@test abs(t2g[3]) < 1e-24 * expected * c2g[3]

    wide = collect(range(0.0, 5.0; length = 12))
    l3g, c3g = _lag_sweep(SFT.L3SFType(), u2, dims, spacing, periodic, wide, 2)
    Test.@test sum(c3g) > 0
    Test.@test maximum(abs, l3g) < 1e-12 * A^3 * maximum(c3g)
end

# Distinct grid harmonics are orthogonal over the cells, so a superposition's cross terms cancel
# exactly and the closed form of one mode adds: D_ab(r) = Σ_m A_m² Σ_p ê_a ê_b (1 − cos(k_m·r)).
# Two orthonormal polarisations per mode make the polarisation sum the transverse projector, so the
# field is divergence-free and Σ_p (ê·r̂)² = 1 − (k̂·r̂)².
Test.@testset "a prescribed spectrum is recovered mode by mode" begin
    dims = (10, 10, 10)
    spacing = (1.0, 0.37, 0.1732)
    periodic = (true, true, true)
    modes = ((1, 2, 3), (3, 1, 2), (2, 3, 1), (1, 1, 2))
    amps = (1.3, 0.8, 0.5, 0.21)

    kvecs = map(m -> 2π .* (m[1] / (dims[1] * spacing[1]),
                           m[2] / (dims[2] * spacing[2]),
                           m[3] / (dims[3] * spacing[3])), modes)
    function _polarisations(k)
        kh = k ./ sqrt(sum(abs2, k))
        a = abs(kh[3]) < 0.9 ? (0.0, 0.0, 1.0) : (1.0, 0.0, 0.0)
        e1 = (kh[2] * a[3] - kh[3] * a[2], kh[3] * a[1] - kh[1] * a[3], kh[1] * a[2] - kh[2] * a[1])
        n1 = sqrt(sum(abs2, e1))
        e1 = e1 ./ n1
        e2 = (kh[2] * e1[3] - kh[3] * e1[2], kh[3] * e1[1] - kh[1] * e1[3],
              kh[1] * e1[2] - kh[2] * e1[1])
        return kh, e1, e2
    end
    pol = map(_polarisations, kvecs)

    # The two polarisations of a mode share its wavevector, so they are not orthogonal over the
    # cells; a quarter turn between their phases makes their cross term vanish exactly.
    Random.seed!(3100)
    phases = zeros(length(modes), 2)
    for m in eachindex(modes)
        phases[m, 1] = 2π * rand()
        phases[m, 2] = phases[m, 1] + π / 2
    end
    u = zeros(3, dims...)
    for (i, j, l) in Iterators.product(map(n -> 1:n, dims)...)
        pos = ((i - 1) * spacing[1], (j - 1) * spacing[2], (l - 1) * spacing[3])
        for m in eachindex(modes)
            kx = sum(kvecs[m] .* pos)
            _, e1, e2 = pol[m]
            for (p, e) in ((1, e1), (2, e2))
                c = amps[m] * cos(kx + phases[m, p])
                u[1, i, j, l] += c * e[1]
                u[2, i, j, l] += c * e[2]
                u[3, i, j, l] += c * e[3]
            end
        end
    end

    # each window holds one lag, which `counts == prod(dims)` verifies rather than assumes
    edges = [0.3364, 0.3564, 0.36, 0.38, 0.6828, 0.7028]
    targets = ((1, (0, 0, 2)), (3, (0, 1, 0)), (5, (0, 0, 4)))

    s2, c2 = _lag_sweep(SFT.S2SFType(), u, dims, spacing, periodic, edges, 3)
    l2, c_l2 = _lag_sweep(SFT.L2SFType(), u, dims, spacing, periodic, edges, 3)
    Test.@test c_l2 == c2

    for (b, lag) in targets
        Test.@test c2[b] == prod(dims)
        rvec = lag .* spacing
        r = sqrt(sum(abs2, rvec))
        rhat = rvec ./ r
        pred_s2 = 0.0
        pred_l2 = 0.0
        for m in eachindex(modes)
            kh, _, _ = pol[m]
            osc = 1 - cos(sum(kvecs[m] .* rvec))
            pred_s2 += amps[m]^2 * 2 * osc
            pred_l2 += amps[m]^2 * (1 - sum(kh .* rhat)^2) * osc
        end
        Test.@test isapprox(s2[b] / c2[b], pred_s2; rtol = 1e-12)
        Test.@test isapprox(l2[b] / c2[b], pred_l2; rtol = 1e-12, atol = 1e-13)
    end
end

Test.@testset "the inertial-range laws invert the moment each is stated for" begin
    r = collect(range(0.2, 3.0; length = 12))
    eps = 0.85
    Test.@test SF.KHM.epsilon_from_four_fifths(r, -(4 / 5) .* eps .* r) ≈ fill(eps, length(r))
    Test.@test SF.KHM.epsilon_from_four_thirds(r, -(4 / 3) .* eps .* r) ≈ fill(eps, length(r))

    eps_theta = 0.42
    LS2 = -(4 / 3) .* eps_theta .* r
    Test.@test SF.KHM.epsilon_theta_from_yaglom(r, LS2) ≈ fill(eps_theta, length(r))
    Test.@test !isapprox(SF.KHM.epsilon_from_four_fifths(r, LS2)[1], eps_theta)
end
