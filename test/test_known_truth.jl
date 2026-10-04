using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT
using Distances: Distances as DI
using Random: Random
using Test: Test

# Results known in closed form ahead of the calculation.

# Neither solenoidal nor irrotational, so both components are nonzero; metres to millimetres changes neither.
Test.@testset "the Helmholtz split does not depend on the unit of length" begin
    edges = collect(10 .^ range(-3, 0; length = 41))
    mids = [sqrt(edges[k] * edges[k + 1]) for k in 1:40]
    counts = ones(UInt32, 40)
    D_LL = mids .^ (2 / 3)
    D_TT = 1.2 .* D_LL
    h = SFC.helmholtz_decompose_2d(edges, D_LL, counts, D_TT, counts)
    h_mm = SFC.helmholtz_decompose_2d(edges .* 1000, D_LL, counts, D_TT, counts)
    Test.@test h_mm.divergent_sums ≈ h.divergent_sums rtol = 1e-10
    Test.@test h_mm.rotational_sums ≈ h.rotational_sums rtol = 1e-10
end

# D_TT = d(r D_LL)/dr is 2-D solenoidal and D_LL = d(r D_TT)/dr irrotational, each all in its own component.
Test.@testset "the Helmholtz split puts a pure field wholly in its own component" begin
    edges = collect(10 .^ range(log10(0.2), log10(20.0); length = 121))
    mids = SF.midpoints(edges)
    counts = ones(UInt32, length(mids))
    D_a = @. mids^2 / (1 + mids^2)^(2 / 3)
    D_b = @. 3mids^2 / (1 + mids^2)^(2 / 3) - (4 / 3) * mids^4 / (1 + mids^2)^(5 / 3)
    sol = SFC.helmholtz_decompose_2d(edges, D_a, counts, D_b, counts)
    Test.@test maximum(abs, sol.divergent_sums) < 1e-3 * maximum(D_a)
    Test.@test isapprox(sol.rotational_sums, D_a .+ D_b; rtol = 1e-3)
    irr = SFC.helmholtz_decompose_2d(edges, D_b, counts, D_a, counts)
    Test.@test maximum(abs, irr.rotational_sums) < 1e-3 * maximum(D_a)
    Test.@test isapprox(irr.divergent_sums, D_a .+ D_b; rtol = 1e-3)
end

# Planar and spherical (one pair frame per pair), and on the sphere the first diagonal entry is L2.
Test.@testset "the tensor trace is the second-order structure function" begin
    Random.seed!(2600)
    N = 60
    serial = CB.SerialBackend()

    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 1.4; length = 6))
    t = SFC.calculate_structure_function_tensor(
        Val(2), x, u, bins, SF.StructureFunctionTensorSumsAndCounts; backend = serial)
    s2 = SFC.calculate_structure_function(
        SFT.S2SFType(), x, u, bins, SF.StructureFunctionSumsAndCounts; backend = serial)
    Test.@test t.counts == s2.counts
    Test.@test isapprox([sum(t.sums[d, d, b] for d in 1:2) for b in 1:(length(bins) - 1)], s2.sums;
                        rtol = 1e-10, atol = 1e-12)

    xs = vcat(reshape(2π .* rand(N), 1, N), reshape((rand(N) .- 0.5) .* 1.4, 1, N))
    us = randn(2, N)
    sbins = collect(range(0.0, 2.4; length = 6))
    ts = SFC.calculate_structure_function_tensor(
        Val(2), xs, us, sbins, SF.StructureFunctionTensorSumsAndCounts; backend = serial,
        distance_metric = DI.SphericalAngle())
    s2s = SFC.calculate_structure_function(
        SFT.S2SFType(), xs, us, sbins, SF.StructureFunctionSumsAndCounts; backend = serial,
        distance_metric = DI.SphericalAngle())
    Test.@test sum(ts.counts) > 0
    Test.@test ts.counts == s2s.counts
    Test.@test isapprox([sum(ts.sums[d, d, b] for d in 1:2) for b in 1:(length(sbins) - 1)],
                        s2s.sums; rtol = 1e-10, atol = 1e-12)

    l2s = SFC.calculate_structure_function(
        SFT.L2SFType(), xs, us, sbins, SF.StructureFunctionSumsAndCounts; backend = serial,
        distance_metric = DI.SphericalAngle())
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

# u = A ê cos(k·x + φ) over whole periods: D_ab(r) = A² ê_a ê_b (1 − cos(k·r)) and odd moments vanish, to round-off.
Test.@testset "a single Fourier mode gives its closed-form structure function" begin
    A = 1.7
    φ = 0.41
    m = 3

    n, d = 32, 0.25
    k = 2π * m / (n * d)
    u = reshape([A * cos(k * (j - 1) * d + φ) for j in 1:n], 1, n)
    bins = [(h - 0.5) * d for h in 1:16]
    s2, c2 = _lag_sweep(SFT.S2SFType(), u, (n,), (d,), (true,), bins, 1)
    Test.@test all(==(n), c2)
    Test.@test isapprox(s2 ./ c2, [A^2 * (1 - cos(2π * m * h / n)) for h in 1:15];
                        rtol = 1e-13, atol = 1e-13)

    l3, c3 = _lag_sweep(SFT.L3SFType(), u, (n,), (d,), (true,), bins, 1)
    Test.@test maximum(abs, l3 ./ c3) < 1e-13 * A^3

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

# A divergence-free superposition of grid harmonics on a 3-D periodic grid: S2 and L2 at single lags, mode by mode.
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

    edges = [0.3364, 0.3564, 0.36, 0.38, 0.6828, 0.7028]
    targets = ((1, (0, 0, 2)), (3, (0, 1, 0)), (5, (0, 0, 4)))
    s2, c2 = _lag_sweep(SFT.S2SFType(), u, dims, spacing, periodic, edges, 3)
    l2, _ = _lag_sweep(SFT.L2SFType(), u, dims, spacing, periodic, edges, 3)

    pred = map(targets) do (_, lag)
        rvec = lag .* spacing
        rhat = rvec ./ sqrt(sum(abs2, rvec))
        osc = [1 - cos(sum(kvecs[m] .* rvec)) for m in eachindex(modes)]
        (s2 = sum(amps[m]^2 * 2 * osc[m] for m in eachindex(modes)),
         l2 = sum(amps[m]^2 * (1 - sum(pol[m][1] .* rhat)^2) * osc[m] for m in eachindex(modes)))
    end
    b = collect(first.(targets))
    Test.@test all(==(prod(dims)), c2[b])
    Test.@test all(isapprox.(s2[b] ./ c2[b], getfield.(pred, :s2); rtol = 1e-12))
    Test.@test all(isapprox.(l2[b] ./ c2[b], getfield.(pred, :l2); rtol = 1e-12, atol = 1e-13))
end

Test.@testset "the inertial-range laws invert the moment each is stated for" begin
    r = collect(range(0.2, 3.0; length = 12))
    eps = 0.85
    Test.@test SF.KHM.epsilon_from_four_fifths(r, -(4 / 5) .* eps .* r) ≈ fill(eps, length(r))
    Test.@test SF.KHM.epsilon_from_four_thirds(r, -(4 / 3) .* eps .* r) ≈ fill(eps, length(r))

    eps_theta = 0.42
    Test.@test SF.KHM.epsilon_theta_from_yaglom(r, -(4 / 3) .* eps_theta .* r) ≈ fill(eps_theta, length(r))
end
