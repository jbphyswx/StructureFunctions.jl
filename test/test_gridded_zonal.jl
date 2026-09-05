using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, HelperFunctions as SFH
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using LinearAlgebra: dot
using Random: Random
using FlowGeometries: FlowGeometries as FG

# `Distances.SphericalAngle` reports the central angle itself, so the geometry it implies has
# unit radius; the schedule must be built on the same sphere for the comparison to mean anything.
const R_UNIT = 1.0
const RE = 6.371e6

# The grid's points as the unstructured entry wants them: (λ, φ) in radians, and the local
# (east, north) velocity, flattened in the order the zonal sweep indexes cells.
function _zonal_points(lats, n_lon, dlon, u)
    D = size(u, 1)
    n_lat = length(lats)
    x = Matrix{Float64}(undef, 2, n_lon * n_lat)
    uu = Matrix{Float64}(undef, D, n_lon * n_lat)
    for j in 1:n_lat, i in 1:n_lon
        k = i + (j - 1) * n_lon
        x[1, k] = (i - 1) * dlon
        x[2, k] = lats[j]
        for c in 1:D
            uu[c, k] = u[c, i, j]
        end
    end
    return x, uu
end

Test.@testset "the geodesic frame does not depend on longitude" begin
    # The claim the whole zonal path rests on, checked against pair_frame itself.
    g = SFH.SphericalGeometry{2}(DI.SphericalAngle(), 1.0)
    worst = 0.0
    for p1 in (-1.2, -0.4, 0.0, 0.3, 1.1), p2 in (-1.0, -0.2, 0.15, 0.9), dl in (0.05, 0.7, 1.9, 3.0)
        _, r0, A0, B0 = SFC.zonal_transport(g, p1, p2, dl, Val(2))
        # rebuild the same pair at a shifted longitude and project onto its own local bases
        for l0 in range(-2π, 4π; length = 17)
            pA, EA, NA = SFC._zonal_basis(l0, p1)
            pB, EB, NB = SFC._zonal_basis(l0 + dl, p2)
            _, r, frame = SFH.pair_frame(g, pA, pB)
            tA, tB, m = frame[1], frame[2], frame[3]
            got = (r, dot(tA, EA), dot(tA, NA), dot(m, EA), dot(m, NA),
                   dot(tB, EB), dot(tB, NB), dot(m, EB), dot(m, NB))
            ref = (r0, A0[1, 1], A0[1, 2], A0[2, 1], A0[2, 2],
                   B0[1, 1], B0[1, 2], B0[2, 1], B0[2, 2])
            worst = max(worst, maximum(abs.(collect(got) .- collect(ref))))
        end
    end
    Test.@test worst < 1e-13
end

Test.@testset "the zonal sweep equals the unstructured spherical path" begin
    # The unstructured path handles a sphere exactly and is independently tested, so it is the oracle.
    for (n_lon, lats, frac) in ((12, collect(range(-0.5, 0.5; length = 5)), 0.45),
                                (9, collect(range(-1.0, -0.2; length = 4)), 0.6),
                                (16, collect(range(0.1, 0.9; length = 6)), 0.35))
        n_lat = length(lats)
        dlon = 2π / n_lon
        Random.seed!(9100 + n_lon * n_lat)
        u = randn(2, n_lon, n_lat)
        x, uu = _zonal_points(lats, n_lon, dlon, u)
        r_max = frac * π * R_UNIT
        bins = collect(range(0.0, r_max; length = 9))
        nb = length(bins) - 1
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)

        for sf in (SFT.L2SFType(), SFT.T2SFType(), SFT.S2SFType(), SFT.L3SFType())
            got_s = zeros(nb); got_c = zeros(Int, nb)
            SFC.gridded_lag_sweep!(got_s, got_c, sf, u, sched, bins, Val(2))
            ref_s = zeros(nb); ref_c = zeros(Int, nb)
            SF.calculate_structure_function!(ref_s, ref_c, sf, x, uu, bins;
                                             distance_metric = DI.SphericalAngle())
            Test.@test got_c == ref_c
            Test.@test isapprox(got_s, ref_s; rtol = 1e-9, atol = 1e-10)
            Test.@test sum(got_c) > 0
        end
    end
end

Test.@testset "the zonal sweep counts every pair once" begin
    n_lon = 10
    lats = collect(range(-0.4, 0.4; length = 4))
    dlon = 2π / n_lon
    Random.seed!(9200)
    u = randn(2, n_lon, length(lats))
    bins = collect(range(0.0, 10.0; length = 4))      # spans the whole unit sphere
    N = n_lon * length(lats)
    # This grid is symmetric about the equator with an even longitude count, so every point has its
    # antipode on it — and an antipodal pair has a separation but no direction, so the geometry
    # refuses it. Count those rather than assuming none exist.
    g = SFH.SphericalGeometry{2}(DI.SphericalAngle(), R_UNIT)
    amb(l, ph) = SA.SVector(cos(ph) * cos(l), cos(ph) * sin(l), sin(ph))
    pts = [amb((i - 1) * dlon, lats[j]) for j in eachindex(lats) for i in 1:n_lon]
    refused = 0
    for a in 1:(N - 1), b in (a + 1):N
        ok, _, _ = SFH.pair_frame(g, pts[a], pts[b])
        ok || (refused += 1)
    end
    Test.@test refused > 0                       # the case this grid is chosen to exercise
    for periodic in (true, false)
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, periodic)
        s = zeros(3); c = zeros(Int, 3)
        SFC.gridded_lag_sweep!(s, c, SFT.S2SFType(), u, sched, bins, Val(2))
        Test.@test sum(c) == N * (N - 1) ÷ 2 - refused
    end
end

Test.@testset "latitude-pair culling changes nothing but the work" begin
    # Rows further apart than the largest bin are skipped whole; the histogram must be unaffected.
    n_lon = 12
    lats = collect(range(-1.2, 1.2; length = 9))
    dlon = 2π / n_lon
    Random.seed!(9300)
    u = randn(2, n_lon, length(lats))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)
    tight = collect(range(0.0, 0.15 * π * R_UNIT; length = 6))
    st = zeros(5); ct = zeros(Int, 5)
    SFC.gridded_lag_sweep!(st, ct, SFT.L2SFType(), u, sched, tight, Val(2))
    x, uu = _zonal_points(lats, n_lon, dlon, u)
    rt_s = zeros(5); rt_c = zeros(Int, 5)
    SF.calculate_structure_function!(rt_s, rt_c, SFT.L2SFType(), x, uu, tight;
                                     distance_metric = DI.SphericalAngle())
    Test.@test ct == rt_c
    Test.@test isapprox(st, rt_s; rtol = 1e-9, atol = 1e-10)
    Test.@test sum(ct) > 0
    Test.@test sum(ct) < length(x[1, :]) * (length(x[1, :]) - 1) ÷ 2   # it really culled
end

Test.@testset "the zonal sweep honours missing cells" begin
    n_lon = 10
    lats = collect(range(-0.3, 0.5; length = 4))
    dlon = 2π / n_lon
    N = n_lon * length(lats)
    Random.seed!(9400)
    u = randn(2, n_lon, length(lats))
    uf = reshape(u, 2, N)
    for k in (3, 17, 28)
        uf[1, k] = NaN
    end
    valid = SFC.field_validity(u, Val(2))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)
    bins = collect(range(0.0, 10.0; length = 4))
    s = zeros(3); c = zeros(Int, 3)
    SFC.gridded_lag_sweep!(s, c, SFT.S2SFType(), u, sched, bins, Val(2); valid)
    Test.@test sum(c) == (N - 3) * (N - 4) ÷ 2
    Test.@test all(isfinite, s)
end

Test.@testset "the schedule keeps its latitude axis type" begin
    lats = range(-0.4, 0.5; length = 5)      # asymmetric: no point's antipode is on this grid
    sched = SFC.ZonalLagSchedule(lats, 8, 2π / 8, R_UNIT, true)
    Test.@test sched.lats === lats                       # a range is not materialised
    Test.@test SFC.n_zonal_cells(sched) == 40
    u = randn(2, 8, 5)
    bins = collect(range(0.0, 10.0; length = 4))
    s = zeros(3); c = zeros(Int, 3)
    SFC.gridded_lag_sweep!(s, c, SFT.S2SFType(), u, sched, bins, Val(2))
    Test.@test sum(c) == 40 * 39 ÷ 2
end

Test.@testset "the sphere radius only scales the separation" begin
    # The frame, and therefore every increment, is a property of the directions alone; the radius
    # turns a central angle into a length. So the same field on a bigger sphere gives the same
    # histogram against proportionally bigger bins.
    n_lon = 12
    lats = collect(range(-0.6, 0.6; length = 5))
    dlon = 2π / n_lon
    Random.seed!(9500)
    u = randn(2, n_lon, length(lats))
    unit_bins = collect(range(0.0, 2.5; length = 7))
    s1 = zeros(6); c1 = zeros(Int, 6)
    SFC.gridded_lag_sweep!(s1, c1, SFT.L2SFType(), u,
                         SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true), unit_bins, Val(2))
    s2 = zeros(6); c2 = zeros(Int, 6)
    SFC.gridded_lag_sweep!(s2, c2, SFT.L2SFType(), u,
                         SFC.ZonalLagSchedule(lats, n_lon, dlon, RE, true), RE .* unit_bins, Val(2))
    Test.@test c1 == c2
    Test.@test isapprox(s1, s2; rtol = 1e-12)
    Test.@test sum(c1) > 0
end

Test.@testset "an antipodal shell is refused, not given an arbitrary direction" begin
    # A lat-lon grid symmetric about the equator with an even longitude count puts every point's
    # antipode on the grid. Those pairs have a separation but no direction — infinitely many great
    # circles join them — so the geometry refuses them rather than returning a round-off direction.
    n_lon, n_lat = 8, 4
    lats = collect(range(-0.6, 0.6; length = n_lat))     # symmetric, so antipodes are on the grid
    dlon = 2π / n_lon
    Random.seed!(9600)
    u = randn(2, n_lon, n_lat)
    bins = collect(range(0.0, π; length = 5))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)
    s = zeros(4); c = zeros(Int, 4)
    SFC.gridded_lag_sweep!(s, c, SFT.S2SFType(), u, sched, bins, Val(2))

    g = SFH.SphericalGeometry{2}(DI.SphericalAngle(), R_UNIT)
    amb(l, ph) = SA.SVector(cos(ph) * cos(l), cos(ph) * sin(l), sin(ph))
    pts = [amb((i - 1) * dlon, lats[j]) for j in 1:n_lat for i in 1:n_lon]
    N = n_lon * n_lat
    refused = 0
    for a in 1:(N - 1), b in (a + 1):N
        ok, _, _ = SFH.pair_frame(g, pts[a], pts[b])
        ok || (refused += 1)
    end
    Test.@test refused == N ÷ 2                          # each point pairs with its own antipode
    Test.@test sum(c) == N * (N - 1) ÷ 2 - refused
    Test.@test all(isfinite, s)
end

Test.@testset "a spherical grid reaches the zonal sweep through the public entry" begin
    # C3 is only a capability if the grid entry routes to it; before this it refused a sphere.
    geo = FG.Geometry.SphericalGeometry(R_UNIT)
    n_lon, n_lat = 12, 5
    lam = range(0.0, step = 2π / n_lon, length = n_lon)
    phi = range(-0.5, 0.5; length = n_lat)
    grid = FG.Grids.StructuredGrid(geo, lam, phi)
    Test.@test FG.Grids.isperiodic(grid, 1)          # a full circle is detected as wrapping
    Random.seed!(9700)
    u = randn(2, n_lon, n_lat)
    bins = collect(range(0.0, 0.9 * π; length = 9))

    got = SFC.calculate_structure_function(
        SFT.L2SFType(), grid, u, bins; output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    x, uu = _zonal_points(collect(phi), n_lon, 2π / n_lon, u)
    ref_s = zeros(8); ref_c = zeros(UInt32, 8)
    SF.calculate_structure_function!(ref_s, ref_c, SFT.L2SFType(), x, uu, bins;
                                     distance_metric = DI.SphericalAngle())
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-9, atol = 1e-10)
    Test.@test sum(got.counts) > 0

    # a regional (non-wrapping) longitude span is handled too, and counts fewer pairs per row
    lam2 = range(0.0, step = 0.05, length = n_lon)
    regional = FG.Grids.StructuredGrid(geo, lam2, phi)
    Test.@test !FG.Grids.isperiodic(regional, 1)
    wide = collect(range(0.0, 10.0; length = 4))
    r2 = SFC.calculate_structure_function(
        SFT.S2SFType(), regional, u, wide; output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    N = n_lon * n_lat
    Test.@test sum(r2.counts) == N * (N - 1) ÷ 2

    # A stretched longitude axis has no frame shared around a circle, so it takes neither the zonal
    # schedule nor the lag one — it falls through to enumerating pairs, which is still exact.
    stretched = FG.Grids.StructuredGrid(geo, [0.0, 0.1, 0.35, 0.9], phi)
    us = randn(2, 4, n_lat)
    r_str = SFC.calculate_structure_function(
        SFT.L2SFType(), stretched, us, bins, UInt32;
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    xs = Matrix{Float64}(undef, 2, 4 * n_lat)
    for (k, I) in enumerate(CartesianIndices((4, n_lat)))
        xs[1, k] = [0.0, 0.1, 0.35, 0.9][I[1]]
        xs[2, k] = phi[I[2]]
    end
    ref_str_s = zeros(8); ref_str_c = zeros(UInt32, 8)
    SF.calculate_structure_function!(ref_str_s, ref_str_c, SFT.L2SFType(), xs,
                                     reshape(us, 2, 4 * n_lat), bins;
                                     distance_metric = DI.SphericalAngle())
    Test.@test r_str.counts == ref_str_c
    Test.@test isapprox(r_str.sums, ref_str_s; rtol = 1e-9, atol = 1e-10)
end
