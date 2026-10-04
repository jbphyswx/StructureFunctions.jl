using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, HelperFunctions as SFH,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT
using OhMyThreads: OhMyThreads
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using Random: Random
using Test: Test

Random.seed!(20260817)

const EARTH_R = 6.371e6

_sp(x, u, bins, m; be = CB.SerialBackend()) = SFC.calculate_structure_functions_single_pass(
    x, u, bins, SFO.StructureFunctionSumsAndCounts; backend = be, distance_metric = m,
)

# A metric with no pair geometry is refused with an error that names the hook to define.
Test.@testset "a metric with no pair geometry is refused" begin
    Test.@test_throws ArgumentError SFH.pair_geometry_for(DI.Cityblock(), Val(2))
    msg = try
        SFH.pair_geometry_for(DI.Cityblock(), Val(2))
    catch e
        sprint(showerror, e)
    end
    Test.@test occursin("pair_geometry_for", msg)
end

# pair_frame's arc matches Haversine in degrees and SphericalAngle in radians, with the same frame for the same pair.
Test.@testset "Spherical separation matches the metric, in both angle conventions" begin
    hav = DI.Haversine(EARTH_R)
    sph = DI.SphericalAngle()
    gh = SFH.pair_geometry_for(hav, Val(2))
    gs = SFH.pair_geometry_for(sph, Val(2))
    cases = map(((10.0, 45.0, 13.0, 47.0), (-170.0, -33.0, 175.0, 12.0),
                 (0.0, 0.0, 0.0, 90.0), (100.0, 60.0, 100.0, 60.5))) do (lo1, la1, lo2, la2)
        p1 = SA.SVector(lo1, la1); p2 = SA.SVector(lo2, la2)
        q1 = SA.SVector(deg2rad(lo1), deg2rad(la1)); q2 = SA.SVector(deg2rad(lo2), deg2rad(la2))
        ok_h, r_h, f_h = SFH.pair_frame(gh, SFH.unit_position(hav, lo1, la1), SFH.unit_position(hav, lo2, la2))
        ok_s, r_s, f_s = SFH.pair_frame(gs, SFH.unit_position(sph, q1[1], q1[2]), SFH.unit_position(sph, q2[1], q2[2]))
        (; ok = ok_h && ok_s, r_h, r_s, f_h, f_s, hav = hav(p1, p2), sph = sph(q1, q2))
    end
    Test.@test all(c -> c.ok, cases)
    Test.@test all(c -> isapprox(c.r_h, c.hav; rtol = 1e-12), cases)
    Test.@test all(c -> isapprox(c.r_s, c.sph; rtol = 1e-12), cases)
    Test.@test all(c -> isapprox(c.r_h, EARTH_R * c.r_s; rtol = 1e-12), cases)
    Test.@test all(c -> isapprox(c.f_h[1], c.f_s[1]; atol = 1e-12) && isapprox(c.f_h[2], c.f_s[2]; atol = 1e-12),
                   cases)
end

# Under transport a rigid rotation has δu_L = 0 on every pair (L2/S2 at the eps² level); a flat lon/lat frame does not.
Test.@testset "Solid-body rotation has no longitudinal increment" begin
    N = 60
    Ω = 7.292e-5
    lon = 360 .* rand(N) .- 180
    lat = 120 .* rand(N) .- 60
    x = permutedims(hcat(lon, lat))
    u = permutedims(hcat(Ω * EARTH_R .* cosd.(lat), zeros(N)))
    bins = collect(range(0.0, 8.0e6; length = 21))

    r = _sp(x, u, bins, DI.Haversine(EARTH_R))
    occ = r.L2.counts .> 0
    Test.@test any(occ)
    Test.@test sum(r.L2.sums[occ]) / sum(r.S2.sums[occ]) < 1e-24

    flat = SFC.calculate_structure_functions_single_pass(
        x, u, collect(range(0.0, 80.0; length = 21)), SFO.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend(), distance_metric = DI.Euclidean(),
    )
    occf = flat.L2.counts .> 0
    Test.@test sum(flat.L2.sums[occf]) / sum(flat.S2.sums[occf]) > 0.01
end

# A radial third component leaves L2 unchanged to eps times the field scale, adds to S2, and keeps S2 = L2 + T2.
Test.@testset "Thin shell: radial component is carried but never transported" begin
    N = 60
    Ω = 7.292e-5
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))
    ue = Ω * EARTH_R .* cosd.(lat)
    u2 = permutedims(hcat(ue, zeros(N)))
    u3 = permutedims(hcat(ue, zeros(N), 3.0 .* sind.(2 .* lat)))
    bins = collect(range(0.0, 8.0e6; length = 21))
    m = DI.Haversine(EARTH_R)

    r2 = _sp(x, u2, bins, m)
    r3 = _sp(x, u3, bins, m)

    Test.@test isapprox(r3.L2.sums, r2.L2.sums; atol = eps() * maximum(r3.S2.sums))
    Test.@test r3.L2.counts == r2.L2.counts
    Test.@test sum(r3.S2.sums) > sum(r2.S2.sums)
    Test.@test r3.S2.sums ≈ r3.L2.sums .+ r3.T2.sums
end

# The threaded single pass, and the L2, T2 and S2 operators, match the serial single-pass invariants on the sphere.
Test.@testset "Spherical geometry: backend agreement" begin
    N = 60
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))
    u = permutedims(hcat(randn(N), randn(N), randn(N)))
    bins = collect(range(0.0, 9.0e6; length = 13))
    m = DI.Haversine(EARTH_R)

    ref = _sp(x, u, bins, m)
    got = _sp(x, u, bins, m; be = CB.ThreadedBackend())
    Test.@test all(got[k].counts == ref[k].counts && got[k].sums ≈ ref[k].sums for k in (:S2, :L2, :T2, :S3, :L3, :L1T2))

    operators = [(SFC.calculate_structure_function(sft, x, u, bins, SFO.StructureFunctionSumsAndCounts;
                                                  backend = CB.SerialBackend(), distance_metric = m), key)
                 for (sft, key) in ((SFT.LongitudinalSecondOrderStructureFunctionType(), :L2),
                                    (SFT.TransverseSecondOrderStructureFunctionType(), :T2),
                                    (SFT.SecondOrderStructureFunctionType(), :S2))]
    Test.@test all(r.counts == ref[key].counts && r.sums ≈ ref[key].sums for (r, key) in operators)
end

# The spherical-minus-tangent-plane L2 discrepancy is O(r/R): a 4x smaller patch cuts it by more than 2.5x.
Test.@testset "Flat limit: spherical converges to Cartesian like r/R" begin
    N = 90
    lat0 = 35.0
    function deviation(halfwidth_deg)
        Random.seed!(4242)
        dlon = halfwidth_deg .* (2 .* rand(N) .- 1)
        dlat = halfwidth_deg .* (2 .* rand(N) .- 1)
        lon = dlon
        lat = lat0 .+ dlat
        uu = permutedims(hcat(randn(N), randn(N)))
        x_sph = permutedims(hcat(lon, lat))
        x_flat = permutedims(hcat(
            EARTH_R .* deg2rad.(dlon) .* cosd(lat0), EARTH_R .* deg2rad.(dlat),
        ))
        rmax = 2.2 * EARTH_R * deg2rad(halfwidth_deg)
        bins = collect(range(0.0, rmax; length = 9))
        a = _sp(x_sph, uu, bins, DI.Haversine(EARTH_R))
        b = _sp(x_flat, uu, bins, DI.Euclidean())
        occ = (a.L2.counts .> 0) .& (b.L2.counts .> 0)
        return maximum(abs.(a.L2.sums[occ] .- b.L2.sums[occ])) / maximum(abs.(b.L2.sums[occ]))
    end

    d_big = deviation(4.0)
    d_small = deviation(1.0)
    Test.@test d_small < d_big
    Test.@test d_big / d_small > 2.5
end

struct DoubledFlatMetric <: DI.PreMetric end
(::DoubledFlatMetric)(a, b) = 2 * sqrt(sum(abs2, a .- b))

struct DoubledFlatGeometry{D} end
SFH.coordinate_width(::DoubledFlatGeometry{D}) where {D} = Val(D)
SFH.pair_geometry_for(::DoubledFlatMetric, ::Val{D}) where {D} = DoubledFlatGeometry{D}()
@inline function SFH.pair_frame(::DoubledFlatGeometry, x1, x2)
    dx = x2 - x1
    return true, 2 * sqrt(LA.dot(dx, dx)), dx
end
@inline SFH.pair_direction(::DoubledFlatGeometry, frame, r) = frame / (r / 2)
@inline SFH.pair_delta(::DoubledFlatGeometry, frame, x1, x2, u1, u2) = u2 - u1

# A user geometry doubling every separation reproduces the Euclidean single-pass and joint histograms on doubled bins.
Test.@testset "User-defined geometry works end to end on every backend" begin
    N = 40
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.05, 1.4; length = 11))

    euc = _sp(x, u, bins, DI.Euclidean())
    for be in (CB.SerialBackend(), CB.ThreadedBackend())
        got = _sp(x, u, 2 .* bins, DoubledFlatMetric(); be = be)
        Test.@test all(got[k].counts == euc[k].counts && got[k].sums ≈ euc[k].sums
                       for k in (:S2, :L2, :T2, :S3, :L3, :L1T2))
    end

    vb = collect(range(-3.0, 3.0; length = 9))
    j_euc = SFC.calculate_structure_function(
        SFT.L2SFType(), x, u, bins, vb; backend = CB.SerialBackend(),
    )
    j_got = SFC.calculate_structure_function(
        SFT.L2SFType(), x, u, 2 .* bins, vb; backend = CB.SerialBackend(),
        distance_metric = DoubledFlatMetric(),
    )
    Test.@test j_got.counts == j_euc.counts
    Test.@test j_got.sums ≈ j_euc.sums
end

# Flat x must match the velocity width; a shell takes two coordinates for both D = 2 and D = 3.
Test.@testset "Shape contract is geometry-aware" begin
    bins = collect(range(0.0, 5.0e6; length = 9))
    Test.@test_throws DimensionMismatch _sp(rand(2, 8), randn(3, 8), bins, DI.Euclidean())
    Test.@test_throws DimensionMismatch _sp(rand(3, 8), randn(2, 8), bins, DI.Euclidean())
    m = DI.Haversine(EARTH_R)
    Test.@test_throws DimensionMismatch _sp(rand(3, 8), randn(3, 8), bins, m)
    lonlat = permutedims(hcat(20 .* rand(8), 20 .* rand(8)))
    Test.@test _sp(lonlat, randn(2, 8), bins, m) isa NamedTuple
    Test.@test _sp(lonlat, randn(3, 8), bins, m) isa NamedTuple
end

# geodesic_frame keeps short pairs to 1e-12; near-antipodal ones are kept within sqrt(eps) at gap ≥ 1e-5 and refused at gap ≤ 1e-9.
Test.@testset "the separation direction is refused where it carries no information" begin
    setprecision(BigFloat, 256)
    Random.seed!(4242)
    exact_frame(p, q) = SFH.geodesic_frame(SA.SVector{3, BigFloat}(BigFloat.(p)),
                                           SA.SVector{3, BigFloat}(BigFloat.(q)))
    function pair_at(gap, sgn)
        p = LA.normalize(SA.SVector(randn(3)...))
        w = LA.normalize(LA.cross(p, LA.normalize(SA.SVector(randn(3)...))))
        return p, LA.normalize(sgn * cos(gap) * p + sin(gap) * w)
    end
    function sample(gap, sgn)
        p, q = pair_at(gap, sgn)
        _, tA, _, _, ok = SFH.geodesic_frame(p, q)
        err = ok ? Float64(LA.norm(SA.SVector{3, BigFloat}(BigFloat.(tA)) - exact_frame(p, q)[2])) : 0.0
        return (; gap, ok, err)
    end

    near = [sample(gap, 1) for gap in (1e-1, 1e-4, 1e-8, 1e-12)]
    Test.@test all(s -> s.ok && s.err < 1e-12, near)

    anti = [sample(gap, -1) for gap in (1e-1, 1e-5, 1e-9, 1e-13, 1e-15) for _ in 1:20]
    Test.@test all(s -> s.err < sqrt(eps(Float64)), anti)
    Test.@test all(s -> !s.ok, filter(s -> s.gap <= 1e-9, anti))
    Test.@test all(s -> s.ok, filter(s -> s.gap >= 1e-5, anti))
end
