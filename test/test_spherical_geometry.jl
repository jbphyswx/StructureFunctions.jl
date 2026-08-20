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
    x, u, bins; backend = be, distance_metric = m,
    output_type = SFO.StructureFunctionSumsAndCounts,
)

Test.@testset "Pair geometry: selection and coordinate width" begin
    Test.@test SFH.pair_geometry_for(DI.Euclidean(), Val(2)) === SFH.FlatGeometry{2}()
    Test.@test SFH.pair_geometry_for(DI.Euclidean(), Val(3)) === SFH.FlatGeometry{3}()
    Test.@test SFH.pair_geometry_for(DI.Haversine(EARTH_R), Val(2)) isa SFH.SphericalGeometry{2}
    Test.@test SFH.pair_geometry_for(DI.SphericalAngle(), Val(3)) isa SFH.SphericalGeometry{3}

    # A point on a shell takes two coordinates whether or not the velocity carries a radial third.
    Test.@test SFH.coordinate_width(SFH.FlatGeometry{2}()) === Val(2)
    Test.@test SFH.coordinate_width(SFH.FlatGeometry{3}()) === Val(3)
    Test.@test SFH.coordinate_width(SFH.pair_geometry_for(DI.Haversine(EARTH_R), Val(2))) === Val(2)
    Test.@test SFH.coordinate_width(SFH.pair_geometry_for(DI.Haversine(EARTH_R), Val(3))) === Val(2)

    # A distance function does not define a direction or a transport rule, so a metric with no
    # geometry is refused rather than assumed flat.
    Test.@test_throws ArgumentError SFH.pair_geometry_for(DI.Cityblock(), Val(2))
    msg = try
        SFH.pair_geometry_for(DI.Cityblock(), Val(2))
    catch e
        sprint(showerror, e)
    end
    Test.@test occursin("pair_geometry_for", msg)
end

Test.@testset "Spherical separation matches the metric, in both angle conventions" begin
    hav = DI.Haversine(EARTH_R)
    sph = DI.SphericalAngle()
    for (lo1, la1, lo2, la2) in ((10.0, 45.0, 13.0, 47.0), (-170.0, -33.0, 175.0, 12.0),
                                 (0.0, 0.0, 0.0, 90.0), (100.0, 60.0, 100.0, 60.5))
        p1 = SA.SVector(lo1, la1); p2 = SA.SVector(lo2, la2)
        ok_h, r_h, f_h = SFH.pair_frame(SFH.pair_geometry_for(hav, Val(2)), p1, p2)
        q1 = SA.SVector(deg2rad(lo1), deg2rad(la1)); q2 = SA.SVector(deg2rad(lo2), deg2rad(la2))
        ok_s, r_s, f_s = SFH.pair_frame(SFH.pair_geometry_for(sph, Val(2)), q1, q2)

        Test.@test ok_h && ok_s
        Test.@test isapprox(r_h, hav(p1, p2); rtol = 1e-12)
        Test.@test isapprox(r_s, sph(q1, q2); rtol = 1e-12)
        # Degrees and radians describe the same physical pair: same arc, same local frame.
        Test.@test isapprox(r_h, EARTH_R * r_s; rtol = 1e-12)
        Test.@test isapprox(f_h[1], f_s[1]; atol = 1e-12)
        Test.@test isapprox(f_h[2], f_s[2]; atol = 1e-12)
    end
end

# A rigid rotation preserves every geodesic distance, so d(σ)/dt = δu_L / R vanishes for EVERY
# pair at every separation. This is the sharpest single test of the parallel transport: it holds
# under transport and fails badly for a frame that is not transported.
Test.@testset "Solid-body rotation has no longitudinal increment" begin
    N = 250
    Ω = 7.292e-5
    lon = 360 .* rand(N) .- 180
    lat = 120 .* rand(N) .- 60
    x = permutedims(hcat(lon, lat))
    u = permutedims(hcat(Ω * EARTH_R .* cosd.(lat), zeros(N)))
    bins = collect(range(0.0, 8.0e6; length = 21))

    r = _sp(x, u, bins, DI.Haversine(EARTH_R))
    occ = r.L2.counts .> 0
    Test.@test any(occ)
    # L2 is a square of δu_L, so machine-zero here is ~eps^2 relative to S2.
    Test.@test sum(r.L2.sums[occ]) / sum(r.S2.sums[occ]) < 1e-24

    # Control: the identical field with a non-transported (flat lon/lat) frame puts a large
    # fraction of the energy into a quantity whose true value is zero.
    flat = SFC.calculate_structure_functions_single_pass(
        x, u, collect(range(0.0, 80.0; length = 21)); backend = CB.SerialBackend(),
        distance_metric = DI.Euclidean(), output_type = SFO.StructureFunctionSumsAndCounts,
    )
    occf = flat.L2.counts .> 0
    Test.@test sum(flat.L2.sums[occf]) / sum(flat.S2.sums[occf]) > 0.01
end

Test.@testset "Thin shell: radial component is carried but never transported" begin
    N = 200
    Ω = 7.292e-5
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))                                   # (2, N)
    ue = Ω * EARTH_R .* cosd.(lat)
    u2 = permutedims(hcat(ue, zeros(N)))                              # (2, N)
    u3 = permutedims(hcat(ue, zeros(N), 3.0 .* sind.(2 .* lat)))      # (3, N) + vertical
    bins = collect(range(0.0, 8.0e6; length = 21))
    m = DI.Haversine(EARTH_R)

    r2 = _sp(x, u2, bins, m)
    r3 = _sp(x, u3, bins, m)

    # The geodesic frame is tangent to the shell, so the radial component is orthogonal to both
    # t̂ and m̂ and cannot enter δu_L at all.
    Test.@test r3.L2.sums == r2.L2.sums
    Test.@test r3.L2.counts == r2.L2.counts
    # It does contribute to the total, and the six invariants stay consistent.
    Test.@test sum(r3.S2.sums) > sum(r2.S2.sums)
    Test.@test r3.S2.sums ≈ r3.L2.sums .+ r3.T2.sums
end

Test.@testset "Spherical geometry: backend agreement" begin
    N = 120
    lon = 300 .* rand(N) .- 150
    lat = 100 .* rand(N) .- 50
    x = permutedims(hcat(lon, lat))
    u = permutedims(hcat(randn(N), randn(N), randn(N)))
    bins = collect(range(0.0, 9.0e6; length = 13))
    m = DI.Haversine(EARTH_R)

    ref = _sp(x, u, bins, m)
    for be in (CB.AutoBackend(), CB.ThreadedBackend())
        got = _sp(x, u, bins, m; be = be)
        for k in (:S2, :L2, :T2, :S3, :L3, :L1T2)
            Test.@test got[k].counts == ref[k].counts
            Test.@test got[k].sums ≈ ref[k].sums
        end
    end

    # Single operators must agree with the single-pass invariants under the same geometry.
    for (sft, key) in ((SFT.LongitudinalSecondOrderStructureFunctionType(), :L2),
                       (SFT.TransverseSecondOrderStructureFunctionType(), :T2),
                       (SFT.SecondOrderStructureFunctionType(), :S2))
        one = SFC.calculate_structure_function(
            sft, x, u, bins; backend = CB.SerialBackend(), distance_metric = m,
            output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
        )
        Test.@test one.counts == ref[key].counts
        Test.@test one.sums ≈ ref[key].sums
    end
end

# On a patch of angular size ε the transported frame and a flat tangent-plane frame differ by
# O(ε) (the meridian convergence). Shrinking the patch must shrink the discrepancy proportionally
# — "the difference is small" is not the claim; the scaling is.
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
        # Local tangent plane at (0, lat0), in metres.
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
    # A 4x smaller patch must cut the O(r/R) discrepancy by at least ~2.5x (allowing slack for
    # the pair set changing with the patch and for the O((r/R)^2) term).
    Test.@test d_big / d_small > 2.5
end

# The geometry interface is the extension point: a user's own manifold works by adding methods,
# with no special-casing anywhere in the package. Demonstrated, not asserted in prose.
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

Test.@testset "User-defined geometry works end to end on every backend" begin
    N = 100
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.05, 1.4; length = 11))

    # Every separation is doubled, so doubled bins must reproduce the Euclidean histogram exactly.
    euc = _sp(x, u, bins, DI.Euclidean())
    for be in (CB.SerialBackend(), CB.AutoBackend())
        got = _sp(x, u, 2 .* bins, DoubledFlatMetric(); be = be)
        for k in (:S2, :L2, :T2, :S3, :L3, :L1T2)
            Test.@test got[k].counts == euc[k].counts
            Test.@test got[k].sums ≈ euc[k].sums
        end
    end

    # And through a different entry point (2D joint) to show nothing is special-cased per-path.
    vb = collect(range(-3.0, 3.0; length = 9))
    j_euc = SFC.calculate_structure_function(
        SFT.L2SFType(), x, u, bins, vb; backend = CB.SerialBackend(), verbose = false,
        show_progress = false,
    )
    j_got = SFC.calculate_structure_function(
        SFT.L2SFType(), x, u, 2 .* bins, vb; backend = CB.SerialBackend(),
        distance_metric = DoubledFlatMetric(), verbose = false, show_progress = false,
    )
    Test.@test j_got.counts == j_euc.counts
    Test.@test j_got.sums ≈ j_euc.sums
end

Test.@testset "Shape contract is geometry-aware" begin
    bins = collect(range(0.0, 5.0e6; length = 9))
    # Flat space: the coordinate count is the velocity dimension.
    Test.@test_throws DimensionMismatch _sp(rand(2, 8), randn(3, 8), bins, DI.Euclidean())
    Test.@test_throws DimensionMismatch _sp(rand(3, 8), randn(2, 8), bins, DI.Euclidean())
    # A shell locates a point with two coordinates, for both D = 2 and D = 3.
    m = DI.Haversine(EARTH_R)
    Test.@test_throws DimensionMismatch _sp(rand(3, 8), randn(3, 8), bins, m)
    lonlat = permutedims(hcat(20 .* rand(8), 20 .* rand(8)))
    Test.@test _sp(lonlat, randn(2, 8), bins, m) isa NamedTuple
    Test.@test _sp(lonlat, randn(3, 8), bins, m) isa NamedTuple
end
