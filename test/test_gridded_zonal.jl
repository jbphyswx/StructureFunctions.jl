using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, HelperFunctions as SFH
using StructureFunctions: MultiFields as MF
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using LinearAlgebra: dot
using Random: Random
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using FFTW: FFTW

# `Distances.SphericalAngle` is the central angle, the separation on the unit sphere.
const R_UNIT = 1.0

# The grid's cells as the unstructured entry wants them: (λ, φ) in radians and local (east, north) components.
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

# The field under the equatorial reflection φ ↦ −φ on a latitude axis symmetric about the equator: (u_E, −u_N).
function _equator_mirror(u::AbstractArray{T, 3}) where {T}
    n_lon, n_lat = size(u, 2), size(u, 3)
    m = similar(u)
    for j in 1:n_lat, i in 1:n_lon
        m[1, i, j] = u[1, i, n_lat + 1 - j]
        m[2, i, j] = -u[2, i, n_lat + 1 - j]
    end
    return m
end

# Ambient position and local east and north unit vectors at longitude λ, latitude φ.
_ambient_basis(λ, φ) = (SA.SVector(cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)), SA.SVector(-sin(λ), cos(λ), 0.0),
                        SA.SVector(-sin(φ) * cos(λ), -sin(φ) * sin(λ), cos(φ)))

# (φ₁, φ₂, Δλ): each latitude and longitude offset once, every pair rebuilt around the whole circle.
const ZONAL_FRAME_CASES = ((-1.2, -1.0, 0.05), (-0.4, 0.9, 3.0), (0.0, -0.2, 0.7), (0.3, 0.15, 1.9), (1.1, 0.9, 0.7))

Test.@testset "the geodesic frame does not depend on longitude" begin
    g = SFH.SphericalGeometry{2}(DI.SphericalAngle(), 1.0)
    worst = 0.0
    for (p1, p2, dl) in ZONAL_FRAME_CASES
        _, r0, A0, B0 = SFC.zonal_transport(g, p1, p2, dl, Val(2))
        for l0 in range(-2π, 4π; length = 8)
            pA, EA, NA = _ambient_basis(l0, p1)
            pB, EB, NB = _ambient_basis(l0 + dl, p2)
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

# (n_lon, lats, r_max / π, operators): each grid and each operator once against the unstructured path.
const ZONAL_SWEEP_CASES = (
    (12, collect(range(-0.5, 0.5; length = 5)), 0.45, (SFT.L2SFType(), SFT.T2SFType())),
    (9, collect(range(-1.0, -0.2; length = 4)), 0.6, (SFT.S2SFType(),)),
    (16, collect(range(0.1, 0.9; length = 6)), 0.35, (SFT.L3SFType(),)),
)

Test.@testset "the zonal sweep equals the unstructured spherical path" begin
    counts_ok, sums_ok = Bool[], Bool[]
    for (n_lon, lats, frac, ops) in ZONAL_SWEEP_CASES
        n_lat = length(lats)
        dlon = 2π / n_lon
        Random.seed!(9100 + n_lon * n_lat)
        u = randn(2, n_lon, n_lat)
        x, uu = _zonal_points(lats, n_lon, dlon, u)
        bins = collect(range(0.0, frac * π * R_UNIT; length = 9))
        nb = length(bins) - 1
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)
        for sf in ops
            got_s = zeros(nb); got_c = zeros(Int, nb)
            SFC.gridded_lag_sweep!(got_s, got_c, sf, u, sched, bins, Val(2))
            ref_s = zeros(nb); ref_c = zeros(Int, nb)
            SFC.calculate_structure_function!(ref_s, ref_c, sf, x, uu, bins;
                                             distance_metric = DI.SphericalAngle())
            push!(counts_ok, got_c == ref_c && sum(ref_c) > 0)
            push!(sums_ok, isapprox(got_s, ref_s; rtol = 1e-9, atol = 1e-10))
        end
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

# (lats, dlon, operators): ascending axes, latitudes running north to south, then longitudes running westward.
const ZONAL_ORDER_CASES = (
    (collect(range(-0.6, 0.6; length = 5)), 2π / 11,
     (SFT.MixedSFType{1, 0, 1}(), SFT.ScalarSFType{3}(), SFT.MixedSFType{1, 0, 2}())),
    (collect(range(0.6, -0.6; length = 5)), 2π / 11, (SFT.MixedSFType{1, 0, 1}(),)),
    (collect(range(-0.6, 0.6; length = 5)), -2π / 11, (SFT.ScalarSFType{3}(),)),
)

Test.@testset "odd scalar moments read each pair in the point path's order, on ascending and descending axes" begin
    # An odd longitude count, so no pair is half a turn apart on a parallel and every pair has a first end.
    n_lon = 11
    Random.seed!(9150)
    counts_ok, sums_ok = Bool[], Bool[]
    for (lats, dlon, ops) in ZONAL_ORDER_CASES
        n_lat = length(lats)
        u = randn(2, n_lon, n_lat)
        th = randn(n_lon, n_lat)
        x, uu = _zonal_points(lats, n_lon, dlon, u)
        f_grid = MF.Fields(vectors = (u,), scalars = (th,))
        f_pts = MF.Fields(vectors = (uu,), scalars = (vec(th),))
        bins = collect(range(0.0, 0.45 * π; length = 7))
        nb = length(bins) - 1
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)
        for sf in ops
            got_s = zeros(nb); got_c = zeros(Int, nb)
            SFC.gridded_lag_sweep!(got_s, got_c, sf, f_grid, sched, bins)
            ref = SFC.calculate_structure_function(sf, x, f_pts, bins, SF.StructureFunctionSumsAndCounts;
                distance_metric = DI.SphericalAngle())
            push!(counts_ok, got_c == Int.(ref.counts))
            push!(sums_ok, any(!iszero, got_s) && isapprox(got_s, ref.sums; rtol = 1e-9, atol = 1e-10))
        end
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

Test.@testset "the zonal sweep counts every pair once, less the antipodal pairs it refuses" begin
    # Symmetric latitudes and an even longitude count put each point's antipode on the grid: N/2 pairs with no direction.
    n_lon = 8
    lats = collect(range(-0.6, 0.6; length = 4))
    dlon = 2π / n_lon
    N = n_lon * length(lats)
    Random.seed!(9200)
    u = randn(2, n_lon, length(lats))
    bins = collect(range(0.0, 3.5; length = 5))
    for periodic in (true, false)
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, periodic)
        s = zeros(4); c = zeros(Int, 4)
        SFC.gridded_lag_sweep!(s, c, SFT.S2SFType(), u, sched, bins, Val(2))
        Test.@test sum(c) == N * (N - 1) ÷ 2 - N ÷ 2 && all(isfinite, s)
    end
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
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)
    bins = collect(range(0.0, 10.0; length = 4))
    s = zeros(3); c = zeros(Int, 3)
    SFC.gridded_lag_sweep!(s, c, SFT.S2SFType(), u, sched, bins, Val(2); valid = SFC.field_validity(u))
    Test.@test sum(c) == (N - 3) * (N - 4) ÷ 2
    Test.@test all(isfinite, s)
end

Test.@testset "a spherical grid entry equals the unstructured path: wrapping, regional and stretched longitudes" begin
    geo = FG.Geometry.SphericalGeometry(R_UNIT)
    n_lon, n_lat = 12, 5
    phi = range(-0.5, 0.5; length = n_lat)
    Random.seed!(9700)
    u = randn(2, n_lon, n_lat)
    bins = collect(range(0.0, 0.9 * π; length = 9))
    us = randn(2, 4, n_lat)
    # (longitudes, field): a whole circle, a regional span, and a stretched axis
    cases = ((range(0.0, step = 2π / n_lon, length = n_lon), u), (range(0.0, step = 0.05, length = n_lon), u),
             ([0.0, 0.1, 0.35, 0.9], us))
    counts_ok, sums_ok = Bool[], Bool[]
    for (lam, field) in cases
        grid = FG.Grids.StructuredGrid(geo, lam, phi)
        got = SFC.calculate_structure_function(SFT.L2SFType(), grid, field, bins, UInt32,
                                               SF.StructureFunctionSumsAndCounts)
        x = Matrix{Float64}(undef, 2, length(lam) * n_lat)
        for (k, I) in enumerate(CartesianIndices((length(lam), n_lat)))
            x[1, k] = lam[I[1]]
            x[2, k] = phi[I[2]]
        end
        ref_s = zeros(8); ref_c = zeros(UInt32, 8)
        SFC.calculate_structure_function!(ref_s, ref_c, SFT.L2SFType(), x, reshape(field, 2, :), bins;
                                         distance_metric = DI.SphericalAngle())
        push!(counts_ok, got.counts == ref_c && sum(ref_c) > 0)
        push!(sums_ok, isapprox(got.sums, ref_s; rtol = 1e-9, atol = 1e-10))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

# Each operator once and each parity on both latitude axes; an odd longitude count keeps antipodes off the grid.
const ZONAL_REFLECTION_CASES = (
    ([-1.0, -0.6, -0.25, 0.25, 0.6, 1.0],
     ((SFT.L2SFType(), 1), (SFT.S2SFType(), 1), (SFT.S3SFType(), 1), (SFT.T3SFType(), -1))),
    ([-0.8, -0.3, 0.0, 0.3, 0.8],
     ((SFT.T2SFType(), 1), (SFT.L3SFType(), 1), (SFT.L1T2SFType(), 1), (SFT.L2T1SFType(), -1))),
)

Test.@testset "the equatorial reflection flips exactly the odd-transverse operators" begin
    # On the sweep, the transform and the point path, each of which also gives the sweep's answer.
    tag = SB.FastFourierTransformSpectralBackend()
    counts_ok, sweep_ok, transform_ok, points_ok = Bool[], Bool[], Bool[], Bool[]
    for (lats, ops) in ZONAL_REFLECTION_CASES
        n_lon = 11
        dlon = 2π / n_lon
        n_lat = length(lats)
        Random.seed!(9400 + n_lat)
        u = randn(2, n_lon, n_lat)
        um = _equator_mirror(u)
        x, uu = _zonal_points(lats, n_lon, dlon, u)
        xm, uum = _zonal_points(lats, n_lon, dlon, um)
        bins = collect(range(0.0, 0.8 * π; length = 9))
        nb = length(bins) - 1
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, R_UNIT, true)
        for (sf, parity) in ops
            s0 = zeros(nb); c0 = zeros(Int, nb)
            SFC.gridded_lag_sweep!(s0, c0, sf, u, sched, bins, Val(2))
            s1 = zeros(nb); c1 = zeros(Int, nb)
            SFC.gridded_lag_sweep!(s1, c1, sf, um, sched, bins, Val(2))
            scale = maximum(abs, s0)
            t0 = zeros(nb); tc0 = zeros(Int, nb)
            SFC.gridded_sweep!(t0, tc0, sf, u, sched, bins, Val(2), tag)
            t1 = zeros(nb); tc1 = zeros(Int, nb)
            SFC.gridded_sweep!(t1, tc1, sf, um, sched, bins, Val(2), tag)
            p0 = zeros(nb); pc0 = zeros(Int, nb)
            SFC.calculate_structure_function!(p0, pc0, sf, x, uu, bins; distance_metric = DI.SphericalAngle())
            p1 = zeros(nb); pc1 = zeros(Int, nb)
            SFC.calculate_structure_function!(p1, pc1, sf, xm, uum, bins; distance_metric = DI.SphericalAngle())
            push!(counts_ok, c1 == c0 && tc0 == c0 && tc1 == c0 && pc0 == c0 && pc1 == c0)
            push!(sweep_ok, any(!iszero, s0) && isapprox(s1, parity .* s0; rtol = 1e-10, atol = 1e-11 * scale))
            push!(transform_ok, isapprox(t0, s0; rtol = 1e-10, atol = 1e-11 * scale) &&
                                isapprox(t1, parity .* t0; rtol = 1e-10, atol = 1e-11 * scale))
            push!(points_ok, isapprox(p0, s0; rtol = 1e-9, atol = 1e-10 * scale) &&
                             isapprox(p1, parity .* p0; rtol = 1e-10, atol = 1e-11 * scale))
        end
    end
    Test.@test all(counts_ok)
    Test.@test all(sweep_ok)
    Test.@test all(transform_ok)
    Test.@test all(points_ok)
end
