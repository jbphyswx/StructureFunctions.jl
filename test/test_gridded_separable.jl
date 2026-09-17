using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using Random: Random

const FFT_TAG = SB.FastFourierTransformSpectralBackend()
const RE = 6.371e6

# An odd operator on a self-reverse lag is exactly zero in the sweep and round-off in the transform, so
# the comparison carries an absolute floor set by the field scale.
_close(got, ref) = isapprox(got, ref; rtol = 1e-9, atol = 1e-10 * max(1.0, maximum(abs, ref)))

# The lat-lon grid's cells as the unstructured entry wants them: (λ, φ) in radians and the local
# (east, north) components, flattened in the order the zonal schedule indexes cells.
function _zonal_points(lats, n_lon, dlon, u)
    W = size(u, 1)
    n_lat = length(lats)
    x = Matrix{Float64}(undef, 2, n_lon * n_lat)
    uu = Matrix{Float64}(undef, W, n_lon * n_lat)
    for j in 1:n_lat, i in 1:n_lon
        k = i + (j - 1) * n_lon
        x[1, k] = (i - 1) * dlon
        x[2, k] = lats[j]
        for c in 1:W
            uu[c, k] = u[c, i, j]
        end
    end
    return x, uu
end

# Cell coordinates of a rectilinear grid in the order `reshape(u, W, :)` flattens them.
function _grid_points(axes)
    dims = map(length, axes)
    x = Matrix{Float64}(undef, length(axes), prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims)), d in eachindex(axes)
        x[d, k] = axes[d][I[d]]
    end
    return x
end

# Bin edges between the separations a point set can produce. Pairs sharing a lag share one exact
# separation, so an edge through it splits a whole shell by rounding: the sweep bins them by the lag's
# separation and a pair loop by each pair's own coordinate difference, which differ by an ulp.
function _separated_bins(x, n_bins)
    N = size(x, 2)
    seps = sort!([sqrt(sum(abs2, x[:, j] .- x[:, i])) for i in 1:(N - 1) for j in (i + 1):N])
    distinct = [seps[1]]
    for s in seps
        s - distinct[end] > 1e-9 && push!(distinct, s)
    end
    idx = unique(round.(Int, range(1, length(distinct) - 1; length = n_bins)))
    return [0.0; [(distinct[i] + distinct[i + 1]) / 2 for i in idx]]
end

function _run(sf, u, sched, bins; valid = SFC.AllValid(), tag = nothing, backend = CB.SerialBackend())
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(bins))
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    if u isa Fields
        tag === nothing ? SFC.gridded_lag_sweep!(s, c, sf, u, sched, bins; valid, backend) :
                          SFC.gridded_sweep!(s, c, sf, u, sched, bins, tag; valid, backend)
    else
        D = size(u, 1)
        tag === nothing ? SFC.gridded_lag_sweep!(s, c, sf, u, sched, bins, Val(D); valid, backend) :
                          SFC.gridded_sweep!(s, c, sf, u, sched, bins, Val(D), tag; valid, backend)
    end
    return s, c
end

function _reference(sf, x, u, bins; metric = DI.Euclidean())
    nb = length(bins) - 1
    s = zeros(nb)
    c = zeros(Int, nb)
    if u isa Fields
        ref = SFC.calculate_structure_function(sf, x, u, bins; distance_metric = metric,
            backend = CB.SerialBackend(), output_type = SFO.StructureFunctionSumsAndCounts,
            verbose = false, show_progress = false)
        return ref.sums, Int.(ref.counts)
    end
    SFC.calculate_structure_function!(s, c, sf, x, u, bins; distance_metric = metric)
    return s, c
end

const ZONAL_OPS = (SFT.L2SFType(), SFT.T2SFType(), SFT.S2SFType(), SFT.L3SFType(), SFT.S3SFType(),
                   SFT.L1T2SFType())

Test.@testset "the zonal transform equals the zonal sweep and the spherical pair loop" begin
    # The sweep is checked against the pair loop, which shares no code with the lag enumeration; the
    # transform is checked against the sweep, which shares the enumeration but not the algebra.
    for (n_lon, dlon, periodic, lats) in (
            (12, 2π / 12, true, [-0.9, -0.5, -0.3, 0.1, 0.35, 0.8]),
            (9, 0.07, false, [-1.0, -0.6, -0.55, -0.2]),
            (16, 2π / 16, true, collect(range(0.1, 0.9; length = 5))),
        )
        Random.seed!(7100 + n_lon)
        u = randn(2, n_lon, length(lats))
        bins = collect(range(0.0, 0.45π; length = 8))
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, periodic)
        x, uu = _zonal_points(lats, n_lon, dlon, u)
        for sf in ZONAL_OPS
            ref_s, ref_c = _reference(sf, x, uu, bins; metric = SFH.SphericalDistance(1.0))
            sw_s, sw_c = _run(sf, u, sched, bins)
            tr_s, tr_c = _run(sf, u, sched, bins; tag = FFT_TAG)
            Test.@test sw_c == ref_c
            Test.@test isapprox(sw_s, ref_s; rtol = 1e-9, atol = 1e-10)
            Test.@test tr_c == sw_c
            Test.@test _close(tr_s, sw_s)
            Test.@test sum(tr_c) > 0
        end

        # missing cells: the mask's own transform carries the counts
        um = copy(u)
        uf = reshape(um, 2, :)
        for k in 1:size(uf, 2)
            rand() < 0.25 && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(um)
        Test.@test !(valid isa SFC.AllValid)
        for sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.S3SFType())
            sw_s, sw_c = _run(sf, um, sched, bins; valid)
            tr_s, tr_c = _run(sf, um, sched, bins; valid, tag = FFT_TAG)
            Test.@test tr_c == sw_c
            Test.@test _close(tr_s, sw_s)
            Test.@test all(isfinite, tr_s)
            Test.@test sum(tr_c) > 0
        end
    end
end

Test.@testset "a scalar field rides the zonal transform" begin
    n_lon, lats = 11, [-0.7, -0.4, 0.0, 0.3, 0.75]
    dlon = 2π / n_lon
    Random.seed!(7200)
    u = randn(2, n_lon, length(lats))
    th = randn(n_lon, length(lats))
    f = Fields(vectors = (u,), scalars = (th,))
    x, uu = _zonal_points(lats, n_lon, dlon, u)
    f_pts = Fields(vectors = (uu,), scalars = (vec(th),))
    bins = collect(range(0.0, 0.5π; length = 7))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, true)
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.MixedSFType{1, 0, 1}(), SFT.ScalarSFType{3}(),
               SFT.ScalarSFType{2}(), SFT.MixedSFType{0, 2, 1}())
        ref_s, ref_c = _reference(sf, x, f_pts, bins; metric = SFH.SphericalDistance(1.0))
        sw_s, sw_c = _run(sf, f, sched, bins)
        tr_s, tr_c = _run(sf, f, sched, bins; tag = FFT_TAG)
        Test.@test sw_c == ref_c
        Test.@test isapprox(sw_s, ref_s; rtol = 1e-9, atol = 1e-10)
        Test.@test tr_c == sw_c
        Test.@test _close(tr_s, sw_s)
        Test.@test any(!iszero, tr_s)
    end
end

Test.@testset "the rectilinear transform equals the sweep and the pair loop" begin
    Random.seed!(7300)
    xs = range(0.0, step = 0.1, length = 9)
    ys = [0.0, 0.13, 0.31, 0.5, 0.52, 0.9]
    zs = [0.0, 0.2, 0.45, 0.6]
    ws = range(0.0, step = 0.15, length = 7)
    cases = (
        # (field axes in order, schedule)
        ((xs, ys), SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (ys,), (1, 2))),
        ((ys, xs), SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (ys,), (2, 1))),
        ((ys, xs, zs), SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (ys, zs), (2, 1, 3))),
        ((xs, zs, ws), SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9, 7), (0.1, 0.15), (false, false)),
                                                  (zs,), (1, 3, 2))),
    )
    for (axes, sched) in cases
        Dg = length(axes)
        dims = map(length, axes)
        u = randn(Dg, dims...)
        x = _grid_points(axes)
        bins = _separated_bins(x, 7)
        ops = Dg == 2 ? (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.L2T1SFType(), SFT.S3SFType()) :
                        (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.S3SFType())
        for sf in ops
            ref_s, ref_c = _reference(sf, x, reshape(u, Dg, :), bins)
            sw_s, sw_c = _run(sf, u, sched, bins)
            tr_s, tr_c = _run(sf, u, sched, bins; tag = FFT_TAG)
            Test.@test sw_c == ref_c
            Test.@test isapprox(sw_s, ref_s; rtol = 1e-10, atol = 1e-12)
            Test.@test tr_c == sw_c
            Test.@test _close(tr_s, sw_s)
        end
        Test.@test sum(_run(SFT.L2SFType(), u, sched, [0.0, 1e3])[2]) == prod(dims) * (prod(dims) - 1) ÷ 2
    end

    # a periodic uniform direction wraps on both routes; the pair loop cannot be the oracle there, but
    # the two routes must agree and every pair must be counted once
    per = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((8,), (0.25,), (true,)), (zs,), (2, 1))
    u = randn(2, 4, 8)
    bins = [0.0; collect(range(0.17, 1.3; length = 7))]
    for sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.T3SFType())
        sw_s, sw_c = _run(sf, u, per, bins)
        tr_s, tr_c = _run(sf, u, per, bins; tag = FFT_TAG)
        Test.@test tr_c == sw_c
        Test.@test _close(tr_s, sw_s)
    end
    Test.@test sum(_run(SFT.L2SFType(), u, per, [0.0, 1e3])[2]) == 32 * 31 ÷ 2

    # a scalar multi-field on a stretched grid
    th = randn(4, 8)
    f = Fields(vectors = (u,), scalars = (th,))
    for sf in (SFT.MixedSFType{1, 0, 1}(), SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{3}())
        sw_s, sw_c = _run(sf, f, per, bins)
        tr_s, tr_c = _run(sf, f, per, bins; tag = FFT_TAG)
        Test.@test tr_c == sw_c
        Test.@test _close(tr_s, sw_s)
        Test.@test any(!iszero, sw_s)
    end
end

Test.@testset "culling the slab pairs and the lags changes nothing but the work" begin
    n_lon, lats = 12, collect(range(-1.2, 1.2; length = 9))
    dlon = 2π / n_lon
    Random.seed!(7400)
    u = randn(2, n_lon, length(lats))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, true)
    tight = collect(range(0.0, 0.12π; length = 5))
    x, uu = _zonal_points(lats, n_lon, dlon, u)
    ref_s, ref_c = _reference(SFT.L2SFType(), x, uu, tight; metric = SFH.SphericalDistance(1.0))
    sw_s, sw_c = _run(SFT.L2SFType(), u, sched, tight)
    tr_s, tr_c = _run(SFT.L2SFType(), u, sched, tight; tag = FFT_TAG)
    Test.@test sw_c == ref_c
    Test.@test isapprox(sw_s, ref_s; rtol = 1e-9, atol = 1e-10)
    Test.@test tr_c == sw_c
    Test.@test _close(tr_s, sw_s)
    Test.@test sum(sw_c) > 0
    n_lat = length(lats)
    Test.@test length(SFC.sweep_items(sched, 0.12π, 1, false)) < n_lat * (n_lat + 1) ÷ 2   # rows skipped
    Test.@test SFC.lag_limits(sched, 1, 1, 0.12π)[1] < n_lon ÷ 2                            # lags skipped
    # the equator row keeps the fewest offsets, a pole-side row the most
    Test.@test SFC.lag_limits(sched, 5, 5, 0.12π)[1] <= SFC.lag_limits(sched, 1, 1, 0.12π)[1]
    Test.@test SFC.lag_limits(sched, 0.12π)[1] >= maximum(I -> SFC.lag_limits(sched, I, I, 0.12π)[1], 1:n_lat)

    ys = [0.0, 0.13, 0.31, 0.5, 0.52, 0.9, 1.4, 1.5]
    rect = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (ys,), (1, 2))
    ur = randn(2, 9, length(ys))
    xr = _grid_points((range(0.0, step = 0.1, length = 9), ys))
    tight_r = _separated_bins(xr, 20)[1:5]
    ref_s, ref_c = _reference(SFT.L2SFType(), xr, reshape(ur, 2, :), tight_r)
    sw_s, sw_c = _run(SFT.L2SFType(), ur, rect, tight_r)
    tr_s, tr_c = _run(SFT.L2SFType(), ur, rect, tight_r; tag = FFT_TAG)
    Test.@test sw_c == ref_c
    Test.@test isapprox(sw_s, ref_s; rtol = 1e-10, atol = 1e-12)
    Test.@test tr_c == sw_c
    Test.@test _close(tr_s, sw_s)
    Test.@test length(SFC.sweep_items(rect, last(tight_r), 1, false)) < 8 * 9 ÷ 2
end

Test.@testset "the threaded backend gives the serial answer" begin
    n_lon, lats = 12, [-0.9, -0.5, -0.3, 0.1, 0.35, 0.8]
    dlon = 2π / n_lon
    Random.seed!(7500)
    u = randn(2, n_lon, length(lats))
    bins = collect(range(0.0, 0.5π; length = 8))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, true)
    for tag in (nothing, FFT_TAG), backend in (CB.ThreadedBackend(), CB.AutoBackend())
        ser_s, ser_c = _run(SFT.L3SFType(), u, sched, bins; tag)
        thr_s, thr_c = _run(SFT.L3SFType(), u, sched, bins; tag, backend)
        Test.@test thr_c == ser_c
        Test.@test _close(thr_s, ser_s)
    end
    # a single-slab schedule splits its lags across the tasks instead
    uni = SFC.UniformLagSchedule((9, 7), (0.1, 0.15), (false, true))
    uu = randn(2, 9, 7)
    ubins = [0.0; collect(range(0.0937, 1.1; length = 7))]
    Test.@test length(SFC.sweep_items(uni, 1.1, 4, true)) == 4
    for tag in (nothing, FFT_TAG)
        ser_s, ser_c = _run(SFT.L2SFType(), uu, uni, ubins; tag)
        thr_s, thr_c = _run(SFT.L2SFType(), uu, uni, ubins; tag, backend = CB.ThreadedBackend())
        Test.@test thr_c == ser_c
        Test.@test _close(thr_s, ser_s)
    end
    # the joint histogram too
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    ax_bins = [prevfloat(0.0); collect(range(0.3011, π - 0.3; length = 5)); π + 1e-9]
    na = length(ax_bins) - 1
    nb = length(ubins) - 1
    rect = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), ([0.0, 0.13, 0.31, 0.5, 0.52, 0.9, 1.4],), (1, 2))
    ur = randn(2, 9, 7)
    ss = zeros(nb, na); sc = zeros(Float64, nb, na)
    SFC.gridded_lag_sweep!(ss, sc, SFT.L2SFType(), ur, rect, ubins, ax_bins, Val(2); second_axis = src)
    ts = zeros(nb, na); tc = zeros(Float64, nb, na)
    SFC.gridded_sweep!(ts, tc, SFT.L2SFType(), ur, rect, ubins, ax_bins, Val(2), FFT_TAG;
                       second_axis = src, backend = CB.ThreadedBackend())
    hs = zeros(nb, na); hc = zeros(Float64, nb, na)
    SFC.gridded_lag_sweep!(hs, hc, SFT.L2SFType(), ur, rect, ubins, ax_bins, Val(2);
                           second_axis = src, backend = CB.ThreadedBackend())
    Test.@test tc == sc
    Test.@test hc == sc
    Test.@test _close(ts, ss)
    Test.@test _close(hs, ss)
    Test.@test sum(sc) > 0
    # a GPU tag has no gridded method, and says so
    Test.@test_throws ArgumentError _run(SFT.L2SFType(), u, sched, bins; backend = CB.GPUBackend(nothing))
end

Test.@testset "the sphere's radius reaches every route" begin
    a = (0.3, -0.2)
    b = (1.1, 0.4)
    Test.@test SFH.SphericalDistance(RE)(a, b) == RE * DI.SphericalAngle()(a, b)
    Test.@test SFH.pair_geometry_for(SFH.SphericalDistance(RE), Val(2)) isa SFH.SphericalGeometry{2}
    Test.@test SFH.pair_geometry_for(SFH.SphericalDistance(RE), Val(2)).radius == RE

    # the unstructured entry with the radius in the metric equals the unit sphere with scaled bins
    n_lon, lats = 12, collect(range(-0.6, 0.6; length = 5))
    dlon = 2π / n_lon
    Random.seed!(7600)
    u = randn(2, n_lon, length(lats))
    x, uu = _zonal_points(lats, n_lon, dlon, u)
    unit_bins = collect(range(0.0, 2.5; length = 7))
    s1, c1 = _reference(SFT.L2SFType(), x, uu, unit_bins; metric = DI.SphericalAngle())
    s2, c2 = _reference(SFT.L2SFType(), x, uu, RE .* unit_bins; metric = SFH.SphericalDistance(RE))
    Test.@test c1 == c2
    Test.@test isapprox(s1, s2; rtol = 1e-12)

    # through the grid entry: the same field on a range longitude axis (zonal schedule) and on a vector
    # one (enumerated pairs) reports the same separations in metres
    geo = FG.Geometry.SphericalGeometry(RE)
    lam = range(0.0, step = dlon, length = n_lon)
    phi = range(-0.6, 0.6; length = length(lats))
    zonal = FG.Grids.StructuredGrid(geo, lam, phi)
    scattered = FG.Grids.StructuredGrid(geo, collect(lam), phi)
    ext = Base.get_extension(SF, :StructureFunctionsFlowGeometriesExt)
    Test.@test Base.invokelatest(ext._lag_schedule, zonal) isa SFC.ZonalLagSchedule
    Test.@test Base.invokelatest(ext._lag_schedule, scattered) isa SFC.ScatteredPairs
    Test.@test Base.invokelatest(ext._lag_schedule, scattered).metric == SFH.SphericalDistance(RE)
    metre_bins = RE .* collect(range(0.0, 1.9; length = 8))
    rz = SFC.calculate_structure_function(SFT.L2SFType(), zonal, u, metre_bins;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    rs = SFC.calculate_structure_function(SFT.L2SFType(), scattered, u, metre_bins;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test rz.counts == rs.counts
    Test.@test isapprox(rz.sums, rs.sums; rtol = 1e-9, atol = 1e-10)
    Test.@test sum(rz.counts) > 0
    Test.@test count(!iszero, rz.counts) > 1              # the bins really are in metres
end

Test.@testset "axis types decide the route, and the routes agree" begin
    ext = Base.get_extension(SF, :StructureFunctionsFlowGeometriesExt)
    sched_of(grid) = Base.invokelatest(ext._lag_schedule, grid)

    # a lat-lon sampling built from a recipe stores its axes as vectors, so it enumerates pairs; the
    # same nodes given as ranges take the zonal schedule; the answers agree
    n_lon, n_lat = 10, 5
    geo = FG.Geometry.SphericalGeometry(1.0)
    recipe = FG.Connectivity.structured_grid(FG.SphericalSampling.LatLonSampling(), n_lat;
                                             geometry = geo, nlon = n_lon)
    ranged = FG.Grids.StructuredGrid(geo, range(0.0, step = 2π / n_lon, length = n_lon),
                                     range(-π / 2, π / 2; length = n_lat))
    Test.@test sched_of(recipe) isa SFC.ScatteredPairs
    Test.@test sched_of(ranged) isa SFC.ZonalLagSchedule
    Random.seed!(7700)
    u = randn(2, n_lon, n_lat)
    bins = collect(range(0.013, 0.9π; length = 7))
    r1 = SFC.calculate_structure_function(SFT.L2SFType(), recipe, u, bins;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    r2 = SFC.calculate_structure_function(SFT.L2SFType(), ranged, u, bins;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test r1.counts == r2.counts
    Test.@test isapprox(r1.sums, r2.sums; rtol = 1e-9, atol = 1e-10)
    Test.@test sum(r1.counts) > 0

    # Cartesian: one uniform axis keeps its lags, none enumerates, and a wrapping coordinate list is refused
    cgeo = FG.Geometry.CartesianGeometry()
    xs = [0.0, 0.1, 0.35, 0.8, 0.85]
    ys = range(0.0, step = 0.2, length = 4)
    Test.@test sched_of(FG.Grids.StructuredGrid(cgeo, xs, ys)) isa SFC.RectilinearLagSchedule{1, 1, 2}
    Test.@test sched_of(FG.Grids.StructuredGrid(cgeo, ys, xs)) isa SFC.RectilinearLagSchedule{1, 1, 2}
    Test.@test sched_of(FG.Grids.StructuredGrid(cgeo, xs, collect(ys))) isa SFC.ScatteredPairs
    Test.@test sched_of(FG.Grids.StructuredGrid(cgeo, ys, ys)) isa SFC.UniformLagSchedule
    Test.@test_throws ArgumentError sched_of(FG.Grids.StructuredGrid(cgeo, xs, ys;
        topology = (FG.Grids.Periodic(), FG.Grids.Bounded())))
    Test.@test_throws ArgumentError sched_of(FG.Grids.StructuredGrid(cgeo, xs, collect(ys);
        topology = (FG.Grids.Bounded(), FG.Grids.Periodic())))
    # and the field's axis order is honoured whichever axis is uniform
    grid_yx = FG.Grids.StructuredGrid(cgeo, ys, xs)
    u_yx = randn(2, length(ys), length(xs))
    x_yx = _grid_points((ys, xs))
    bins_yx = _separated_bins(x_yx, 6)
    got = SFC.calculate_structure_function(SFT.L3SFType(), grid_yx, u_yx, bins_yx, UInt32, FFT_TAG;
        output_type = SFO.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    ref_s, ref_c = _reference(SFT.L3SFType(), x_yx, reshape(u_yx, 2, :), bins_yx)
    Test.@test got.counts == ref_c
    Test.@test _close(got.sums, ref_s)
end

Test.@testset "the transform tag is served wherever there is a lag, and refused where there is none" begin
    Random.seed!(7800)
    n_lon, lats = 12, [-0.9, -0.5, -0.3, 0.1, 0.35, 0.8]
    sched = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    u = randn(2, n_lon, length(lats))
    bins = collect(range(0.0, 0.5π; length = 8))
    au_s, au_c = _run(SFT.L2SFType(), u, sched, bins; tag = SB.AutoSpectralBackend())
    sw_s, sw_c = _run(SFT.L2SFType(), u, sched, bins)
    Test.@test au_c == sw_c
    Test.@test _close(au_s, sw_s)
    ds_s, ds_c = _run(SFT.L2SFType(), u, sched, bins; tag = SB.DirectSumSpectralBackend())
    Test.@test ds_c == sw_c
    Test.@test ds_s == sw_s
    # a norm is no polynomial, on this schedule as on any
    Test.@test_throws ArgumentError _run(SFT.FullVectorStructureFunctionType{3}(), u, sched, bins; tag = FFT_TAG)
    # no uniform direction, no transform
    pts = randn(2, 20)
    scat = SFC.ScatteredPairs(pts, DI.Euclidean())
    err = try
        SFC.gridded_sweep!(zeros(3), zeros(Int, 3), SFT.L2SFType(), reshape(randn(2, 20), 2, :), scat,
                           [0.0, 1.0, 2.0, 3.0], Val(2), Val(1), Val(0), FFT_TAG)
        nothing
    catch e
        e
    end
    Test.@test err isa ArgumentError
    Test.@test occursin("ScatteredPairs", err.msg)
    # the angle-resolved histogram has no meaning on a sphere, on either route
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    ax = [prevfloat(0.0), 1.0, 2.0, π + 1e-9]
    Test.@test_throws ArgumentError SFC.gridded_sweep!(zeros(7, 3), zeros(7, 3), SFT.L2SFType(), u, sched,
                                                       bins, ax, Val(2), FFT_TAG; second_axis = src)
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(zeros(7, 3), zeros(7, 3), SFT.L2SFType(), u, sched,
                                                           bins, ax, Val(2); second_axis = src)
end

Test.@testset "uniform_lag_box agrees with the lag limits it describes" begin
    # A device launch indexes its work items by division on one lag box when a schedule answers
    # `true`, and through the prefix sum of the pairs' box volumes when it answers `false`. A `true`
    # that is not the truth silently drops every lag outside the first pair's box, so the trait is
    # checked against `lag_limits` itself, on every schedule and at several radii.
    lats = collect(range(-1.2, 1.2; length = 9))
    zonal = SFC.ZonalLagSchedule(lats, 16, 2π / 16, 1.0, true)
    uniform = SFC.UniformLagSchedule((12, 10), (0.1, 0.2), (true, false))
    rect = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)),
                                      (collect(range(0.0, 1.0; length = 7)),), (2, 1))
    for s in (uniform, rect, zonal), r_max in (0.05, 0.3, 1.0, 3.0)
        items = SFC.sweep_items(s, r_max, 1, false)
        isempty(items) && continue
        limits = unique(SFC.lag_limits(s, it[1], it[2], r_max) for it in items)
        # the direction the device depends on: `true` must mean one box for every pair
        SFC.uniform_lag_box(s) && Test.@test length(limits) == 1
    end
    # and the sphere must answer `false`, because at a radius that saturates nothing a row pair near
    # a pole reaches more longitude offsets than one at the equator
    zonal_limits = unique(SFC.lag_limits(zonal, it[1], it[2], 0.3)
                          for it in SFC.sweep_items(zonal, 0.3, 1, false))
    Test.@test length(zonal_limits) > 1
    Test.@test SFC.uniform_lag_box(zonal) == false
    Test.@test SFC.uniform_lag_box(uniform) && SFC.uniform_lag_box(rect)
end
