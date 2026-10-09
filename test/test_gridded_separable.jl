using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Distances: Distances as DI
using StaticArrays: StaticArrays as SA
using Random: Random

const FFT_TAG = SB.FastFourierTransformSpectralBackend()
const RE = 6.371e6

# Absolute floor at the field scale: an odd operator on a self-reverse lag is exactly zero only in the sweep.
_close(got, ref) = isapprox(got, ref; rtol = 1e-9, atol = 1e-10 * max(1.0, maximum(abs, ref)))

# The lat-lon grid's cells as the unstructured entry wants them: (λ, φ) in radians and local (east, north) components.
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

# Bin edges midway between the distinct separations of a point set, so no shell sits on an edge.
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
        ref = SFC.calculate_structure_function(sf, x, u, bins, SFO.StructureFunctionSumsAndCounts;
            distance_metric = metric, backend = CB.SerialBackend())
        return ref.sums, Int.(ref.counts)
    end
    SFC.calculate_structure_function!(s, c, sf, x, u, bins; distance_metric = metric, backend = CB.SerialBackend())
    return s, c
end

# (n_lon, dlon, periodic, lats, operator): each grid once, complete and masked.
const SEPARABLE_ZONAL_CASES = (
    (12, 2π / 12, true, [-0.9, -0.5, -0.3, 0.1, 0.35, 0.8], SFT.L2SFType()),
    (9, 0.07, false, [-1.0, -0.6, -0.55, -0.2], SFT.L3SFType()),
    (16, 2π / 16, true, collect(range(0.1, 0.9; length = 5)), SFT.L2SFType()),
)

Test.@testset "the zonal transform equals the spherical pair loop, and the masked sweep on a masked field" begin
    complete_ok, masked_ok = Bool[], Bool[]
    for (n_lon, dlon, periodic, lats, sf) in SEPARABLE_ZONAL_CASES
        Random.seed!(7100 + n_lon)
        u = randn(2, n_lon, length(lats))
        bins = collect(range(0.0, 0.45π; length = 8))
        sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, periodic)
        x, uu = _zonal_points(lats, n_lon, dlon, u)
        ref_s, ref_c = _reference(sf, x, uu, bins; metric = SFH.SphericalDistance(1.0))
        tr_s, tr_c = _run(sf, u, sched, bins; tag = FFT_TAG)
        push!(complete_ok, tr_c == ref_c && sum(ref_c) > 0 && _close(tr_s, ref_s))

        um = copy(u)
        uf = reshape(um, 2, :)
        for k in 1:size(uf, 2)
            rand() < 0.25 && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(um)
        sw_s, sw_c = _run(SFT.L2SFType(), um, sched, bins; valid)
        tr_s, tr_c = _run(SFT.L2SFType(), um, sched, bins; valid, tag = FFT_TAG)
        push!(masked_ok, tr_c == sw_c && sum(sw_c) > 0 && all(isfinite, tr_s) && _close(tr_s, sw_s))
    end
    Test.@test all(complete_ok)
    Test.@test all(masked_ok)
end

Test.@testset "a scalar field rides the zonal transform" begin
    # A transported vector times an odd power of the scalar, against the spherical pair loop.
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
    sf = SFT.MixedSFType{1, 0, 1}()
    ref_s, ref_c = _reference(sf, x, f_pts, bins; metric = SFH.SphericalDistance(1.0))
    tr_s, tr_c = _run(sf, f, sched, bins; tag = FFT_TAG)
    Test.@test tr_c == ref_c
    Test.@test any(!iszero, ref_s) && _close(tr_s, ref_s)
end

const SEPARABLE_XS = range(0.0, step = 0.1, length = 9)
const SEPARABLE_YS = [0.0, 0.13, 0.31, 0.5, 0.52, 0.9]
const SEPARABLE_ZS = [0.0, 0.2, 0.45, 0.6]
const SEPARABLE_WS = range(0.0, step = 0.15, length = 7)
# (field axes in order, schedule, operator): each layout once, the signed transverse operator on both axis orders.
const SEPARABLE_RECT_CASES = (
    ((SEPARABLE_XS, SEPARABLE_YS),
     SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (SEPARABLE_YS,), (1, 2)),
     SFT.L2T1SFType()),
    ((SEPARABLE_YS, SEPARABLE_XS),
     SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (SEPARABLE_YS,), (2, 1)),
     SFT.L2T1SFType()),
    ((SEPARABLE_YS, SEPARABLE_XS, SEPARABLE_ZS),
     SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (SEPARABLE_YS, SEPARABLE_ZS),
                                (2, 1, 3)),
     SFT.L2SFType()),
    ((SEPARABLE_XS, SEPARABLE_ZS, SEPARABLE_WS),
     SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9, 7), (0.1, 0.15), (false, false)), (SEPARABLE_ZS,),
                                (1, 3, 2)),
     SFT.L2SFType()),
)

Test.@testset "the rectilinear transform equals the sweep and the pair loop" begin
    Random.seed!(7300)
    sweep_ok, transform_ok = Bool[], Bool[]
    for (axes, sched, sf) in SEPARABLE_RECT_CASES
        Dg = length(axes)
        dims = map(length, axes)
        N = prod(dims)
        u = randn(Dg, dims...)
        x = _grid_points(axes)
        bins = _separated_bins(x, 7)
        ref_s, ref_c = _reference(sf, x, reshape(u, Dg, :), bins)
        sw_s, sw_c = _run(sf, u, sched, bins)
        tr_s, tr_c = _run(sf, u, sched, bins; tag = FFT_TAG)
        all_pairs = sum(_run(SFT.L2SFType(), u, sched, [0.0, 1e3])[2]) == N * (N - 1) ÷ 2
        push!(sweep_ok, all_pairs && sw_c == ref_c && sum(ref_c) > 0 &&
                        isapprox(sw_s, ref_s; rtol = 1e-10, atol = 1e-12))
        push!(transform_ok, tr_c == sw_c && _close(tr_s, sw_s))
    end
    # a periodic uniform axis beside a stretched one, on a vector field and on a multi-field
    per = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((8,), (0.25,), (true,)), (SEPARABLE_ZS,), (2, 1))
    u = randn(2, 4, 8)
    bins = [0.0; collect(range(0.17, 1.3; length = 7))]
    push!(sweep_ok, sum(_run(SFT.L2SFType(), u, per, [0.0, 1e3])[2]) == 32 * 31 ÷ 2)
    sw_s, sw_c = _run(SFT.L2T1SFType(), u, per, bins)
    tr_s, tr_c = _run(SFT.L2T1SFType(), u, per, bins; tag = FFT_TAG)
    push!(transform_ok, tr_c == sw_c && _close(tr_s, sw_s))
    f = Fields(vectors = (u,), scalars = (randn(4, 8),))
    sw_s, sw_c = _run(SFT.MixedSFType{1, 0, 1}(), f, per, bins)
    tr_s, tr_c = _run(SFT.MixedSFType{1, 0, 1}(), f, per, bins; tag = FFT_TAG)
    push!(transform_ok, tr_c == sw_c && any(!iszero, sw_s) && _close(tr_s, sw_s))
    Test.@test all(sweep_ok)
    Test.@test all(transform_ok)
end

Test.@testset "bins shorter than the grid: the zonal and rectilinear routes equal the pair loop" begin
    n_lon, lats = 12, collect(range(-1.2, 1.2; length = 9))
    dlon = 2π / n_lon
    Random.seed!(7400)
    u = randn(2, n_lon, length(lats))
    x, uu = _zonal_points(lats, n_lon, dlon, u)
    ys = [0.0, 0.13, 0.31, 0.5, 0.52, 0.9, 1.4, 1.5]
    ur = randn(2, 9, length(ys))
    xr = _grid_points((range(0.0, step = 0.1, length = 9), ys))
    tight_r = _separated_bins(xr, 20)[1:5]
    # (schedule, field, points, bins, metric, rtol, atol): a sphere's rows and a stretched axis's slabs beyond the bins
    cases = ((SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, true), u, uu, x, collect(range(0.0, 0.12π; length = 5)),
              SFH.SphericalDistance(1.0), 1e-9, 1e-10),
             (SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((9,), (0.1,), (false,)), (ys,), (1, 2)), ur,
              reshape(ur, 2, :), xr, tight_r, DI.Euclidean(), 1e-10, 1e-12))
    sweep_ok, transform_ok = Bool[], Bool[]
    for (sched, field, flat, pts, bins, metric, rtol, atol) in cases
        ref_s, ref_c = _reference(SFT.L2SFType(), pts, flat, bins; metric)
        sw_s, sw_c = _run(SFT.L2SFType(), field, sched, bins)
        tr_s, tr_c = _run(SFT.L2SFType(), field, sched, bins; tag = FFT_TAG)
        push!(sweep_ok, sw_c == ref_c && sum(ref_c) > 0 && isapprox(sw_s, ref_s; rtol, atol))
        push!(transform_ok, tr_c == sw_c && _close(tr_s, sw_s))
    end
    Test.@test all(sweep_ok)
    Test.@test all(transform_ok)
end

Test.@testset "a sphere's radius scales the separations of every route and nothing else" begin
    # The unit sphere's pair loop with radian bins is the oracle for the metric, the zonal and the enumerated routes.
    a = (0.3, -0.2)
    b = (1.1, 0.4)
    Test.@test SFH.SphericalDistance(RE)(a, b) == RE * DI.SphericalAngle()(a, b)
    n_lon, lats = 12, collect(range(-0.6, 0.6; length = 5))
    dlon = 2π / n_lon
    Random.seed!(7600)
    u = randn(2, n_lon, length(lats))
    x, uu = _zonal_points(lats, n_lon, dlon, u)
    unit_bins = collect(range(0.0, 2.5; length = 7))
    s1, c1 = _reference(SFT.L2SFType(), x, uu, unit_bins; metric = DI.SphericalAngle())
    s2, c2 = _reference(SFT.L2SFType(), x, uu, RE .* unit_bins; metric = SFH.SphericalDistance(RE))
    geo = FG.Geometry.SphericalGeometry(RE)
    lam = range(0.0, step = dlon, length = n_lon)
    phi = range(-0.6, 0.6; length = length(lats))
    rz = SFC.calculate_structure_function(SFT.L2SFType(), FG.Grids.StructuredGrid(geo, lam, phi), u,
                                          RE .* unit_bins, SFO.StructureFunctionSumsAndCounts;
                                          backend = CB.SerialBackend())
    rs = SFC.calculate_structure_function(SFT.L2SFType(), FG.Grids.StructuredGrid(geo, collect(lam), phi), u,
                                          RE .* unit_bins, SFO.StructureFunctionSumsAndCounts;
                                          backend = CB.SerialBackend())
    Test.@test count(!iszero, c1) > 1 && c2 == c1 && rz.counts == c1 && rs.counts == c1
    Test.@test isapprox(s2, s1; rtol = 1e-12) && isapprox(rz.sums, s1; rtol = 1e-9, atol = 1e-10) &&
               isapprox(rs.sums, s1; rtol = 1e-9, atol = 1e-10)
end

Test.@testset "the transform is refused without a uniform direction, and the angle histogram on a sphere" begin
    Random.seed!(7800)
    scat = SFC.ScatteredPairs(randn(2, 20), DI.Euclidean())
    Test.@test_throws ArgumentError SFC.gridded_sweep!(zeros(3), zeros(Int, 3), SFT.L2SFType(), randn(2, 20), scat,
                                                       [0.0, 1.0, 2.0, 3.0], Val(2), Val(1), Val(0), FFT_TAG)
    n_lon, lats = 12, [-0.9, -0.5, -0.3, 0.1, 0.35, 0.8]
    sched = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    u = randn(2, n_lon, length(lats))
    bins = collect(range(0.0, 0.5π; length = 8))
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    ax = [prevfloat(0.0), 1.0, 2.0, π + 1e-9]
    Test.@test_throws ArgumentError SFC.gridded_sweep!(zeros(7, 3), zeros(7, 3), SFT.L2SFType(), u, sched,
                                                       bins, ax, Val(2), FFT_TAG; second_axis = src)
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(zeros(7, 3), zeros(7, 3), SFT.L2SFType(), u, sched,
                                                           bins, ax, Val(2); second_axis = src)
end
