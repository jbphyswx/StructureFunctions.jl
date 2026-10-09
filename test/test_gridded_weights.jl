using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const FFT_TAG = SB.FastFourierTransformSpectralBackend()
const SERIAL = CB.SerialBackend()
const DEVICE = CB.GPUBackend(KA.CPU())
const RAW = SF.StructureFunctionSumsAndCounts
const L2, L3 = SFT.L2SFType(), SFT.L3SFType()

# One run on either gridded route from a packed field and a schedule, into Float64 counts.
function _weighted_run(sf, data, sched, bins; valid = SFC.AllValid(), weights = nothing, tag = nothing,
                       backend = SERIAL)
    nb = length(bins) - 1
    s, c = zeros(nb), zeros(nb)
    if tag === nothing
        SFC.gridded_lag_sweep!(s, c, sf, data, sched, bins, Val(2), Val(1), Val(0); valid, weights, backend)
    else
        SFC.gridded_sweep!(s, c, sf, data, sched, bins, Val(2), Val(1), Val(0), tag; valid, weights, backend)
    end
    return s, c
end

function _weighted_run(sf, f::Fields, sched, bins; weights = nothing, tag = nothing)
    nb = length(bins) - 1
    s, c = zeros(nb), zeros(nb)
    if tag === nothing
        SFC.gridded_lag_sweep!(s, c, sf, f, sched, bins; weights, backend = SERIAL)
    else
        SFC.gridded_sweep!(s, c, sf, f, sched, bins, tag; weights, backend = SERIAL)
    end
    return s, c
end

# The weighted pair statistic on flat points: Σ w_i w_j v_ij and Σ w_i w_j over the pairs of each (lo, hi] bin.
function _flat_pair_loop(sf, x, u, bins, w)
    nb = length(bins) - 1
    s, c = zeros(nb), zeros(nb)
    N = size(x, 2)
    for i in 1:(N - 1), j in (i + 1):N
        dx = x[:, j] .- x[:, i]
        r = LA.norm(dx)
        b = searchsortedfirst(bins, r) - 1
        1 <= b <= nb || continue
        ww = w[i] * w[j]
        s[b] += ww * sf(u[:, j] .- u[:, i], dx ./ r)
        c[b] += ww
    end
    return s, c
end

# Grid coordinates as a point list, in the cell order the packed field uses.
_grid_points(dims, spacing) = reshape([(I[d] - 1) * spacing[d] for d in 1:2, I in CartesianIndices(dims)], 2, :)

# Bin edges between the separations a point set can produce, so no shell sits on an edge.
function _separated_bins(x, n_bins)
    N = size(x, 2)
    d = sort!([LA.norm(x[:, j] .- x[:, i]) for i in 1:(N - 1) for j in (i + 1):N])
    dist = Float64[d[1]]
    for v in d
        v - dist[end] > 1e-9 && push!(dist, v)
    end
    picks = unique(round.(Int, range(1, length(dist) - 1; length = n_bins)))
    return [0.0; [(dist[k] + dist[k + 1]) / 2 for k in picks]]
end

_close(a, b) = isapprox(a, b; rtol = 1e-9, atol = 1e-10 * max(1.0, maximum(abs, b)))

# Bin averages agree where both are defined; an empty bin is NaN on both sides.
_same_average(a, b) = all(((x, y),) -> (isnan(x) && isnan(y)) || isapprox(x, y; rtol = 1e-9), zip(a, b))

# Random weights give the weighted pair loop on the lag sweep, on the transform (an odd operator) and at the point entry.
Test.@testset "random weights equal the weighted pair loop on grids and points" begin
    Random.seed!(9510)
    dims, spacing = (9, 6), (0.1, 0.2)
    N = prod(dims)
    data = randn(2, N)
    w = 0.5 .+ rand(N)
    x = _grid_points(dims, spacing)
    bins = _separated_bins(x, 8)
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
    for (sf, route) in ((L2, :sweep), (L3, :transform), (L2, :points))
        ref_s, ref_c = _flat_pair_loop(sf, x, data, bins, w)
        got_s, got_c = route === :points ?
            (r = SFC.calculate_structure_function(sf, x, data, bins, Float64, RAW; backend = SERIAL, weights = w);
             (r.sums, r.counts)) :
            _weighted_run(sf, data, sched, bins; weights = w, tag = route === :transform ? FFT_TAG : nothing)
        Test.@test sum(ref_c) > 0 && _close(got_s, ref_s) && isapprox(got_c, ref_c; rtol = 1e-11)
    end
end

# With masked cells and wrapping axes the weighted transform equals the weighted lag sweep, on the host and the device.
Test.@testset "the weighted transform equals the weighted sweep with masks and wrapping, on the device too" begin
    Random.seed!(9520)
    for (dims, spacing, periodic, sf, backend) in (((8, 8), (0.25, 0.25), (true, true), L2, SERIAL),
                                                   ((10, 7), (0.15, 0.15), (true, false), L3, DEVICE))
        N = prod(dims)
        uf = randn(2, N)
        for k in 1:N
            rand() < 0.3 && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(reshape(uf, 2, dims...))
        w = 0.5 .+ rand(N)
        sched = SFC.UniformLagSchedule(dims, spacing, periodic)
        bins = collect(range(0.0, 0.7 * sum(d -> spacing[d] * dims[d], 1:2); length = 9))
        ref_s, ref_c = _weighted_run(sf, uf, sched, bins; valid, weights = w)
        got_s, got_c = _weighted_run(sf, uf, sched, bins; valid, weights = w, tag = FFT_TAG, backend)
        Test.@test all(isfinite, ref_s) && _close(got_s, ref_s)
        Test.@test sum(ref_c) > 0 && isapprox(got_c, ref_c; rtol = 1e-11)
    end
end

# A multi-field's weighted sweep equals its transform and its point entry: a mixed, a scalar and a two-vector operator.
Test.@testset "multi-fields carry weights on every route" begin
    Random.seed!(9530)
    dims, spacing = (9, 7), (0.1, 0.15)
    N = prod(dims)
    u, θ, a = randn(2, dims...), randn(dims...), randn(2, dims...)
    x = _grid_points(dims, spacing)
    bins = _separated_bins(x, 8)
    w = 0.5 .+ rand(N)
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
    mixed, pair = Fields(vectors = (u,), scalars = (θ,)), Fields(vectors = (u, a))
    for (f, sf, route) in ((mixed, SFT.MixedSFType{1, 0, 2}(), :transform), (mixed, SFT.ScalarSFType{2}(), :points),
                           (pair, SFT.VectorDotSFType(1, 2), :points))
        ref_s, ref_c = _weighted_run(sf, f, sched, bins; weights = w)
        got_s, got_c = route === :transform ? _weighted_run(sf, f, sched, bins; weights = w, tag = FFT_TAG) :
            (r = SFC.calculate_structure_function(sf, x, f, bins, Float64, RAW; backend = SERIAL, weights = w);
             (r.sums, r.counts))
        Test.@test sum(ref_c) > 0 && _close(got_s, ref_s) && isapprox(got_c, ref_c; rtol = 1e-11)
    end
end

# On the sphere the weighted zonal sweep equals the transform, the device transform and the point entry.
Test.@testset "weights on the sphere: zonal sweep, transform, device and the point entry agree" begin
    Random.seed!(9540)
    n_lon, n_lat = 15, 7
    dlon = 2π / n_lon
    lats = sort(0.9 .* (π .* rand(n_lat) .- π / 2))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, true)
    N = n_lon * n_lat
    data = randn(2, N)
    w = 0.5 .+ rand(N)
    x = reshape([d == 1 ? (i - 1) * dlon : lats[j] for d in 1:2, i in 1:n_lon, j in 1:n_lat], 2, :)
    bins = collect(range(0.0, π; length = 8)) .+ 1e-3
    ref_s, ref_c = _weighted_run(L2, data, sched, bins; weights = w)
    runs = (_weighted_run(L2, data, sched, bins; weights = w, tag = FFT_TAG),
            _weighted_run(L2, data, sched, bins; weights = w, tag = FFT_TAG, backend = DEVICE),
            (r = SFC.calculate_structure_function(L2, x, data, bins, Float64, RAW; backend = SERIAL, weights = w,
                                                  distance_metric = SFH.SphericalDistance(1.0)); (r.sums, r.counts)))
    Test.@test sum(ref_c) > 0
    Test.@test all(((s, c),) -> _close(s, ref_s) && isapprox(c, ref_c; rtol = 1e-11), runs)
end

Test.@testset "cell_measure gives each cell its area, and as weights an area average on every route" begin
    # A lat-lon cell's area is ∝ cos φ, longitude running fastest; a uniform Cartesian cell's is hx·hy.
    Random.seed!(9550)
    geo = FG.Geometry.SphericalGeometry(1.0)
    n_lon, n_lat = 15, 8
    lam = range(0.0, step = 2π / n_lon, length = n_lon)
    phi = range(-π / 2 + π / (2n_lat), step = π / n_lat, length = n_lat)
    grid = FG.Grids.StructuredGrid(geo, lam, phi)
    w = SFC.cell_measure(grid)
    N = n_lon * n_lat
    ratio = w ./ vec([cos(φ) for _ in lam, φ in phi])
    Test.@test all(r -> isapprox(r, ratio[1]; rtol = 1e-12), ratio)
    u = randn(2, n_lon, n_lat)
    bins = collect(range(0.0, π; length = 8)) .+ 1e-3
    got = SFC.calculate_structure_function(L2, grid, u, bins, Float64, RAW; weights = w, backend = SERIAL)
    coords = FG.Grids.materialize(grid)
    x = Matrix(hcat(coords[1], coords[2])')
    ref = SFC.calculate_structure_function(L2, x, reshape(u, 2, N), bins, Float64, RAW; weights = w,
                                           backend = SERIAL, distance_metric = SFH.SphericalDistance(1.0))
    Test.@test _close(got.sums, ref.sums)
    Test.@test got.counts ≈ ref.counts rtol = 1e-11
    tr = SFC.calculate_structure_function(L2, grid, u, bins, FFT_TAG, Float64, RAW; weights = w, backend = SERIAL)
    Test.@test _close(tr.sums, got.sums)
    Test.@test tr.counts ≈ got.counts rtol = 1e-11

    cgrid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 7),
                                    range(0.0, step = 0.2, length = 5))
    cw = SFC.cell_measure(cgrid)
    Test.@test all(v -> isapprox(v, 0.1 * 0.2; rtol = 1e-12), cw)
    uc = randn(2, 7, 5)
    cbins = collect(range(0.0, 1.2; length = 7)) .+ 1e-3
    cgot = SFC.calculate_structure_function(L2, cgrid, uc, cbins, Float64, RAW; weights = cw, backend = SERIAL)
    cplain = SFC.calculate_structure_function(L2, cgrid, uc, cbins, RAW; backend = SERIAL)
    Test.@test _same_average(cgot.sums ./ cgot.counts, cplain.sums ./ cplain.counts)
    Test.@test cgot.counts ≈ cplain.counts .* cw[1]^2
end

Test.@testset "weights are refused where they cannot be honoured" begin
    Random.seed!(9560)
    dims, spacing = (6, 5), (0.1, 0.1)
    N = prod(dims)
    u = randn(2, dims...)
    data = reshape(u, 2, N)
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
    bins = collect(range(0.0, 0.5; length = 5))
    w = 0.5 .+ rand(N)
    s = zeros(4)
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(s, zeros(Int, 4), L2, data, sched, bins, Val(2), Val(1),
                                                            Val(0); weights = w)
    Test.@test_throws DimensionMismatch SFC.gridded_lag_sweep!(s, zeros(4), L2, data, sched, bins, Val(2), Val(1),
                                                                Val(0); weights = w[1:(end - 1)])
    bad = copy(w)
    bad[3] = NaN
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(s, zeros(4), L2, data, sched, bins, Val(2), Val(1), Val(0);
                                                            weights = bad)
    Test.@test_throws ArgumentError SFC.gridded_sweep!(s, zeros(Int, 4), L2, data, sched, bins, Val(2), Val(1), Val(0),
                                                        FFT_TAG; weights = w)
    x = _grid_points(dims, spacing)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, x, data, bins; weights = w, backend = SERIAL)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 6),
                                   range(0.0, step = 0.1, length = 5))
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, grid, u, bins; weights = SFC.cell_measure(grid))
end

Test.@testset "the value-binned joint histogram takes pair weights" begin
    # The weighted joint (distance × value) histogram equals a weighted pair loop; integer counts are refused.
    Random.seed!(2024)
    N, nb, nv = 60, 6, 5
    x = rand(2, N) .* 3
    u = randn(2, N)
    w = 0.25 .+ rand(N)
    dbins = collect(range(0.0, 3.0; length = nb + 1))
    vbins = collect(range(-4.0, 4.0; length = nv + 1))
    bs, bc = zeros(nb, nv), zeros(nb, nv)
    for i in 1:(N - 1), j in (i + 1):N
        dx = SA.SVector(x[1, j] - x[1, i], x[2, j] - x[2, i])
        r = sqrt(LA.dot(dx, dx))
        db = SFH.digitize(r, dbins)
        1 <= db <= nb || continue
        v = L2(SA.SVector(u[1, j] - u[1, i], u[2, j] - u[2, i]), dx ./ r)
        vb = SFH.digitize(v, vbins)
        1 <= vb <= nv || continue
        bs[db, vb] += w[i] * w[j] * v
        bc[db, vb] += w[i] * w[j]
    end
    same(a, b) = maximum(abs, a .- b) <= 1e-10 * max(maximum(abs, b), 1e-10)
    r = SFC.calculate_structure_function(L2, x, u, dbins, vbins, Float64; backend = SERIAL, weights = w)
    Test.@test sum(bc) > 0 && same(r.sums, bs) && same(r.counts, bc)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, x, u, dbins, vbins; weights = w,
                                                                     backend = SERIAL)
end
