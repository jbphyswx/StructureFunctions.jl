using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using Random: Random

const FFT_TAG = SB.FastFourierTransformSpectralBackend()
const SERIAL = CB.SerialBackend()
const THREADED = CB.ThreadedBackend()
const DEVICE = CB.GPUBackend(KA.CPU())
const RAW = SF.StructureFunctionSumsAndCounts

# One run on either gridded route from a packed field and a schedule, into Float64 counts.
function _weighted_run(sf, data, sched, bins, vD, vV, vK; valid = SFC.AllValid(), weights = nothing,
                       tag = nothing, backend = SERIAL)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(bins))
    s = zeros(Float64, nb)
    c = zeros(Float64, nb)
    if tag === nothing
        SFC.gridded_lag_sweep!(s, c, sf, data, sched, bins, vD, vV, vK; valid, weights, backend)
    else
        SFC.gridded_sweep!(s, c, sf, data, sched, bins, vD, vV, vK, tag; valid, weights, backend)
    end
    return s, c
end

function _weighted_run(sf, f::Fields, sched, bins; weights = nothing, tag = nothing, backend = SERIAL)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(bins))
    s = zeros(Float64, nb)
    c = zeros(Float64, nb)
    if tag === nothing
        SFC.gridded_lag_sweep!(s, c, sf, f, sched, bins; weights, backend)
    else
        SFC.gridded_sweep!(s, c, sf, f, sched, bins, tag; weights, backend)
    end
    return s, c
end

# The weighted pair statistic on flat points: Σ w_i w_j v_ij and Σ w_i w_j over the pairs of each (lo, hi] bin.
function _flat_pair_loop(sf, x, u, bins, w)
    nb = length(bins) - 1
    s = zeros(nb)
    c = zeros(nb)
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
function _grid_points(dims, spacing)
    Dg = length(dims)
    N = prod(dims)
    x = zeros(Float64, Dg, N)
    for (k, I) in enumerate(CartesianIndices(dims)), d in 1:Dg
        x[d, k] = (I[d] - 1) * spacing[d]
    end
    return x
end

# Bin edges between the separations a point set can produce, so no shell sits on an edge.
function _separated_bins(x, n_bins)
    N = size(x, 2)
    d = Float64[]
    for i in 1:(N - 1), j in (i + 1):N
        push!(d, LA.norm(x[:, j] .- x[:, i]))
    end
    sort!(d)
    dist = Float64[d[1]]
    for v in d
        v - dist[end] > 1e-9 && push!(dist, v)
    end
    picks = unique(round.(Int, range(1, length(dist) - 1; length = n_bins)))
    return [0.0; [(dist[k] + dist[k + 1]) / 2 for k in picks]]
end

# The sphere's cells as the unstructured entry wants them: (λ, φ) in radians, flattened as the zonal sweep indexes cells.
function _sphere_points(lats, n_lon, dlon)
    n_lat = length(lats)
    x = Matrix{Float64}(undef, 2, n_lon * n_lat)
    for j in 1:n_lat, i in 1:n_lon
        k = i + (j - 1) * n_lon
        x[1, k] = (i - 1) * dlon
        x[2, k] = lats[j]
    end
    return x
end

_close(a, b) = isapprox(a, b; rtol = 1e-9, atol = 1e-10 * max(1.0, maximum(abs, b)))

# Bin averages agree where both are defined; an empty bin is NaN on both sides.
_same_average(a, b) = all(((x, y),) -> (isnan(x) && isnan(y)) || isapprox(x, y; rtol = 1e-9), zip(a, b))

# Each operator on one route: the lag sweep, the transform, or the point entry on a serial or threaded backend.
const WEIGHT_ROUTES_2D = ((SFT.L2SFType(), :sweep), (SFT.T2SFType(), :transform), (SFT.L3SFType(), SERIAL),
                          (SFT.S3SFType(), THREADED), (SFT.T3SFType(), :sweep),
                          (SFT.ProjectedStructureFunctionType{2, 2}(), :transform))
const WEIGHT_ROUTES_3D = ((SFT.L2SFType(), :transform), (SFT.T3SFType(), :sweep), (SFT.S3SFType(), SERIAL))

# Sums and counts of `sf` on `route` over the grid `sched` of the points `x`, as a `RAW` result's are.
function _route_run(sf, route, x, data, sched, bins, ::Val{Dg}; weights = nothing) where {Dg}
    route === :sweep && return _weighted_run(sf, data, sched, bins, Val(Dg), Val(1), Val(0); weights)
    route === :transform && return _weighted_run(sf, data, sched, bins, Val(Dg), Val(1), Val(0); weights, tag = FFT_TAG)
    r = SFC.calculate_structure_function(sf, x, data, bins, Float64, RAW; backend = route, weights)
    return r.sums, r.counts
end

Test.@testset "random weights equal the weighted pair loop on grids and points" begin
    Random.seed!(9510)
    sums_ok, counts_ok = Bool[], Bool[]
    for (dims, spacing, routes) in (((9, 6), (0.1, 0.2), WEIGHT_ROUTES_2D),
                                    ((5, 4, 4), (0.2, 0.25, 0.3), WEIGHT_ROUTES_3D))
        Dg = length(dims)
        N = prod(dims)
        u = randn(Dg, dims...)
        data = reshape(u, Dg, N)
        w = 0.5 .+ rand(N)
        x = _grid_points(dims, spacing)
        bins = _separated_bins(x, 8)
        sched = SFC.UniformLagSchedule(dims, spacing, ntuple(_ -> false, Dg))
        for (sf, route) in routes
            ref_s, ref_c = _flat_pair_loop(sf, x, data, bins, w)
            got_s, got_c = _route_run(sf, route, x, data, sched, bins, Val(Dg); weights = w)
            push!(sums_ok, _close(got_s, ref_s))
            push!(counts_ok, sum(ref_c) > 0 && isapprox(got_c, ref_c; rtol = 1e-11))
        end
    end
    Test.@test all(sums_ok)
    Test.@test all(counts_ok)
end

const WEIGHT_MASKED_TRANSFORM_CASES = (
    ((8, 8), (0.25, 0.25), (true, true), SFT.L2SFType(), SERIAL),
    ((10, 7), (0.15, 0.15), (true, false), SFT.L3SFType(), DEVICE),
    ((6, 6, 4), (0.2, 0.2, 0.3), (true, false, true), SFT.S3SFType(), DEVICE),
    ((8, 8), (0.25, 0.25), (true, true), SFT.T3SFType(), SERIAL),
    ((10, 7), (0.15, 0.15), (true, false), SFT.ProjectedStructureFunctionType{4, 0}(), SERIAL),
)

Test.@testset "the weighted transform equals the weighted sweep with masks and wrapping, on the device too" begin
    Random.seed!(9520)
    sums_ok, counts_ok = Bool[], Bool[]
    for (dims, spacing, periodic, sf, backend) in WEIGHT_MASKED_TRANSFORM_CASES
        Dg = length(dims)
        N = prod(dims)
        u = randn(Dg, dims...)
        uf = reshape(u, Dg, N)
        for k in 1:N
            rand() < 0.3 && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(u)
        w = 0.5 .+ rand(N)
        sched = SFC.UniformLagSchedule(dims, spacing, periodic)
        bins = collect(range(0.0, 0.7 * sum(d -> spacing[d] * dims[d], 1:Dg); length = 9))
        ref_s, ref_c = _weighted_run(sf, uf, sched, bins, Val(Dg), Val(1), Val(0); valid, weights = w)
        got_s, got_c = _weighted_run(sf, uf, sched, bins, Val(Dg), Val(1), Val(0);
                                     valid, weights = w, tag = FFT_TAG, backend)
        push!(sums_ok, all(isfinite, ref_s) && _close(got_s, ref_s))
        push!(counts_ok, sum(ref_c) > 0 && isapprox(got_c, ref_c; rtol = 1e-11))
    end
    Test.@test all(sums_ok)
    Test.@test all(counts_ok)
end

const WEIGHT_FIELD_ROUTES = ((:mixed, SFT.MixedSFType{1, 0, 2}(), :transform), (:mixed, SFT.ScalarSFType{2}(), SERIAL),
                             (:mixed, SFT.MixedSFType{1, 0, 1}(), THREADED), (:mixed, SFT.L2SFType(), :transform),
                             (:pair, SFT.VectorDotSFType(1, 2), SERIAL), (:pair, SFT.L2SFType(), THREADED))

Test.@testset "multi-fields carry weights on every route" begin
    Random.seed!(9530)
    dims, spacing = (9, 7), (0.1, 0.15)
    N = prod(dims)
    u = randn(2, dims...)
    θ = randn(dims...)
    a = randn(2, dims...)
    x = _grid_points(dims, spacing)
    bins = _separated_bins(x, 8)
    w = 0.5 .+ rand(N)
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
    fields = (mixed = Fields(vectors = (u,), scalars = (θ,)), pair = Fields(vectors = (u, a)))
    agree = Bool[]
    for (key, sf, route) in WEIGHT_FIELD_ROUTES
        f = fields[key]
        ref_s, ref_c = _weighted_run(sf, f, sched, bins; weights = w)
        got_s, got_c = if route === :transform
            _weighted_run(sf, f, sched, bins; weights = w, tag = FFT_TAG)
        else
            got = SFC.calculate_structure_function(sf, x, f, bins, Float64, RAW; backend = route, weights = w)
            got.sums, got.counts
        end
        push!(agree, sum(ref_c) > 0 && _close(got_s, ref_s) && isapprox(got_c, ref_c; rtol = 1e-11))
    end
    Test.@test all(agree)
end

const WEIGHT_SPHERE_ROUTES = ((SFT.L2SFType(), :transform, SERIAL), (SFT.L2SFType(), :points, SERIAL),
                              (SFT.T2SFType(), :transform, DEVICE), (SFT.L3SFType(), :sweep, THREADED),
                              (SFT.S3SFType(), :points, THREADED))

Test.@testset "weights on the sphere: zonal sweep, transform, device and the point entry agree" begin
    Random.seed!(9540)
    n_lon, n_lat = 15, 7
    dlon = 2π / n_lon
    lats = sort(0.9 .* (π .* rand(n_lat) .- π / 2))
    sched = SFC.ZonalLagSchedule(lats, n_lon, dlon, 1.0, true)
    N = n_lon * n_lat
    u = randn(2, n_lon, n_lat)
    data = reshape(u, 2, N)
    w = 0.5 .+ rand(N)
    x = _sphere_points(lats, n_lon, dlon)
    bins = collect(range(0.0, π; length = 8)) .+ 1e-3
    agree = Bool[]
    for (sf, route, backend) in WEIGHT_SPHERE_ROUTES
        ref_s, ref_c = _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0); weights = w)
        got_s, got_c = if route === :points
            pts = SFC.calculate_structure_function(sf, x, data, bins, Float64, RAW; backend, weights = w,
                                                   distance_metric = SFH.SphericalDistance(1.0))
            pts.sums, pts.counts
        else
            _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0); weights = w,
                          tag = route === :transform ? FFT_TAG : nothing, backend)
        end
        push!(agree, sum(ref_c) > 0 && _close(got_s, ref_s) && isapprox(got_c, ref_c; rtol = 1e-11))
    end
    Test.@test all(agree)
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
    got = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, Float64, RAW; weights = w, backend = SERIAL)
    coords = FG.Grids.materialize(grid)
    x = Matrix(hcat(coords[1], coords[2])')
    ref = SFC.calculate_structure_function(SFT.L2SFType(), x, reshape(u, 2, N), bins, Float64, RAW; weights = w,
                                           backend = SERIAL, distance_metric = SFH.SphericalDistance(1.0))
    Test.@test _close(got.sums, ref.sums)
    Test.@test got.counts ≈ ref.counts rtol = 1e-11
    tr = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, FFT_TAG, Float64, RAW; weights = w,
                                          backend = SERIAL)
    Test.@test _close(tr.sums, got.sums)
    Test.@test tr.counts ≈ got.counts rtol = 1e-11

    cgrid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 7),
                                    range(0.0, step = 0.2, length = 5))
    cw = SFC.cell_measure(cgrid)
    Test.@test all(v -> isapprox(v, 0.1 * 0.2; rtol = 1e-12), cw)
    uc = randn(2, 7, 5)
    cbins = collect(range(0.0, 1.2; length = 7)) .+ 1e-3
    cgot = SFC.calculate_structure_function(SFT.L2SFType(), cgrid, uc, cbins, Float64, RAW; weights = cw,
                                            backend = SERIAL)
    cplain = SFC.calculate_structure_function(SFT.L2SFType(), cgrid, uc, cbins, RAW; backend = SERIAL)
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
    L2 = SFT.L2SFType()
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

Test.@testset "the device and distributed point paths carry weights" begin
    # Edges shifted clear of every lattice separation, where the backends' distance arithmetic must agree.
    Random.seed!(9570)
    dims, spacing = (6, 5), (0.1, 0.1)
    N = prod(dims)
    data = randn(2, N)
    w = 0.5 .+ rand(N)
    x = _grid_points(dims, spacing)
    bins = collect(range(0.0, 0.5; length = 5)) .+ 0.013
    ref = SFC.calculate_structure_function(SFT.L2SFType(), x, data, bins, Float64, RAW; weights = w, backend = SERIAL)
    agree = Bool[]
    for backend in (DEVICE, CB.DistributedBackend())
        got = SFC.calculate_structure_function(SFT.L2SFType(), x, data, bins, Float64, RAW; weights = w, backend)
        push!(agree, isapprox(collect(got.sums), collect(ref.sums); rtol = 1e-10) &&
                     isapprox(collect(got.counts), collect(ref.counts); rtol = 1e-10))
    end
    Test.@test sum(ref.counts) > 0 && all(agree)
end

Test.@testset "the value-binned joint histogram takes pair weights" begin
    # Serial and threaded weighted joint (distance × value) histograms equal a weighted pair loop; integer counts are refused.
    Random.seed!(2024)
    N, nb, nv = 60, 6, 5
    x = rand(2, N) .* 3
    u = randn(2, N)
    w = 0.25 .+ rand(N)
    dbins = collect(range(0.0, 3.0; length = nb + 1))
    vbins = collect(range(-4.0, 4.0; length = nv + 1))
    op = SFT.L2SFType()
    bs = zeros(Float64, nb, nv)
    bc = zeros(Float64, nb, nv)
    for i in 1:(N - 1), j in (i + 1):N
        dx = SA.SVector(x[1, j] - x[1, i], x[2, j] - x[2, i])
        r = sqrt(LA.dot(dx, dx))
        db = SFH.digitize(r, dbins)
        1 <= db <= nb || continue
        du = SA.SVector(u[1, j] - u[1, i], u[2, j] - u[2, i])
        v = op(du, dx ./ r)
        vb = SFH.digitize(v, vbins)
        1 <= vb <= nv || continue
        bs[db, vb] += w[i] * w[j] * v
        bc[db, vb] += w[i] * w[j]
    end
    same(a, b) = maximum(abs, a .- b) <= 1e-10 * max(maximum(abs, b), 1e-10)
    agree = Bool[]
    for be in (CB.SerialBackend(), CB.ThreadedBackend())
        r = SFC.calculate_structure_function(op, x, u, dbins, vbins, Float64; backend = be, weights = w)
        push!(agree, same(r.sums, bs) && same(r.counts, bc))
    end
    Test.@test sum(bc) > 0 && all(agree)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        op, x, u, dbins, vbins; weights = w)
end
