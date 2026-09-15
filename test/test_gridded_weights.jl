using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, Fields, HarmonicNodes
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
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

# The weighted pair statistic written out on flat points: Σ w_i w_j v_ij and Σ w_i w_j over the pairs
# each (lo, hi] bin holds.
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

# The sphere's cells as the unstructured entry wants them: (λ, φ) in radians, flattened as the zonal
# sweep indexes cells.
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

const WEIGHT_OPS = (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.S3SFType(), SFT.T3SFType(),
                    SFT.ProjectedStructureFunctionType{2, 2}())

Test.@testset "weights of one reproduce the unweighted results" begin
    Random.seed!(9500)
    dims, spacing = (9, 6), (0.1, 0.2)
    N = prod(dims)
    u = randn(2, dims...)
    data = reshape(u, 2, N)
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
    x = _grid_points(dims, spacing)
    bins = _separated_bins(x, 8)
    ones_w = ones(N)
    for sf in WEIGHT_OPS
        ref_s, ref_c = _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0))
        got_s, got_c = _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0); weights = ones_w)
        Test.@test got_s == ref_s
        Test.@test got_c == ref_c
        tr_s, tr_c = _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0); tag = FFT_TAG)
        tw_s, tw_c = _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0); tag = FFT_TAG, weights = ones_w)
        Test.@test _close(tw_s, tr_s)
        Test.@test tw_c ≈ tr_c rtol = 1e-12
        for backend in (SERIAL, THREADED)
            ref = SFC.calculate_structure_function(sf, x, data, bins, Float64; backend, verbose = false,
                                                   show_progress = false, output_type = RAW)
            got = SFC.calculate_structure_function(sf, x, data, bins, Float64; backend, weights = ones_w,
                                                   verbose = false, show_progress = false, output_type = RAW)
            Test.@test got.sums == ref.sums
            Test.@test got.counts == ref.counts
        end
    end
    θ = randn(dims...)
    f = Fields(vectors = (u,), scalars = (θ,))
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{2}(), SFT.VectorDotSFType(1, 1)),
        backend in (SERIAL, THREADED)

        ref = SFC.calculate_structure_function(sf, x, f, bins, Float64; backend, verbose = false,
                                               show_progress = false, output_type = RAW)
        got = SFC.calculate_structure_function(sf, x, f, bins, Float64; backend, weights = ones_w,
                                               verbose = false, show_progress = false, output_type = RAW)
        Test.@test got.sums == ref.sums
        Test.@test got.counts == ref.counts
    end
end

Test.@testset "random weights equal the weighted pair loop on grids and points" begin
    Random.seed!(9510)
    for (dims, spacing, ops) in (((9, 6), (0.1, 0.2), WEIGHT_OPS),
                                 ((5, 4, 4), (0.2, 0.25, 0.3), (SFT.L2SFType(), SFT.T3SFType(), SFT.S3SFType())))
        Dg = length(dims)
        N = prod(dims)
        u = randn(Dg, dims...)
        data = reshape(u, Dg, N)
        w = 0.5 .+ rand(N)
        x = _grid_points(dims, spacing)
        bins = _separated_bins(x, 8)
        sched = SFC.UniformLagSchedule(dims, spacing, ntuple(_ -> false, Dg))
        for sf in ops
            ref_s, ref_c = _flat_pair_loop(sf, x, data, bins, w)
            Test.@test sum(ref_c) > 0
            for tag in (nothing, FFT_TAG)
                got_s, got_c = _weighted_run(sf, data, sched, bins, Val(Dg), Val(1), Val(0); weights = w, tag)
                Test.@test _close(got_s, ref_s)
                Test.@test got_c ≈ ref_c rtol = 1e-11
            end
            for backend in (SERIAL, THREADED)
                got = SFC.calculate_structure_function(sf, x, data, bins, Float64; backend, weights = w,
                                                       verbose = false, show_progress = false, output_type = RAW)
                Test.@test _close(got.sums, ref_s)
                Test.@test got.counts ≈ ref_c rtol = 1e-11
            end
        end
    end
end

Test.@testset "the weighted transform equals the weighted sweep with masks and wrapping, on the device too" begin
    Random.seed!(9520)
    for (dims, spacing, periodic) in (((8, 8), (0.25, 0.25), (true, true)),
                                      ((10, 7), (0.15, 0.15), (true, false)),
                                      ((6, 6, 4), (0.2, 0.2, 0.3), (true, false, true)))
        Dg = length(dims)
        N = prod(dims)
        u = randn(Dg, dims...)
        uf = reshape(u, Dg, N)
        for k in 1:N
            rand() < 0.3 && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(u)
        Test.@test !(valid isa SFC.AllValid)
        w = 0.5 .+ rand(N)
        sched = SFC.UniformLagSchedule(dims, spacing, periodic)
        bins = collect(range(0.0, 0.7 * sum(d -> spacing[d] * dims[d], 1:Dg); length = 9))
        for sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.S3SFType(), SFT.T3SFType(),
                   SFT.ProjectedStructureFunctionType{4, 0}())
            ref_s, ref_c = _weighted_run(sf, uf, sched, bins, Val(Dg), Val(1), Val(0); valid, weights = w)
            Test.@test all(isfinite, ref_s)
            Test.@test sum(ref_c) > 0
            for backend in (SERIAL, DEVICE)
                got_s, got_c = _weighted_run(sf, uf, sched, bins, Val(Dg), Val(1), Val(0);
                                             valid, weights = w, tag = FFT_TAG, backend)
                Test.@test _close(got_s, ref_s)
                Test.@test got_c ≈ ref_c rtol = 1e-11
            end
        end
    end
end

Test.@testset "channel bundles carry weights on every route" begin
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
    cases = (
        (Fields(vectors = (u,), scalars = (θ,)),
         (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{2}(), SFT.MixedSFType{1, 0, 1}(), SFT.L2SFType())),
        (Fields(vectors = (u, a)), (SFT.VectorDotSFType(1, 2), SFT.L2SFType())),
    )
    for (f, ops) in cases, sf in ops
        ref_s, ref_c = _weighted_run(sf, f, sched, bins; weights = w)
        Test.@test sum(ref_c) > 0
        tr_s, tr_c = _weighted_run(sf, f, sched, bins; weights = w, tag = FFT_TAG)
        Test.@test _close(tr_s, ref_s)
        Test.@test tr_c ≈ ref_c rtol = 1e-11
        for backend in (SERIAL, THREADED)
            got = SFC.calculate_structure_function(sf, x, f, bins, Float64; backend, weights = w,
                                                   verbose = false, show_progress = false, output_type = RAW)
            Test.@test _close(got.sums, ref_s)
            Test.@test got.counts ≈ ref_c rtol = 1e-11
        end
    end
end

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
    for sf in (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.S3SFType())
        ref_s, ref_c = _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0); weights = w)
        Test.@test sum(ref_c) > 0
        for (tag, backend) in ((FFT_TAG, SERIAL), (FFT_TAG, DEVICE), (nothing, THREADED))
            got_s, got_c = _weighted_run(sf, data, sched, bins, Val(2), Val(1), Val(0); weights = w, tag, backend)
            Test.@test _close(got_s, ref_s)
            Test.@test got_c ≈ ref_c rtol = 1e-11
        end
        for backend in (SERIAL, THREADED)
            pts = SFC.calculate_structure_function(sf, x, data, bins, Float64; backend, weights = w,
                                                   distance_metric = SF.SphericalDistance(1.0), verbose = false,
                                                   show_progress = false, output_type = RAW)
            Test.@test _close(pts.sums, ref_s)
            Test.@test pts.counts ≈ ref_c rtol = 1e-11
        end
    end
end

Test.@testset "cell_measure feeds a grid's cell areas as weights" begin
    Random.seed!(9550)
    geo = FG.Geometry.SphericalGeometry(1.0)
    n_lon, n_lat = 15, 8
    lam = range(0.0, step = 2π / n_lon, length = n_lon)
    phi = range(-π / 2 + π / (2n_lat), step = π / n_lat, length = n_lat)
    grid = FG.Grids.StructuredGrid(geo, lam, phi)
    w = SF.cell_measure(grid)
    N = n_lon * n_lat
    Test.@test length(w) == N
    Test.@test all(>(0), w)
    Test.@test w[1] < w[n_lon * (n_lat ÷ 2) + 1]
    u = randn(2, n_lon, n_lat)
    bins = collect(range(0.0, π; length = 8)) .+ 1e-3
    got = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, Float64; weights = w, backend = SERIAL,
                                           verbose = false, show_progress = false, output_type = RAW)
    coords = FG.Grids.materialize(grid)
    x = Matrix(hcat(coords[1], coords[2])')
    ref = SFC.calculate_structure_function(SFT.L2SFType(), x, reshape(u, 2, N), bins, Float64; weights = w,
                                           backend = SERIAL, distance_metric = SF.SphericalDistance(1.0),
                                           verbose = false, show_progress = false, output_type = RAW)
    Test.@test _close(got.sums, ref.sums)
    Test.@test got.counts ≈ ref.counts rtol = 1e-11
    tr = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins, Float64, FFT_TAG; weights = w,
                                          backend = SERIAL, verbose = false, show_progress = false, output_type = RAW)
    Test.@test _close(tr.sums, got.sums)
    Test.@test tr.counts ≈ got.counts rtol = 1e-11
    plain = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, bins; backend = SERIAL, verbose = false,
                                             show_progress = false, output_type = RAW)
    Test.@test !(got.sums ./ got.counts ≈ plain.sums ./ plain.counts)

    # a Cartesian grid's cells are all alike, so its measure changes no average
    cgrid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 7),
                                    range(0.0, step = 0.2, length = 5))
    cw = SF.cell_measure(cgrid)
    Test.@test all(v -> v ≈ cw[1], cw)
    uc = randn(2, 7, 5)
    cbins = collect(range(0.0, 1.2; length = 7)) .+ 1e-3
    cgot = SFC.calculate_structure_function(SFT.L2SFType(), cgrid, uc, cbins, Float64; weights = cw, backend = SERIAL,
                                            verbose = false, show_progress = false, output_type = RAW)
    cplain = SFC.calculate_structure_function(SFT.L2SFType(), cgrid, uc, cbins; backend = SERIAL, verbose = false,
                                              show_progress = false, output_type = RAW)
    Test.@test _same_average(cgot.sums ./ cgot.counts, cplain.sums ./ cplain.counts)
    Test.@test cgot.counts ≈ cplain.counts .* cw[1]^2
end

Test.@testset "area-weighted hard bins agree with the harmonic route's area average" begin
    geo = FG.Geometry.SphericalGeometry(1.0)
    n_lon, n_lat = 96, 48
    lam = range(0.0, step = 2π / n_lon, length = n_lon)
    phi = range(-π / 2 + π / (2n_lat), step = π / n_lat, length = n_lat)
    grid = FG.Grids.StructuredGrid(geo, lam, phi)
    # the gradient of Φ = cos²φ cos 2λ, a degree-2 harmonic: u_E = ∂_λΦ / cos φ, u_N = ∂_φΦ
    u = Array{Float64}(undef, 2, n_lon, n_lat)
    for (j, φ) in enumerate(phi), (i, λ) in enumerate(lam)
        u[1, i, j] = -2 * cos(φ) * sin(2λ)
        u[2, i, j] = -2 * cos(φ) * sin(φ) * cos(2λ)
    end
    w = SF.cell_measure(grid)
    nodes = HarmonicNodes(24, 48; taper = SF.GaussianTaper(π / 48))
    harm = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, nodes, SB.DirectSumSpectralBackend();
                                            verbose = false, output_type = RAW)
    β = nodes.separations
    half = 0.04
    edges = sort(vcat(0.0, β .- half, β .+ half))
    zon = SFC.calculate_structure_function(SFT.L2SFType(), grid, u, edges, Float64; weights = w, backend = SERIAL,
                                           verbose = false, show_progress = false, output_type = RAW)
    hard = zon.sums ./ zon.counts
    soft = harm.sums ./ harm.counts
    compared = 0
    for (k, b) in enumerate(β)
        0.7 <= b <= 2.4 || continue
        Test.@test isapprox(hard[2k], soft[k]; rtol = 4e-2)
        compared += 1
    end
    Test.@test compared >= 8
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
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, x, data, bins; weights = w, backend = SERIAL,
                                                                     verbose = false, show_progress = false)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, x, data, bins, Float64; weights = w,
                                                                     backend = DEVICE, verbose = false,
                                                                     show_progress = false)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, x, data, bins, Float64; weights = w,
                                                                     backend = CB.DistributedBackend(),
                                                                     verbose = false, show_progress = false)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, x, data, bins,
                                                                     collect(range(-3.0, 3.0; length = 5)), Float64;
                                                                     weights = w, backend = SERIAL, verbose = false,
                                                                     show_progress = false)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 6),
                                   range(0.0, step = 0.1, length = 5))
    Test.@test_throws ArgumentError SFC.calculate_structure_function(L2, grid, u, bins; weights = SF.cell_measure(grid),
                                                                     verbose = false, show_progress = false)
end
