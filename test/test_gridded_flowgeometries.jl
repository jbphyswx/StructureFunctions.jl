using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using Random: Random
using StaticArrays: StaticArrays as SA
using StructureFunctions: HelperFunctions as SFH

const SF1D = SFT.L2SFType()

# Bin edges placed between the separations the grid can produce. Pairs sharing a lag share one exact
# separation, so an edge through it splits a whole shell by rounding: the sweep bins them by the
# lag's separation and a pair loop by each pair's own coordinate difference, which differ by an ulp.
function _separated_bins(nx, ny, hx, hy, n_bins)
    seps = sort!(unique!([sqrt((i * hx)^2 + (j * hy)^2)
                          for i in 0:(nx - 1) for j in 0:(ny - 1) if !(i == 0 && j == 0)]))
    idx = unique(round.(Int, range(1, length(seps) - 1; length = n_bins)))
    return [0.0; [(seps[i] + seps[i + 1]) / 2 for i in idx]]
end

# The grid's own cell coordinates, in the order `reshape(u, D, N)` flattens them.
function _points(grid)
    c = FG.Grids.coordinates(grid)
    dims = ntuple(d -> length(c[d]), Val(length(c)))
    Dg = length(dims)
    x = Matrix{Float64}(undef, Dg, prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims)), d in 1:Dg
        x[d, k] = c[d][I[d]]
    end
    return x
end

Test.@testset "a uniform Cartesian grid gives the unstructured answer" begin
    # The adapter's whole job is to hand the sweep the right dims, spacing and topology. A bounded
    # grid has plain Euclidean separations, so the existing pair loop over the grid's own points is
    # the oracle.
    geo = FG.Geometry.CartesianGeometry()
    for (nx, ny, hx, hy) in ((9, 6, 0.1, 0.2), (7, 7, 0.25, 0.25), (5, 8, 0.3, 0.15))
        grid = FG.Grids.StructuredGrid(geo, range(0.0, step = hx, length = nx),
                                       range(0.0, step = hy, length = ny))
        Random.seed!(5100 + nx * ny)
        u = randn(2, nx, ny)
        bins = _separated_bins(nx, ny, hx, hy, 8)
        nb = length(bins) - 1

        got = SFC.calculate_structure_function(
            SF1D, grid, u, bins; output_type = SF.StructureFunctionSumsAndCounts,
            verbose = false, show_progress = false,
        )
        ref_s = zeros(Float64, nb)
        ref_c = zeros(UInt32, nb)
        SF.calculate_structure_function!(ref_s, ref_c, SF1D, _points(grid),
                                         reshape(u, 2, nx * ny), bins)
        Test.@test got.counts == ref_c
        Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
        Test.@test sum(got.counts) > 0
    end
end

Test.@testset "the adapter reads the grid's topology" begin
    # A periodic direction must reach the sweep as periodic; the count is the tell, since wrapping
    # changes which separations exist but never how many pairs there are.
    geo = FG.Geometry.CartesianGeometry()
    nx, ny = 8, 6
    ax, ay = range(0.0, step = 0.25, length = nx), range(0.0, step = 0.25, length = ny)
    Random.seed!(5200)
    u = randn(2, nx, ny)
    # spans the bounded grid's 2.15 diagonal, and fine enough that wrapping visibly moves pairs
    bins = collect(range(0.0, 2.4; length = 13))

    bounded = FG.Grids.StructuredGrid(geo, ax, ay)
    wrapped = FG.Grids.StructuredGrid(geo, ax, ay; topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
    Test.@test FG.Grids.periodic_flags(bounded) == (false, false)
    Test.@test FG.Grids.periodic_flags(wrapped) == (true, false)

    rb = SFC.calculate_structure_function(SF1D, bounded, u, bins;
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    rw = SFC.calculate_structure_function(SF1D, wrapped, u, bins;
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    N = nx * ny
    Test.@test sum(rb.counts) == N * (N - 1) ÷ 2
    Test.@test sum(rw.counts) == N * (N - 1) ÷ 2
    # wrapping shortens the long separations, so the pairs move to nearer bins
    Test.@test rb.counts != rw.counts
end

Test.@testset "the adapter refuses only what is genuinely ill-posed" begin
    geo = FG.Geometry.CartesianGeometry()
    bins = collect(range(0.0, 1.0; length = 5))

    # A sphere's (λ, φ) lag is not a constant separation, so it does not take the lag schedule — it
    # takes the zonal one, where the shared quantity is a circle of longitude rather than a lag.
    # Exercised in depth in test_gridded_zonal.jl; here only that it is routed, not refused.
    sph = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0),
                                  range(0.0, step = 0.1, length = 6),
                                  range(-0.3, step = 0.1, length = 5))
    r_sph = SFC.calculate_structure_function(
        SF1D, sph, randn(2, 6, 5), collect(range(0.0, 3.2; length = 5)), UInt32;
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test sum(r_sph.counts) == 30 * 29 ÷ 2

    uniform = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.2, length = 4),
                                      range(0.0, step = 0.2, length = 4))
    Test.@test_throws DimensionMismatch SFC.calculate_structure_function(
        SF1D, uniform, randn(2, 4), bins; verbose = false, show_progress = false)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        SF1D, uniform, randn(2, 4, 4), bins; nonsense = 1, verbose = false, show_progress = false)
end

Test.@testset "grids with no structure to exploit are enumerated, not refused" begin
    # A stretched axis, a curvilinear mesh and a pixelized sphere share no separation between pairs.
    # Each is still a valid question, so the adapter routes it to the pair loop rather than erroring.
    Random.seed!(5300)

    # a stretched Cartesian axis: the answer must equal the unstructured path on the same points
    geo = FG.Geometry.CartesianGeometry()
    xs = [0.0, 0.1, 0.35, 0.8, 0.85]
    ys = range(0.0, step = 0.2, length = 4)
    stretched = FG.Grids.StructuredGrid(geo, xs, ys)
    u = randn(2, length(xs), length(ys))
    bins = collect(range(0.0, 1.5; length = 7))
    got = SFC.calculate_structure_function(
        SF1D, stretched, u, bins, UInt32; output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    N = length(xs) * length(ys)
    x = Matrix{Float64}(undef, 2, N)
    for (k, I) in enumerate(CartesianIndices((length(xs), length(ys))))
        x[1, k] = xs[I[1]]
        x[2, k] = ys[I[2]]
    end
    ref_s = zeros(6); ref_c = zeros(UInt32, 6)
    SF.calculate_structure_function!(ref_s, ref_c, SF1D, x, reshape(u, 2, N), bins)
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
    Test.@test sum(got.counts) > 0

    # the schedule chosen really is the structureless one, not a lag sweep
    sched = Base.invokelatest(
        getfield(Base.get_extension(SF, :StructureFunctionsFlowGeometriesExt), :_lag_schedule),
        stretched)
    Test.@test sched isa SFC.ScatteredPairs
    Test.@test SFC.n_scattered_cells(sched) == N

    # while a uniform grid still gets the lag sweep
    uniform = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.2, length = 5), ys)
    sched_u = Base.invokelatest(
        getfield(Base.get_extension(SF, :StructureFunctionsFlowGeometriesExt), :_lag_schedule),
        uniform)
    Test.@test sched_u isa SFC.UniformLagSchedule
end

Test.@testset "a pixelized sphere is enumerated too" begin
    # HEALPix has no axes at all — its coordinates are closed-form in the pixel id — so nothing is
    # shared between pairs and the adapter must fall through to the pair loop.
    grid = FG.Grids.HEALPixGrid(FG.Geometry.SphericalGeometry(1.0), 2)
    n = length(FG.Grids.mask(grid))
    Random.seed!(5400)
    u = randn(2, n)
    bins = collect(range(0.0, 3.2; length = 6))
    got = SFC.calculate_structure_function(
        SF1D, grid, u, bins, UInt32; output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    # Every pair except those the geometry itself refuses: at exactly antipodal separation there is
    # no unique geodesic, and `pair_frame` reports the pair as degenerate rather than inventing one.
    coords0 = FG.Grids.materialize(grid)
    gg = SFH.SphericalGeometry{2}(SFC.DI.SphericalAngle(), 1.0)
    ambient(l, p) = SA.SVector(cos(p) * cos(l), cos(p) * sin(l), sin(p))
    degenerate = 0
    for i in 1:(n - 1), j in (i + 1):n
        ok, _, _ = SFH.pair_frame(gg, ambient(coords0[1][i], coords0[2][i]),
                                      ambient(coords0[1][j], coords0[2][j]))
        ok || (degenerate += 1)
    end
    Test.@test sum(got.counts) == n * (n - 1) ÷ 2 - degenerate
    Test.@test all(isfinite, got.sums)

    # the same points through the unstructured entry give the same answer
    coords = FG.Grids.materialize(grid)
    x = permutedims(hcat(coords...))
    ref_s = zeros(5); ref_c = zeros(UInt32, 5)
    SF.calculate_structure_function!(ref_s, ref_c, SF1D, x, u, bins;
                                     distance_metric = SFC.DI.SphericalAngle())
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
end
