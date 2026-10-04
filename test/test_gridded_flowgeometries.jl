using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using FlowGeometries: FlowGeometries as FG
using SpectralBackends: SpectralBackends as SB
using FFTW: FFTW
using Random: Random

const SF1D = SFT.L2SFType()

# Cell coordinates of the grid with these axes, in the order `reshape(u, D, :)` flattens them.
function _axes_points(axes)
    dims = map(length, axes)
    x = Matrix{Float64}(undef, length(axes), prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims)), d in eachindex(axes)
        x[d, k] = axes[d][I[d]]
    end
    return x
end

# Bin edges midway between the distinct separations of the points `x`, separations within 1e-9 merged.
function _separated_bins_points(x, n_bins)
    N = size(x, 2)
    seps = sort!([sqrt(sum(abs2, x[:, j] .- x[:, i])) for i in 1:(N - 1) for j in (i + 1):N])
    distinct = [seps[1]]
    for s in seps
        s - distinct[end] > 1e-9 && push!(distinct, s)
    end
    idx = unique(round.(Int, range(1, length(distinct) - 1; length = n_bins)))
    return [0.0; [(distinct[i] + distinct[i + 1]) / 2 for i in idx]]
end

# Pairs per (lo, hi] bin of a uniform grid, separations taken by the minimum image along periodic directions.
function _minimum_image_counts(dims, spacing, periodic, bins)
    c = zeros(Int, length(bins) - 1)
    ci = CartesianIndices(dims)
    for k1 in 1:(length(ci) - 1), k2 in (k1 + 1):length(ci)
        r2 = sum(eachindex(dims)) do d
            m = ci[k2][d] - ci[k1][d]
            if periodic[d]
                m = mod(m, dims[d])
                m > dims[d] ÷ 2 && (m -= dims[d])
            end
            (m * spacing[d])^2
        end
        b = searchsortedfirst(bins, sqrt(r2)) - 1
        1 <= b <= length(c) && (c[b] += 1)
    end
    return c
end

Test.@testset "a uniform Cartesian grid gives the unstructured answer" begin
    # Unequal counts and spacings per axis, so a transposed or misread axis changes the answer.
    geo = FG.Geometry.CartesianGeometry()
    nx, ny, hx, hy = 9, 6, 0.1, 0.2
    ax, ay = range(0.0, step = hx, length = nx), range(0.0, step = hy, length = ny)
    grid = FG.Grids.StructuredGrid(geo, ax, ay)
    Random.seed!(5100 + nx * ny)
    u = randn(2, nx, ny)
    x = _axes_points((ax, ay))
    bins = _separated_bins_points(x, 8)
    nb = length(bins) - 1
    got = SFC.calculate_structure_function(SF1D, grid, u, bins, SF.StructureFunctionSumsAndCounts)
    ref_s = zeros(Float64, nb)
    ref_c = zeros(UInt32, nb)
    SFC.calculate_structure_function!(ref_s, ref_c, SF1D, x, reshape(u, 2, nx * ny), bins)
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
end

Test.@testset "a periodic topology wraps its direction" begin
    # Bounded and x-periodic grids each bin their pairs as a minimum-image pair count does.
    geo = FG.Geometry.CartesianGeometry()
    nx, ny = 8, 6
    ax, ay = range(0.0, step = 0.25, length = nx), range(0.0, step = 0.25, length = ny)
    Random.seed!(5200)
    u = randn(2, nx, ny)
    bins = collect(range(0.0, 2.4; length = 13)) .+ 0.0137
    bounded = FG.Grids.StructuredGrid(geo, ax, ay)
    wrapped = FG.Grids.StructuredGrid(geo, ax, ay; topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
    rb = SFC.calculate_structure_function(SF1D, bounded, u, bins, SF.StructureFunctionSumsAndCounts)
    rw = SFC.calculate_structure_function(SF1D, wrapped, u, bins, SF.StructureFunctionSumsAndCounts)
    Test.@test rb.counts == _minimum_image_counts((nx, ny), (0.25, 0.25), (false, false), bins)
    Test.@test rw.counts == _minimum_image_counts((nx, ny), (0.25, 0.25), (true, false), bins)
end

Test.@testset "a mis-shaped field and an unknown keyword are refused" begin
    geo = FG.Geometry.CartesianGeometry()
    bins = collect(range(0.0, 1.0; length = 5))
    uniform = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.2, length = 4),
                                      range(0.0, step = 0.2, length = 4))
    Test.@test_throws DimensionMismatch SFC.calculate_structure_function(
        SF1D, uniform, randn(2, 4), bins)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        SF1D, uniform, randn(2, 4, 4), bins; nonsense = 1)
end

Test.@testset "stretched and irregular axes give the unstructured answer" begin
    # A stretched axis first, then second under the transform; then no uniform axis; a periodic stretched axis is refused.
    Random.seed!(5300)
    geo = FG.Geometry.CartesianGeometry()
    xs = [0.0, 0.1, 0.35, 0.8, 0.85]
    ys = range(0.0, step = 0.2, length = 4)
    exact(g, r) = isapprox(g, r; rtol = 1e-10, atol = 1e-12)
    rounded(g, r) = isapprox(g, r; rtol = 1e-9, atol = 1e-10 * max(1.0, maximum(abs, r)))
    cases = (((xs, ys), SF1D, SB.AutoSpectralBackend(), exact),
             ((ys, xs), SFT.L2T1SFType(), SB.FastFourierTransformSpectralBackend(), rounded),
             ((xs, collect(ys)), SF1D, SB.AutoSpectralBackend(), exact))
    counts_ok, sums_ok = Bool[], Bool[]
    for (axes, sf, tag, close) in cases
        grid = FG.Grids.StructuredGrid(geo, axes...)
        x = _axes_points(axes)
        bins = _separated_bins_points(x, 6)
        nb = length(bins) - 1
        u = randn(2, map(length, axes)...)
        got = SFC.calculate_structure_function(sf, grid, u, bins, tag, UInt32, SF.StructureFunctionSumsAndCounts)
        ref_s = zeros(nb); ref_c = zeros(UInt32, nb)
        SFC.calculate_structure_function!(ref_s, ref_c, sf, x, reshape(u, 2, :), bins)
        push!(counts_ok, got.counts == ref_c && sum(ref_c) > 0)
        push!(sums_ok, close(got.sums, ref_s))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
    bins = collect(range(0.0, 1.0; length = 5))
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        SF1D, FG.Grids.StructuredGrid(geo, xs, ys; topology = (FG.Grids.Periodic(), FG.Grids.Bounded())),
        randn(2, 5, 4), bins)
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        SF1D, FG.Grids.StructuredGrid(geo, xs, collect(ys); topology = (FG.Grids.Bounded(), FG.Grids.Periodic())),
        randn(2, 5, 4), bins)
end

Test.@testset "a pixelized sphere gives the unstructured answer, less its antipodal pairs" begin
    # Every HEALPix pixel's antipode is a pixel, and those n/2 pairs have no direction.
    grid = FG.Grids.HEALPixGrid(FG.Geometry.SphericalGeometry(1.0), 2)
    n = length(FG.Grids.mask(grid))
    Random.seed!(5400)
    u = randn(2, n)
    bins = collect(range(0.0, 3.2; length = 6))
    got = SFC.calculate_structure_function(
        SF1D, grid, u, bins, UInt32, SF.StructureFunctionSumsAndCounts)
    Test.@test sum(got.counts) == n * (n - 1) ÷ 2 - n ÷ 2
    coords = FG.Grids.materialize(grid)
    x = permutedims(hcat(coords...))
    ref_s = zeros(5); ref_c = zeros(UInt32, 5)
    SFC.calculate_structure_function!(ref_s, ref_c, SF1D, x, u, bins;
                                     distance_metric = SFC.DI.SphericalAngle())
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
end
