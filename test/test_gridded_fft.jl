using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Random: Random

const QUADRATIC = (SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType(), SFT.T2ComponentSFType())

function _run(sf, u, dims, spacing, periodic, bins, D, backend)
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    sched = SFC.UniformLagSchedule(dims, spacing, periodic)
    if backend === nothing
        SFC.gridded_lag_sweep!(s, c, sf, u, sched, bins, Val(D))
    else
        SFC.gridded_sweep!(s, c, sf, u, sched, bins, Val(D), backend)
    end
    return s, c
end

# The transform and the lag sweep are two algorithms for one definition, so they must agree to
# round-off on the same data. The sweep is the reference because it is checked against the
# unstructured pair loop and against brute force in test_gridded.jl.
Test.@testset "the transform agrees with the lag sweep" begin
    T = Float64
    for (dims, spacing, periodic) in (((9, 6), (0.1, 0.2), (false, false)),
                                      ((8, 8), (0.25, 0.25), (true, true)),
                                      ((10, 7), (0.15, 0.15), (true, false)),
                                      ((12,), (0.3,), (false,)),
                                      ((6, 5, 4), (0.2, 0.2, 0.3), (false, false, false)))
        Dg = length(dims)
        Random.seed!(6100 + prod(dims) + Dg)
        u = randn(T, Dg, dims...)
        r_max = 0.7 * sum(d -> spacing[d] * dims[d], 1:Dg)
        bins = collect(range(0.0, r_max; length = 9))
        for sf in QUADRATIC
            Dg == 1 && sf === SFT.T2ComponentSFType() && continue   # no transverse subspace on a line
            ref_s, ref_c = _run(sf, u, dims, spacing, periodic, bins, Dg, nothing)
            got_s, got_c = _run(sf, u, dims, spacing, periodic, bins, Dg,
                                SB.FastFourierTransformSpectralBackend())
            Test.@test got_c == ref_c
            Test.@test isapprox(got_s, ref_s; rtol = 1e-8, atol = 1e-10)
            Test.@test sum(got_c) > 0
        end
    end
end

Test.@testset "the transform counts every pair once" begin
    T = Float64
    for (dims, periodic) in (((6, 4), (false, false)), ((6, 4), (true, true)), ((5, 5), (true, false)))
        Dg = length(dims)
        spacing = ntuple(_ -> T(0.5), Dg)
        N = prod(dims)
        Random.seed!(6200 + N)
        u = randn(T, Dg, dims...)
        bins = collect(range(0.0, 1e3; length = 4))
        _, c = _run(SFT.S2SFType(), u, dims, spacing, periodic, bins, Dg,
                    SB.FastFourierTransformSpectralBackend())
        Test.@test sum(c) == N * (N - 1) ÷ 2
    end
end

Test.@testset "the transform refuses what it cannot express" begin
    T = Float64
    dims = (6, 5)
    spacing = (0.2, 0.2)
    periodic = (false, false)
    u = randn(T, 2, dims...)
    bins = collect(range(0.0, 1.0; length = 5))
    # a third-order operator is not a contraction of the second-order tensor
    for sf in (SFT.L3SFType(), SFT.S3SFType())
        Test.@test_throws ArgumentError _run(sf, u, dims, spacing, periodic, bins, 2,
                                             SB.FastFourierTransformSpectralBackend())
        # ...and the lag sweep still does it
        _, c = _run(sf, u, dims, spacing, periodic, bins, 2, nothing)
        Test.@test sum(c) > 0
    end
end

Test.@testset "the algorithm tags select as documented" begin
    T = Float64
    dims = (8, 6)
    spacing = (0.2, 0.25)
    periodic = (false, false)
    Random.seed!(6300)
    u = randn(T, 2, dims...)
    bins = collect(range(0.0, 1.2; length = 7))
    ref_s, ref_c = _run(SFT.L2SFType(), u, dims, spacing, periodic, bins, 2, nothing)

    # the direct sum IS the lag sweep, so it must be bit-identical, not merely close
    ds_s, ds_c = _run(SFT.L2SFType(), u, dims, spacing, periodic, bins, 2,
                      SB.DirectSumSpectralBackend())
    Test.@test ds_c == ref_c
    Test.@test ds_s == ref_s

    # auto picks one of the two exact algorithms, so it must agree with both
    au_s, au_c = _run(SFT.L2SFType(), u, dims, spacing, periodic, bins, 2, SB.AutoSpectralBackend())
    Test.@test au_c == ref_c
    Test.@test isapprox(au_s, ref_s; rtol = 1e-8, atol = 1e-10)

    # auto on an operator no transform expresses still works, by sweeping
    l3_s, l3_c = _run(SFT.L3SFType(), u, dims, spacing, periodic, bins, 2, SB.AutoSpectralBackend())
    ref3_s, ref3_c = _run(SFT.L3SFType(), u, dims, spacing, periodic, bins, 2, nothing)
    Test.@test l3_c == ref3_c
    Test.@test l3_s == ref3_s
end

Test.@testset "the grid entry takes the tag positionally" begin
    geo = FG.Geometry.CartesianGeometry()
    nx, ny = 9, 7
    grid = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.2, length = nx),
                                   range(0.0, step = 0.2, length = ny))
    Random.seed!(6400)
    u = randn(2, nx, ny)
    bins = collect(range(0.0, 1.4; length = 8))
    swept = SFC.calculate_structure_function(
        SFT.L2SFType(), grid, u, bins; output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    transformed = SFC.calculate_structure_function(
        SFT.L2SFType(), grid, u, bins, UInt32, SB.FastFourierTransformSpectralBackend();
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test transformed.counts == swept.counts
    Test.@test isapprox(transformed.sums, swept.sums; rtol = 1e-8, atol = 1e-10)
    Test.@test sum(swept.counts) > 0
end
