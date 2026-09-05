using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using StaticArrays: StaticArrays as SA
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using Random: Random

# Every pair whose two ends both hold a datum, counted directly. Independent of lags and transforms.
function _brute_masked(sf, u, dims::NTuple{Dg, Int}, spacing::NTuple{Dg, T},
                       valid, bins) where {Dg, T}
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    D = size(u, 1)
    N = prod(dims)
    uf = reshape(u, D, N)
    ci = collect(CartesianIndices(dims))
    sums = zeros(Float64, nb)
    counts = zeros(Int, nb)
    for k1 in 1:(N - 1), k2 in (k1 + 1):N
        (valid[k1] && valid[k2]) || continue
        dx = SA.SVector{D, T}(ntuple(d -> d <= Dg ? T(ci[k2][d] - ci[k1][d]) * spacing[d] : zero(T),
                                     Val(D)))
        r2 = sum(abs2, dx)
        b = SFC.squared_digitize(plan, r2)
        1 <= b <= nb || continue
        du = SA.SVector{D, T}(ntuple(c -> uf[c, k2] - uf[c, k1], Val(D)))
        sums[b] += SFT._sf_raw(sf, du, dx, r2)
        counts[b] += 1
    end
    return sums, counts
end

function _run(sf, u, dims, spacing, periodic, bins, D, valid, backend)
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    sched = SFC.UniformLagSchedule(dims, spacing, periodic)
    if backend === nothing
        SFC.gridded_lag_sweep!(s, c, sf, u, sched, bins, Val(D); valid)
    else
        SFC.gridded_sweep!(s, c, sf, u, sched, bins, Val(D), backend; valid)
    end
    return s, c
end

Test.@testset "field_validity reports what is usable" begin
    u = randn(2, 4, 3)
    Test.@test SFC.field_validity(u, Val(2)) isa SFC.AllValid
    u[1, 2, 2] = NaN
    v = SFC.field_validity(u, Val(2))
    Test.@test !(v isa SFC.AllValid)
    Test.@test count(v) == 11                          # one cell of twelve lost
    Test.@test !v[2 + (2 - 1) * 4]
    # an infinite component is no more usable than a missing one
    u2 = randn(2, 3, 3); u2[2, 1, 1] = Inf
    Test.@test count(SFC.field_validity(u2, Val(2))) == 8
    # a cell the grid says does not exist is excluded even where the field is finite
    cm = trues(12); cm[5] = false
    Test.@test count(SFC.field_validity(randn(2, 4, 3), Val(2), cm)) == 11
    # AllValid answers for any index, so a complete field needs no array
    Test.@test SFC.AllValid()[1] && SFC.AllValid()[10^9]
end

Test.@testset "a masked sweep matches brute force" begin
    T = Float64
    for (dims, frac) in (((9, 6), 0.15), ((7, 7), 0.3), ((11,), 0.2), ((5, 4, 3), 0.25))
        Dg = length(dims)
        spacing = ntuple(_ -> T(0.2), Dg)
        N = prod(dims)
        Random.seed!(8100 + N + Dg)
        u = randn(T, Dg, dims...)
        # knock out a scattered fraction of cells, as an instrument or a coastline would
        uf = reshape(u, Dg, N)
        for k in 1:N
            rand() < frac && (uf[1, k] = NaN)
        end
        valid = SFC.field_validity(u, Val(Dg))
        Test.@test !(valid isa SFC.AllValid)
        bins = collect(range(0.0, 0.7 * maximum(d -> spacing[d] * dims[d], 1:Dg); length = 7))
        for sf in (SFT.L2SFType(), SFT.L3SFType())
            got_s, got_c = _run(sf, u, dims, spacing, ntuple(_ -> false, Dg), bins, Dg, valid, nothing)
            ref_s, ref_c = _brute_masked(sf, u, dims, spacing, valid, bins)
            Test.@test got_c == ref_c
            Test.@test isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12)
            Test.@test sum(got_c) > 0
            Test.@test all(isfinite, got_s)            # NaN must not leak out of an empty cell
        end
    end
end

Test.@testset "the masked transform matches the masked sweep" begin
    # The two compute the pair count differently — the sweep counts as it goes, the transform reads
    # the mask autocorrelation — so agreeing on counts is a real check, not a tautology.
    T = Float64
    for (dims, periodic) in (((10, 8), (false, false)), ((8, 8), (true, true)), ((12, 6), (true, false)))
        Dg = 2
        spacing = (T(0.2), T(0.25))
        N = prod(dims)
        Random.seed!(8200 + N)
        u = randn(T, Dg, dims...)
        uf = reshape(u, Dg, N)
        for k in 1:N
            rand() < 0.2 && (uf[2, k] = NaN)
        end
        valid = SFC.field_validity(u, Val(Dg))
        bins = collect(range(0.0, 1.2; length = 8))
        for sf in (SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType())
            ref_s, ref_c = _run(sf, u, dims, spacing, periodic, bins, Dg, valid, nothing)
            got_s, got_c = _run(sf, u, dims, spacing, periodic, bins, Dg, valid,
                                SB.FastFourierTransformSpectralBackend())
            Test.@test got_c == ref_c
            Test.@test isapprox(got_s, ref_s; rtol = 1e-8, atol = 1e-10)
            Test.@test all(isfinite, got_s)
            Test.@test sum(got_c) > 0
        end
    end
end

Test.@testset "masking removes exactly the pairs it should" begin
    T = Float64
    dims = (8, 6)
    spacing = (T(0.25), T(0.25))
    periodic = (false, false)
    N = prod(dims)
    Random.seed!(8300)
    u = randn(T, 2, dims...)
    bins = collect(range(0.0, 1e3; length = 4))
    _, c_full = _run(SFT.S2SFType(), u, dims, spacing, periodic, bins, 2, SFC.AllValid(), nothing)
    Test.@test sum(c_full) == N * (N - 1) ÷ 2

    # dropping one cell removes exactly the N-1 pairs it took part in
    v = trues(N); v[13] = false
    _, c_one = _run(SFT.S2SFType(), u, dims, spacing, periodic, bins, 2, v, nothing)
    Test.@test sum(c_one) == (N - 1) * (N - 2) ÷ 2
    Test.@test sum(c_full) - sum(c_one) == N - 1
end

Test.@testset "a grid entry honours the grid's own mask" begin
    geo = FG.Geometry.CartesianGeometry()
    nx, ny = 9, 7
    ax = range(0.0, step = 0.2, length = nx)
    ay = range(0.0, step = 0.2, length = ny)
    Random.seed!(8400)
    u = randn(2, nx, ny)
    bins = collect(range(0.0, 1e3; length = 4))

    cellmask = trues(nx, ny)
    cellmask[3, 4] = false
    cellmask[7, 2] = false
    holed = FG.Grids.StructuredGrid(geo, ax, ay, cellmask)
    whole = FG.Grids.StructuredGrid(geo, ax, ay)

    r_whole = SFC.calculate_structure_function(SFT.L2SFType(), whole, u, bins;
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    r_holed = SFC.calculate_structure_function(SFT.L2SFType(), holed, u, bins;
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    N = nx * ny
    Test.@test sum(r_whole.counts) == N * (N - 1) ÷ 2
    Test.@test sum(r_holed.counts) == (N - 2) * (N - 3) ÷ 2

    # a NaN in the field is excluded the same way, with no mask on the grid at all
    u2 = copy(u); u2[1, 5, 5] = NaN
    r_nan = SFC.calculate_structure_function(SFT.L2SFType(), whole, u2, bins;
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test sum(r_nan.counts) == (N - 1) * (N - 2) ÷ 2
    Test.@test all(isfinite, r_nan.sums)
end
