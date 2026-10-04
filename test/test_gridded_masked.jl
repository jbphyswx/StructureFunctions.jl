using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using StaticArrays: StaticArrays as SA
using FlowGeometries: FlowGeometries as FG
using Random: Random

# Every pair whose two ends both hold a datum, counted directly.
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
        sums[b] += sf(du, dx / sqrt(r2))
        counts[b] += 1
    end
    return sums, counts
end

Test.@testset "field_validity marks the cells whose components are all finite and that the grid holds" begin
    held(v, n) = findall(k -> !v[k], 1:n)
    u = randn(2, 4, 3)
    Test.@test held(SFC.field_validity(u), 12) == Int[]
    u[1, 2, 2] = NaN
    Test.@test held(SFC.field_validity(u), 12) == [6]
    u2 = randn(2, 3, 3); u2[2, 1, 1] = Inf
    Test.@test held(SFC.field_validity(u2), 9) == [1]
    cm = trues(12); cm[5] = false
    Test.@test held(SFC.field_validity(randn(2, 4, 3), cm), 12) == [5]
end

# (dims, fraction of cells knocked out, operator): each grid shape once.
const _MASKED_SWEEP_CASES = (((9, 6), 0.15, SFT.L2SFType()), ((7, 7), 0.3, SFT.L3SFType()),
                             ((11,), 0.2, SFT.L2SFType()), ((5, 4, 3), 0.25, SFT.L3SFType()))

Test.@testset "a masked sweep matches brute force" begin
    T = Float64
    counts_ok, sums_ok = Bool[], Bool[]
    for (dims, frac, sf) in _MASKED_SWEEP_CASES
        Dg = length(dims)
        spacing = ntuple(_ -> T(0.2), Dg)
        N = prod(dims)
        Random.seed!(8100 + N + Dg)
        u = randn(T, Dg, dims...)
        uf = reshape(u, Dg, N)
        for k in 1:N
            rand() < frac && (uf[1, k] = NaN)
        end
        held = vec(all(isfinite, uf; dims = 1))
        bins = collect(range(0.0, 0.7 * maximum(d -> spacing[d] * dims[d], 1:Dg); length = 7))
        nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(bins))
        got_s, got_c = zeros(nb), zeros(Int, nb)
        SFC.gridded_lag_sweep!(got_s, got_c, sf, u, SFC.UniformLagSchedule(dims, spacing, ntuple(_ -> false, Dg)),
                               bins, Val(Dg); valid = SFC.field_validity(u))
        ref_s, ref_c = _brute_masked(sf, u, dims, spacing, held, bins)
        push!(counts_ok, got_c == ref_c && sum(ref_c) > 0)
        push!(sums_ok, isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

Test.@testset "a grid entry honours the grid's own mask and the field's missing values" begin
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
    N = nx * ny
    r_holed = SFC.calculate_structure_function(SFT.L2SFType(), holed, u, bins, SF.StructureFunctionSumsAndCounts)
    Test.@test sum(r_holed.counts) == (N - 2) * (N - 3) ÷ 2
    u2 = copy(u); u2[1, 5, 5] = NaN
    r_nan = SFC.calculate_structure_function(SFT.L2SFType(), whole, u2, bins, SF.StructureFunctionSumsAndCounts)
    Test.@test sum(r_nan.counts) == (N - 1) * (N - 2) ÷ 2
    Test.@test all(isfinite, r_nan.sums)
end
