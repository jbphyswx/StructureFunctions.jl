using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Random: Random
using StaticArrays: StaticArrays as SA

# Cell coordinates of a uniform grid, flattened in the same order `reshape(u, D, N)` uses.
function _grid_points(dims::NTuple{Dg, Int}, spacing::NTuple{Dg, T}, origin::NTuple{Dg, T}) where {Dg, T}
    N = prod(dims)
    x = Matrix{T}(undef, Dg, N)
    for (k, I) in enumerate(CartesianIndices(dims))
        for d in 1:Dg
            x[d, k] = origin[d] + (I[d] - 1) * spacing[d]
        end
    end
    return x
end

# Brute force over unordered pairs with the minimum image on periodic directions, averaging the
# operator over the equal-length images where a direction half-turns. Independent of the lag
# machinery: it enumerates pairs, not lags, and builds each image set from the pair's own offset.
function _brute_force_histogram(sf, u, dims::NTuple{Dg, Int}, spacing::NTuple{Dg, T},
                                periodic::NTuple{Dg, Bool}, bins) where {Dg, T}
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    D = size(u, 1)
    N = prod(dims)
    uf = reshape(u, D, N)
    ci = collect(CartesianIndices(dims))
    sums = zeros(Float64, nb)
    counts = zeros(Int, nb)
    for k1 in 1:(N - 1), k2 in (k1 + 1):N
        I1, I2 = ci[k1], ci[k2]
        δ = ntuple(Val(Dg)) do d
            m = I2[d] - I1[d]
            if periodic[d]
                n = dims[d]
                m = mod(m, n)
                m > n ÷ 2 && (m -= n)
            end
            T(m) * spacing[d]
        end
        dx = SA.SVector{D, T}(ntuple(d -> d <= Dg ? δ[d] : zero(T), Val(D)))
        r2 = sum(abs2, dx)
        b = SFC.squared_digitize(plan, r2)
        1 <= b <= nb || continue
        du = SA.SVector{D, T}(ntuple(c -> uf[c, k2] - uf[c, k1], Val(D)))
        # directions this pair half-turns: both signs are minimal, so both are equally its direction
        amb = [d for d in 1:Dg if periodic[d] && iseven(dims[d]) &&
               abs(I2[d] - I1[d]) % dims[d] == dims[d] ÷ 2]
        acc = 0.0
        for m in 0:((1 << length(amb)) - 1)
            dxm = SA.SVector{D, T}(ntuple(Val(D)) do d
                j = findfirst(==(d), amb)
                (j !== nothing && (m >> (j - 1)) & 1 == 1) ? -dx[d] : dx[d]
            end)
            acc += SFT._sf_raw(sf, du, dxm, r2)
        end
        sums[b] += acc / (1 << length(amb))
        counts[b] += 1
    end
    return sums, counts
end

# Bin edges placed between the separations a grid can produce. Pairs sharing a lag share one exact
# separation, so an edge through it splits a whole shell by rounding: the sweep bins them by the
# lag's separation and a pair loop by each pair's own coordinate difference, which differ by an ulp.
function _separated_bins(dims::NTuple{Dg, Int}, spacing::NTuple{Dg, T}, n_bins::Int) where {Dg, T}
    seps = Float64[]
    for I in CartesianIndices(ntuple(d -> 0:(dims[d] - 1), Val(Dg)))
        h = Tuple(I)
        all(iszero, h) && continue
        push!(seps, sqrt(sum(d -> (h[d] * spacing[d])^2, 1:Dg)))
    end
    sort!(seps)
    unique!(seps)
    idx = unique(round.(Int, range(1, length(seps) - 1; length = n_bins)))
    return [0.0; [(seps[i] + seps[i + 1]) / 2 for i in idx]]
end

function _sweep(sf, u, dims, spacing, periodic, bins, D)
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    sched = SFC.UniformLagSchedule(dims, spacing, periodic)
    SFC.gridded_lag_sweep!(s, c, sf, u, sched, bins, Val(D))
    return s, c
end

Test.@testset "lag sweep on a bounded grid equals the unstructured path" begin
    # The strongest oracle available: on a bounded grid the separations are plain Euclidean, so the
    # existing pair loop over the same points must agree exactly in counts.
    for (dims, spacing) in (((7,), (0.25,)),
                            ((9, 6), (0.1, 0.2)),
                            ((5, 4, 3), (0.3, 0.3, 0.5)),
                            ((8, 5), (0.15, -0.25)))     # a descending axis reports negative spacing
        Dg = length(dims)
        T = Float64
        origin = ntuple(_ -> zero(T), Dg)
        N = prod(dims)
        Random.seed!(4200 + N + Dg)
        x = _grid_points(dims, spacing, origin)
        u = randn(T, Dg, dims...)
        periodic = ntuple(_ -> false, Dg)
        bins = _separated_bins(dims, abs.(spacing), 8)
        nb = length(bins) - 1

        for sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.S2SFType())
            got_s, got_c = _sweep(sf, u, dims, spacing, periodic, bins, Dg)
            ref_s = zeros(Float64, nb)
            ref_c = zeros(Int, nb)
            SF.calculate_structure_function!(ref_s, ref_c, sf, x, reshape(u, Dg, N), bins)
            Test.@test got_c == ref_c
            Test.@test isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12)
            Test.@test sum(got_c) > 0
        end
    end
end

Test.@testset "lag sweep counts every pair once" begin
    # With bins spanning every separation the histogram must hold exactly N(N-1)/2 pairs, on a
    # bounded grid and on a periodic one, where the minimum image makes the largest separation
    # smaller but changes no count.
    T = Float64
    for dims in ((6, 4), (5, 5), (4, 4), (7,), (3, 4, 2))
        Dg = length(dims)
        spacing = ntuple(_ -> T(0.5), Dg)
        N = prod(dims)
        Random.seed!(4300 + N)
        u = randn(T, Dg, dims...)
        bins = collect(range(0.0, 1e3; length = 4))
        for periodic in (ntuple(_ -> false, Dg), ntuple(_ -> true, Dg),
                         ntuple(d -> isodd(d), Dg))
            _, c = _sweep(SFT.L2SFType(), u, dims, spacing, periodic, bins, Dg)
            Test.@test sum(c) == N * (N - 1) ÷ 2
        end
    end
end

Test.@testset "lag sweep matches a brute-force minimum image" begin
    # Periodicity is the part the unstructured path cannot check: it has no wrap concept. Even
    # lengths are included because a half-turn lag is its own reverse and is the one case that names
    # each pair twice.
    T = Float64
    for (dims, periodic) in (((6, 4), (true, true)),
                             ((6, 4), (true, false)),
                             ((5, 5), (true, true)),
                             ((8,), (true,)),
                             ((4, 4, 2), (true, true, true)))
        Dg = length(dims)
        spacing = ntuple(d -> T(0.1 * d + 0.1), Dg)
        Random.seed!(4400 + prod(dims) + Dg)
        u = randn(T, Dg, dims...)
        r_max = 0.6 * sum(d -> spacing[d] * dims[d], 1:Dg)
        bins = collect(range(0.0, r_max; length = 7))
        for sf in (SFT.L2SFType(), SFT.L3SFType())
            got_s, got_c = _sweep(sf, u, dims, spacing, periodic, bins, Dg)
            ref_s, ref_c = _brute_force_histogram(sf, u, dims, spacing, periodic, bins)
            Test.@test got_c == ref_c
            Test.@test isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12)
            Test.@test sum(got_c) > 0
        end
    end
end

Test.@testset "lag sweep with more field components than grid directions" begin
    # A horizontal slice of a 3-component flow: the lag lies in the grid's plane and is zero along
    # the extra component, which still enters the transverse energy.
    T = Float64
    dims = (7, 5)
    spacing = (0.2, 0.2)
    periodic = (false, false)
    Random.seed!(4500)
    u = randn(T, 3, dims...)
    bins = collect(range(0.0, 2.0; length = 6))     # spans the 1.44 diagonal, so every pair bins
    s3, c3 = _sweep(SFT.S2SFType(), u, dims, spacing, periodic, bins, 3)
    Test.@test sum(c3) == prod(dims) * (prod(dims) - 1) ÷ 2

    # dropping the third component must leave the counts alone and lower the second-order sum
    s2, c2 = _sweep(SFT.S2SFType(), u[1:2, :, :], dims, spacing, periodic, bins, 2)
    Test.@test c2 == c3
    Test.@test all(s2 .<= s3 .+ 1e-12)
end

Test.@testset "lag sweep rejects mismatched shapes" begin
    T = Float64
    dims = (5, 4)
    sched = SFC.UniformLagSchedule(dims, (0.1, 0.1), (false, false))
    bins = collect(range(0.0, 1.0; length = 5))
    s = zeros(4); c = zeros(Int, 4)
    Test.@test_throws DimensionMismatch SFC.gridded_lag_sweep!(
        s, c, SFT.L2SFType(), randn(T, 2, 5, 5), sched, bins, Val(2))
    Test.@test_throws DimensionMismatch SFC.gridded_lag_sweep!(
        zeros(3), zeros(Int, 3), SFT.L2SFType(), randn(T, 2, dims...), sched, bins, Val(2))
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(
        s, c, SFT.L2SFType(), randn(T, 1, dims...), sched, bins, Val(1))
end

Test.@testset "a shell exactly on a bin edge falls in the bin below it" begin
    # Bins are half-open (e_i, e_{i+1}], and a uniform grid puts whole shells of pairs at exactly one
    # separation, so an edge placed on one decides a whole shell at once. Pinned because the sweep
    # binds them together where a pair loop lets each pair's round-off decide.
    T = Float64
    dims = (5, 5)
    spacing = (T(1), T(1))
    periodic = (false, false)
    Random.seed!(4600)
    u = randn(T, 2, dims...)
    bins = [0.0, 1.0, 2.0, 3.0]                    # 1.0 is exactly the nearest-neighbour separation
    s, c = _sweep(SFT.S2SFType(), u, dims, spacing, periodic, bins, 2)
    # the axis-aligned neighbours are at exactly r = 1.0, and there are 2·5·4 of them
    Test.@test c[1] == 2 * 5 * 4
    # r = 2.0 lands on the second edge, so those pairs are in bin 2, not bin 3
    Test.@test c[2] > 0
    Test.@test sum(c) > 0
end
