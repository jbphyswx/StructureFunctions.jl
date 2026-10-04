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

# Pair loop under the minimum image, a half-turning pair averaged over its equal-length images.
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
        amb = [d for d in 1:Dg if periodic[d] && iseven(dims[d]) &&
               abs(I2[d] - I1[d]) % dims[d] == dims[d] ÷ 2]
        acc = 0.0
        for m in 0:((1 << length(amb)) - 1)
            dxm = SA.SVector{D, T}(ntuple(Val(D)) do d
                j = findfirst(==(d), amb)
                (j !== nothing && (m >> (j - 1)) & 1 == 1) ? -dx[d] : dx[d]
            end)
            acc += sf(du, dxm / sqrt(r2))
        end
        sums[b] += acc / (1 << length(amb))
        counts[b] += 1
    end
    return sums, counts
end

# Bin edges midway between the distinct separations a grid produces, so no shell sits on an edge.
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

const _BOUNDED_GRID_CASES = (((7,), (0.25,), SFT.L2SFType()),
                             ((9, 6), (0.1, 0.2), SFT.L3SFType()),
                             ((5, 4, 3), (0.3, 0.3, 0.5), SFT.S2SFType()),
                             ((8, 5), (0.15, -0.25), SFT.L2SFType()))

Test.@testset "lag sweep on a bounded grid equals the unstructured path" begin
    counts_ok, sums_ok = Bool[], Bool[]
    for (dims, spacing, sf) in _BOUNDED_GRID_CASES
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
        got_s, got_c = _sweep(sf, u, dims, spacing, periodic, bins, Dg)
        ref_s = zeros(Float64, nb)
        ref_c = zeros(Int, nb)
        SFC.calculate_structure_function!(ref_s, ref_c, sf, x, reshape(u, Dg, N), bins)
        push!(counts_ok, got_c == ref_c && sum(ref_c) > 0)
        push!(sums_ok, isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

const _MINIMUM_IMAGE_CASES = (((6, 4), (true, true), SFT.L3SFType()), ((6, 4), (true, false), SFT.L2SFType()),
                              ((5, 5), (true, true), SFT.L2SFType()), ((8,), (true,), SFT.L2SFType()),
                              ((4, 4, 2), (true, true, true), SFT.L3SFType()))

Test.@testset "lag sweep matches a brute-force minimum image" begin
    # Every wrapped pair falls in these bins, half-turning ones included, each counted once as the pair loop does.
    T = Float64
    counts_ok, sums_ok = Bool[], Bool[]
    for (dims, periodic, sf) in _MINIMUM_IMAGE_CASES
        Dg = length(dims)
        spacing = ntuple(d -> T(0.1 * d + 0.1), Dg)
        Random.seed!(4400 + prod(dims) + Dg)
        u = randn(T, Dg, dims...)
        r_max = 0.6 * sum(d -> spacing[d] * dims[d], 1:Dg)
        bins = collect(range(0.0, r_max; length = 7))
        got_s, got_c = _sweep(sf, u, dims, spacing, periodic, bins, Dg)
        ref_s, ref_c = _brute_force_histogram(sf, u, dims, spacing, periodic, bins)
        push!(counts_ok, got_c == ref_c && sum(ref_c) > 0)
        push!(sums_ok, isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

Test.@testset "lag sweep with more field components than grid directions" begin
    # A 3-component field on a bounded 2-D grid counts every pair, and its energy is the in-plane one plus the third's.
    T = Float64
    dims = (7, 5)
    spacing = (0.2, 0.2)
    periodic = (false, false)
    Random.seed!(4500)
    u = randn(T, 3, dims...)
    bins = collect(range(0.0, 2.0; length = 6))
    s3, c3 = _sweep(SFT.S2SFType(), u, dims, spacing, periodic, bins, 3)
    Test.@test sum(c3) == prod(dims) * (prod(dims) - 1) ÷ 2
    s2, c2 = _sweep(SFT.S2SFType(), u[1:2, :, :], dims, spacing, periodic, bins, 2)
    third = zeros(T, 2, dims...)
    third[2, :, :] .= u[3, :, :]
    s_third, _ = _sweep(SFT.S2SFType(), third, dims, spacing, periodic, bins, 2)
    Test.@test c2 == c3
    Test.@test isapprox(s3, s2 .+ s_third; rtol = 1e-12)
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

# Joint histogram by value under the minimum image: each of a pair's M equal-length images adds a 1/M share.
function _brute_force_value_histogram(sf, u, dims::NTuple{Dg, Int}, spacing::NTuple{Dg, T},
                                      periodic::NTuple{Dg, Bool}, bins, vbins) where {Dg, T}
    plan = SFC.squared_digitize_plan(bins)
    nb, nv = SFC.n_histogram_bins(plan), length(vbins) - 1
    D = size(u, 1)
    N = prod(dims)
    uf = reshape(u, D, N)
    ci = collect(CartesianIndices(dims))
    sums, counts = zeros(nb, nv), zeros(nb, nv)
    for k1 in 1:(N - 1), k2 in (k1 + 1):N
        I1, I2 = ci[k1], ci[k2]
        δ = ntuple(Val(Dg)) do d
            m = I2[d] - I1[d]
            if periodic[d]
                m = mod(m, dims[d])
                m > dims[d] ÷ 2 && (m -= dims[d])
            end
            T(m) * spacing[d]
        end
        dx = SA.SVector{D, T}(ntuple(d -> d <= Dg ? δ[d] : zero(T), Val(D)))
        r2 = sum(abs2, dx)
        b = SFC.squared_digitize(plan, r2)
        1 <= b <= nb || continue
        du = SA.SVector{D, T}(ntuple(c -> uf[c, k2] - uf[c, k1], Val(D)))
        amb = [d for d in 1:Dg if periodic[d] && iseven(dims[d]) && abs(I2[d] - I1[d]) % dims[d] == dims[d] ÷ 2]
        M = 1 << length(amb)
        for m in 0:(M - 1)
            dxm = SA.SVector{D, T}(ntuple(Val(D)) do d
                j = findfirst(==(d), amb)
                (j !== nothing && (m >> (j - 1)) & 1 == 1) ? -dx[d] : dx[d]
            end)
            val = sf(du, dxm / sqrt(r2))
            vb = searchsortedfirst(vbins, val) - 1
            1 <= vb <= nv || continue
            sums[b, vb] += val / M
            counts[b, vb] += 1 / M
        end
    end
    return sums, counts
end

const _VALUE_JOINT_CASES = ((SFT.L2SFType(), :plain), (SFT.T2SFType(), :masked), (SFT.S3SFType(), :plain),
                            (SFT.L3SFType(), :weighted))
const _VALUE_JOINT_PERIODIC_CASES = (((6, 4), (true, true), SFT.L3SFType()), ((6, 5), (true, false), SFT.L2SFType()))

Test.@testset "the joint histogram by value equals the pair loop and the minimum-image brute force" begin
    T = Float64
    vax = SFC.InvariantValueAxis()
    vbins = collect(range(-1.5, 1.5; length = 9))
    dims, spacing = (9, 6), (0.1, 0.2)
    N = prod(dims)
    x = _grid_points(dims, spacing, (0.0, 0.0))
    bins = _separated_bins(dims, spacing, 6)
    nb = length(bins) - 1
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
    Random.seed!(4700)
    counts_ok, sums_ok = Bool[], Bool[]
    for (sf, variant) in _VALUE_JOINT_CASES
        u = randn(T, 2, dims...)
        variant === :masked && (u[1, 3, 2] = NaN; u[2, 7, 5] = NaN)
        w = variant === :weighted ? 0.5 .+ rand(N) : nothing
        valid = SFC.field_validity(u)
        keep = vec(all(isfinite, reshape(u, 2, N); dims = 1))
        got_s, got_c = zeros(nb, 8), zeros(nb, 8)
        SFC.gridded_lag_sweep!(got_s, got_c, sf, u, sched, bins, vbins, Val(2); valid, weights = w, second_axis = vax)
        ref_s, ref_c = zeros(nb, 8), zeros(nb, 8)
        SFC.calculate_structure_function!(ref_s, ref_c, sf, x[:, keep], reshape(u, 2, N)[:, keep], bins, vbins;
                                         weights = w === nothing ? nothing : w[keep])
        push!(counts_ok, sum(ref_c) > 0 && isapprox(got_c, ref_c; rtol = 1e-12))
        push!(sums_ok, isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12))
    end
    for (pdims, periodic, sf) in _VALUE_JOINT_PERIODIC_CASES
        pspacing = (0.1, 0.12)
        psched = SFC.UniformLagSchedule(pdims, pspacing, periodic)
        pbins = collect(range(0.0, 0.45; length = 6)) .+ 0.0137
        u = randn(T, 2, pdims...)
        got_s, got_c = zeros(5, 8), zeros(5, 8)
        SFC.gridded_lag_sweep!(got_s, got_c, sf, u, psched, pbins, vbins, Val(2); second_axis = vax)
        ref_s, ref_c = _brute_force_value_histogram(sf, u, pdims, pspacing, periodic, pbins, vbins)
        push!(counts_ok, sum(ref_c) > 0 && isapprox(got_c, ref_c; rtol = 1e-12))
        push!(sums_ok, isapprox(got_s, ref_s; rtol = 1e-10, atol = 1e-12))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
    # half-turn lags split a pair between two value bins, which integer counts cannot hold
    pdims, periodic, _ = first(_VALUE_JOINT_PERIODIC_CASES)
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(
        zeros(5, 8), zeros(Int, 5, 8), SFT.L2SFType(), randn(T, 2, pdims...),
        SFC.UniformLagSchedule(pdims, (0.1, 0.12), periodic), collect(range(0.0, 0.45; length = 6)) .+ 0.0137,
        vbins, Val(2); second_axis = vax)
    ub = randn(T, 2, dims..., 2)
    bs, bc = zeros(nb, 8, 2), zeros(nb, 8, 2)
    SFC.gridded_lag_sweep_batch!(bs, bc, SFT.S3SFType(), ub, sched, bins, vbins, Val(2); second_axis = vax)
    ss, sc = zeros(nb, 8, 2), zeros(nb, 8, 2)
    for t in 1:2
        SFC.gridded_lag_sweep!(view(ss, :, :, t), view(sc, :, :, t), SFT.S3SFType(), ub[:, :, :, t], sched, bins,
                               vbins, Val(2); second_axis = vax)
    end
    Test.@test bc == sc
    Test.@test isapprox(bs, ss; rtol = 1e-12, atol = 1e-14)
end

Test.@testset "a shell exactly on a bin edge falls in the bin below it" begin
    # Unit 5×5 grid, bins (lo, hi]: r = 1 holds 40 pairs; √2 32 and 2 30; √5 48, √8 18 and 3 20.
    T = Float64
    dims = (5, 5)
    spacing = (T(1), T(1))
    periodic = (false, false)
    Random.seed!(4600)
    u = randn(T, 2, dims...)
    bins = [0.0, 1.0, 2.0, 3.0]
    _, c = _sweep(SFT.S2SFType(), u, dims, spacing, periodic, bins, 2)
    Test.@test c == [40, 32 + 30, 48 + 18 + 20]
end
