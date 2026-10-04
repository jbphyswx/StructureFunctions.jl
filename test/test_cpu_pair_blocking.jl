using Test: Test
using StructureFunctions: StructureFunctions, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH, LinearBinEdges, LogBinEdges
using Random: Random
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads

const _SERIAL, _THREADED = CB.SerialBackend(), CB.ThreadedBackend()
const _RAW = StructureFunctions.StructureFunctionSumsAndCounts

_comp(x, D) = ntuple(d -> collect(view(x, d, :)), D)

"""Per-operator sums and counts over `edges` of the pairs `(i, j > i)`, `i ∈ ilist`, by brute force."""
function _brute_histogram(ops, x, u, edges, ilist)
    D, N = size(x)
    e = collect(edges)
    nb = length(e) - 1
    s, c = zeros(length(ops), nb), zeros(Int, nb)
    for i in ilist, j in (i + 1):N
        dx = SA.SVector{D}(ntuple(d -> x[d, j] - x[d, i], D))
        r = sqrt(sum(abs2, dx))
        b = searchsortedfirst(e, r) - 1
        1 <= b <= nb || continue
        δu = SA.SVector{D}(ntuple(d -> u[d, j] - u[d, i], D))
        s[:, b] .+= [op(δu, dx / r) for op in ops]
        c[b] += 1
    end
    return s, c
end

const _SCHEDULE_CASES = (
    (kernel = :pair, D = 2, bins = :linear, sf = SFT.L2SFType(), tile = 1, window = SFC.WholeRun(), stride = 1),
    (kernel = :pair, D = 3, bins = :log, sf = SFT.L3SFType(), tile = 7, window = SFC.BlockRun(), stride = 1),
    (kernel = :pair, D = 2, bins = :log, sf = SFT.S2SFType(), tile = 80, window = SFC.WholeRun(), stride = 3),
    (kernel = :single_pass, D = 3, bins = :linear, sf = nothing, tile = 13, window = SFC.WholeRun(), stride = 1),
    (kernel = :single_pass, D = 2, bins = :log, sf = nothing, tile = 39, window = SFC.BlockRun(), stride = 3),
)

# Under every tile, buffer window and outer index list the pair kernels sum exactly the pairs they are given.
Test.@testset "pair kernels give the brute-force histogram under every block schedule" begin
    N, FT = 40, Float64
    for (; kernel, D, bins, sf, tile, window, stride) in _SCHEDULE_CASES
        Random.seed!(4242 + D + tile)
        x = rand(FT, D, N); u = randn(FT, D, N)
        edges = bins === :linear ? LinearBinEdges(range(FT(0.0), FT(1.6); length = 17)) :
                LogBinEdges(FT(0.02), FT(1.6), 17)
        plan = SFC.squared_digitize_plan(edges)
        nb = SFC.n_histogram_bins(plan)
        ilist = 1:stride:(N - 1)
        blocks = SFC.pair_blocks(N, ilist; tile)
        L = window isa SFC.WholeRun ? N : tile
        if kernel === :pair
            s, c = zeros(FT, nb), zeros(UInt32, nb)
            SFC._pf_simd_pairs!(s, c, sf, _comp(x, D), _comp(u, D), plan, Val(D), Vector{FT}(undef, L),
                                Vector{FT}(undef, L), Vector{Int32}(undef, L), Vector{Int32}(undef, L), window, blocks,
                                SFC.NoWeights())
            sref, cref = _brute_histogram((sf,), x, u, edges, ilist)
            Test.@test c == cref && isapprox(s, vec(sref); rtol = 1e-12, atol = 1e-14)
        else
            s, c = zeros(FT, SFC.SINGLE_PASS_N, nb), zeros(UInt32, SFC.SINGLE_PASS_N, nb)
            SFC._pf_sp_simd_pairs!(s, c, _comp(x, D), _comp(u, D), plan, Val(D), Vector{FT}(undef, L),
                                   Vector{FT}(undef, L), Vector{FT}(undef, L), Vector{Int32}(undef, L),
                                   Vector{Int32}(undef, L), window, blocks)
            sref, cref = _brute_histogram(values(SFC.SINGLE_PASS_OPERATORS), x, u, edges, ilist)
            Test.@test all(==(cref), eachrow(c)) && isapprox(s, sref; rtol = 1e-12, atol = 1e-14)
        end
    end
    x, u = rand(2, N), randn(2, N)
    plan = SFC.squared_digitize_plan(LinearBinEdges(0.0, 1.6, 17))
    nb = SFC.n_histogram_bins(plan)
    short = N - 1
    Test.@test_throws ArgumentError SFC._pf_sp_simd_pairs!(
        zeros(SFC.SINGLE_PASS_N, nb), zeros(UInt32, SFC.SINGLE_PASS_N, nb), _comp(x, 2), _comp(u, 2), plan, Val(2),
        zeros(short), zeros(short), zeros(short), zeros(Int32, short), zeros(Int32, short), SFC.BlockRun(),
        SFC.pair_blocks(N, 1:(N - 1); tile = N))
end

const _CULLED_SCHEDULE_CASES = (
    (D = 1, span = 2, cut = 0.03, parts = (N -> [1:(N - 1)])),
    (D = 1, span = 2, cut = 0.03, parts = (N -> [1:0, 1:(N - 1)])),
    (D = 2, span = 1, cut = 0.1, parts = (N -> [k:min(k + 6, N - 1) for k in 1:7:(N - 1)])),
    (D = 2, span = 1, cut = 0.1, parts = (N -> [k:13:(N - 1) for k in 1:13])),
    (D = 3, span = 3, cut = 0.25, parts = (N -> [collect(k:5:(N - 1)) for k in 1:5])),
)

# A culled schedule, whole or split over parts of its outer indices, sweeps every pair within the cutoff exactly once.
Test.@testset "culled block schedules sweep every in-range pair once" begin
    N = 150
    for (; D, span, cut, parts) in _CULLED_SCHEDULE_CASES
        Random.seed!(2601 + D + span)
        xc = ntuple(_ -> rand(N), D)
        grid = SFC.build_cell_grid(xc, cut, span)
        xp = SFC.apply_perm(xc, grid.perm)
        near = [(i, j) for i in 1:(N - 1) for j in (i + 1):N if sum(d -> (xp[d][i] - xp[d][j])^2, 1:D) <= cut^2]
        swept = [(i, j) for irange in parts(N) for (ir, jr) in SFC.pair_blocks(N, irange; grid)
                 for i in ir for j in jr if j > i]
        Test.@test allunique(swept) && issubset(near, Set(swept))
    end
end

# A pair at the largest bin's separation lies within the cutoff: the unit chord on a sphere, the separation when flat.
Test.@testset "cull_cutoff bounds every separation within the largest bin" begin
    R = 6.371e6
    g = SFH.SphericalGeometry{2}(DI.Haversine(R), R)
    chord(σ) = sqrt((cos(σ) - 1)^2 + sin(σ)^2)
    Test.@test all(r_max -> chord(r_max / R) <= SFC.cull_cutoff(g, r_max) + 1e-12, (1.0e4, 1.0e6, 5.0e6))
    Test.@test SFC.cull_cutoff(g, π * R) >= 2
    Test.@test SFC.cull_cutoff(SFH.FlatGeometry{2}(), 3.5) >= 3.5
end

struct UnboundedTestGeometry{D} end
SFH.coordinate_width(::UnboundedTestGeometry{D}) where {D} = Val(D)

# An unbounded last bin, or a geometry with no bound, is swept whole by AutoCulling and refused by AlwaysCulling.
Test.@testset "culling declines where it cannot skip pairs" begin
    Random.seed!(11)
    N = 20
    x, u = rand(2, N), randn(2, N)
    tight = LinearBinEdges(range(0.0, 0.05; length = 9))
    unbounded = StructureFunctions.InfPaddedBinEdges(tight)
    padded(pol) = SFC.calculate_structure_function(SFT.L2SFType(), x, u, unbounded, _RAW; backend = _SERIAL,
                                                   culling = pol)
    Test.@test sum(padded(SFC.AutoCulling()).counts) == N * (N - 1) ÷ 2
    Test.@test_throws ArgumentError padded(SFC.AlwaysCulling())
    xc, g = _comp(x, 2), UnboundedTestGeometry{2}()
    Test.@test SFC.cull_grid_for(xc, g, tight, SFC.AutoCulling()) === nothing
    Test.@test_throws ArgumentError SFC.cull_grid_for(xc, g, tight, SFC.AlwaysCulling())
end

"""Positions, velocities, edges and keywords: a unit box, or a sphere patch across the antimeridian."""
function _cull_inputs(points, edges, N)
    if points === :sphere
        lon = mod.(350 .+ 20 .* rand(N), 360) .- 180
        lat = 20 .* rand(N) .- 10
        return permutedims(hcat(lon, lat)), randn(2, N), collect(range(1.0e4, 3.0e5; length = 9)),
               (; distance_metric = DI.Haversine(6.371e6))
    end
    D = points === :flat2 ? 2 : points === :flat3 ? 3 : 4
    r_max = edges === :wide ? 1.5 : D == 2 ? 0.08 : 0.2
    return rand(D, N), randn(D, N), collect(range(0.0, r_max; length = 9)), (;)
end

"""Sums and counts of a public `entry` under the culling policy `pol` on `backend`."""
function _cull_entry(entry, x, u, bins, kw, pol, backend)
    sf, vbins = SFT.L2SFType(), collect(range(-4.0, 4.0; length = 9))
    nb, nv, n6 = length(bins) - 1, length(vbins) - 1, SFC.SINGLE_PASS_N
    if entry === :sf1d
        r = SFC.calculate_structure_function(sf, x, u, bins, _RAW; backend, culling = pol, kw...)
        return r.sums, r.counts
    elseif entry === :joint
        r = SFC.calculate_structure_function(sf, x, u, bins, vbins; backend, culling = pol, kw...)
        return r.sums, r.counts
    elseif entry === :sp1d
        return SFC.calculate_structure_functions_single_pass!(zeros(n6, nb), zeros(UInt32, n6, nb), x, u, bins;
                                                              backend, culling = pol, kw...)
    else
        return SFC.calculate_structure_functions_single_pass_2d!(zeros(n6, nb, nv), zeros(UInt32, n6, nb, nv), x, u,
                                                                 bins, vbins; backend, culling = pol, kw...)
    end
end

const _PUBLIC_CULL_CASES = (
    (entry = :sf1d, points = :flat2, edges = :near, pol = SFC.AutoCulling(), backend = _THREADED),
    (entry = :sp2d, points = :flat2, edges = :near, pol = SFC.AutoCulling(), backend = _SERIAL),
    (entry = :sp1d, points = :flat3, edges = :near, pol = SFC.AlwaysCulling(), backend = _SERIAL),
    (entry = :joint, points = :sphere, edges = :near, pol = SFC.AlwaysCulling(), backend = _SERIAL),
    (entry = :joint, points = :flat4, edges = :near, pol = SFC.AutoCulling(), backend = _SERIAL),
    (entry = :sf1d, points = :flat4, edges = :wide, pol = SFC.AlwaysCulling(), backend = _SERIAL),
)

# Each public entry gives the same sums and counts culled as unculled, in 2- to 4-D flat boxes and on a sphere.
Test.@testset "culling does not change a public entry's result" begin
    Random.seed!(4)
    N = 200
    for (; entry, points, edges, pol, backend) in _PUBLIC_CULL_CASES
        x, u, bins, kw = _cull_inputs(points, edges, N)
        s_ref, c_ref = _cull_entry(entry, x, u, bins, kw, SFC.NoCulling(), _SERIAL)
        s, c = _cull_entry(entry, x, u, bins, kw, pol, backend)
        Test.@test c == c_ref && isapprox(s, s_ref; rtol = 1e-9, atol = 1e-12)
    end
end

"""One batch entry's sums and counts over `B` slices, under the culling policy `pol` on backend `be`."""
function _batch_cull_run(entry, x, u, w, bins, vbins, B, pol, be)
    FT = Float64
    CT = w === nothing ? UInt32 : FT
    nb, nv = length(bins) - 1, length(vbins) - 1
    L2 = SFT.L2SFType()
    if entry === :sf1d
        s, c = zeros(FT, nb, B), zeros(CT, nb, B)
        SFC.calculate_structure_function_batch!(s, c, L2, x, u, bins; backend = be, culling = pol, weights = w)
    elseif entry === :joint
        s, c = zeros(FT, nb, nv, B), zeros(CT, nb, nv, B)
        SFC.calculate_structure_function_2d_batch!(s, c, L2, x, u, bins, vbins; backend = be, culling = pol,
                                                   weights = w)
    elseif entry === :sp1d
        s, c = zeros(FT, SFC.SINGLE_PASS_N, nb, B), zeros(CT, SFC.SINGLE_PASS_N, nb, B)
        SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, bins; backend = be, culling = pol, weights = w)
    else
        s, c = zeros(FT, SFC.SINGLE_PASS_N, nb, nv, B), zeros(CT, SFC.SINGLE_PASS_N, nb, nv, B)
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, x, u, bins, vbins; backend = be, culling = pol,
                                                                weights = w)
    end
    return s, c
end

"""The same histograms, one unculled serial point-entry call per slice."""
function _batch_slice_oracle(entry, x, u, w, bins, vbins, B)
    FT = Float64
    CT = w === nothing ? UInt32 : FT
    nb, nv = length(bins) - 1, length(vbins) - 1
    L2 = SFT.L2SFType()
    kw = (; backend = CB.SerialBackend(), culling = SFC.NoCulling(), weights = w)
    dims = entry === :sf1d ? (nb,) : entry === :joint ? (nb, nv) :
           entry === :sp1d ? (SFC.SINGLE_PASS_N, nb) : (SFC.SINGLE_PASS_N, nb, nv)
    s, c = zeros(FT, dims..., B), zeros(CT, dims..., B)
    for t in 1:B
        xt = ndims(x) == 2 ? x : x[:, :, t]
        ut = u[:, :, t]
        st, ct = selectdim(s, ndims(s), t), selectdim(c, ndims(c), t)
        if entry === :sf1d
            SFC.calculate_structure_function!(st, ct, L2, xt, ut, bins; kw...)
        elseif entry === :joint
            SFC.calculate_structure_function!(st, ct, L2, xt, ut, bins, vbins; kw...)
        elseif entry === :sp1d
            SFC.calculate_structure_functions_single_pass!(st, ct, xt, ut, bins; kw...)
        else
            SFC.calculate_structure_functions_single_pass_2d!(st, ct, xt, ut, bins, vbins; kw...)
        end
    end
    return s, c
end

const _BATCH_CULL_CASES = (
    (entry = :sf1d, D = 2, fixed = true, weighted = true, be = _SERIAL, pol = SFC.AlwaysCulling()),
    (entry = :joint, D = 3, fixed = false, weighted = false, be = _THREADED, pol = SFC.AutoCulling()),
    (entry = :sp1d, D = 2, fixed = true, weighted = false, be = _SERIAL, pol = SFC.AutoCulling()),
    (entry = :sp2d, D = 2, fixed = true, weighted = true, be = _SERIAL, pol = SFC.AlwaysCulling()),
)

# Each batch entry, culled, equals one unculled point-entry call per slice, on shared and on per-slice positions.
Test.@testset "batch kernels cull without changing the result" begin
    FT = Float64
    N, B = 150, 2
    bins = collect(FT, range(0.0, 0.08; length = 9))
    vbins = collect(FT, range(-0.02, 0.05; length = 11))
    for (; entry, D, fixed, weighted, be, pol) in _BATCH_CULL_CASES
        Random.seed!(990 + D + 2fixed + 4weighted)
        x = fixed ? rand(FT, D, N) : rand(FT, D, N, B)
        u = randn(FT, D, N, B) .* 0.1
        w = weighted ? rand(FT, N) .+ 0.5 : nothing
        s_ref, c_ref = _batch_slice_oracle(entry, x, u, w, bins, vbins, B)
        s, c = _batch_cull_run(entry, x, u, w, bins, vbins, B, pol, be)
        Test.@test (weighted ? isapprox(c, c_ref; rtol = 1e-12) : c == c_ref) &&
                   isapprox(s, s_ref; rtol = 1e-9, atol = 1e-12)
    end
end

const _TENSOR_CULL_CASES = (
    (P = 2, layout = :point, be = _SERIAL, pol = SFC.AlwaysCulling()),
    (P = 3, layout = :shared, be = _THREADED, pol = SFC.AutoCulling()),
    (P = 2, layout = :varying, be = _SERIAL, pol = SFC.AutoCulling()),
)

# The moment tensor of point, shared-position and per-slice-position fields is the same culled as unculled.
Test.@testset "tensor kernels cull without changing the result" begin
    Random.seed!(368)
    N, B = 150, 2
    bins = collect(range(0.0, 0.08; length = 9))
    TRAW = StructureFunctions.StructureFunctionObjects.StructureFunctionTensorSumsAndCounts
    for (; P, layout, be, pol) in _TENSOR_CULL_CASES
        x = layout === :varying ? rand(2, N, B) : rand(2, N)
        u = layout === :point ? randn(2, N) : randn(2, N, B)
        run(culling, backend) = SFC.calculate_structure_function_tensor(Val(P), x, u, bins, TRAW; backend, culling)
        r = run(SFC.NoCulling(), _SERIAL)
        got = run(pol, be)
        Test.@test got.counts == r.counts && isapprox(got.sums, r.sums; rtol = 1e-9, atol = 1e-12)
    end
end

# Block ids map row by row onto tile pairs ti ≤ tj, for Int32 ids too, and exactly past Float64's exact integers.
Test.@testset "tile_for enumerates exactly the upper triangle" begin
    ns = (1, 2, 7, 33)
    nblocks(n) = n * (n + 1) ÷ 2
    Test.@test all(n -> SFC.n_pair_blocks(SFC.FullUpperTriangle(n)) == nblocks(n), ns)
    Test.@test all(n -> [SFC.tile_for(SFC.FullUpperTriangle(n), k) for k in 1:nblocks(n)] ==
                        [(ti, tj) for ti in 1:n for tj in ti:n], ns)
    Test.@test all(n -> all(k -> SFC.tile_for(SFC.FullUpperTriangle(n), Int32(k)) ===
                                 Int32.(SFC.tile_for(SFC.FullUpperTriangle(n), k)), 1:nblocks(n)), ns)
    n = 2^27
    s = SFC.FullUpperTriangle(n)
    row_start(ti) = (ti - 1) * n - (ti - 1) * (ti - 2) ÷ 2 + 1
    Test.@test all(ti -> SFC.tile_for(s, row_start(ti)) == (ti, ti) &&
                         SFC.tile_for(s, row_start(ti) - 1) == (ti - 1, n), (2, 3, n ÷ 2, n - 1, n))
end

# A work list hands back, in order, the tile pairs packed into it.
Test.@testset "TilePairWorkList unpacks what pack_tile_pair packs" begin
    n = 41
    pairs = [(ti, tj) for tj in 1:n for ti in 1:tj if (ti * 7 + tj) % 3 == 0]
    s = SFC.TilePairWorkList(Int32[SFC.pack_tile_pair(ti, tj, n) for (ti, tj) in pairs], Int32(n))
    Test.@test SFC.n_pair_blocks(s) == length(pairs)
    Test.@test all(k -> SFC.tile_for(s, k) == pairs[k], eachindex(pairs))
end
