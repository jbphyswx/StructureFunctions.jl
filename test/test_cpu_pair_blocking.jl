using Test: Test
using StructureFunctions: StructureFunctions, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH
using Random: Random
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB

const _SERIAL = CB.SerialBackend()
const _RAW = StructureFunctions.StructureFunctionSumsAndCounts
const _OP = SFT.L2SFType()

_comp(x) = ntuple(d -> x[d, :], Val(2))

"""Per-operator sums and counts over `edges` of the pairs `(i, j > i)`, `i ∈ ilist`, by brute force."""
function _brute_histogram(ops, x, u, edges, ilist)
    D, N = size(x)
    nb = length(edges) - 1
    s, c = zeros(length(ops), nb), zeros(Int, nb)
    for i in ilist, j in (i + 1):N
        dx = SA.SVector{D}(ntuple(d -> x[d, j] - x[d, i], D))
        r = sqrt(sum(abs2, dx))
        b = searchsortedfirst(edges, r) - 1
        1 <= b <= nb || continue
        δu = SA.SVector{D}(ntuple(d -> u[d, j] - u[d, i], D))
        s[:, b] .+= [op(δu, dx / r) for op in ops]
        c[b] += 1
    end
    return s, c
end

# Under every tile and outer index list the pair kernels sum exactly the pairs they are given; a buffer shorter than a
# block is refused.
Test.@testset "pair kernels give the brute-force histogram under every block schedule" begin
    N = 40
    Random.seed!(4242)
    x, u = rand(2, N), randn(2, N)
    edges = collect(range(0.0, 1.6; length = 17))
    plan = SFC.squared_digitize_plan(edges)
    nb = SFC.n_histogram_bins(plan)
    bufs(L) = (zeros(L), zeros(L), zeros(Int32, L), zeros(Int32, L))
    for (tile, stride) in ((1, 1), (7, 1), (80, 3))
        ilist = 1:stride:(N - 1)
        s, c = zeros(nb), zeros(UInt32, nb)
        SFC._pf_simd_pairs!(s, c, _OP, _comp(x), _comp(u), plan, Val(2), bufs(min(N, tile))...,
                            SFC.pair_blocks(N, ilist; tile), SFC.NoWeights())
        sref, cref = _brute_histogram((_OP,), x, u, edges, ilist)
        Test.@test c == cref && isapprox(s, vec(sref); rtol = 1e-12, atol = 1e-14)
    end
    spbufs(L) = (zeros(L), zeros(L), zeros(L), zeros(Int32, L), zeros(Int32, L))
    for (tile, stride) in ((13, 1), (39, 3))
        ilist = 1:stride:(N - 1)
        s, c = zeros(SFC.SINGLE_PASS_N, nb), zeros(UInt32, SFC.SINGLE_PASS_N, nb)
        SFC._pf_sp_simd_pairs!(s, c, _comp(x), _comp(u), plan, Val(2), spbufs(min(N, tile))...,
                               SFC.pair_blocks(N, ilist; tile))
        sref, cref = _brute_histogram(values(SFC.SINGLE_PASS_OPERATORS), x, u, edges, ilist)
        Test.@test all(==(cref), eachrow(c)) && isapprox(s, sref; rtol = 1e-12, atol = 1e-14)
    end
    Test.@test_throws ArgumentError SFC._pf_sp_simd_pairs!(
        zeros(SFC.SINGLE_PASS_N, nb), zeros(UInt32, SFC.SINGLE_PASS_N, nb), _comp(x), _comp(u), plan, Val(2),
        spbufs(N - 1)..., SFC.pair_blocks(N, 1:(N - 1); tile = N))
end

# A culled schedule, whole or split over parts of its outer indices (an empty part, consecutive runs, strides, an index
# vector), sweeps every pair within the cutoff exactly once.
Test.@testset "culled block schedules sweep every in-range pair once" begin
    N, cut = 150, 0.1
    Random.seed!(2601)
    xc = (rand(N), rand(N))
    for (span, parts) in ((1, [1:0, 1:(N - 1)]), (2, [k:min(k + 6, N - 1) for k in 1:7:(N - 1)]),
                          (1, [k:13:(N - 1) for k in 1:13]), (3, [collect(k:5:(N - 1)) for k in 1:5]))
        grid = SFC.build_cell_grid(xc, cut, span)
        xp = SFC.apply_perm(xc, grid.perm)
        near = [(i, j) for i in 1:(N - 1) for j in (i + 1):N if (xp[1][i] - xp[1][j])^2 + (xp[2][i] - xp[2][j])^2 <= cut^2]
        swept = [(i, j) for irange in parts for (ir, jr) in SFC.pair_blocks(N, irange; grid) for i in ir for j in jr
                 if j > i]
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

# An unbounded last bin, a geometry with no bound, or a cutoff whose grid has more cells than an Int counts, is swept
# whole by AutoCulling and refused by AlwaysCulling.
Test.@testset "culling declines where it cannot skip pairs" begin
    Random.seed!(11)
    N = 20
    x, u = rand(2, N), randn(2, N)
    tight = collect(range(0.0, 0.05; length = 9))
    unbounded = StructureFunctions.InfPaddedBinEdges(tight)
    padded(pol) = SFC.calculate_structure_function(_OP, x, u, unbounded, _RAW; backend = _SERIAL, culling = pol)
    Test.@test sum(padded(SFC.AutoCulling()).counts) == N * (N - 1) ÷ 2
    Test.@test_throws ArgumentError padded(SFC.AlwaysCulling())
    xc = _comp(x)
    Test.@test SFC.cull_grid_for(xc, UnboundedTestGeometry{2}(), tight, SFC.AutoCulling()) === nothing
    Test.@test_throws ArgumentError SFC.cull_grid_for(xc, UnboundedTestGeometry{2}(), tight, SFC.AlwaysCulling())
    far, fine = (1e7 .* rand(N), 1e7 .* rand(N)), collect(range(0.0, 1e-3; length = 3))
    Test.@test SFC.cull_grid_for(far, SFH.FlatGeometry{2}(), fine, SFC.AutoCulling()) === nothing
    Test.@test_throws ArgumentError SFC.cull_grid_for(far, SFH.FlatGeometry{2}(), fine, SFC.AlwaysCulling())
end

"""Positions, velocities, edges and keywords: a unit box of `D` dimensions, or a sphere patch across the antimeridian."""
function _cull_inputs(points, N)
    if points === :sphere
        lon = mod.(350 .+ 20 .* rand(N), 360) .- 180
        lat = 20 .* rand(N) .- 10
        return permutedims(hcat(lon, lat)), randn(2, N), collect(range(1.0e4, 3.0e5; length = 9)),
               (; distance_metric = DI.Haversine(6.371e6))
    end
    D = points === :flat2 ? 2 : 4
    return rand(D, N), randn(D, N), collect(range(0.0, D == 2 ? 0.08 : 0.2; length = 9)), (;)
end

"""Sums and counts of a public `entry` under the culling policy `pol` on `backend`."""
function _cull_entry(entry, x, u, bins, kw, pol, backend)
    vbins = collect(range(-4.0, 4.0; length = 9))
    if entry === :sf1d
        r = SFC.calculate_structure_function(_OP, x, u, bins, _RAW; backend, culling = pol, kw...)
        return r.sums, r.counts
    elseif entry === :joint
        r = SFC.calculate_structure_function(_OP, x, u, bins, vbins; backend, culling = pol, kw...)
        return r.sums, r.counts
    end
    r = entry === :sp1d ? SFC.calculate_structure_functions_single_pass(x, u, bins, _RAW; backend, culling = pol, kw...) :
        SFC.calculate_structure_functions_single_pass_2d(x, u, bins, vbins; backend, culling = pol, kw...)
    invs = (:S2, :L2, :T2, :S3, :L3, :L1T2)
    return stack(r[k].sums for k in invs), stack(r[k].counts for k in invs)
end

# Each public entry gives the same sums and counts culled as unculled: every family on the 2-D SIMD kernels, and the
# scalar kernels of a 4-D box and of a sphere.
const _PUBLIC_CULL_CASES = (
    (entry = :sf1d, points = :flat2), (entry = :joint, points = :flat2), (entry = :sp1d, points = :flat2),
    (entry = :sp2d, points = :flat2), (entry = :sf1d, points = :flat4), (entry = :joint, points = :flat4),
    (entry = :sp2d, points = :flat4), (entry = :sp1d, points = :sphere), (entry = :joint, points = :sphere),
)

Test.@testset "culling does not change a public entry's result" begin
    Random.seed!(4)
    N = 100
    for (; entry, points) in _PUBLIC_CULL_CASES
        x, u, bins, kw = _cull_inputs(points, N)
        s_ref, c_ref = _cull_entry(entry, x, u, bins, kw, SFC.NoCulling(), _SERIAL)
        s, c = _cull_entry(entry, x, u, bins, kw, SFC.AlwaysCulling(), _SERIAL)
        Test.@test c == c_ref && isapprox(s, s_ref; rtol = 1e-9, atol = 1e-12)
    end
end

"""One batch entry's sums and counts over `B` slices under the culling policy `pol` on backend `be`."""
function _batch_cull_run(entry, x, u, w, bins, vbins, B, pol, be)
    CT = w === nothing ? UInt32 : Float64
    kw = w === nothing ? (; backend = be, culling = pol) : (; backend = be, culling = pol, weights = w)
    nb, nv = length(bins) - 1, length(vbins) - 1
    dims = entry === :sf1d ? (nb,) : entry === :joint ? (nb, nv) :
           entry === :sp1d ? (SFC.SINGLE_PASS_N, nb) : (SFC.SINGLE_PASS_N, nb, nv)
    s, c = zeros(dims..., B), zeros(CT, dims..., B)
    if entry === :sf1d
        SFC.calculate_structure_function_batch!(s, c, _OP, x, u, bins; kw...)
    elseif entry === :joint
        SFC.calculate_structure_function_2d_batch!(s, c, _OP, x, u, bins, vbins; kw...)
    elseif entry === :sp1d
        SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, bins; kw...)
    else
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, x, u, bins, vbins; kw...)
    end
    return s, c
end

# Each batch family gives the same result culled as unculled: one grid for shared positions, one per slice for
# per-slice positions, and the pair weights sorted with the points.
const _BATCH_CULL_CASES = (
    (entry = :sf1d, shared = true, weighted = true),
    (entry = :joint, shared = false, weighted = false),
    (entry = :sp1d, shared = true, weighted = false),
    (entry = :sp2d, shared = false, weighted = false),
)

Test.@testset "batch kernels cull without changing the result" begin
    N, B = 150, 2
    bins = collect(range(0.0, 0.08; length = 9))
    vbins = collect(range(-0.02, 0.05; length = 11))
    for (; entry, shared, weighted) in _BATCH_CULL_CASES
        Random.seed!(990 + 2shared + 4weighted)
        x = shared ? rand(2, N) : rand(2, N, B)
        u = randn(2, N, B) .* 0.1
        w = weighted ? rand(N) .+ 0.5 : nothing
        s_ref, c_ref = _batch_cull_run(entry, x, u, w, bins, vbins, B, SFC.NoCulling(), _SERIAL)
        s, c = _batch_cull_run(entry, x, u, w, bins, vbins, B, SFC.AlwaysCulling(), _SERIAL)
        Test.@test (weighted ? isapprox(c, c_ref; rtol = 1e-12) : c == c_ref) &&
                   isapprox(s, s_ref; rtol = 1e-9, atol = 1e-12)
    end
end

# The moment tensor of point, shared-position and per-slice-position fields is the same culled as unculled.
Test.@testset "tensor kernels cull without changing the result" begin
    Random.seed!(368)
    N, B = 150, 2
    bins = collect(range(0.0, 0.08; length = 9))
    TRAW = StructureFunctions.StructureFunctionObjects.StructureFunctionTensorSumsAndCounts
    for layout in (:point, :shared, :varying)
        x = layout === :varying ? rand(2, N, B) : rand(2, N)
        u = layout === :point ? randn(2, N) : randn(2, N, B)
        run(culling) = SFC.calculate_structure_function_tensor(Val(2), x, u, bins, TRAW; backend = _SERIAL, culling)
        r = run(SFC.NoCulling())
        got = run(SFC.AlwaysCulling())
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
