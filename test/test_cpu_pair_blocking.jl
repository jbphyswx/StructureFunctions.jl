using Test: Test
using StructureFunctions: StructureFunctions, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH, LinearBinEdges, LogBinEdges
using Random: Random
using Distances: Distances as DI
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads     # loads the threaded extension used below

# Run the pair kernel over an explicit block schedule, returning the histogram.
function _run_blocks(sf, xc, uc, bins, ::Val{D}, blocks, N, ::Type{FT}) where {D, FT}
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    s = zeros(FT, nb); c = zeros(UInt32, nb)
    SFC._pf_simd_pairs!(s, c, sf, xc, uc, plan, Val(D),
        Vector{FT}(undef, N), Vector{FT}(undef, N), Vector{Int32}(undef, N), Vector{Int32}(undef, N), SFC.WholeRun(),
        blocks, SFC.NoWeights())
    return s, c
end

_comp(x, D) = ntuple(d -> collect(view(x, d, :)), D)

Test.@testset "CPU pair loop is invariant to the block schedule" begin
    for D in (2, 3), N in (37, 500, 1000), sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.S2SFType())
        Random.seed!(4242 + N + D)
        FT = Float64
        x = rand(FT, D, N); u = randn(FT, D, N)
        xc, uc = _comp(x, D), _comp(u, D)
        for bins in (LinearBinEdges(range(FT(0.0), FT(1.6); length = 17)),
                     LogBinEdges(FT(0.02), FT(1.6), 17))
            ilist = 1:(N - 1)
            sref, cref = _run_blocks(sf, xc, uc, bins, Val(D),
                SFC.pair_blocks(N, ilist; tile = N), N, FT)
            for tile in (1, 7, 64, 333, N - 1, N, 2N)
                s, c = _run_blocks(sf, xc, uc, bins, Val(D),
                    SFC.pair_blocks(N, ilist; tile = tile), N, FT)
                Test.@test c == cref
                Test.@test isapprox(s, sref; rtol = 1e-12, atol = 1e-14)
            end
            # a strided outer list must block identically too
            il2 = 1:3:(N - 1)
            s2r, c2r = _run_blocks(sf, xc, uc, bins, Val(D), SFC.pair_blocks(N, il2; tile = N), N, FT)
            s2, c2 = _run_blocks(sf, xc, uc, bins, Val(D), SFC.pair_blocks(N, il2; tile = 64), N, FT)
            Test.@test c2 == c2r
            Test.@test isapprox(s2, s2r; rtol = 1e-12, atol = 1e-14)
        end
    end
    let N = 400, D = 2, FT = Float64
        Random.seed!(9)
        x = rand(FT, D, N); u = randn(FT, D, N)
        xc, uc = _comp(x, D), _comp(u, D)
        bins = LinearBinEdges(range(FT(0.0), FT(10.0); length = 5))   # wide: captures every pair
        _, c = _run_blocks(SFT.L2SFType(), xc, uc, bins, Val(D),
            SFC.pair_blocks(N, 1:(N - 1); tile = 37), N, FT)
        Test.@test sum(c) == N * (N - 1) ÷ 2
    end
end

Test.@testset "culling does not change the answer" begin
    FT = Float64
    for D in (2, 3), r_max in (FT(0.5), FT(0.2), FT(0.07)), span in (1, 2, 3)
        Random.seed!(31 + D + round(Int, 1000r_max) + span)
        N = 1200
        x = rand(FT, D, N); u = randn(FT, D, N)
        bins = LinearBinEdges(range(FT(0.0), r_max; length = 13))
        sf = SFT.L2SFType()

        xc, uc = _comp(x, D), _comp(u, D)
        sref, cref = _run_blocks(sf, xc, uc, bins, Val(D),
            SFC.pair_blocks(N, 1:(N - 1)), N, FT)

        cutoff = SFC.cull_cutoff(SFH.FlatGeometry{D}(), r_max)
        grid = SFC.build_cell_grid(xc, cutoff, span)
        xp = SFC.apply_perm(xc, grid.perm)
        up = SFC.apply_perm(uc, grid.perm)
        s, c = _run_blocks(sf, xp, up, bins, Val(D),
            SFC.pair_blocks(N, 1:(N - 1); grid = grid), N, FT)

        Test.@test c == cref                                  # counts must be exactly equal
        Test.@test isapprox(s, sref; rtol = 1e-9, atol = 1e-12)
        Test.@test sum(c) < N * (N - 1) ÷ 2                   # and it really culled something
    end
end

Test.@testset "cull_cutoff bounds the true separation" begin
    # On a shell the kernel sees ambient unit positions, so the cutoff is the unit chord.
    R = 6.371e6
    g = SFH.SphericalGeometry{2}(DI.Haversine(R), R)
    for r_max in (1.0e4, 1.0e6, 5.0e6)
        cut = SFC.cull_cutoff(g, r_max)
        σ = r_max / R
        Test.@test cut ≈ 2 * sin(σ / 2)
        # a pair at exactly r_max must sit at or inside the cutoff
        p = [1.0, 0.0, 0.0]
        q = [cos(σ), sin(σ), 0.0]
        Test.@test sqrt(sum(abs2, q .- p)) <= cut + 1e-12
    end
    Test.@test SFC.cull_cutoff(g, π * R) == 2.0               # saturates at the diameter
    Test.@test SFC.cull_cutoff(SFH.FlatGeometry{2}(), 3.5) == 3.5
end

struct UnboundedTestGeometry{D} end
SFH.coordinate_width(::UnboundedTestGeometry{D}) where {D} = Val(D)

Test.@testset "a geometry declaring no bound is not required to have one" begin
    # Culling must stay opt-in: a user-defined geometry with no `cull_cutoff` method has to keep
    # working (declining to cull), not error. Requiring the method is how the geometry extension
    # point gets broken.
    Test.@test SFC.cull_cutoff(UnboundedTestGeometry{2}(), 1.0) === nothing
    Random.seed!(11)
    xc = (collect(rand(400)), collect(rand(400)))
    bins = LinearBinEdges(range(0.0, 0.05; length = 9))
    g = UnboundedTestGeometry{2}()
    Test.@test SFC.cull_grid_for(xc, g, bins, SFC.AutoCulling()) === nothing
    Test.@test SFC.cull_grid_for(xc, g, bins, SFC.NoCulling()) === nothing
    # ...but an explicit demand must say why it cannot be honoured
    Test.@test_throws ArgumentError SFC.cull_grid_for(xc, g, bins, SFC.AlwaysCulling())
end

Test.@testset "culling policy" begin
    Random.seed!(1)
    x = rand(2, 5000)
    xc = (collect(view(x, 1, :)), collect(view(x, 2, :)))
    g = SFH.FlatGeometry{2}()
    tight = LinearBinEdges(range(0.0, 0.05; length = 9))    # stencil much smaller than the grid
    wide = LinearBinEdges(range(0.0, 0.6; length = 9))      # stencil spans the grid
    unbounded = StructureFunctions.InfPaddedBinEdges(tight)

    Test.@test SFC.cull_grid_for(xc, g, tight, SFC.NoCulling()) === nothing
    Test.@test SFC.cull_grid_for(xc, g, tight, SFC.AutoCulling()) !== nothing
    # a cutoff spanning the grid removes no pairs, so :auto declines and pays for no sort
    Test.@test SFC.cull_grid_for(xc, g, wide, SFC.AutoCulling()) === nothing
    Test.@test SFC.cull_grid_for(xc, g, wide, SFC.AlwaysCulling()) !== nothing

    # An unbounded last bin is reported, so no pair can be skipped.
    Test.@test SFC.cull_grid_for(xc, g, unbounded, SFC.AutoCulling()) === nothing
    Test.@test_throws ArgumentError SFC.cull_grid_for(xc, g, unbounded, SFC.AlwaysCulling())

    grid = SFC.cull_grid_for(xc, g, tight, SFC.AutoCulling())
    Test.@test sort(grid.perm) == collect(1:5000)          # a permutation, nothing dropped
    Test.@test grid.run_starts[end] == 5001                # every point landed in a cell
    Test.@test SFC.n_occupied_cells(grid) <= 5000           # only occupied cells are stored
    Test.@test issorted(grid.cell_ids)                      # the run lookup relies on this
    # the occupied runs must tile 1:N exactly, with no gap or overlap
    Test.@test vcat([collect(SFC.occupied_cell(grid, k)[2])
                     for k in 1:SFC.n_occupied_cells(grid)]...) == collect(1:5000)
end

# The wired entry, on bins tight enough that AutoCulling actually engages — the default path in
# the rest of the suite uses wide bins, where AutoCulling declines and the culled code never runs.
Test.@testset "_pf_simd_partial! culls without changing the result" begin
    FT = Float64
    for D in (2, 3)
        Random.seed!(808 + D)
        N = 3000
        x = rand(FT, D, N)
        u = randn(FT, D, N)
        xv = ntuple(d -> view(x, d, :), D)
        uv = ntuple(d -> view(u, d, :), D)
        g = SFH.FlatGeometry{D}()
        bins = LinearBinEdges(range(FT(0.0), FT(0.08); length = 9))
        nb = SFC.n_histogram_bins(bins)

        # confirm the policy really engages here, else this testset proves nothing
        xc = ntuple(d -> collect(xv[d]), D)
        Test.@test SFC.cull_grid_for(xc, g, bins, SFC.AutoCulling()) !== nothing

        s_ref = zeros(FT, nb); c_ref = zeros(UInt32, nb)
        SFC._pf_simd_partial!(s_ref, c_ref, SFT.L2SFType(), xv, uv, bins, Val(D), nothing,
            SFC.NoCulling(); geometry = g)
        s_cull = zeros(FT, nb); c_cull = zeros(UInt32, nb)
        SFC._pf_simd_partial!(s_cull, c_cull, SFT.L2SFType(), xv, uv, bins, Val(D), nothing,
            SFC.AutoCulling(); geometry = g)

        Test.@test c_cull == c_ref
        Test.@test isapprox(s_cull, s_ref; rtol = 1e-9, atol = 1e-12)
        Test.@test sum(c_ref) > 0                          # the bins are not all empty
    end
end

Test.@testset "single-pass kernel is invariant to the block schedule" begin
    FT = Float64
    for D in (2, 3), (N, dmax) in ((41, 1.6), (700, 1.6), (3000, 0.08))
        Random.seed!(77 + N + D)
        x = rand(FT, D, N)
        u = randn(FT, D, N)
        bins = LinearBinEdges(range(FT(0.0), FT(dmax); length = 17))
        nb = SFC.n_histogram_bins(bins)
        engages = SFC.cull_grid_for(_comp(x, D), SFH.FlatGeometry{D}(), bins, SFC.AutoCulling()) !== nothing
        Test.@test engages == (dmax < 1)

        run(pol) = begin
            s = zeros(FT, SFC.SINGLE_PASS_N, nb); c = zeros(UInt32, SFC.SINGLE_PASS_N, nb)
            SFC._sp_simd_partial!(s, c, x, u, bins, Val(D), nothing, pol)
            (s, c)
        end
        s_ref, c_ref = run(SFC.NoCulling())
        s_cull, c_cull = run(SFC.AutoCulling())
        Test.@test c_cull == c_ref
        Test.@test isapprox(s_cull, s_ref; rtol = 1e-9, atol = 1e-12)
        Test.@test sum(view(c_ref, 1, :)) > 0

        # tiling alone (no culling): every tile size gives the same histogram, whether the buffers span
        # every point or one tile
        xc = _comp(x, D); uc = _comp(u, D)
        plan = SFC.squared_digitize_plan(bins)
        tile_run(tile, window, L) = begin
            s = zeros(FT, SFC.SINGLE_PASS_N, nb); c = zeros(UInt32, SFC.SINGLE_PASS_N, nb)
            SFC._pf_sp_simd_pairs!(s, c, xc, uc, plan, Val(D),
                Vector{FT}(undef, L), Vector{FT}(undef, L), Vector{FT}(undef, L),
                Vector{Int32}(undef, L), Vector{Int32}(undef, L), window, SFC.pair_blocks(N, 1:(N - 1); tile = tile))
            (s, c)
        end
        s0, c0 = tile_run(N, SFC.WholeRun(), N)
        for tile in (1, 13, 128, N), (window, L) in ((SFC.WholeRun(), N), (SFC.BlockRun(), tile))
            s1, c1 = tile_run(tile, window, L)
            Test.@test c1 == c0
            Test.@test isapprox(s1, s0; rtol = 1e-12, atol = 1e-14)
        end
        Test.@test_throws ArgumentError tile_run(128, SFC.BlockRun(), min(N, 128) - 1)
    end
end

Test.@testset "2D kernels are invariant to the block schedule" begin
    FT = Float64
    for D in (2, 3), (N, dmax) in ((43, 1.5), (600, 1.5), (3000, 0.08))
        Random.seed!(303 + N + D)
        x = rand(FT, D, N)
        u = randn(FT, D, N)
        dist = LinearBinEdges(range(FT(0.0), FT(dmax); length = 13))
        val = LinearBinEdges(range(FT(-4.0), FT(4.0); length = 11))
        n_dist = SFC.n_histogram_bins(dist)
        n_val = SFC.n_histogram_bins(val)
        engages = SFC.cull_grid_for(_comp(x, D), SFH.FlatGeometry{D}(), dist, SFC.AutoCulling()) !== nothing
        Test.@test engages == (dmax < 1)

        # joint 2D
        joint(pol) = begin
            s = zeros(FT, n_dist, n_val); c = zeros(UInt32, n_dist, n_val)
            SFC._pf_2d_simd_partial!(s, c, SFT.L2SFType(), _comp(x, D), _comp(u, D),
                dist, val, Val(D), nothing, pol)
            (s, c)
        end
        sj_ref, cj_ref = joint(SFC.NoCulling())
        sj, cj = joint(SFC.AutoCulling())
        Test.@test cj == cj_ref
        Test.@test isapprox(sj, sj_ref; rtol = 1e-9, atol = 1e-12)
        Test.@test sum(cj_ref) > 0

        # single-pass 2D
        vb = ntuple(_ -> val, SFC.SINGLE_PASS_N)
        sp2d(pol) = begin
            s = zeros(FT, SFC.SINGLE_PASS_N, n_dist, n_val); c = zeros(UInt32, SFC.SINGLE_PASS_N, n_dist, n_val)
            SFC._sp2d_accumulate_range!(s, c, x, u, dist, SFC.digitize_plan(vb), DI.Euclidean(), n_dist, n_val,
                                        nothing, pol)
            (s, c)
        end
        s_ref, c_ref = sp2d(SFC.NoCulling())
        s_cull, c_cull = sp2d(SFC.AutoCulling())
        Test.@test c_cull == c_ref
        Test.@test isapprox(s_cull, s_ref; rtol = 1e-9, atol = 1e-12)
        Test.@test sum(c_ref) > 0
    end
end

Test.@testset "batch kernels cull without changing the result" begin
    # Shared positions are sorted once for every slice, varying ones per slice; on bins tight enough
    # that the cull engages on every slice, every policy must count the same pairs.
    FT = Float64
    N, B = 1500, 3
    bins = collect(FT, range(0.0, 0.08; length = 9))
    vbins = collect(FT, range(-0.02, 0.05; length = 11))
    nb, nv = length(bins) - 1, length(vbins) - 1
    L2 = SFT.L2SFType()
    for D in (2, 3), fixed in (true, false), weighted in (false, true)
        Random.seed!(990 + D + 2fixed + 4weighted)
        x = fixed ? rand(FT, D, N) : rand(FT, D, N, B)
        u = randn(FT, D, N, B) .* 0.1
        w = weighted ? rand(FT, N) .+ 0.5 : nothing
        CT = weighted ? FT : UInt32
        slices = fixed ? (x,) : Tuple(x[:, :, b] for b in 1:B)
        Test.@test all(xs -> SFC.cull_grid_for(_comp(xs, D), SFH.FlatGeometry{D}(), bins,
                                               SFC.AutoCulling()) !== nothing, slices)
        runs = (
            (pol, be) -> (s = zeros(FT, nb, B); c = zeros(CT, nb, B);
                SFC.calculate_structure_function_batch!(s, c, L2, x, u, bins; backend = be, culling = pol,
                    weights = w); (s, c)),
            (pol, be) -> (s = zeros(FT, nb, nv, B); c = zeros(CT, nb, nv, B);
                SFC.calculate_structure_function_2d_batch!(s, c, L2, x, u, bins, vbins; backend = be,
                    culling = pol, weights = w); (s, c)),
            (pol, be) -> (s = zeros(FT, SFC.SINGLE_PASS_N, nb, B); c = zeros(CT, SFC.SINGLE_PASS_N, nb, B);
                SFC.calculate_structure_functions_single_pass_batch!(s, c, x, u, bins; backend = be,
                    culling = pol, weights = w); (s, c)),
            (pol, be) -> (s = zeros(FT, SFC.SINGLE_PASS_N, nb, nv, B);
                c = zeros(CT, SFC.SINGLE_PASS_N, nb, nv, B);
                SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, x, u, bins, vbins;
                    backend = be, culling = pol, weights = w); (s, c)),
        )
        for run in runs
            s_ref, c_ref = run(SFC.NoCulling(), CB.SerialBackend())
            Test.@test sum(c_ref) > 0
            for be in (CB.SerialBackend(), CB.ThreadedBackend()),
                pol in (SFC.NoCulling(), SFC.AlwaysCulling(), SFC.AutoCulling())
                s, c = run(pol, be)
                Test.@test weighted ? isapprox(c, c_ref; rtol = 1e-12) : c == c_ref
                Test.@test isapprox(s, s_ref; rtol = 1e-9, atol = 1e-12)
            end
        end
    end
end

Test.@testset "tensor kernels cull without changing the result" begin
    # Shared positions sorted once for every slice, varying ones per slice.
    N, B = 1500, 3
    bins = collect(range(0.0, 0.08; length = 9))
    TRAW = StructureFunctions.StructureFunctionObjects.StructureFunctionTensorSumsAndCounts
    for P in (2, 3), (x, u) in ((rand(2, N), randn(2, N)), (rand(2, N), randn(2, N, B)),
                                (rand(2, N, B), randn(2, N, B))),
        backend in (CB.SerialBackend(), CB.ThreadedBackend())
        run(pol) = SFC.calculate_structure_function_tensor(Val(P), x, u, bins, TRAW; backend, culling = pol)
        ref = run(SFC.NoCulling())
        Test.@test sum(ref.counts) > 0
        for pol in (SFC.AlwaysCulling(), SFC.AutoCulling())
            got = run(pol)
            Test.@test got.counts == ref.counts
            Test.@test isapprox(got.sums, ref.sums; rtol = 1e-9, atol = 1e-12)
        end
    end
end

Test.@testset "culling on a sphere does not change the answer" begin
    # The kernels see ambient unit positions, so the cull grid lives in 3-space around a 2-D
    # shell: cells vastly outnumber points, which is exactly what the sparse grid is for.
    FT = Float64
    R = 6.371e6
    for r_max in (3.0e5, 1.0e5)
        Random.seed!(515 + round(Int, r_max))
        N = 2500
        lon = rand(FT, N) .* 360 .- 180
        lat = rand(FT, N) .* 120 .- 60
        x = permutedims(hcat(lon, lat))
        u = randn(FT, 2, N)
        bins = collect(FT, range(1.0e4, r_max; length = 9))
        metric = DI.Haversine(R)
        nb = SFC.n_histogram_bins(bins)

        run(pol) = begin
            s = zeros(FT, SFC.SINGLE_PASS_N, nb); c = zeros(UInt32, SFC.SINGLE_PASS_N, nb)
            SFC._accumulate_single_pass_1d!(s, c, x, u, bins;
                distance_metric = metric, culling = pol)
            (s, c)
        end
        s_ref, c_ref = run(SFC.NoCulling())
        s_cull, c_cull = run(SFC.AutoCulling())

        # confirm the grid really engages, else this proves nothing
        geom = SFH.pair_geometry_for(metric, Val(2))
        xk, _ = SFH.prepare_pair_inputs(geom, x, u)
        g, _, _ = SFC.cull_sorted_matrices(xk, u, geom, SFC.BinEdges(bins), SFC.AutoCulling())
        Test.@test g !== nothing
        Test.@test prod(g.dims) > SFC.n_occupied_cells(g)   # sparse, as expected on a shell

        Test.@test c_cull == c_ref
        Test.@test isapprox(s_cull, s_ref; rtol = 1e-9, atol = 1e-12)
        Test.@test sum(view(c_ref, 1, :)) > 0
    end
end

Test.@testset "culling through the public entry" begin
    FT = Float64
    Random.seed!(4)
    N = 2500
    x = rand(FT, 2, N)
    u = randn(FT, 2, N)
    bins = collect(FT, range(0.0, 0.08; length = 9))
    sf = SFT.L2SFType()

    # Empty bins finalize to NaN, so agreement means the same NaN pattern and equal finite values.
    function same_result(a, b)
        isnan.(a.values) == isnan.(b.values) || return false
        keep = .!isnan.(a.values)
        any(keep) || return false
        return isapprox(a.values[keep], b.values[keep]; rtol = 1e-9, atol = 1e-12)
    end

    ref = SFC.calculate_structure_function(sf, x, u, bins;
        backend = CB.SerialBackend(), culling = SFC.NoCulling())
    cull = SFC.calculate_structure_function(sf, x, u, bins;
        backend = CB.SerialBackend(), culling = SFC.AutoCulling())
    Test.@test same_result(cull, ref)
    Test.@test cull.distance == ref.distance
    Test.@test any(.!isnan.(ref.values))               # the bins are not all empty

    thr = SFC.calculate_structure_function(sf, x, u, bins;
        backend = CB.ThreadedBackend(), culling = SFC.AutoCulling())
    Test.@test same_result(thr, ref)

    # The scalar kernels, for a curved metric and for a flat width past three, cull too.
    RAW = StructureFunctions.StructureFunctionSumsAndCounts
    lon = rand(N) .* 360 .- 180
    lat = rand(N) .* 120 .- 60
    xs = permutedims(hcat(lon, lat))
    us = randn(2, N)
    sbins = collect(range(1.0e4, 3.0e5; length = 9))
    vbins = collect(range(-4.0, 4.0; length = 9))
    m = DI.Haversine(6.371e6)
    x4, u4 = rand(4, N), randn(4, N)
    bins4 = collect(range(0.0, 0.2; length = 9))
    Test.@test SFC.cull_grid_for(_comp(x4, 4), SFH.FlatGeometry{4}(), bins4, SFC.AutoCulling()) !== nothing
    x64, u64 = rand(64, 600), randn(64, 600)
    bins64 = collect(range(0.0, 3.0; length = 9))
    for backend in (CB.SerialBackend(), CB.ThreadedBackend())
        for (xk, uk, bk, kw) in ((xs, us, sbins, (; distance_metric = m)), (x4, u4, bins4, (;)),
                                 (x64, u64, bins64, (;)))
            one_d(pol) = SFC.calculate_structure_function(sf, xk, uk, bk, RAW; backend, culling = pol,
                kw...)
            joint(pol) = SFC.calculate_structure_function(sf, xk, uk, bk, vbins; backend, culling = pol,
                kw...)
            ref1, ref2 = one_d(SFC.NoCulling()), joint(SFC.NoCulling())
            Test.@test sum(ref1.counts) > 0
            for pol in (SFC.AlwaysCulling(), SFC.AutoCulling())
                got1, got2 = one_d(pol), joint(pol)
                Test.@test got1.counts == ref1.counts
                Test.@test isapprox(got1.sums, ref1.sums; rtol = 1e-9, atol = 1e-12)
                Test.@test got2.counts == ref2.counts
                Test.@test isapprox(got2.sums, ref2.sums; rtol = 1e-9, atol = 1e-12)
            end
        end
    end
end
Test.@testset "tile_for enumerates exactly the upper triangle" begin
    # B2a: every GPU kernel maps its linear block id through this. It must be a bijection onto
    # {(ti, tj) : ti <= tj}, in the fixed order the kernels' block-private accumulators rely on.
    for n in (1, 2, 3, 7, 16, 33, 157, 512)
        s = SFC.FullUpperTriangle(n)
        nb = SFC.n_pair_blocks(s)
        Test.@test nb == n * (n + 1) ÷ 2
        seen = [SFC.tile_for(s, k) for k in 1:nb]
        Test.@test seen == [(ti, tj) for ti in 1:n for tj in ti:n]   # the exact row-by-row order
        # Int32 ids, as the device hands them out, give the same pairs
        Test.@test all(k -> SFC.tile_for(s, Int32(k)) === (Int32(seen[k][1]), Int32(seen[k][2])), 1:nb)
    end
    # a tile count past what the tests above can enumerate: every row boundary, exactly
    n = 40_000
    s = SFC.FullUpperTriangle(n)
    row_start(ti) = (ti - 1) * n - (ti - 1) * (ti - 2) ÷ 2 + 1
    Test.@test all(ti -> SFC.tile_for(s, row_start(ti)) == (ti, ti) &&
                         SFC.tile_for(s, row_start(ti) - 1) == (ti - 1, n), 2:n)
    Test.@test SFC.tile_for(s, SFC.n_pair_blocks(s)) == (n, n)
end

Test.@testset "TilePairWorkList unpacks what pack_tile_pair packs" begin
    n = 41
    pairs = [(ti, tj) for tj in 1:n for ti in 1:tj if (ti * 7 + tj) % 3 == 0]   # an arbitrary subset
    packed = Int32[SFC.pack_tile_pair(ti, tj, n) for (ti, tj) in pairs]
    s = SFC.TilePairWorkList(packed, Int32(n))
    Test.@test SFC.n_pair_blocks(s) == length(pairs)
    Test.@test all(k -> SFC.tile_for(s, k) == pairs[k], eachindex(pairs))
    Test.@test all(p -> 1 <= p <= n * n, packed)                # packing stays in one id space
end

Test.@testset "culled block schedule covers every in-range pair in one dimension" begin
    # One dimension has a single stencil row, reached along the sorted axis alone.
    FT = Float64
    N = 800
    Random.seed!(2601)
    xc = (rand(FT, N),)
    cut = FT(0.03)
    grid = SFC.build_cell_grid(xc, cut, 2)
    Test.@test length(grid.offsets) == 1
    xp = SFC.apply_perm(xc, grid.perm)
    covered = falses(N, N)
    for (ir, jr) in SFC.CulledBlockPairs(grid), i in ir, j in jr
        j > i && (covered[i, j] = true)
    end
    missed = count(((i, j),) -> abs(xp[1][i] - xp[1][j]) <= cut && !covered[i, j],
                   ((i, j) for i in 1:(N - 1) for j in (i + 1):N))
    Test.@test missed == 0
    Test.@test count(covered) < N * (N - 1) ÷ 2
end

Test.@testset "a culled schedule narrowed to parts of the outer indices sweeps the same pairs" begin
    FT = Float64
    N = 1500
    Random.seed!(2602)
    xc = (rand(FT, N), rand(FT, N))
    grid = SFC.build_cell_grid(xc, FT(0.05), 1)
    pairs_of(irange) = sort!([(i, j) for (ir, jr) in SFC.pair_blocks(N, irange; grid) for i in ir for j in jr if j > i])
    whole = pairs_of(1:(N - 1))
    Test.@test length(whole) < N * (N - 1) ÷ 2
    for parts in ([1:(N - 1)], [k:min(k + 6, N - 1) for k in 1:7:(N - 1)], [k:min(k + 99, N - 1) for k in 1:100:(N - 1)],
                  [k:13:(N - 1) for k in 1:13], [collect(k:5:(N - 1)) for k in 1:5], [1:0, 1:(N - 1)])
        Test.@test sort!(reduce(vcat, map(pairs_of, parts))) == whole
    end
end

