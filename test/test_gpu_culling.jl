using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH
using KernelAbstractions: KernelAbstractions as KA
using ComputationalBackends: ComputationalBackends as CB
using Distances: Distances as DI
using Random: Random

const BE = KA.CPU()
const SF1D = SFT.L2SFType()
const TIGHT = collect(range(0.0, 0.06; length = 9))          # AutoCulling engages on the unit square
const WIDE = collect(range(0.0, 1.5; length = 9))            # same bin count, other edges
const NB = length(TIGHT) - 1
const FLAT2 = SFH.FlatGeometry{2}()
const GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)

function run1d(x, u, bins, pol, ws)
    s = zeros(NB)
    c = zeros(UInt32, NB)
    SFC.gpu_calculate_structure_function!(s, c, SF1D, BE, x, u, bins; geometry = FLAT2, workspace = ws, culling = pol)
    return s, c
end

# A culled launch equals the full sweep exactly in counts, for the 1D, joint and single-pass 2D histograms.
Test.@testset "GPU culling, with and without a workspace" begin
    Random.seed!(77)
    N = 300
    x = rand(2, N)
    u = randn(2, N)

    s_ref, c_ref = run1d(x, u, TIGHT, SFC.NoCulling(), SFC.GPUSFWorkspace(BE, TIGHT))
    Test.@test sum(c_ref) > 0
    s_cull, c_cull = run1d(x, u, TIGHT, SFC.AutoCulling(), SFC.GPUSFWorkspace(BE, TIGHT))
    Test.@test c_cull == c_ref
    Test.@test isapprox(s_cull, s_ref; rtol = 1e-10, atol = 1e-12)
    s_nows, c_nows = run1d(x, u, TIGHT, SFC.AlwaysCulling(), nothing)
    Test.@test c_nows == c_ref
    Test.@test isapprox(s_nows, s_ref; rtol = 1e-10, atol = 1e-12)

    val = collect(range(-4.0, 4.0; length = 9))
    ref2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val, UInt32;
        geometry = FLAT2, workspace = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d), culling = SFC.NoCulling())
    got2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val, UInt32;
        geometry = FLAT2, workspace = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d), culling = SFC.AutoCulling())
    Test.@test got2.counts == ref2.counts
    Test.@test isapprox(got2.sums, ref2.sums; rtol = 1e-10, atol = 1e-12)

    vb = ntuple(_ -> val, SFC.SINGLE_PASS_N)
    n_val = length(val) - 1
    runsp(pol) = begin
        s = zeros(SFC.SINGLE_PASS_N, NB, n_val); c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, n_val)
        SFC.gpu_calculate_structure_functions_single_pass_2d!(s, c, BE, x, u, TIGHT, vb;
            geometry = FLAT2, workspace = SFC.GPUSFWorkspace(BE, TIGHT, vb; kind = :single_pass_2d), culling = pol)
        (s, c)
    end
    sp_s0, sp_c0 = runsp(SFC.NoCulling())
    sp_s1, sp_c1 = runsp(SFC.AutoCulling())
    Test.@test sp_c1 == sp_c0
    Test.@test isapprox(sp_s1, sp_s0; rtol = 1e-10, atol = 1e-12)
end

# A culling workspace gives the full sweep's answer on reuse, after refresh!, after release!, and refuses other bins.
Test.@testset "a culling workspace stays correct across reuse, refresh! and release!" begin
    Random.seed!(78)
    N = 300
    x = rand(2, N)
    u = randn(2, N)
    ref = SFC.GPUSFWorkspace(BE, TIGHT)
    ws = SFC.GPUSFWorkspace(BE, TIGHT)
    run1d(x, u, TIGHT, SFC.AutoCulling(), ws)

    u2 = randn(2, N)
    s_ref, c_ref = run1d(x, u2, TIGHT, SFC.NoCulling(), ref)
    s2, c2 = run1d(x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test c2 == c_ref
    Test.@test isapprox(s2, s_ref; rtol = 1e-10, atol = 1e-12)

    x .= rand(2, N)
    SFC.refresh!(ws)
    s_ref, c_ref = run1d(x, u2, TIGHT, SFC.NoCulling(), ref)
    s3, c3 = run1d(x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test c3 == c_ref
    Test.@test isapprox(s3, s_ref; rtol = 1e-10, atol = 1e-12)

    Test.@test_throws ArgumentError run1d(x, u2, WIDE, SFC.AutoCulling(), ws)

    xs = 0.1 .* x
    s_ref, c_ref = run1d(xs, u2, TIGHT, SFC.NoCulling(), ref)
    run1d(xs, u2, TIGHT, SFC.AutoCulling(), ws)
    s4, c4 = run1d(xs, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test c4 == c_ref
    Test.@test isapprox(s4, s_ref; rtol = 1e-10, atol = 1e-12)

    SFC.release!(ws)
    s_ref, c_ref = run1d(x, u2, TIGHT, SFC.NoCulling(), ref)
    s5, c5 = run1d(x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test c5 == c_ref
    Test.@test isapprox(s5, s_ref; rtol = 1e-10, atol = 1e-12)
end

# The device grid has the host grid's cells, runs and point order, over the widest axes when extents differ.
Test.@testset "the device cull grid equals the host grid" begin
    for (D, N, frac) in ((2, 300, 0.06), (3, 250, 0.15), (5, 200, 0.4))
        Random.seed!(2700 + N)
        x = rand(D, N) .* ntuple(d -> d, D)
        geom = SFH.FlatGeometry{D}()
        bins = collect(range(0.0, frac; length = 5))
        host = SFC.cull_grid_for(ntuple(d -> view(x, d, :), D), geom, bins, SFC.AlwaysCulling())
        dev = GE._gpu_cull_grid(BE, x, geom, SFC.cull_cutoff_for(geom, bins, SFC.AlwaysCulling()),
                                SFC.AlwaysCulling())
        Test.@test (D, dev.origin, dev.dims, dev.inv_h, dev.cell_ids, dev.run_starts, dev.perm, dev.offsets) ==
                   (D, host.origin, host.dims, host.inv_h, host.cell_ids, host.run_starts, host.perm, host.offsets)
    end
end

# On a sphere the cull grid is built over the unit positions with the chord of the largest angle as its cutoff.
Test.@testset "a sphere culls on the device by its chord bound" begin
    Random.seed!(81)
    N = 300
    x = vcat(2π .* rand(1, N), asin.(2 .* rand(1, N) .- 1))
    u = randn(2, N)
    bins = collect(range(0.0, 0.12; length = 9))
    nb = length(bins) - 1
    run(pol, be) = begin
        s, c = zeros(nb), zeros(UInt32, nb)
        SFC.calculate_structure_function!(s, c, SF1D, x, u, bins; backend = be, distance_metric = DI.SphericalAngle(),
                                          culling = pol)
        (s, c)
    end
    s_ser, c_ser = run(SFC.NoCulling(), CB.SerialBackend())
    s_cull, c_cull = run(SFC.AlwaysCulling(), CB.GPUBackend(BE))
    Test.@test sum(c_ser) > 0
    Test.@test c_cull == c_ser
    Test.@test isapprox(s_cull, s_ser; rtol = 1e-10)
end

# Every pair inside the cutoff, in the permuted order the tiles are cut from, has its canonical tile pair listed once.
Test.@testset "the device work list covers every in-range pair" begin
    Random.seed!(2600)
    for (D, N, frac, tile) in ((2, 300, 0.06, 128), (3, 300, 0.15, 64), (1, 300, 0.02, 32))
        x = rand(D, N)
        xc = ntuple(d -> x[d, :], D)
        cut = SFC.cull_cutoff(SFH.FlatGeometry{D}(), frac)
        grid = SFC.build_cell_grid(xc, cut, 2)
        xp = SFC.apply_perm(xc, grid.perm)
        wl = SFC.gpu_tile_worklist(grid, N, tile)
        n_tiles = cld(N, tile)
        canonical = issorted(wl.pairs) && allunique(wl.pairs) &&
                    all(k -> (1 <= SFC.tile_for(wl, k)[1] <= SFC.tile_for(wl, k)[2] <= n_tiles),
                        1:SFC.n_pair_blocks(wl))
        listed = Set(wl.pairs)
        missed = 0
        for i in 1:(N - 1), j in (i + 1):N
            r2 = sum(d -> (xp[d][i] - xp[d][j])^2, 1:D)
            r2 <= cut^2 || continue
            ti, tj = minmax(cld(i, tile), cld(j, tile))
            SFC.pack_tile_pair(eltype(wl.pairs)(ti), eltype(wl.pairs)(tj),
                               eltype(wl.pairs)(n_tiles)) in listed || (missed += 1)
        end
        Test.@test (D, tile, canonical, missed) == (D, tile, true, 0)
    end
end

"""Batch family `name`: output shape, workspace constructor, batch call, and serial call on one slice `(xt, ut)`."""
function cull_batch_family(name, x, u, val, B, kw)
    vb = ntuple(_ -> val, SFC.SINGLE_PASS_N)
    nv, NI = length(val) - 1, SFC.SINGLE_PASS_N
    ser = CB.SerialBackend()
    name === :batch1d && return ((NB, B), (() -> SFC.GPUSFWorkspace(BE, TIGHT)),
        ((s, c, be, ws, pol) -> SFC.calculate_structure_function_batch!(
            s, c, SF1D, x, u, TIGHT; backend = be, workspace = ws, culling = pol, kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_function!(s, c, SF1D, xt, ut, TIGHT; backend = ser, kw...)))
    name === :joint && return ((NB, nv, B), (() -> SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d)),
        ((s, c, be, ws, pol) -> SFC.calculate_structure_function_2d_batch!(
            s, c, SF1D, x, u, TIGHT, val; backend = be, workspace = ws, culling = pol, kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_function!(s, c, SF1D, xt, ut, TIGHT, val; backend = ser, kw...)))
    name === :sp1d && return ((NI, NB, B), (() -> SFC.GPUSFWorkspace(BE, TIGHT; kind = :single_pass)),
        ((s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_batch!(
            s, c, x, u, TIGHT; backend = be, workspace = ws, culling = pol, kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_functions_single_pass!(s, c, xt, ut, TIGHT; backend = ser, kw...)))
    name === :sp2d && return ((NI, NB, nv, B), (() -> SFC.GPUSFWorkspace(BE, TIGHT, vb; kind = :single_pass_2d)),
        ((s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_2d_batch!(
            s, c, x, u, TIGHT, vb; backend = be, workspace = ws, culling = pol, kw...)),
        ((s, c, xt, ut) -> SFC.calculate_structure_functions_single_pass_2d!(s, c, xt, ut, TIGHT, vb; backend = ser,
                                                                             kw...)))
    error("unknown batch family $name")
end

const CULL_BATCH_CASES = (
    (:batch1d, true, false, SFC.AutoCulling(), true),
    (:sp1d, false, true, SFC.AlwaysCulling(), true),
    (:joint, true, true, SFC.AlwaysCulling(), false),
    (:sp2d, false, false, SFC.NoCulling(), true),
)

# A batch culls shared positions once and varying positions per slice, and every policy gives the serial answer.
Test.@testset "the device batch culls" begin
    Random.seed!(79)
    N, B = 300, 2
    val = collect(range(-4.0, 4.0; length = 9))
    dev = CB.GPUBackend(BE)
    for (name, shared, weighted, pol, with_ws) in CULL_BATCH_CASES
        x = shared ? rand(2, N) : rand(2, N, B)
        u = randn(2, N, B)
        kw = weighted ? (; weights = 0.5 .+ rand(N)) : (;)
        CT = weighted ? Float64 : UInt32
        shape, workspace, run!, slice! = cull_batch_family(name, x, u, val, B, kw)
        rs, rc = zeros(shape), zeros(CT, shape)
        for b in 1:B
            s, c = zeros(shape[1:(end - 1)]), zeros(CT, shape[1:(end - 1)])
            slice!(s, c, shared ? x : x[:, :, b], u[:, :, b])
            selectdim(rs, length(shape), b) .= s
            selectdim(rc, length(shape), b) .= c
        end
        Test.@test sum(rc) > 0
        gs, gc = zeros(shape), zeros(CT, shape)
        run!(gs, gc, dev, with_ws ? workspace() : nothing, pol)
        agree = weighted ? isapprox(gc, rc; rtol = 1e-10) : gc == rc
        Test.@test (name, shared, weighted, pol, agree, isapprox(gs, rs; rtol = 1e-10, atol = 1e-12)) ==
                   (name, shared, weighted, pol, true, true)
    end
end
