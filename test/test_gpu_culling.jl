using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using KernelAbstractions: KernelAbstractions as KA
using ComputationalBackends: ComputationalBackends as CB
using Distances: Distances as DI
using Random: Random

const BE = KA.CPU()
const SF1D = SFT.L2SFType()
const TIGHT = collect(range(0.0, 0.06; length = 9))          # AutoCulling engages on the unit square
const WIDE = collect(range(0.0, 1.5; length = 9))            # same bin count, no culling
const NB = length(TIGHT) - 1

function run1d(x, u, bins, pol, ws)
    s = zeros(NB)
    c = zeros(UInt32, NB)
    SFC.gpu_calculate_structure_function!(s, c, SF1D, BE, x, u, bins; workspace = ws, culling = pol)
    return s, c
end

# The one work list a launch on this backend built, at whatever tile size that launcher uses.
launched_list(ws) = only(values(ws.lazy.cull.schedules))

# The prologue sorts the points on the device and hands every launcher of the call the memo; each
# launcher takes its tile-pair list from it at its own tile size and enumerates only those tile pairs. A
# workspace keeps the memo between calls. Results must equal the full sweep exactly in counts.
Test.@testset "GPU culling, with and without a workspace" begin
    Random.seed!(77)
    N = 3000
    x = rand(2, N)
    u = randn(2, N)

    ws_ref = SFC.GPUSFWorkspace(BE, TIGHT)
    s_ref, c_ref = run1d(x, u, TIGHT, SFC.NoCulling(), ws_ref)
    Test.@test ws_ref.lazy.cull === nothing
    ws = SFC.GPUSFWorkspace(BE, TIGHT)
    s_cull, c_cull = run1d(x, u, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull isa SFC.GPUCullMemo                             # it engaged
    Test.@test length(ws.lazy.cull.schedules) == 1                          # one launcher, one tile size
    wl = launched_list(ws)
    n_tiles = Int(wl.n_tiles)
    Test.@test SFC.n_pair_blocks(wl) < n_tiles * (n_tiles + 1) ÷ 2
    Test.@test c_cull == c_ref
    Test.@test isapprox(s_cull, s_ref; rtol = 1e-10, atol = 1e-12)
    Test.@test sum(c_ref) > 0

    # a later call on the SAME workspace with points that do not cull must not see the stale memo
    run1d(0.1 .* x, u, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull isa SFC.GPUNoCullMemo

    # the workspace digitizes with the bins it was built for, so it refuses any other bins
    Test.@test_throws ArgumentError run1d(x, u, WIDE, SFC.AutoCulling(), ws)

    # without a workspace the call builds its own memo
    for pol in (SFC.AutoCulling(), SFC.AlwaysCulling())
        s_nows, c_nows = run1d(x, u, TIGHT, pol, nothing)
        Test.@test (pol, c_nows == c_ref, isapprox(s_nows, s_ref; rtol = 1e-10, atol = 1e-12)) == (pol, true, true)
    end

    # joint distance x value
    val = collect(range(-4.0, 4.0; length = 9))
    wj_ref = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d)
    ref2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val, UInt32;
        workspace = wj_ref, culling = SFC.NoCulling())
    wj = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d)
    got2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val, UInt32;
        workspace = wj, culling = SFC.AutoCulling())
    Test.@test wj.lazy.cull isa SFC.GPUCullMemo && length(wj.lazy.cull.schedules) == 1
    Test.@test got2.counts == ref2.counts
    Test.@test isapprox(got2.sums, ref2.sums; rtol = 1e-10, atol = 1e-12)

    # six-invariant distance x value
    vb = ntuple(_ -> val, SFC.SINGLE_PASS_N)
    n_val = length(val) - 1
    runsp(pol, w) = begin
        s = zeros(SFC.SINGLE_PASS_N, NB, n_val); c = zeros(UInt32, SFC.SINGLE_PASS_N, NB, n_val)
        SFC.gpu_calculate_structure_functions_single_pass_2d!(s, c, BE, x, u, TIGHT, vb;
            workspace = w, culling = pol)
        (s, c)
    end
    wsp_ref = SFC.GPUSFWorkspace(BE, TIGHT, vb; kind = :single_pass_2d)
    sp_s0, sp_c0 = runsp(SFC.NoCulling(), wsp_ref)
    wsp = SFC.GPUSFWorkspace(BE, TIGHT, vb; kind = :single_pass_2d)
    sp_s1, sp_c1 = runsp(SFC.AutoCulling(), wsp)
    Test.@test wsp.lazy.cull isa SFC.GPUCullMemo && length(wsp.lazy.cull.schedules) == 1
    Test.@test sp_c1 == sp_c0
    Test.@test isapprox(sp_s1, sp_s0; rtol = 1e-10, atol = 1e-12)
end

# The prologue's grid and permutation are memoised on the workspace, keyed on the kernel
# coordinate-array identity, the cutoff and the policy; the per-tile-size lists are built from the
# grid on first use. In-place coordinate changes require `refresh!`, avoiding a device-wide equality
# scan on every prepared execution.
Test.@testset "the cull memo is reused for the same points and invalidated otherwise" begin
    Random.seed!(78)
    N = 2500
    x = rand(2, N)
    u = randn(2, N)
    ref = SFC.GPUSFWorkspace(BE, TIGHT)
    ws = SFC.GPUSFWorkspace(BE, TIGHT)
    Test.@test ws.lazy.cull === nothing
    run1d(x, u, TIGHT, SFC.AutoCulling(), ws)
    memo = ws.lazy.cull
    Test.@test memo isa SFC.GPUCullMemo
    Test.@test memo.source === x                           # keyed on the caller's array
    wl = launched_list(ws)

    # same points, new fields: reused, the list is not rebuilt, and still the full sweep's answer
    u2 = randn(2, N)
    s_ref, c_ref = run1d(x, u2, TIGHT, SFC.NoCulling(), ref)
    s2, c2 = run1d(x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull === memo
    Test.@test launched_list(ws) === wl
    Test.@test c2 == c_ref
    Test.@test isapprox(s2, s_ref; rtol = 1e-10, atol = 1e-12)

    # the caller mutates its coordinates in place and explicitly invalidates preparation
    x .= rand(2, N)
    SFC.refresh!(ws)
    s_ref, c_ref = run1d(x, u2, TIGHT, SFC.NoCulling(), ref)
    s3, c3 = run1d(x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull !== memo
    Test.@test c3 == c_ref
    Test.@test isapprox(s3, s_ref; rtol = 1e-10, atol = 1e-12)
    memo = ws.lazy.cull

    # a workspace refuses bins it was not built for, so its memo is never read at another cutoff
    Test.@test_throws ArgumentError run1d(x, u2, collect(range(0.0, 0.03; length = 9)), SFC.AutoCulling(), ws)

    # a different policy is a different decision
    run1d(x, u2, TIGHT, SFC.AlwaysCulling(), ws)
    Test.@test ws.lazy.cull !== memo
    Test.@test ws.lazy.cull.policy === SFC.AlwaysCulling()

    # a second tile size gets its own list from the same grid, built once
    memo = ws.lazy.cull
    tile2 = 2 * only(keys(memo.schedules))
    s_other = SFC.schedule_for(memo, N, tile2)
    Test.@test length(memo.schedules) == 2
    Test.@test Int(s_other.n_tiles) == cld(N, tile2)
    Test.@test SFC.schedule_for(memo, N, tile2) === s_other
    Test.@test_throws ArgumentError SFC.schedule_for(memo, N + 1, tile2)

    # points the stencil already spans are not culled: the workspace stores that decision for those points
    run1d(0.1 .* x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull isa SFC.GPUNoCullMemo

    SFC.release!(ws)
    Test.@test ws.lazy.cull === nothing
end

const GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)

# The device grid is the host grid: same cells, same runs, and the same order of points, since both sorts
# are stable on this backend.
Test.@testset "the device cull grid equals the host grid" begin
    for (D, N, frac) in ((2, 1500, 0.06), (3, 1200, 0.15), (5, 900, 0.4))
        Random.seed!(2700 + N)
        x = rand(D, N) .* ntuple(d -> d, D)                   # unequal extents choose the widest axes
        geom = SF.HelperFunctions.FlatGeometry{D}()
        bins = collect(range(0.0, frac; length = 5))
        host = SFC.cull_grid_for(ntuple(d -> view(x, d, :), D), geom, bins, SFC.AlwaysCulling())
        dev = GE._gpu_cull_grid(BE, x, geom, SFC.cull_cutoff_for(geom, bins, SFC.AlwaysCulling()),
                                SFC.AlwaysCulling())
        Test.@test (D, dev.origin, dev.dims, dev.inv_h) == (D, host.origin, host.dims, host.inv_h)
        Test.@test (D, dev.cell_ids, dev.run_starts, dev.perm) == (D, host.cell_ids, host.run_starts, host.perm)
        Test.@test (D, dev.offsets) == (D, host.offsets)
    end
end

# On a sphere the grid is built over the unit positions and the cutoff is the chord of the largest angle.
Test.@testset "a sphere culls on the device by its chord bound" begin
    Random.seed!(81)
    N = 3000
    x = vcat(2π .* rand(1, N), asin.(2 .* rand(1, N) .- 1))
    u = randn(2, N)
    bins = collect(range(0.0, 0.12; length = 9))
    nb = length(bins) - 1
    run(pol, ws, be) = begin
        s, c = zeros(nb), zeros(UInt32, nb)
        SFC.calculate_structure_function!(s, c, SF1D, x, u, bins; backend = be, distance_metric = DI.SphericalAngle(),
                                          culling = pol, (ws === nothing ? (;) : (; workspace = ws))...)
        (s, c)
    end
    s_ser, c_ser = run(SFC.NoCulling(), nothing, CB.SerialBackend())
    s_full, c_full = run(SFC.NoCulling(), nothing, CB.GPUBackend(BE))
    ws = SFC.GPUSFWorkspace(BE, bins)
    s_ws, c_ws = run(SFC.AlwaysCulling(), ws, CB.GPUBackend(BE))
    s_own, c_own = run(SFC.AlwaysCulling(), nothing, CB.GPUBackend(BE))
    Test.@test ws.lazy.cull isa SFC.GPUCullMemo && length(ws.lazy.cull.grid.dims) == 3
    wl = launched_list(ws)
    n_tiles = Int(wl.n_tiles)
    Test.@test SFC.n_pair_blocks(wl) < n_tiles * (n_tiles + 1) ÷ 2
    Test.@test c_ws == c_full && c_own == c_full && c_full == c_ser
    Test.@test isapprox(s_ws, s_full; rtol = 1e-10) && isapprox(s_own, s_full; rtol = 1e-10)
    Test.@test isapprox(s_full, s_ser; rtol = 1e-10)
    Test.@test sum(c_ser) > 0
end

# A grid prepared for one call pays only when one slice's pairs reach the threshold of the grid's coordinates; the
# slices sharing it do not count, and a workspace keeps it for later calls.
Test.@testset "AutoCulling prepares a grid for a single call only when one slice's pairs pay for it" begin
    cull_of(x, u, pol, ws) =
        GE._gpu_cull_and_permute!(ws, BE, x, u, SF.HelperFunctions.FlatGeometry{size(x, 1)}(), TIGHT, pol)[4]
    pairs(n) = n * (n - 1) ÷ 2
    Test.@test pairs(3000) < GE.GPU_CULL_COLD_MIN_PAIRS[2] <= pairs(20_000) < GE.GPU_CULL_COLD_MIN_PAIRS[3] <= pairs(30_000)
    Test.@test cull_of(rand(2, 3000), randn(2, 3000), SFC.AutoCulling(), nothing) === nothing
    Test.@test cull_of(rand(2, 20_000), randn(2, 20_000), SFC.AutoCulling(), nothing) isa SFC.GPUCullMemo
    Test.@test cull_of(rand(3, 20_000), randn(3, 20_000), SFC.AutoCulling(), nothing) === nothing
    Test.@test cull_of(rand(3, 30_000), randn(3, 30_000), SFC.AutoCulling(), nothing) isa SFC.GPUCullMemo
    Test.@test cull_of(rand(2, 3000), randn(2, 3000, 100), SFC.AutoCulling(), nothing) === nothing
    Test.@test cull_of(rand(2, 3000), randn(2, 3000), SFC.AlwaysCulling(), nothing) isa SFC.GPUCullMemo
    Test.@test cull_of(rand(2, 3000), randn(2, 3000), SFC.AutoCulling(), SFC.GPUSFWorkspace(BE, TIGHT)) isa SFC.GPUCullMemo
end

# Exactness of the work list: brute-force every pair inside the cutoff, in the permuted order the tiles are
# cut from, and require its canonical tile pair to be listed.
Test.@testset "the device work list covers every in-range pair" begin
    for (D, N, frac, tile) in ((2, 1500, 0.06, 128), (3, 1200, 0.15, 64), (2, 700, 0.3, 32),
                               (1, 900, 0.02, 64))
        Random.seed!(2600 + N)
        x = rand(D, N)
        xc = ntuple(d -> x[d, :], D)
        cut = SFC.cull_cutoff(SF.HelperFunctions.FlatGeometry{D}(), frac)
        grid = SFC.build_cell_grid(xc, cut, 2)
        xp = SFC.apply_perm(xc, grid.perm)
        wl = SFC.gpu_tile_worklist(grid, N, tile)
        n_tiles = cld(N, tile)
        Test.@test wl.n_tiles == n_tiles
        Test.@test issorted(wl.pairs) && allunique(wl.pairs)
        Test.@test all(k -> (1 <= SFC.tile_for(wl, k)[1] <= SFC.tile_for(wl, k)[2] <= n_tiles),
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
        Test.@test missed == 0
        Test.@test SFC.n_pair_blocks(wl) < n_tiles * (n_tiles + 1) ÷ 2   # it really culled
    end
end

# A batch culls too: shared positions once for every slice, positions varying per slice one slice at a
# time. Every policy gives the serial answer, with or without a workspace.
Test.@testset "the device batch culls" begin
    Random.seed!(79)
    N, B = 1200, 3
    val = collect(range(-4.0, 4.0; length = 9))
    vb = ntuple(_ -> val, SFC.SINGLE_PASS_N)
    nv, NI = length(val) - 1, SFC.SINGLE_PASS_N
    ser, dev = CB.SerialBackend(), CB.GPUBackend(BE)
    for shared in (true, false), weighted in (false, true)
        x = shared ? rand(2, N) : rand(2, N, B)
        u = randn(2, N, B)
        kw = weighted ? (; weights = 0.5 .+ rand(N)) : (;)
        CT = weighted ? Float64 : UInt32
        families = (
            (:batch1d, (NB, B), () -> SFC.GPUSFWorkspace(BE, TIGHT),
             (s, c, be, ws, pol) -> SFC.calculate_structure_function_batch!(
                 s, c, SF1D, x, u, TIGHT; backend = be, workspace = ws, culling = pol, kw...)),
            (:joint, (NB, nv, B), () -> SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d),
             (s, c, be, ws, pol) -> SFC.calculate_structure_function_2d_batch!(
                 s, c, SF1D, x, u, TIGHT, val; backend = be, workspace = ws, culling = pol, kw...)),
            (:sp1d, (NI, NB, B), () -> SFC.GPUSFWorkspace(BE, TIGHT; kind = :single_pass),
             (s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_batch!(
                 s, c, x, u, TIGHT; backend = be, workspace = ws, culling = pol, kw...)),
            (:sp2d, (NI, NB, nv, B), () -> SFC.GPUSFWorkspace(BE, TIGHT, vb; kind = :single_pass_2d),
             (s, c, be, ws, pol) -> SFC.calculate_structure_functions_single_pass_2d_batch!(
                 s, c, x, u, TIGHT, vb; backend = be, workspace = ws, culling = pol, kw...)),
        )
        for (name, shape, workspace, run!) in families
            rs, rc = zeros(shape), zeros(CT, shape)
            run!(rs, rc, ser, nothing, SFC.NoCulling())
            Test.@test sum(rc) > 0
            for pol in (SFC.NoCulling(), SFC.AutoCulling(), SFC.AlwaysCulling())
                ws = workspace()
                gs, gc = zeros(shape), zeros(CT, shape)
                run!(gs, gc, dev, ws, pol)
                agree = weighted ? isapprox(gc, rc; rtol = 1e-10) : gc == rc
                Test.@test (name, shared, weighted, pol, agree, isapprox(gs, rs; rtol = 1e-10, atol = 1e-12)) ==
                           (name, shared, weighted, pol, true, true)
                Test.@test (name, shared, weighted, pol, ws.lazy.cull isa SFC.GPUCullMemo) ==
                           (name, shared, weighted, pol, !(pol isa SFC.NoCulling))
            end
            for pol in (SFC.AutoCulling(), SFC.AlwaysCulling())
                gs, gc = zeros(shape), zeros(CT, shape)
                run!(gs, gc, dev, nothing, pol)
                agree = weighted ? isapprox(gc, rc; rtol = 1e-10) : gc == rc
                Test.@test (name, shared, weighted, pol, agree, isapprox(gs, rs; rtol = 1e-10, atol = 1e-12)) ==
                           (name, shared, weighted, pol, true, true)
            end
        end
    end
end
