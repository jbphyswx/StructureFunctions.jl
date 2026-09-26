using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using KernelAbstractions: KernelAbstractions as KA
using ComputationalBackends: ComputationalBackends as CB
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
launched_list(ws) = only(values(ws.lazy.active.schedules))

# Culling on the GPU path rides on the workspace: the prologue sorts the points and publishes the
# memo as `workspace.lazy.active`; each launcher takes its tile-pair list from it at its own tile
# size and enumerates only those tile pairs. Results must equal the full sweep exactly in counts.
Test.@testset "GPU culling through a workspace" begin
    Random.seed!(77)
    N = 3000
    x = rand(2, N)
    u = randn(2, N)

    ws_ref = SFC.GPUSFWorkspace(BE, TIGHT)
    s_ref, c_ref = run1d(x, u, TIGHT, SFC.NoCulling(), ws_ref)
    Test.@test ws_ref.lazy.active === nothing
    ws = SFC.GPUSFWorkspace(BE, TIGHT)
    s_cull, c_cull = run1d(x, u, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.active !== nothing                                    # it engaged
    Test.@test length(ws.lazy.active.schedules) == 1                        # one launcher, one tile size
    wl = launched_list(ws)
    n_tiles = Int(wl.n_tiles)
    Test.@test SFC.n_pair_blocks(wl) < n_tiles * (n_tiles + 1) ÷ 2
    Test.@test c_cull == c_ref
    Test.@test isapprox(s_cull, s_ref; rtol = 1e-10, atol = 1e-12)
    Test.@test sum(c_ref) > 0

    # a later call on the SAME workspace with points that do not cull must not see the stale memo
    run1d(0.1 .* x, u, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.active === nothing

    # the workspace digitizes with the bins it was built for, so it refuses any other bins
    Test.@test_throws ArgumentError run1d(x, u, WIDE, SFC.AutoCulling(), ws)

    # an explicit demand without a workspace cannot be honoured and must say so
    Test.@test_throws ArgumentError run1d(x, u, TIGHT, SFC.AlwaysCulling(), nothing)

    # joint distance x value
    val = collect(range(-4.0, 4.0; length = 9))
    wj_ref = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d)
    ref2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val, UInt32;
        workspace = wj_ref, culling = SFC.NoCulling())
    wj = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d)
    got2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val, UInt32;
        workspace = wj, culling = SFC.AutoCulling())
    Test.@test wj.lazy.active !== nothing && length(wj.lazy.active.schedules) == 1
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
    Test.@test wsp.lazy.active !== nothing && length(wsp.lazy.active.schedules) == 1
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
    Test.@test memo !== nothing
    Test.@test ws.lazy.active === memo
    Test.@test memo.x !== x                                # the workspace owns its key
    wl = launched_list(ws)

    # same points, new fields: reused, the list is not rebuilt, and still the full sweep's answer
    u2 = randn(2, N)
    s_ref, c_ref = run1d(x, u2, TIGHT, SFC.NoCulling(), ref)
    s2, c2 = run1d(x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull === memo
    Test.@test ws.lazy.active === memo
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

    # points the stencil already spans are not culled: the call publishes no memo and stores that
    # decision for those points
    run1d(0.1 .* x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.active === nothing
    Test.@test ws.lazy.cull isa SFC.GPUNoCullMemo

    SFC.release!(ws)
    Test.@test ws.lazy.cull === nothing
    Test.@test ws.lazy.active === nothing
end

# A batch culls through its workspace too: shared positions once for every slice, positions varying per
# slice one slice at a time. Every policy gives the serial answer.
Test.@testset "the device batch culls through a workspace" begin
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
            # Without a workspace there is nowhere to hold the work list: `Auto` sweeps every pair,
            # an explicit `AlwaysCulling()` raises.
            gs, gc = zeros(shape), zeros(CT, shape)
            run!(gs, gc, dev, nothing, SFC.AutoCulling())
            Test.@test (name, shared, weighted, gc == rc || isapprox(gc, rc; rtol = 1e-10)) ==
                       (name, shared, weighted, true)
            Test.@test_throws ArgumentError run!(zeros(shape), zeros(CT, shape), dev, nothing, SFC.AlwaysCulling())
        end
    end
end
