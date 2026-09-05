using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using KernelAbstractions: KernelAbstractions as KA
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

    # a later call on the SAME workspace with bins that do not cull must not see the stale memo
    run1d(x, u, WIDE, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.active === nothing

    # an explicit demand without a workspace cannot be honoured and must say so
    Test.@test_throws ArgumentError run1d(x, u, TIGHT, SFC.AlwaysCulling(), nothing)

    # joint distance x value
    val = collect(range(-4.0, 4.0; length = 9))
    wj_ref = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d)
    ref2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val;
        workspace = wj_ref, culling = SFC.NoCulling())
    wj = SFC.GPUSFWorkspace(BE, TIGHT, val; kind = :joint2d)
    got2 = SFC.gpu_calculate_structure_function_2d(SF1D, BE, x, u, TIGHT, val;
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
# coordinates, the cutoff and the policy; the per-tile-size lists are built from the grid on first
# use. A hit must be exactly as correct as a rebuild, and anything that could change the grid must
# miss.
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

    # the caller mutates its coordinates in place: a miss, and the new points are what is culled
    x .= rand(2, N)
    s_ref, c_ref = run1d(x, u2, TIGHT, SFC.NoCulling(), ref)
    s3, c3 = run1d(x, u2, TIGHT, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull !== memo
    Test.@test c3 == c_ref
    Test.@test isapprox(s3, s_ref; rtol = 1e-10, atol = 1e-12)
    memo = ws.lazy.cull

    # a different cutoff is a different grid
    tighter = collect(range(0.0, 0.03; length = 9))
    run1d(x, u2, tighter, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.cull !== memo
    Test.@test ws.lazy.cull.cutoff == last(tighter)
    memo = ws.lazy.cull

    # a different policy is a different decision
    run1d(x, u2, tighter, SFC.AlwaysCulling(), ws)
    Test.@test ws.lazy.cull !== memo
    Test.@test ws.lazy.cull.policy === SFC.AlwaysCulling()

    # bins that do not cull publish no memo for the call and leave the stored one for the next
    run1d(x, u2, WIDE, SFC.AutoCulling(), ws)
    Test.@test ws.lazy.active === nothing
    Test.@test ws.lazy.cull !== nothing

    # a second tile size gets its own list from the same grid, built once
    memo = ws.lazy.cull
    tile2 = 2 * only(keys(memo.schedules))
    s_other = SFC.schedule_for(memo, N, tile2)
    Test.@test length(memo.schedules) == 2
    Test.@test Int(s_other.n_tiles) == cld(N, tile2)
    Test.@test SFC.schedule_for(memo, N, tile2) === s_other
    Test.@test_throws ArgumentError SFC.schedule_for(memo, N + 1, tile2)

    SFC.release!(ws)
    Test.@test ws.lazy.cull === nothing
    Test.@test ws.lazy.active === nothing
end
