using Test: Test
using Random: Random
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, HelperFunctions as SFH

const IR_EXT = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const IR_BE = KA.CPU()

# The share in range over every pair of `sched`, of `tile`-point tiles, and every slice, and the share of the
# estimator's draws that are pairs.
function ir_exact(x3, dig, NB, geom, sched, N, tile)
    B, vW = size(x3, 3), SFH.coordinate_width(geom)
    point(i, b) = SA.SVector(ntuple(a -> x3[a, i, b], vW))
    pairs = hits = 0
    for b in 1:B, k in 1:SFC.n_pair_blocks(sched)
        ti, tj = SFC.tile_for(sched, k)
        for i in ((ti - 1) * tile + 1):min(ti * tile, N), j in ((tj - 1) * tile + 1):min(tj * tile, N)
            (ti < tj || i < j) || continue
            ok, d, _ = SFH.pair_frame(geom, point(i, b), point(j, b))
            bin = SFH.digitize(d, dig)
            pairs += 1
            hits += ok && 1 <= bin <= NB
        end
    end
    return hits / pairs, pairs / (B * SFC.n_pair_blocks(sched) * tile^2)
end

# The estimate against the exact share, within five of its standard errors, over the schedule a launch of `tile`-point
# tiles sweeps: every pair, or the tile pairs a cull grid lists, and every slice of varying positions.
function ir_agrees(x, u, bins, geom, culling, tile = IR_EXT.SF_GPU_TILE)
    NB = length(bins) - 1
    xs, _, _, cull = IR_EXT._gpu_cull_and_permute!(nothing, IR_BE, x, u, geom, bins, culling)
    dig = IR_EXT._gpu_digitizer(IR_BE, bins, Val(:sf1d))
    N = size(x, 2)
    f = SFC.gpu_in_range_fraction(IR_BE, xs, dig, NB, geom, cull, tile)
    exact, valid = ir_exact(reshape(xs, size(xs, 1), N, :), dig, NB, geom, SFC.schedule_for(cull, N, tile), N, tile)
    n = SFC.GPU_IN_RANGE_GROUPS * SFC.GPU_IN_RANGE_GROUP * valid
    return (culled = cull !== nothing, close = abs(f - exact) <= 5 * sqrt(max(exact * (1 - exact), 1 / n) / n),
            repeatable = f == SFC.gpu_in_range_fraction(IR_BE, xs, dig, NB, geom, cull, tile))
end

Test.@testset "the in-range estimate matches the share over the pairs a sweep visits" begin
    rng = Random.Xoshiro(5)
    for D in (2, 3), rmax in (0.05, 0.3, 1.5)
        x, u = rand(rng, D, 1000), randn(rng, D, 1000)
        bins = SF.LinearBinEdges(rmax / 50, rmax, 17)
        r = ir_agrees(x, u, bins, SFH.FlatGeometry{D}(), SFC.NoCulling())
        Test.@test (D, rmax, r.close, r.repeatable) == (D, rmax, true, true)
        if rmax <= 0.3
            for tile in (IR_EXT.SF_GPU_TILE, 384)
                r = ir_agrees(x, u, bins, SFH.FlatGeometry{D}(), SFC.AlwaysCulling(), tile)
                Test.@test (D, rmax, tile, r.culled, r.close, r.repeatable) == (D, rmax, tile, true, true, true)
            end
        end
    end
    x, u = rand(rng, Float32, 2, 60), randn(rng, Float32, 2, 60)
    r = ir_agrees(x, u, SF.LinearBinEdges(0.0f0, 0.3f0, 9), SFH.FlatGeometry{2}(), SFC.NoCulling())
    Test.@test (r.close, r.repeatable) == (true, true)
    x, u = rand(rng, 2, 700, 3), randn(rng, 2, 700, 3)
    r = ir_agrees(x, u, SF.LinearBinEdges(0.0, 0.4, 9), SFH.FlatGeometry{2}(), SFC.NoCulling())
    Test.@test (r.close, r.repeatable) == (true, true)
    x = vcat(2π .* rand(rng, 1, 1500), asin.(2 .* rand(rng, 1, 1500) .- 1))
    geom = SFH.pair_geometry_for(DI.SphericalAngle(), Val(2))
    xk, uk = SFH.prepare_pair_inputs(geom, x, randn(rng, 2, 1500))
    r = ir_agrees(xk, uk, SF.LinearBinEdges(0.0, 0.3, 9), geom, SFC.AlwaysCulling())
    Test.@test (r.culled, r.close, r.repeatable) == (true, true, true)
end

Test.@testset "a tally holds each work group's draws and the estimate is its share" begin
    rng = Random.Xoshiro(6)
    x = rand(rng, 2, 1000)
    bins = SF.LinearBinEdges(0.0, 0.3, 9)
    geom, dig = SFH.FlatGeometry{2}(), IR_EXT._gpu_digitizer(IR_BE, bins, Val(:sf1d))
    tally = SFC.gpu_in_range_tally!(fill(Int32(-1), 2, SFC.GPU_IN_RANGE_GROUPS), IR_BE, x, dig, 8, geom, nothing,
                                    IR_EXT.SF_GPU_TILE)
    Test.@test all(0 .<= tally[2, :] .<= tally[1, :] .<= SFC.GPU_IN_RANGE_GROUP)
    Test.@test SFC.in_range_share(tally) == SFC.gpu_in_range_fraction(IR_BE, x, dig, 8, geom, nothing, IR_EXT.SF_GPU_TILE)
    Test.@test_throws DimensionMismatch SFC.gpu_in_range_tally!(zeros(Int32, 2, 3), IR_BE, x, dig, 8, geom, nothing,
                                                                IR_EXT.SF_GPU_TILE)
end
