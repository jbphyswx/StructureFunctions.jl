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

# Whether the estimate is within five standard errors of the exact share over the pairs a `tile`-point sweep visits.
function ir_close(x, u, bins, geom, culling, tile = IR_EXT.SF_GPU_TILE)
    NB = length(bins) - 1
    xs, _, _, cull = IR_EXT._gpu_cull_and_permute!(nothing, IR_BE, x, u, geom, bins, culling)
    dig = IR_EXT._gpu_digitizer(IR_BE, bins, Val(:sf1d))
    N = size(x, 2)
    f = SFC.gpu_in_range_fraction(IR_BE, xs, dig, NB, geom, cull, tile)
    exact, valid = ir_exact(reshape(xs, size(xs, 1), N, :), dig, NB, geom, SFC.schedule_for(cull, N, tile), N, tile)
    n = SFC.GPU_IN_RANGE_GROUPS * SFC.GPU_IN_RANGE_GROUP * valid
    return abs(f - exact) <= 5 * sqrt(max(exact * (1 - exact), 1 / n) / n)
end

# Each case once: D = 2 and 3, culled or not, the default tile and one past N, Float32, slices, and a sphere.
Test.@testset "the in-range estimate matches the share over the pairs a sweep visits" begin
    rng = Random.Xoshiro(5)
    N = 300
    x, u = rand(rng, 2, N), randn(rng, 2, N)
    Test.@test ir_close(x, u, SF.LinearBinEdges(0.001, 0.05, 17), SFH.FlatGeometry{2}(), SFC.AlwaysCulling())
    x, u = rand(rng, 3, N), randn(rng, 3, N)
    Test.@test ir_close(x, u, SF.LinearBinEdges(0.006, 0.3, 17), SFH.FlatGeometry{3}(), SFC.AlwaysCulling(), 384)
    x, u = rand(rng, Float32, 2, N), randn(rng, Float32, 2, N)
    Test.@test ir_close(x, u, SF.LinearBinEdges(0.03f0, 1.5f0, 17), SFH.FlatGeometry{2}(), SFC.NoCulling())
    x, u = rand(rng, 2, N, 3), randn(rng, 2, N, 3)
    Test.@test ir_close(x, u, SF.LinearBinEdges(0.0, 0.4, 9), SFH.FlatGeometry{2}(), SFC.NoCulling())
    x = vcat(2π .* rand(rng, 1, N), asin.(2 .* rand(rng, 1, N) .- 1))
    geom = SFH.pair_geometry_for(DI.SphericalAngle(), Val(2))
    xk, uk = SFH.prepare_pair_inputs(geom, x, randn(rng, 2, N))
    Test.@test ir_close(xk, uk, SF.LinearBinEdges(0.0, 0.3, 9), geom, SFC.AlwaysCulling())
end
