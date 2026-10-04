using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StaticArrays: StaticArrays as SA
using Random: Random

const SF2 = SFT.L2SFType()

function _joint(sf, x, u, dist_bins, ax_bins, source)
    return SFC.serial_calculate_structure_function(
        sf, x, u, dist_bins, ax_bins, UInt32; geometry = SFH.FlatGeometry{size(u, 1)}(), second_axis = source)
end

# A pair and its reverse share one angle, in [0, π) in two dimensions and in [0, π/2] in three.
Test.@testset "the angle axis folds a pair and its reverse together" begin
    Random.seed!(7100)
    src = SFC.SeparationAngleAxis(SA.SVector(0.7, -0.3))
    dxs = [SA.SVector(randn(), randn()) for _ in 1:16]
    angles = [SFC.axis_quantity(src, dx, sum(abs2, dx)) for dx in dxs]
    Test.@test angles ≈ [SFC.axis_quantity(src, -dx, sum(abs2, dx)) for dx in dxs]
    Test.@test all(a -> 0 <= a < π, angles)

    src3 = SFC.SeparationAngleAxis(SA.SVector(0.0, 0.0, 1.0))
    dxs3 = [SA.SVector(randn(), randn(), randn()) for _ in 1:16]
    angles3 = [SFC.axis_quantity(src3, dx, sum(abs2, dx)) for dx in dxs3]
    Test.@test angles3 ≈ [SFC.axis_quantity(src3, -dx, sum(abs2, dx)) for dx in dxs3]
    Test.@test all(a -> 0 <= a <= π / 2 + 1e-12, angles3)
end

# The angle is measured from the given reference axis: zero along it, a right angle across it.
Test.@testset "the angle axis reads the geometry it should" begin
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    Test.@test SFC.axis_quantity(src, SA.SVector(2.0, 0.0), 4.0) ≈ 0.0 atol = 1e-12
    Test.@test SFC.axis_quantity(src, SA.SVector(0.0, 3.0), 9.0) ≈ π / 2
    Test.@test SFC.axis_quantity(src, SA.SVector(1.0, 1.0), 2.0) ≈ π / 4
    rot = SFC.SeparationAngleAxis(SA.SVector(0.0, 1.0))
    Test.@test SFC.axis_quantity(rot, SA.SVector(0.0, 3.0), 9.0) ≈ 0.0 atol = 1e-12
end

# Summed over the angle, S(r, θ) is S(r), bin for bin and pair for pair.
Test.@testset "marginalizing the angle recovers the plain structure function" begin
    Random.seed!(7200)
    for D in (2, 3)
        N = 100
        x = rand(D, N)
        u = randn(D, N)
        dist_bins = collect(range(0.0, 1.2; length = 7))
        e = D == 2 ? SA.SVector(1.0, 0.0) : SA.SVector(0.0, 0.0, 1.0)
        hi = D == 2 ? π : π / 2
        ax_bins = collect(range(0.0, hi + 1e-9; length = 9))

        joint = _joint(SF2, x, u, dist_bins, ax_bins, SFC.SeparationAngleAxis(e))
        nb = length(dist_bins) - 1
        ref_s = zeros(Float64, nb)
        ref_c = zeros(UInt32, nb)
        SFC.calculate_structure_function!(ref_s, ref_c, SF2, x, u, dist_bins;
                                         backend = CB.SerialBackend())
        Test.@test vec(sum(joint.counts; dims = 2)) == ref_c
        Test.@test isapprox(vec(sum(joint.sums; dims = 2)), ref_s; rtol = 1e-10, atol = 1e-12)
        Test.@test sum(joint.counts) > 0
    end
end

# u = (sin kx, 0) has no increment between points sharing an x, the only pairs in the bin about π/2 on this grid.
Test.@testset "an anisotropic field puts its signal in the predicted angular bin" begin
    n = 12
    xs = range(0.0, 1.0; length = n)
    pts = Matrix{Float64}(undef, 2, n * n)
    fld = zeros(2, n * n)
    k = 2π * 3
    for (idx, I) in enumerate(CartesianIndices((n, n)))
        px, py = xs[I[1]], xs[I[2]]
        pts[1, idx] = px
        pts[2, idx] = py
        fld[1, idx] = sin(k * px)
    end
    dist_bins = collect(range(0.0, 0.35; length = 5))
    ax_bins = [0.0, π / 2 - 0.02, π / 2 + 0.02, π]
    joint = _joint(SF2, pts, fld, dist_bins, ax_bins, SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)))

    perpendicular = sum(joint.sums[:, 2])
    oblique = sum(joint.sums[:, 1]) + sum(joint.sums[:, 3])
    Test.@test perpendicular == 0.0
    Test.@test oblique > 0
    Test.@test all(sum(joint.counts[:, a]) > 0 for a in 1:3)
end

# The value axis source bins exactly what the joint entry without a source bins.
Test.@testset "binning the operator value is unchanged" begin
    Random.seed!(7400)
    N = 100
    x = rand(2, N)
    u = randn(2, N)
    dist_bins = collect(range(0.0, 1.0; length = 6))
    val_bins = collect(range(0.0, 4.0; length = 9))
    with_source = _joint(SF2, x, u, dist_bins, val_bins, SFC.InvariantValueAxis())
    plain = SFC.serial_calculate_structure_function(SF2, x, u, dist_bins, val_bins, UInt32;
                                                    geometry = SFH.FlatGeometry{2}())
    Test.@test with_source.counts == plain.counts
    Test.@test with_source.sums == plain.sums
    Test.@test sum(plain.counts) > 0
end

# Each slice of a device batch with its own positions bins the angle of its own separations, as the point entry does.
Test.@testset "the angle axis on a device batch with positions varying per slice" begin
    Random.seed!(7600)
    N, B = 100, 2
    dist_bins = collect(range(0.0, 1.0; length = 7))
    ax_bins = collect(range(prevfloat(0.0), π; length = 5))
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    x, u = rand(2, N, B), randn(2, N, B)
    got = SFC.calculate_structure_function(SF2, x, u, dist_bins, ax_bins; backend = CB.GPUBackend(KA.CPU()),
        second_axis = src)
    refs = [SFC.calculate_structure_function(SF2, x[:, :, b], u[:, :, b], dist_bins, ax_bins;
        backend = CB.SerialBackend(), second_axis = src) for b in 1:B]
    Test.@test got.counts == cat((r.counts for r in refs)...; dims = 3)
    Test.@test got.sums ≈ cat((r.sums for r in refs)...; dims = 3)
    value_binned = SFC.calculate_structure_function(SF2, x[:, :, 1], u[:, :, 1], dist_bins, ax_bins;
        backend = CB.SerialBackend())
    Test.@test value_binned.counts != refs[1].counts
end

# On a sphere each pair's direction lives in its own frame, so every entry refuses an angle to a fixed axis.
Test.@testset "an angle axis is refused where the direction is not shared" begin
    Random.seed!(7500)
    x = [0.1 0.2 0.35; -0.2 0.05 0.3]
    dist_bins = collect(range(0.0, 2.0; length = 4))
    ax_bins = collect(range(0.0, π; length = 4))
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    Test.@test_throws ArgumentError SFC.serial_calculate_structure_function(
        SF2, x, randn(2, 3), dist_bins, ax_bins, UInt32;
        geometry = SFH.pair_geometry_for(SFC.DI.SphericalAngle(), Val(2)), second_axis = src)
    for (u, backend) in ((randn(2, 3), CB.GPUBackend(KA.CPU())), (randn(2, 3, 2), CB.SerialBackend()),
                         (randn(2, 3, 2), CB.GPUBackend(KA.CPU())))
        Test.@test_throws ArgumentError SFC.calculate_structure_function(SF2, x, u, dist_bins, ax_bins;
            backend, distance_metric = SFC.DI.SphericalAngle(), second_axis = src)
    end
end
