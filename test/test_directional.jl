using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using Random: Random

const SF2 = SFT.L2SFType()

function _joint(sf, x, u, dist_bins, ax_bins, source)
    return SFC.serial_calculate_structure_function(
        sf, x, u, dist_bins, ax_bins; second_axis = source, verbose = false, show_progress = false)
end

Test.@testset "the angle axis folds a pair and its reverse together" begin
    # Swapping a pair's ends flips the separation, and no structure function distinguishes the two,
    # so the angle must not either. Checked on the source directly, over the whole circle.
    e = SA.SVector(0.7, -0.3)
    src = SFC.SeparationAngleAxis(e)
    Random.seed!(7100)
    for _ in 1:200
        dx = SA.SVector(randn(), randn())
        r2 = sum(abs2, dx)
        Test.@test SFC.axis_quantity(src, dx, r2) ≈ SFC.axis_quantity(src, -dx, r2)
        Test.@test 0 <= SFC.axis_quantity(src, dx, r2) < π
    end
    # in three dimensions the fold is onto the polar angle, so the range halves
    e3 = SA.SVector(0.0, 0.0, 1.0)
    src3 = SFC.SeparationAngleAxis(e3)
    for _ in 1:200
        dx = SA.SVector(randn(), randn(), randn())
        r2 = sum(abs2, dx)
        Test.@test SFC.axis_quantity(src3, dx, r2) ≈ SFC.axis_quantity(src3, -dx, r2)
        Test.@test 0 <= SFC.axis_quantity(src3, dx, r2) <= π / 2 + 1e-12
    end
end

Test.@testset "the angle axis reads the geometry it should" begin
    # A separation along the reference axis is angle zero; perpendicular is a right angle.
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    Test.@test SFC.axis_quantity(src, SA.SVector(2.0, 0.0), 4.0) ≈ 0.0 atol = 1e-12
    Test.@test SFC.axis_quantity(src, SA.SVector(0.0, 3.0), 9.0) ≈ π / 2
    Test.@test SFC.axis_quantity(src, SA.SVector(1.0, 1.0), 2.0) ≈ π / 4
    # and it is measured from the axis given, not from x
    rot = SFC.SeparationAngleAxis(SA.SVector(0.0, 1.0))
    Test.@test SFC.axis_quantity(rot, SA.SVector(0.0, 3.0), 9.0) ≈ 0.0 atol = 1e-12
end

Test.@testset "marginalizing the angle recovers the plain structure function" begin
    # S(r, θ) summed over θ must be S(r), bin for bin and pair for pair: the angle axis re-sorts the
    # same pairs, it does not select among them.
    Random.seed!(7200)
    for D in (2, 3)
        N = 300
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
        SF.calculate_structure_function!(ref_s, ref_c, SF2, x, u, dist_bins;
                                         backend = CB.SerialBackend())
        Test.@test vec(sum(joint.counts; dims = 2)) == ref_c
        Test.@test isapprox(vec(sum(joint.sums; dims = 2)), ref_s; rtol = 1e-10, atol = 1e-12)
        Test.@test sum(joint.counts) > 0
    end
end

Test.@testset "an anisotropic field puts its signal in the predicted angular bin" begin
    # u = (sin(k·x), 0) with k along x varies only along x, so δu vanishes for separations
    # perpendicular to x. The longitudinal second-order structure function must therefore be large
    # in the angle bin containing 0 and near zero in the one containing π/2.
    Random.seed!(7300)
    n = 40
    xs = range(0.0, 1.0; length = n)
    pts = Matrix{Float64}(undef, 2, n * n)
    fld = zeros(2, n * n)
    k = 2π * 3
    for (idx, I) in enumerate(CartesianIndices((n, n)))
        px, py = xs[I[1]], xs[I[2]]
        pts[1, idx] = px
        pts[2, idx] = py
        fld[1, idx] = sin(k * px)          # varies along x only
    end
    dist_bins = collect(range(0.0, 0.35; length = 5))
    # An angle bin narrow around π/2 holds only the exactly-perpendicular separations: the next
    # achievable angle on this grid is atan(39/1), which is 0.026 away, outside the bin.
    ax_bins = [0.0, π / 2 - 0.02, π / 2 + 0.02, π]
    joint = _joint(SF2, pts, fld, dist_bins, ax_bins, SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)))

    perpendicular = sum(joint.sums[:, 2])
    oblique = sum(joint.sums[:, 1]) + sum(joint.sums[:, 3])
    # a field varying only along x has no increment at all between points sharing an x
    Test.@test perpendicular == 0.0
    Test.@test oblique > 0
    Test.@test sum(joint.counts[:, 2]) > 0          # the perpendicular pairs are present, not absent
    Test.@test all(sum(joint.counts[:, a]) > 0 for a in 1:3)
end

Test.@testset "binning the operator value is unchanged" begin
    # The default source must give exactly what the joint entry gave before an axis source existed.
    Random.seed!(7400)
    N = 200
    x = rand(2, N)
    u = randn(2, N)
    dist_bins = collect(range(0.0, 1.0; length = 6))
    val_bins = collect(range(0.0, 4.0; length = 9))
    with_source = _joint(SF2, x, u, dist_bins, val_bins, SFC.InvariantValueAxis())
    plain = SFC.serial_calculate_structure_function(SF2, x, u, dist_bins, val_bins;
                                                    verbose = false, show_progress = false)
    Test.@test with_source.counts == plain.counts
    Test.@test with_source.sums == plain.sums
    Test.@test sum(plain.counts) > 0
end

Test.@testset "an angle axis is refused where the direction is not shared" begin
    # On a sphere each pair's direction lives in its own frame, so an angle to one fixed reference
    # axis is not a property of the pair. Refused by name rather than approximated.
    Random.seed!(7500)
    x = [0.1 0.2 0.35; -0.2 0.05 0.3]
    u = randn(2, 3)
    Test.@test_throws ArgumentError SFC.serial_calculate_structure_function(
        SF2, x, u, collect(range(0.0, 2.0; length = 4)), collect(range(0.0, π; length = 4));
        distance_metric = SFC.DI.SphericalAngle(),
        second_axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)),
        verbose = false, show_progress = false)
end
