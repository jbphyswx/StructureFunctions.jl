using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using StaticArrays: StaticArrays as SA
using Random: Random

const SF2 = SFT.L2SFType()

function _joint(sf, x, u, dist_bins, ax_bins, source)
    return SFC.serial_calculate_structure_function(
        sf, x, u, dist_bins, ax_bins, UInt32; second_axis = source)
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
        SFC.calculate_structure_function!(ref_s, ref_c, SF2, x, u, dist_bins;
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
    plain = SFC.serial_calculate_structure_function(SF2, x, u, dist_bins, val_bins, UInt32)
    Test.@test with_source.counts == plain.counts
    Test.@test with_source.sums == plain.sums
    Test.@test sum(plain.counts) > 0
end

Test.@testset "the angle axis reaches every backend through the public entry" begin
    # Every testset above calls the serial kernel directly, so none of them sees a backend that
    # takes the keyword and bins something else. This one goes through the entry a user calls.
    Random.seed!(7450)
    N = 200
    x = rand(2, N)
    u = randn(2, N)
    dist_bins = collect(range(0.0, 1.0; length = 7))
    ax_bins = collect(range(prevfloat(0.0), π; length = 5))
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    through(backend) = SFC.calculate_structure_function(
        SF2, x, u, dist_bins, ax_bins; backend = backend, second_axis = src)

    ref = through(CB.SerialBackend())
    Test.@test sum(ref.counts) > 0
    # The control that makes the rows below mean something: binning the angle must not reproduce
    # the default value binning, so a backend that drops the keyword cannot pass by accident.
    value_binned = SFC.calculate_structure_function(
        SF2, x, u, dist_bins, ax_bins; backend = CB.SerialBackend())
    Test.@test value_binned.counts != ref.counts

    thr = through(CB.ThreadedBackend())
    Test.@test thr.counts == ref.counts
    Test.@test thr.sums ≈ ref.sums

    dev = through(CB.GPUBackend(KA.CPU()))
    Test.@test dev.counts == ref.counts
    Test.@test dev.sums ≈ ref.sums

    # A histogram wider than the shared-memory cap takes the device's global-atomic route, which
    # is a different kernel and was unreachable until its launcher was given the geometry.
    wide = collect(range(prevfloat(0.0), π; length = 201))
    wide_ref = SFC.calculate_structure_function(SF2, x, u, dist_bins, wide;
        backend = CB.SerialBackend(), second_axis = src)
    wide_dev = SFC.calculate_structure_function(SF2, x, u, dist_bins, wide;
        backend = CB.GPUBackend(KA.CPU()), second_axis = src)
    Test.@test wide_dev.counts == wide_ref.counts
    Test.@test wide_dev.sums ≈ wide_ref.sums

    # Three coordinates on that same route: the global-atomic kernels read the width off the
    # geometry, and reading two of three silently returns a two-dimensional answer.
    x3, u3 = rand(3, N), randn(3, N)
    src3 = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0, 0.0))
    ref3 = SFC.calculate_structure_function(SF2, x3, u3, dist_bins, wide;
        backend = CB.SerialBackend(), second_axis = src3)
    dev3 = SFC.calculate_structure_function(SF2, x3, u3, dist_bins, wide;
        backend = CB.GPUBackend(KA.CPU()), second_axis = src3)
    Test.@test sum(ref3.counts) > 0
    Test.@test dev3.counts == ref3.counts
    Test.@test dev3.sums ≈ ref3.sums

    # On a curved metric each pair's direction is in its own frame, so the device refuses too.
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        SF2, x, u, dist_bins, ax_bins; backend = CB.GPUBackend(KA.CPU()),
        second_axis = src, distance_metric = SFC.DI.SphericalAngle())
end

Test.@testset "the angle axis on a trailing batch axis" begin
    # Positions every slice shares are binned by angle once per pair, varying ones per slice; each
    # slice must equal the point entry on that slice.
    Random.seed!(7600)
    N, B = 300, 3
    dist_bins = collect(range(0.0, 1.0; length = 7))
    ax_bins = collect(range(prevfloat(0.0), π; length = 5))
    src = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
    for x in (rand(2, N), rand(2, N, B)),
        backend in (CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        u = randn(2, N, B)
        got = SFC.calculate_structure_function(SF2, x, u, dist_bins, ax_bins; backend = backend,
            second_axis = src)
        for b in 1:B
            ref = SFC.calculate_structure_function(SF2, ndims(x) == 2 ? x : x[:, :, b], u[:, :, b],
                dist_bins, ax_bins; backend = CB.SerialBackend(), second_axis = src)
            Test.@test got.counts[:, :, b] == ref.counts
            Test.@test got.sums[:, :, b] ≈ ref.sums
        end
        value_binned = SFC.calculate_structure_function(SF2, x, u, dist_bins, ax_bins; backend = backend)
        Test.@test value_binned.counts != got.counts
    end

    # three coordinates: summing the angle recovers the batch distance histogram
    x3, u3 = rand(3, N, B), randn(3, N, B)
    ax3 = collect(range(prevfloat(0.0), π / 2 + 1e-9; length = 5))
    joint3 = SFC.calculate_structure_function(SF2, x3, u3, dist_bins, ax3; backend = CB.SerialBackend(),
        second_axis = SFC.SeparationAngleAxis(SA.SVector(0.0, 0.0, 1.0)))
    plain3 = SFC.calculate_structure_function(SF2, x3, u3, dist_bins, SF.StructureFunctionSumsAndCounts;
        backend = CB.SerialBackend())
    Test.@test dropdims(sum(joint3.counts; dims = 2); dims = 2) == plain3.counts
    Test.@test sum(plain3.counts) > 0

    for backend in (CB.SerialBackend(), CB.GPUBackend(KA.CPU()))
        Test.@test_throws ArgumentError SFC.calculate_structure_function(SF2, [0.1 0.2 0.35; -0.2 0.05 0.3],
            randn(2, 3, B), collect(range(0.0, 2.0; length = 4)), collect(range(0.0, π; length = 4));
            backend, distance_metric = SFC.DI.SphericalAngle(), second_axis = src)
    end
end

Test.@testset "an angle axis is refused where the direction is not shared" begin
    # On a sphere each pair's direction lives in its own frame, so an angle to one fixed reference
    # axis is not a property of the pair, so it is refused by name.
    Random.seed!(7500)
    x = [0.1 0.2 0.35; -0.2 0.05 0.3]
    u = randn(2, 3)
    Test.@test_throws ArgumentError SFC.serial_calculate_structure_function(
        SF2, x, u, collect(range(0.0, 2.0; length = 4)), collect(range(0.0, π; length = 4)), UInt32;
        distance_metric = SFC.DI.SphericalAngle(),
        second_axis = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)))
end
