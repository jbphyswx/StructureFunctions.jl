using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, MultiFields as MF, HelperFunctions as SFH
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using OhMyThreads: OhMyThreads
using Distances: Distances as DI
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

# Every pair, evaluated straight from the inputs. Independent of the packing and of the sweep.
function _brute(op, x, vectors, scalars, bins)
    N = size(x, 2)
    D = isempty(vectors) ? 0 : size(first(vectors), 1)
    nb = length(bins) - 1
    sums = zeros(Float64, nb)
    counts = zeros(Int, nb)
    for i in 1:(N - 1), j in (i + 1):N
        dx = SA.SVector{size(x, 1)}(x[:, j] - x[:, i])
        r = sqrt(LA.dot(dx, dx))
        b = searchsortedfirst(bins, r) - 1
        1 <= b <= nb || continue
        rh = dx / r
        dv = [SA.SVector{D}(v[:, j] - v[:, i]) for v in vectors]
        ds = [s[j] - s[i] for s in scalars]
        sums[b] += op(dv, ds, rh)
        counts[b] += 1
    end
    return sums, counts
end

_run(op, x, f, bins; kw...) = SFC.calculate_structure_function(
    op, x, f, bins, UInt32, SF.StructureFunctionSumsAndCounts; culling = SFC.NoCulling(), kw...)

const ONE_VECTOR_FIELD_CASES = ((SFT.L2SFType(), CB.SerialBackend()), (SFT.T2SFType(), CB.ThreadedBackend()),
                                (SFT.S2SFType(), CB.SerialBackend()), (SFT.L3SFType(), CB.SerialBackend()))

Test.@testset "a field of one vector field is that velocity's structure function" begin
    Random.seed!(1200)
    x = rand(2, 80)
    u = randn(2, 80)
    bins = collect(range(0.0, 1.5; length = 7))
    for (op, backend) in ONE_VECTOR_FIELD_CASES
        got = _run(op, x, MF.Fields(vectors = (u,)), bins; backend)
        ref_s, ref_c = _brute((dv, ds, rh) -> op(dv[1], rh), x, (u,), (), bins)
        Test.@test (op, got.counts == ref_c, isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)) == (op, true, true)
    end
end

Test.@testset "packing lays fields out as declared" begin
    Random.seed!(1300)
    u = randn(3, 6)
    a = randn(3, 6)
    th = randn(6)
    ph = randn(6)
    f = MF.Fields(vectors = (u, a), scalars = (th, ph))
    Test.@test MF.field_dimension(f) == 3
    Test.@test MF.n_vector_fields(f) == 2
    Test.@test MF.n_scalar_fields(f) == 2
    d = MF.packed(f)
    Test.@test size(d) == (3 * 2 + 2, 6)
    Test.@test d[1:3, :] == u
    Test.@test d[4:6, :] == a
    Test.@test d[7, :] == th
    Test.@test d[8, :] == ph
end

Test.@testset "the multi-field refuses what it cannot mean" begin
    Test.@test_throws ArgumentError MF.Fields()
    Test.@test_throws DimensionMismatch MF.Fields(vectors = (randn(2, 5), randn(3, 5)))
    Test.@test_throws DimensionMismatch MF.Fields(vectors = (randn(2, 5),), scalars = (randn(4),))
    Test.@test_throws ArgumentError MF.Fields(vectors = (randn(5),))
    Test.@test_throws DimensionMismatch MF.Fields(vectors = (randn(2, 5, 2),), scalars = (randn(5, 3),))
    # a grid-shaped field is one vector field over the flattened cells
    Test.@test size(MF.packed(MF.Fields(vectors = (randn(2, 5, 2),), scalars = (randn(5, 2),)))) == (3, 10)
end

Test.@testset "the scalar structure function matches brute force" begin
    Random.seed!(1400)
    N = 70
    x = rand(2, N)
    th = randn(N)
    bins = collect(range(0.0, 1.5; length = 7))   # spans the unit square diagonal
    f = MF.Fields(vectors = (randn(2, N),), scalars = (th,))
    for P in (2, 3)
        got = _run(SFT.ScalarSFType{P}(), x, f, bins)
        # an odd power reads the pair from the lower to the upper end along the first separating axis
        orient(rh) = isodd(P) ? (rh[1] != 0 ? sign(rh[1]) : sign(rh[2])) : 1.0
        ref_s, ref_c = _brute((dv, ds, rh) -> orient(rh) * ds[1]^P, x, (), (th,), bins)
        Test.@test got.counts == ref_c
        Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
        Test.@test sum(got.counts) == N * (N - 1) ÷ 2
    end
end

Test.@testset "Yaglom's mixed moment matches brute force" begin
    # ⟨δu_L (δθ)²⟩ — the velocity part read from a transported field, the scalar part from a
    # differenced one, so the two never share a frame by accident.
    Random.seed!(1500)
    N = 70
    x = rand(2, N)
    u = randn(2, N)
    th = randn(N)
    bins = collect(range(0.0, 1.5; length = 7))   # spans the unit square diagonal
    f = MF.Fields(vectors = (u,), scalars = (th,))
    got = _run(SFT.MixedSFType{1, 0, 2}(), x, f, bins)
    ref_s, ref_c = _brute((dv, ds, rh) -> LA.dot(dv[1], rh) * ds[1]^2, x, (u,), (th,), bins)
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)

    # and the first-order flux of the tracer itself
    got1 = _run(SFT.MixedSFType{1, 0, 1}(), x, f, bins)
    orient(rh) = rh[1] != 0 ? sign(rh[1]) : sign(rh[2])     # odd in θ: read along the first separating axis
    ref1_s, _ = _brute((dv, ds, rh) -> orient(rh) * LA.dot(dv[1], rh) * ds[1], x, (u,), (th,), bins)
    Test.@test isapprox(got1.sums, ref1_s; rtol = 1e-10, atol = 1e-12)
end

const SCALARS_ALONE_CASES = ((2, SFT.ScalarSFType{2}(), (ds -> ds[1]^2)),
                             (3, SFT.ScalarDotSFType(1, 2), (ds -> ds[1] * ds[2])))

Test.@testset "a field of scalars alone is located by its coordinates" begin
    # With no vector field the points' coordinate count sets the geometry; checked against brute force.
    Random.seed!(1550)
    N = 60
    for (Dx, op, value) in SCALARS_ALONE_CASES
        x = rand(Dx, N)
        th = randn(N)
        ph = randn(N)
        bins = collect(range(0.0, 1.2; length = 6))
        f = MF.Fields(scalars = (th, ph))
        got = _run(op, x, f, bins)
        ref_s, ref_c = _brute((dv, ds, rh) -> value(ds), x, (), (th, ph), bins)
        Test.@test got.counts == ref_c
        Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
    end
end

# (Dx, metric, operator, backends checked against the serial answer)
const ODD_MOMENT_CASES = (
    (2, DI.Euclidean(), SFT.ScalarSFType{3}(), (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))),
    (3, DI.Euclidean(), SFT.MixedSFType{1, 0, 1}(), ()),
    (2, DI.Haversine(6.371e6), SFT.MixedSFType{1, 0, 1}(), (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))),
)

Test.@testset "odd scalar moments do not depend on how the points are ordered" begin
    # Each pair is read from its lower to its upper end along the first separating coordinate, on every backend.
    Random.seed!(1570)
    N = 60
    for (Dx, metric, sf, backends) in ODD_MOMENT_CASES
        x = Dx == 2 && metric isa DI.Haversine ? vcat(120 .* rand(1, N) .- 60, 100 .* rand(1, N) .- 50) : rand(Dx, N)
        u = randn(Dx == 3 ? 3 : 2, N)
        th = randn(N)
        bins = metric isa DI.Haversine ? collect(range(0.0, 1.2e7; length = 6)) :
                                         collect(range(0.0, 1.5; length = 6))
        perm = Random.randperm(N)
        f = MF.Fields(vectors = (u,), scalars = (th,))
        fp = MF.Fields(vectors = (u[:, perm],), scalars = (th[perm],))
        a = SFC.calculate_structure_function(sf, x, f, bins, SF.StructureFunctionSumsAndCounts;
            backend = CB.SerialBackend(), distance_metric = metric)
        b = SFC.calculate_structure_function(sf, x[:, perm], fp, bins, SF.StructureFunctionSumsAndCounts;
            backend = CB.SerialBackend(), distance_metric = metric)
        Test.@test a.counts == b.counts
        Test.@test isapprox(a.sums, b.sums; rtol = 1e-10, atol = 1e-12)
        Test.@test any(!iszero, a.sums)
        for backend in backends
            d = SFC.calculate_structure_function(sf, x[:, perm], fp, bins, SF.StructureFunctionSumsAndCounts;
                backend, distance_metric = metric)
            Test.@test d.counts == a.counts
            Test.@test isapprox(d.sums, a.sums; rtol = 1e-10, atol = 1e-12)
        end
        if metric isa DI.Euclidean
            ref_s, ref_c = _brute((dv, ds, rh) -> (rh[1] != 0 ? sign(rh[1]) : sign(rh[2])) *
                (sf isa SFT.ScalarSFType ? ds[1]^3 : LA.dot(dv[1], rh) * ds[1]), x, (u,), (th,), bins)
            Test.@test a.counts == ref_c
            Test.@test isapprox(a.sums, ref_s; rtol = 1e-10, atol = 1e-12)
        end
    end
end

Test.@testset "cross-field moments are what an advective structure function is" begin
    # ⟨δu · δ𝓐⟩ and ⟨δω δ𝓐_ω⟩ — second-order moments between two different fields.
    Random.seed!(1600)
    N = 60
    x = rand(2, N)
    u = randn(2, N)
    adv = randn(2, N)
    w = randn(N)
    advw = randn(N)
    bins = collect(range(0.0, 1.3; length = 6))

    fv = MF.Fields(vectors = (u, adv))
    got = _run(SFT.VectorDotSFType(1, 2), x, fv, bins)
    ref_s, ref_c = _brute((dv, ds, rh) -> LA.dot(dv[1], dv[2]), x, (u, adv), (), bins)
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)

    fs = MF.Fields(vectors = (u,), scalars = (w, advw))
    gots = _run(SFT.ScalarDotSFType(1, 2), x, fs, bins)
    refs_s, _ = _brute((dv, ds, rh) -> ds[1] * ds[2], x, (u,), (w, advw), bins)
    Test.@test isapprox(gots.sums, refs_s; rtol = 1e-10, atol = 1e-12)

    # the diagonal of the vector cross-moment IS the second-order structure function
    diag = _run(SFT.VectorDotSFType(1, 1), x, fv, bins)
    s2 = _run(SFT.S2SFType(), x, MF.Fields(vectors = (u,)), bins)
    Test.@test diag.counts == s2.counts
    Test.@test isapprox(diag.sums, s2.sums; rtol = 1e-12)
end

Test.@testset "asking for a field a field does not carry says so" begin
    Random.seed!(1700)
    x = rand(2, 20)
    u = randn(2, 20)
    bins = collect(range(0.0, 1.0; length = 4))
    # a plain velocity has no scalar field
    Test.@test_throws ArgumentError _run(SFT.ScalarSFType{2}(), x, MF.Fields(vectors = (u,)), bins)
    # and only one vector field
    Test.@test_throws ArgumentError _run(SFT.VectorDotSFType(1, 2), x, MF.Fields(vectors = (u,)), bins)
end

Test.@testset "fields are transported on a sphere, scalars are not" begin
    # Every vector field is transported as a bare velocity is; a scalar is differenced as it stands.
    Random.seed!(1800)
    N = 50
    x = vcat(reshape(2π .* rand(N), 1, N), reshape((rand(N) .- 0.5) .* 1.4, 1, N))
    u = randn(2, N)
    th = randn(N)
    bins = collect(range(0.0, 2.4; length = 6))
    metric = SFC.DI.SphericalAngle()

    bare_s = zeros(5); bare_c = zeros(UInt32, 5)
    SFC.calculate_structure_function!(bare_s, bare_c, SFT.L2SFType(), x, u, bins;
                                     distance_metric = metric, backend = CB.SerialBackend())
    multi = SFC.calculate_structure_function(
        SFT.L2SFType(), x, MF.Fields(vectors = (u,)), bins, UInt32, SF.StructureFunctionSumsAndCounts;
        distance_metric = metric, backend = CB.SerialBackend())
    Test.@test multi.counts == bare_c
    Test.@test isapprox(multi.sums, bare_s; rtol = 1e-12)

    with_tracer = SFC.calculate_structure_function(
        SFT.L2SFType(), x, MF.Fields(vectors = (u,), scalars = (th,)), bins, UInt32, SF.StructureFunctionSumsAndCounts;
        distance_metric = metric)
    Test.@test with_tracer.counts == bare_c
    Test.@test isapprox(with_tracer.sums, bare_s; rtol = 1e-12)

    scalar_only = SFC.calculate_structure_function(
        SFT.ScalarSFType{2}(), x, MF.Fields(scalars = (th,)), bins, UInt32, SF.StructureFunctionSumsAndCounts;
        distance_metric = metric)
    ref_s = zeros(5); ref_c = zeros(Int, 5)
    for i in 1:(N - 1), j in (i + 1):N
        r = SFC.DI.SphericalAngle()(view(x, :, i), view(x, :, j))
        b = searchsortedfirst(bins, r) - 1
        1 <= b <= 5 || continue
        ref_s[b] += (th[j] - th[i])^2
        ref_c[b] += 1
    end
    Test.@test scalar_only.counts == UInt32.(ref_c)
    Test.@test isapprox(scalar_only.sums, ref_s; rtol = 1e-10, atol = 1e-12)
    Test.@test sum(scalar_only.counts) > 0

    yag = SFC.calculate_structure_function(
        SFT.MixedSFType{1, 0, 2}(), x, MF.Fields(vectors = (u,), scalars = (th,)), bins, UInt32,
        SF.StructureFunctionSumsAndCounts; distance_metric = metric)
    Test.@test all(isfinite, yag.sums)
    Test.@test yag.counts == bare_c
end

Test.@testset "the threaded backend gives the serial answer" begin
    # Every pair is swept exactly once across the tasks; only the summation order may differ from serial.
    Random.seed!(1900)
    N = 200
    x = rand(2, N)
    u = randn(2, N)
    adv = randn(2, N)
    th = randn(N)
    bins = collect(range(0.0, 1.5; length = 8))   # spans the unit square diagonal
    nb = length(bins) - 1

    for (f, op) in ((MF.Fields(vectors = (u,), scalars = (th,)), SFT.MixedSFType{1, 0, 2}()),
                    (MF.Fields(vectors = (u, adv)), SFT.VectorDotSFType(1, 2)),
                    (MF.Fields(vectors = (u,), scalars = (th,)), SFT.ScalarSFType{2}()),
                    (MF.Fields(vectors = (u,)), SFT.L2SFType()))
        ser_s = zeros(nb); ser_c = zeros(Int, nb)
        SFC.serial_calculate_structure_function!(ser_s, ser_c, op, x, f, bins; geometry = SFH.FlatGeometry{2}(),
                                                 culling = SFC.NoCulling())
        thr_s = zeros(nb); thr_c = zeros(Int, nb)
        SFC.threaded_calculate_structure_function!(thr_s, thr_c, op, x, f, bins; geometry = SFH.FlatGeometry{2}(),
                                                   culling = SFC.NoCulling())
        Test.@test thr_c == ser_c
        Test.@test isapprox(thr_s, ser_s; rtol = 1e-10, atol = 1e-12)
        Test.@test sum(thr_c) == N * (N - 1) ÷ 2
    end
end

const DEVICE_FIELD_CASES = ((SFT.MixedSFType{1, 0, 2}(), false, SFC.NoCulling()),
                            (SFT.MixedSFType{1, 2, 1}(), true, SFC.AlwaysCulling()),
                            (SFT.VectorDotSFType(1, 2), true, SFC.NoCulling()),
                            (SFT.ScalarDotSFType(1, 2), false, SFC.AlwaysCulling()),
                            (SFT.ScalarSFType{3}(2), false, SFC.NoCulling()))

Test.@testset "the device gives the serial answer, culled with or without a workspace, and adds" begin
    Random.seed!(1950)
    N = 300
    x = rand(2, N)
    f = MF.Fields(vectors = (randn(2, N), randn(2, N)), scalars = (randn(N), randn(N)))
    w = 0.5 .+ rand(N)
    bins = collect(range(0.0, 0.2; length = 7))
    nb = length(bins) - 1
    dev = CB.GPUBackend(KA.CPU())
    RAW = SF.StructureFunctionSumsAndCounts
    for (op, weighted, pol) in DEVICE_FIELD_CASES
        CT = weighted ? Float64 : Int
        kw = weighted ? (; weights = w) : (;)
        ref = SFC.calculate_structure_function(op, x, f, bins, CT, RAW; backend = CB.SerialBackend(), kw...)
        case = (nameof(typeof(op)), weighted, pol)
        Test.@test sum(ref.counts) > 0
        ws = SFC.GPUSFWorkspace(KA.CPU(), bins)
        got = SFC.calculate_structure_function(op, x, f, bins, CT, RAW; backend = dev, workspace = ws,
                                               culling = pol, kw...)
        Test.@test (case, isapprox(got.counts, ref.counts; rtol = 1e-12)) == (case, true)
        Test.@test (case, isapprox(got.sums, ref.sums; rtol = 1e-10, atol = 1e-12)) == (case, true)
        if pol isa SFC.AlwaysCulling
            got = SFC.calculate_structure_function(op, x, f, bins, CT, RAW; backend = dev,
                                                   culling = SFC.AlwaysCulling(), kw...)
            Test.@test (case, isapprox(got.counts, ref.counts; rtol = 1e-12),
                        isapprox(got.sums, ref.sums; rtol = 1e-10, atol = 1e-12)) == (case, true, true)
        else
            s, c = zeros(nb), zeros(CT, nb)
            for _ in 1:2
                SFC.calculate_structure_function!(s, c, op, x, f, bins; backend = dev, culling = pol, kw...)
            end
            Test.@test (case, isapprox(c, 2 .* ref.counts; rtol = 1e-12)) == (case, true)
            Test.@test (case, isapprox(s, 2 .* ref.sums; rtol = 1e-10, atol = 1e-12)) == (case, true)
        end
    end
    # on a sphere the vector fields are transported and the scalars are not
    xs = vcat(reshape(2π .* rand(N), 1, N), reshape((rand(N) .- 0.5) .* 1.4, 1, N))
    sbins = collect(range(0.0, 0.6; length = 6))
    for op in (SFT.MixedSFType{1, 0, 2}(), SFT.VectorDotSFType(1, 2))
        ref = SFC.calculate_structure_function(op, xs, f, sbins, Int, RAW; backend = CB.SerialBackend(),
                                               distance_metric = DI.SphericalAngle())
        got = SFC.calculate_structure_function(op, xs, f, sbins, Int, RAW; backend = dev,
                                               distance_metric = DI.SphericalAngle())
        Test.@test sum(ref.counts) > 0
        Test.@test got.counts == ref.counts
        Test.@test isapprox(got.sums, ref.sums; rtol = 1e-10, atol = 1e-12)
    end
end
