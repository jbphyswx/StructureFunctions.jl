using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, MultiFields as MF
using StructureFunctions.StructureFunctionTypes: MixedSFType, ScalarSFType, VectorDotSFType, ScalarDotSFType,
    MixedStructureFunctionType
using StaticArrays: StaticArrays as SA
using LinearAlgebra: dot
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
        r = sqrt(dot(dx, dx))
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

_run(op, x, f, bins) = SFC.calculate_structure_function(
    op, x, f, bins, UInt32; output_type = SF.StructureFunctionSumsAndCounts,
    verbose = false, show_progress = false)

Test.@testset "a field of one vector field is the array path" begin
    # The adapter must be a no-op for what callers already pass: same kernel, same answer, bit for bit.
    Random.seed!(1200)
    x = rand(2, 80)
    u = randn(2, 80)
    bins = collect(range(0.0, 1.5; length = 7))   # spans the unit square diagonal
    for op in (SFT.L2SFType(), SFT.T2SFType(), SFT.S2SFType(), SFT.L3SFType())
        bare = SFC.calculate_structure_function(op, x, u, bins, UInt32;
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
        multi = _run(op, x, MF.Fields(vectors = (u,)), bins)
        Test.@test multi.counts == bare.counts
        Test.@test multi.sums == bare.sums          # identical, not merely close
    end
    # and the packing itself copies nothing it need not
    Test.@test MF.packed(MF.Fields(vectors = (u,))) == u
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
    ref_s, ref_c = _brute((dv, ds, rh) -> dot(dv[1], rh) * ds[1]^2, x, (u,), (th,), bins)
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)

    # and the first-order flux of the tracer itself
    got1 = _run(SFT.MixedSFType{1, 0, 1}(), x, f, bins)
    orient(rh) = rh[1] != 0 ? sign(rh[1]) : sign(rh[2])     # odd in θ: read along the first separating axis
    ref1_s, _ = _brute((dv, ds, rh) -> orient(rh) * dot(dv[1], rh) * ds[1], x, (u,), (th,), bins)
    Test.@test isapprox(got1.sums, ref1_s; rtol = 1e-10, atol = 1e-12)
end

Test.@testset "a field of scalars alone is located by its coordinates" begin
    # With no vector field there is no velocity dimension to read the geometry from; the points
    # carry it. Checked against brute force on two- and three-dimensional points.
    Random.seed!(1550)
    N = 60
    for Dx in (2, 3)
        x = rand(Dx, N)
        th = randn(N)
        ph = randn(N)
        bins = collect(range(0.0, 1.2; length = 6))
        f = MF.Fields(scalars = (th, ph))
        got = _run(SFT.ScalarSFType{2}(), x, f, bins)
        ref_s, ref_c = _brute((dv, ds, rh) -> ds[1]^2, x, (), (th, ph), bins)
        Test.@test got.counts == ref_c
        Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
        gotx = _run(SFT.ScalarDotSFType(1, 2), x, f, bins)
        refx_s, _ = _brute((dv, ds, rh) -> ds[1] * ds[2], x, (), (th, ph), bins)
        Test.@test isapprox(gotx.sums, refx_s; rtol = 1e-10, atol = 1e-12)
    end
end

Test.@testset "odd scalar moments do not depend on how the points are ordered" begin
    # ⟨δu_L δθ⟩ and ⟨(δθ)³⟩ change sign when a pair is read from its other end, so their value is fixed
    # by reading every pair from the lower to the upper end along the first separating coordinate —
    # never by the order the points arrive in.
    Random.seed!(1570)
    N = 60
    for (Dx, metric) in ((2, DI.Euclidean()), (3, DI.Euclidean()), (2, DI.Haversine(6.371e6)))
        x = Dx == 2 && metric isa DI.Haversine ? vcat(120 .* rand(1, N) .- 60, 100 .* rand(1, N) .- 50) : rand(Dx, N)
        u = randn(Dx == 3 ? 3 : 2, N)
        th = randn(N)
        bins = metric isa DI.Haversine ? collect(range(0.0, 1.2e7; length = 6)) :
                                         collect(range(0.0, 1.5; length = 6))
        perm = Random.randperm(N)
        f = MF.Fields(vectors = (u,), scalars = (th,))
        fp = MF.Fields(vectors = (u[:, perm],), scalars = (th[perm],))
        for sf in (SFT.MixedSFType{1, 0, 1}(), SFT.ScalarSFType{3}())
            a = SFC.calculate_structure_function(sf, x, f, bins; backend = CB.SerialBackend(),
                distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
                verbose = false, show_progress = false)
            b = SFC.calculate_structure_function(sf, x[:, perm], fp, bins; backend = CB.SerialBackend(),
                distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
                verbose = false, show_progress = false)
            Test.@test a.counts == b.counts
            Test.@test isapprox(a.sums, b.sums; rtol = 1e-10, atol = 1e-12)
            Test.@test any(!iszero, a.sums)
            c = SFC.calculate_structure_function(sf, x[:, perm], fp, bins; backend = CB.ThreadedBackend(),
                distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
                verbose = false, show_progress = false)
            Test.@test isapprox(c.sums, a.sums; rtol = 1e-10, atol = 1e-12)
            d = SFC.calculate_structure_function(sf, x[:, perm], fp, bins;
                backend = CB.GPUBackend(KA.CPU()), distance_metric = metric,
                output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
            Test.@test d.counts == a.counts
            Test.@test isapprox(d.sums, a.sums; rtol = 1e-10, atol = 1e-12)
            if metric isa DI.Euclidean
                # the reading is the lexicographic one on the displacement, checked pair by pair
                ref_s, ref_c = _brute((dv, ds, rh) -> (rh[1] != 0 ? sign(rh[1]) : sign(rh[2])) *
                    (sf isa SFT.ScalarSFType ? ds[1]^3 : dot(dv[1], rh) * ds[1]), x, (u,), (th,), bins)
                Test.@test a.counts == ref_c
                Test.@test isapprox(a.sums, ref_s; rtol = 1e-10, atol = 1e-12)
            end
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
    ref_s, ref_c = _brute((dv, ds, rh) -> dot(dv[1], dv[2]), x, (u, adv), (), bins)
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
    # A vector field is carried as an ambient 3-vector on a sphere, so a multi-field must widen every
    # vector field exactly as the array path widens the one it has. A scalar has nothing to
    # transport and passes through untouched.
    Random.seed!(1800)
    N = 50
    x = vcat(reshape(2π .* rand(N), 1, N), reshape((rand(N) .- 0.5) .* 1.4, 1, N))
    u = randn(2, N)
    th = randn(N)
    bins = collect(range(0.0, 2.4; length = 6))
    metric = SFC.DI.SphericalAngle()

    # one vector field: the same kernel as the array path, so identical to the last bit
    # Both pinned to the same backend: the claim is that the multi-field takes the *same kernel*, and a
    # different backend would differ in summation order alone, which would not test that.
    bare_s = zeros(5); bare_c = zeros(UInt32, 5)
    SFC.calculate_structure_function!(bare_s, bare_c, SFT.L2SFType(), x, u, bins;
                                     distance_metric = metric, backend = CB.SerialBackend())
    multi = SFC.calculate_structure_function(
        SFT.L2SFType(), x, MF.Fields(vectors = (u,)), bins, UInt32; distance_metric = metric,
        backend = CB.SerialBackend(),
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test multi.counts == bare_c
    Test.@test multi.sums == bare_s

    # a scalar rides along without disturbing the velocity part: L2SF on the multi-field must still equal
    # L2SF on the velocity alone
    with_tracer = SFC.calculate_structure_function(
        SFT.L2SFType(), x, MF.Fields(vectors = (u,), scalars = (th,)), bins, UInt32;
        distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    Test.@test with_tracer.counts == bare_c
    Test.@test isapprox(with_tracer.sums, bare_s; rtol = 1e-12)

    # the scalar structure function on a sphere: transport-free, so it is the plain difference
    scalar_only = SFC.calculate_structure_function(
        SFT.ScalarSFType{2}(), x, MF.Fields(scalars = (th,)), bins, UInt32;
        distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
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

    # Yaglom on a sphere runs and stays finite; its velocity half is transported, so it is not the
    # flat answer
    yag = SFC.calculate_structure_function(
        SFT.MixedSFType{1, 0, 2}(), x, MF.Fields(vectors = (u,), scalars = (th,)), bins, UInt32;
        distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    Test.@test all(isfinite, yag.sums)
    Test.@test yag.counts == bare_c
end

Test.@testset "the threaded backend gives the serial answer" begin
    # Multi-field across threads: the setup happens once above the task loop and each task sweeps
    # its own outer indices, so the only thing that may differ from serial is summation order.
    Random.seed!(1900)
    N = 400
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
        SFC.serial_calculate_structure_function!(ser_s, ser_c, op, x, f, bins;
                                                 verbose = false, show_progress = false)
        thr_s = zeros(nb); thr_c = zeros(Int, nb)
        SFC.threaded_calculate_structure_function!(thr_s, thr_c, op, x, f, bins;
                                                   verbose = false, show_progress = false)
        Test.@test thr_c == ser_c
        Test.@test isapprox(thr_s, ser_s; rtol = 1e-10, atol = 1e-12)
        Test.@test sum(thr_c) == N * (N - 1) ÷ 2
    end
end

Test.@testset "the field operators are exported" begin
    Test.@test MixedSFType === MixedStructureFunctionType
    Test.@test MixedSFType{1, 0, 2}() === SFT.MixedSFType{1, 0, 2}()
    Test.@test ScalarSFType{2}() === SFT.ScalarSFType{2}()
    Test.@test VectorDotSFType(1, 2) === SFT.VectorDotSFType(1, 2)
    Test.@test ScalarDotSFType(1, 2) === SFT.ScalarDotSFType(1, 2)
end
