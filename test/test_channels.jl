using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, Channels as CH
using StaticArrays: StaticArrays as SA
using LinearAlgebra: dot
using OhMyThreads: OhMyThreads
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

Test.@testset "a field of one vector channel is the array path" begin
    # The adapter must be a no-op for what callers already pass: same kernel, same answer, bit for bit.
    Random.seed!(1200)
    x = rand(2, 80)
    u = randn(2, 80)
    bins = collect(range(0.0, 1.5; length = 7))   # spans the unit square diagonal
    for op in (SFT.L2SFType(), SFT.T2SFType(), SFT.S2SFType(), SFT.L3SFType())
        bare = SFC.calculate_structure_function(op, x, u, bins, UInt32;
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
        bundled = _run(op, x, SF.Fields(vectors = (u,)), bins)
        Test.@test bundled.counts == bare.counts
        Test.@test bundled.sums == bare.sums          # identical, not merely close
    end
    # and the packing itself copies nothing it need not
    Test.@test CH.packed(SF.Fields(vectors = (u,))) == u
end

Test.@testset "packing lays channels out as declared" begin
    Random.seed!(1300)
    u = randn(3, 6)
    a = randn(3, 6)
    th = randn(6)
    ph = randn(6)
    f = SF.Fields(vectors = (u, a), scalars = (th, ph))
    Test.@test CH.channel_dimension(f) == 3
    Test.@test CH.n_vector_channels(f) == 2
    Test.@test CH.n_scalar_channels(f) == 2
    d = CH.packed(f)
    Test.@test size(d) == (3 * 2 + 2, 6)
    Test.@test d[1:3, :] == u
    Test.@test d[4:6, :] == a
    Test.@test d[7, :] == th
    Test.@test d[8, :] == ph
end

Test.@testset "the bundle refuses what it cannot mean" begin
    Test.@test_throws ArgumentError SF.Fields()
    Test.@test_throws DimensionMismatch SF.Fields(vectors = (randn(2, 5), randn(3, 5)))
    Test.@test_throws DimensionMismatch SF.Fields(vectors = (randn(2, 5),), scalars = (randn(4),))
    Test.@test_throws ArgumentError SF.Fields(vectors = (randn(2, 5, 2),))
end

Test.@testset "the scalar structure function matches brute force" begin
    Random.seed!(1400)
    N = 70
    x = rand(2, N)
    th = randn(N)
    bins = collect(range(0.0, 1.5; length = 7))   # spans the unit square diagonal
    f = SF.Fields(vectors = (randn(2, N),), scalars = (th,))
    for P in (2, 3)
        got = _run(SFT.ScalarSFType{P}(), x, f, bins)
        ref_s, ref_c = _brute((dv, ds, rh) -> ds[1]^P, x, (), (th,), bins)
        Test.@test got.counts == ref_c
        Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)
        Test.@test sum(got.counts) == N * (N - 1) ÷ 2
    end
end

Test.@testset "Yaglom's mixed moment matches brute force" begin
    # ⟨δu_L (δθ)²⟩ — the velocity part read from a transported channel, the scalar part from a
    # differenced one, so the two never share a frame by accident.
    Random.seed!(1500)
    N = 70
    x = rand(2, N)
    u = randn(2, N)
    th = randn(N)
    bins = collect(range(0.0, 1.5; length = 7))   # spans the unit square diagonal
    f = SF.Fields(vectors = (u,), scalars = (th,))
    got = _run(SFT.MixedSFType{1, 0, 2}(), x, f, bins)
    ref_s, ref_c = _brute((dv, ds, rh) -> dot(dv[1], rh) * ds[1]^2, x, (u,), (th,), bins)
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)

    # and the first-order flux of the tracer itself
    got1 = _run(SFT.MixedSFType{1, 0, 1}(), x, f, bins)
    ref1_s, _ = _brute((dv, ds, rh) -> dot(dv[1], rh) * ds[1], x, (u,), (th,), bins)
    Test.@test isapprox(got1.sums, ref1_s; rtol = 1e-10, atol = 1e-12)
end

Test.@testset "cross-channel moments are what an advective structure function is" begin
    # ⟨δu · δ𝓐⟩ and ⟨δω δ𝓐_ω⟩ — second-order moments between two different channels.
    Random.seed!(1600)
    N = 60
    x = rand(2, N)
    u = randn(2, N)
    adv = randn(2, N)
    w = randn(N)
    advw = randn(N)
    bins = collect(range(0.0, 1.3; length = 6))

    fv = SF.Fields(vectors = (u, adv))
    got = _run(SFT.VectorDotSFType(1, 2), x, fv, bins)
    ref_s, ref_c = _brute((dv, ds, rh) -> dot(dv[1], dv[2]), x, (u, adv), (), bins)
    Test.@test got.counts == ref_c
    Test.@test isapprox(got.sums, ref_s; rtol = 1e-10, atol = 1e-12)

    fs = SF.Fields(vectors = (u,), scalars = (w, advw))
    gots = _run(SFT.ScalarDotSFType(1, 2), x, fs, bins)
    refs_s, _ = _brute((dv, ds, rh) -> ds[1] * ds[2], x, (u,), (w, advw), bins)
    Test.@test isapprox(gots.sums, refs_s; rtol = 1e-10, atol = 1e-12)

    # the diagonal of the vector cross-moment IS the second-order structure function
    diag = _run(SFT.VectorDotSFType(1, 1), x, fv, bins)
    s2 = _run(SFT.S2SFType(), x, SF.Fields(vectors = (u,)), bins)
    Test.@test diag.counts == s2.counts
    Test.@test isapprox(diag.sums, s2.sums; rtol = 1e-12)
end

Test.@testset "asking for a channel a field does not carry says so" begin
    Random.seed!(1700)
    x = rand(2, 20)
    u = randn(2, 20)
    bins = collect(range(0.0, 1.0; length = 4))
    # a plain velocity has no scalar channel
    Test.@test_throws ArgumentError _run(SFT.ScalarSFType{2}(), x, SF.Fields(vectors = (u,)), bins)
    # and only one vector channel
    Test.@test_throws ArgumentError _run(SFT.VectorDotSFType(1, 2), x, SF.Fields(vectors = (u,)), bins)
end

Test.@testset "channels are transported on a sphere, scalars are not" begin
    # A vector channel is carried as an ambient 3-vector on a sphere, so a bundle must widen every
    # vector channel exactly as the array path widens the one it has. A scalar has nothing to
    # transport and passes through untouched.
    Random.seed!(1800)
    N = 50
    x = vcat(reshape(2π .* rand(N), 1, N), reshape((rand(N) .- 0.5) .* 1.4, 1, N))
    u = randn(2, N)
    th = randn(N)
    bins = collect(range(0.0, 2.4; length = 6))
    metric = SFC.DI.SphericalAngle()

    # one vector channel: the same kernel as the array path, so identical to the last bit
    bare_s = zeros(5); bare_c = zeros(UInt32, 5)
    SF.calculate_structure_function!(bare_s, bare_c, SFT.L2SFType(), x, u, bins;
                                     distance_metric = metric)
    bundled = SFC.calculate_structure_function(
        SFT.L2SFType(), x, SF.Fields(vectors = (u,)), bins, UInt32; distance_metric = metric,
        output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false)
    Test.@test bundled.counts == bare_c
    Test.@test bundled.sums == bare_s

    # a scalar rides along without disturbing the velocity part: L2SF on the bundle must still equal
    # L2SF on the velocity alone
    with_tracer = SFC.calculate_structure_function(
        SFT.L2SFType(), x, SF.Fields(vectors = (u,), scalars = (th,)), bins, UInt32;
        distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    Test.@test with_tracer.counts == bare_c
    Test.@test isapprox(with_tracer.sums, bare_s; rtol = 1e-12)

    # the scalar structure function on a sphere: transport-free, so it is the plain difference
    scalar_only = SFC.calculate_structure_function(
        SFT.ScalarSFType{2}(), x, SF.Fields(scalars = (th,)), bins, UInt32;
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
        SFT.MixedSFType{1, 0, 2}(), x, SF.Fields(vectors = (u,), scalars = (th,)), bins, UInt32;
        distance_metric = metric, output_type = SF.StructureFunctionSumsAndCounts,
        verbose = false, show_progress = false)
    Test.@test all(isfinite, yag.sums)
    Test.@test yag.counts == bare_c
end

Test.@testset "the threaded backend gives the serial answer" begin
    # Multi-channel across threads: the setup happens once above the task loop and each task sweeps
    # its own outer indices, so the only thing that may differ from serial is summation order.
    Random.seed!(1900)
    N = 400
    x = rand(2, N)
    u = randn(2, N)
    adv = randn(2, N)
    th = randn(N)
    bins = collect(range(0.0, 1.5; length = 8))   # spans the unit square diagonal
    nb = length(bins) - 1

    for (f, op) in ((SF.Fields(vectors = (u,), scalars = (th,)), SFT.MixedSFType{1, 0, 2}()),
                    (SF.Fields(vectors = (u, adv)), SFT.VectorDotSFType(1, 2)),
                    (SF.Fields(vectors = (u,), scalars = (th,)), SFT.ScalarSFType{2}()),
                    (SF.Fields(vectors = (u,)), SFT.L2SFType()))
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
