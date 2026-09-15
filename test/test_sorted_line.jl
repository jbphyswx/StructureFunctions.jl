using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    Channels as CH, Fields
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using OhMyThreads: OhMyThreads
using Random: Random

const RAW = SF.StructureFunctionSumsAndCounts
const SERIAL = CB.SerialBackend()
const THREADED = CB.ThreadedBackend()
const AUTO = CB.AutoBackend()

const LINE_OPS = (
    SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.S3SFType(), SFT.L1T2SFType(),
    SFT.ProjectedStructureFunctionType{4, 0}(), SFT.ProjectedStructureFunctionType{2, 2}(),
    SFT.FullVectorStructureFunctionType{2}(), SFT.FullVectorStructureFunctionType{4}(),
)

# One pair's increment, read from point `i` to point `j`: the plain vector for one vector channel, else
# the channel bundle.
function _line_increment(data, i, j, ::Val{D}, ::Val{V}, ::Val{K}) where {D, V, K}
    T = eltype(data)
    V == 1 && K == 0 && return SA.SVector{D, T}(ntuple(d -> data[d, j] - data[d, i], Val(D)))
    vectors = ntuple(Val(V)) do a
        o = (a - 1) * D
        SA.SVector{D, T}(ntuple(d -> data[o + d, j] - data[o + d, i], Val(D)))
    end
    scalars = ntuple(c -> data[V * D + c, j] - data[V * D + c, i], Val(K))
    return CH.ChannelIncrement{D, V, K, T}(vectors, scalars)
end

# The pair statistic written out on a line: Σ w_i w_j sf(δu, r̂) and Σ w_i w_j over the pairs each (lo, hi]
# bin holds, every pair read from its lower to its upper coordinate.
function _line_pair_loop(sf, x::AbstractVector, data::AbstractMatrix, bins, w, vD::Val{D}, vV::Val{V}, vK) where {D, V}
    nb = length(bins) - 1
    s = zeros(nb)
    c = zeros(nb)
    N = length(x)
    Dr = V == 0 ? 1 : D
    r̂ = SA.SVector{Dr, Float64}(ntuple(d -> d == 1 ? 1.0 : 0.0, Dr))
    for i in 1:(N - 1), j in (i + 1):N
        lo, hi = x[i] <= x[j] ? (i, j) : (j, i)
        b = searchsortedfirst(bins, x[hi] - x[lo]) - 1
        1 <= b <= nb || continue
        ww = w[i] * w[j]
        s[b] += ww * sf(_line_increment(data, lo, hi, vD, vV, vK), r̂)
        c[b] += ww
    end
    return s, c
end

_close(a, b; rtol = 1e-12) = isapprox(a, b; rtol, atol = rtol * max(maximum(abs, b), 1e-300))

Test.@testset "every polynomial operator on a line equals the pair loop" begin
    Random.seed!(4210)
    N = 300
    x = rand(N) .* 10.0
    u = randn(1, N)
    bins = [0.0; sort(rand(7)) .* 4.0]
    ones_w = ones(N)
    x1 = reshape(x, 1, :)
    for sf in LINE_OPS
        ref_s, ref_c = _line_pair_loop(sf, x, u, bins, ones_w, Val(1), Val(1), Val(0))
        Test.@test sum(ref_c) > 0
        for backend in (SERIAL, THREADED, AUTO)
            got = SFC.calculate_structure_function(sf, x1, u, bins; backend, verbose = false,
                                                   show_progress = false, output_type = RAW)
            Test.@test got.counts == UInt32.(ref_c)
            Test.@test _close(got.sums, ref_s)
        end
    end
    # the averaged result object comes out of the same route
    res = SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins; backend = SERIAL, verbose = false,
                                           show_progress = false)
    ref_s, ref_c = _line_pair_loop(SFT.L2SFType(), x, u, bins, ones_w, Val(1), Val(1), Val(0))
    Test.@test _close(res.values, ref_s ./ ref_c)
end

Test.@testset "channel bundles on a line" begin
    Random.seed!(4220)
    N = 240
    x = rand(N) .* 6.0
    x1 = reshape(x, 1, :)
    u = randn(1, N)
    a = randn(1, N)
    θ = randn(N)
    φ = randn(N)
    bins = [0.0; sort(rand(6)) .* 3.0]
    ones_w = ones(N)
    cases = (
        (Fields(vectors = (u,), scalars = (θ,)),
         (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{2}(), SFT.ScalarSFType{1}(), SFT.ScalarSFType{3}(),
          SFT.MixedSFType{2, 0, 1}(), SFT.MixedSFType{1, 0, 1}(), SFT.L2SFType())),
        (Fields(vectors = (u, a)), (SFT.VectorDotSFType(1, 2), SFT.L3SFType())),
        (Fields(scalars = (θ, φ)), (SFT.ScalarDotSFType(1, 2), SFT.ScalarSFType{4}(2), SFT.ScalarSFType{3}(1))),
    )
    for (f, ops) in cases, sf in ops
        D, V, K = CH.channel_dimension(f), CH.n_vector_channels(f), CH.n_scalar_channels(f)
        ref_s, ref_c = _line_pair_loop(sf, x, CH.packed(f), bins, ones_w, Val(D), Val(V), Val(K))
        Test.@test sum(ref_c) > 0
        for backend in (SERIAL, THREADED)
            got = SFC.calculate_structure_function(sf, x1, f, bins; backend, verbose = false,
                                                   show_progress = false, output_type = RAW)
            Test.@test got.counts == UInt32.(ref_c)
            Test.@test _close(got.sums, ref_s)
        end
    end
end

Test.@testset "pair weights on a line" begin
    Random.seed!(4230)
    N = 260
    x = rand(N) .* 8.0
    x1 = reshape(x, 1, :)
    u = randn(1, N)
    θ = randn(N)
    w = 0.5 .+ rand(N)
    bins = [0.0; sort(rand(7)) .* 3.5]
    for sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.ProjectedStructureFunctionType{4, 0}())
        ref_s, ref_c = _line_pair_loop(sf, x, u, bins, w, Val(1), Val(1), Val(0))
        for backend in (SERIAL, THREADED)
            got = SFC.calculate_structure_function(sf, x1, u, bins, Float64; backend, weights = w,
                                                   verbose = false, show_progress = false, output_type = RAW)
            Test.@test _close(got.counts, ref_c)
            Test.@test _close(got.sums, ref_s)
        end
    end
    f = Fields(vectors = (u,), scalars = (θ,))
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{3}())
        ref_s, ref_c = _line_pair_loop(sf, x, CH.packed(f), bins, w, Val(1), Val(1), Val(1))
        got = SFC.calculate_structure_function(sf, x1, f, bins, Float64; backend = SERIAL, weights = w,
                                               verbose = false, show_progress = false, output_type = RAW)
        Test.@test _close(got.counts, ref_c)
        Test.@test _close(got.sums, ref_s)
    end
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins; backend = SERIAL,
                                                                     weights = w, verbose = false, show_progress = false)
end

Test.@testset "the order of the points and coincident points change nothing" begin
    Random.seed!(4240)
    N = 320
    x = rand(1:60, N) .* 0.1
    u = randn(1, N)
    θ = randn(N)
    bins = [0.0; sort(rand(6)) .* 3.0]
    ones_w = ones(N)
    perm = Random.randperm(N)
    for sf in (SFT.L3SFType(), SFT.S2SFType())
        ref_s, ref_c = _line_pair_loop(sf, x, u, bins, ones_w, Val(1), Val(1), Val(0))
        Test.@test sum(ref_c) > 0
        sorted_res = SFC.calculate_structure_function(sf, reshape(sort(x), 1, :), u[:, sortperm(x)], bins;
                                                      backend = SERIAL, verbose = false, show_progress = false,
                                                      output_type = RAW)
        shuffled_res = SFC.calculate_structure_function(sf, reshape(x[perm], 1, :), u[:, perm], bins;
                                                        backend = SERIAL, verbose = false, show_progress = false,
                                                        output_type = RAW)
        Test.@test sorted_res.counts == UInt32.(ref_c)
        Test.@test shuffled_res.counts == UInt32.(ref_c)
        Test.@test _close(sorted_res.sums, ref_s)
        Test.@test _close(shuffled_res.sums, ref_s)
    end
    # an odd scalar moment reads every pair from its lower to its upper coordinate, whatever the input order
    f = Fields(vectors = (u,), scalars = (θ,))
    fp = Fields(vectors = (u[:, perm],), scalars = (θ[perm],))
    for sf in (SFT.ScalarSFType{3}(), SFT.MixedSFType{1, 0, 1}())
        ref_s, ref_c = _line_pair_loop(sf, x, CH.packed(f), bins, ones_w, Val(1), Val(1), Val(1))
        got = SFC.calculate_structure_function(sf, reshape(x[perm], 1, :), fp, bins; backend = SERIAL,
                                               verbose = false, show_progress = false, output_type = RAW)
        Test.@test got.counts == UInt32.(ref_c)
        Test.@test _close(got.sums, ref_s)
        Test.@test maximum(abs, ref_s) > 0
    end
end

Test.@testset "what the sorted route does not take stays on the pair loop, and it refuses by name" begin
    Random.seed!(4250)
    N = 150
    x = rand(N) .* 5.0
    x1 = reshape(x, 1, :)
    u = randn(1, N)
    bins = [0.0; sort(rand(5)) .* 2.5]
    ones_w = ones(N)
    cubic = SFT.FullVectorStructureFunctionType{3}()
    Test.@test !SFT.is_polynomial_operator(cubic)
    ref_s, ref_c = _line_pair_loop(cubic, x, u, bins, ones_w, Val(1), Val(1), Val(0))
    for backend in (SERIAL, THREADED)
        got = SFC.calculate_structure_function(cubic, x1, u, bins; backend, verbose = false, show_progress = false,
                                               output_type = RAW)
        Test.@test got.counts == UInt32.(ref_c)
        Test.@test _close(got.sums, ref_s)
    end
    nb = length(bins) - 1
    Test.@test_throws ArgumentError SFC.sorted_line_sweep!(zeros(nb), zeros(Int, nb), cubic, x, u, bins,
                                                           Val(1), Val(1), Val(0))
    Test.@test_throws DimensionMismatch SFC.sorted_line_sweep!(zeros(nb), zeros(Int, nb), SFT.L2SFType(), x,
                                                               randn(2, N), bins, Val(1), Val(1), Val(0))
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins; backend = SERIAL,
                                                                     culling = SFC.AlwaysCulling(), verbose = false,
                                                                     show_progress = false)
    # the serial entry on a one-dimensional list returns a result
    direct = SFC.serial_calculate_structure_function(SFT.L2SFType(), x1, u, bins; verbose = false, show_progress = false)
    ref_s, ref_c = _line_pair_loop(SFT.L2SFType(), x, u, bins, ones_w, Val(1), Val(1), Val(0))
    Test.@test direct isa RAW
    Test.@test direct.counts == UInt32.(ref_c)
    Test.@test _close(direct.sums, ref_s)
end

Test.@testset "the sorted route is linear in the points" begin
    Random.seed!(4260)
    bins = collect(range(0.0, 0.5; length = 21))
    small = rand(1, 2000)
    SFC.calculate_structure_function(SFT.L2SFType(), small, randn(1, 2000), bins, Int64; backend = SERIAL,
                                     verbose = false, show_progress = false)
    N = 100_000
    x = rand(1, N) .* 100.0
    u = randn(1, N)
    t = @elapsed res = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, Int64; backend = SERIAL,
                                                        verbose = false, show_progress = false, output_type = RAW)
    Test.@test sum(res.counts) > 0
    Test.@test t < 2.0
end
