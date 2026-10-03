using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    MultiFields as MF, HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
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

# One pair's increment, read from point `i` to point `j`: the plain vector for one vector field, else
# the multi-field.
function _line_increment(data, i, j, ::Val{D}, ::Val{V}, ::Val{K}) where {D, V, K}
    T = eltype(data)
    V == 1 && K == 0 && return SA.SVector{D, T}(ntuple(d -> data[d, j] - data[d, i], Val(D)))
    vectors = ntuple(Val(V)) do a
        o = (a - 1) * D
        SA.SVector{D, T}(ntuple(d -> data[o + d, j] - data[o + d, i], Val(D)))
    end
    scalars = ntuple(c -> data[V * D + c, j] - data[V * D + c, i], Val(K))
    return MF.FieldIncrement{D, V, K, T}(vectors, scalars)
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

# Σ |sf(δu, r̂)| over the pairs of each bin, the scale a sum of values of either sign is compared at.
function _line_pair_abs(sf, x::AbstractVector, data::AbstractMatrix, bins)
    s = zeros(length(bins) - 1)
    r̂ = SA.SVector(1.0)
    for i in 1:(length(x) - 1), j in (i + 1):length(x)
        lo, hi = x[i] <= x[j] ? (i, j) : (j, i)
        b = searchsortedfirst(bins, x[hi] - x[lo]) - 1
        1 <= b <= length(s) && (s[b] += abs(sf(_line_increment(data, lo, hi, Val(1), Val(1), Val(0)), r̂)))
    end
    return s
end

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
            got = SFC.calculate_structure_function(sf, x1, u, bins, RAW; backend)
            Test.@test got.counts == UInt32.(ref_c)
            Test.@test _close(got.sums, ref_s)
        end
    end
    # the averaged result object comes out of the same route
    res = SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins; backend = SERIAL)
    ref_s, ref_c = _line_pair_loop(SFT.L2SFType(), x, u, bins, ones_w, Val(1), Val(1), Val(0))
    Test.@test _close(res.values, ref_s ./ ref_c)
end

Test.@testset "multi-fields on a line" begin
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
        D, V, K = MF.field_dimension(f), MF.n_vector_fields(f), MF.n_scalar_fields(f)
        ref_s, ref_c = _line_pair_loop(sf, x, MF.packed(f), bins, ones_w, Val(D), Val(V), Val(K))
        Test.@test sum(ref_c) > 0
        for backend in (SERIAL, THREADED)
            got = SFC.calculate_structure_function(sf, x1, f, bins, RAW; backend)
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
            got = SFC.calculate_structure_function(sf, x1, u, bins, Float64, RAW; backend, weights = w)
            Test.@test _close(got.counts, ref_c)
            Test.@test _close(got.sums, ref_s)
        end
    end
    f = Fields(vectors = (u,), scalars = (θ,))
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{3}())
        ref_s, ref_c = _line_pair_loop(sf, x, MF.packed(f), bins, w, Val(1), Val(1), Val(1))
        got = SFC.calculate_structure_function(sf, x1, f, bins, Float64, RAW; backend = SERIAL, weights = w)
        Test.@test _close(got.counts, ref_c)
        Test.@test _close(got.sums, ref_s)
    end
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins; backend = SERIAL,
                                                                     weights = w)
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
        sorted_res = SFC.calculate_structure_function(sf, reshape(sort(x), 1, :), u[:, sortperm(x)], bins, RAW;
                                                      backend = SERIAL)
        shuffled_res = SFC.calculate_structure_function(sf, reshape(x[perm], 1, :), u[:, perm], bins, RAW;
                                                        backend = SERIAL)
        Test.@test sorted_res.counts == UInt32.(ref_c)
        Test.@test shuffled_res.counts == UInt32.(ref_c)
        Test.@test _close(sorted_res.sums, ref_s)
        Test.@test _close(shuffled_res.sums, ref_s)
    end
    # an odd scalar moment reads every pair from its lower to its upper coordinate, whatever the input order
    f = Fields(vectors = (u,), scalars = (θ,))
    fp = Fields(vectors = (u[:, perm],), scalars = (θ[perm],))
    for sf in (SFT.ScalarSFType{3}(), SFT.MixedSFType{1, 0, 1}())
        ref_s, ref_c = _line_pair_loop(sf, x, MF.packed(f), bins, ones_w, Val(1), Val(1), Val(1))
        got = SFC.calculate_structure_function(sf, reshape(x[perm], 1, :), fp, bins, RAW; backend = SERIAL)
        Test.@test got.counts == UInt32.(ref_c)
        Test.@test _close(got.sums, ref_s)
        Test.@test maximum(abs, ref_s) > 0
    end
end

Test.@testset "a field on a large offset keeps its moments on a line, in either precision" begin
    Random.seed!(4270)
    N = 3000
    bins = collect(range(0.0, 1.0; length = 9))
    ones_w = ones(N)
    for (T, rtol) in ((Float32, 1e-3), (Float64, 1e-9))
        x = T.(rand(N) .* 10.0)
        u = T.(1000 .+ 0.01 .* randn(1, N))
        for sf in (SFT.L2SFType(), SFT.L3SFType(), SFT.ProjectedStructureFunctionType{4, 0}())
            xf, uf = Float64.(x), Float64.(u)
            ref_s, ref_c = _line_pair_loop(sf, xf, uf, bins, ones_w, Val(1), Val(1), Val(0))
            scale = _line_pair_abs(sf, xf, uf, bins)
            for backend in (SERIAL, THREADED)
                got = SFC.calculate_structure_function(sf, reshape(x, 1, :), u, bins, RAW; backend)
                Test.@test (T, sf, got.counts == UInt32.(ref_c)) == (T, sf, true)
                Test.@test (T, sf, all(abs.(got.sums .- ref_s) .<= rtol .* scale)) == (T, sf, true)
            end
        end
    end
end

Test.@testset "the device sorted line gives the host's answer" begin
    Random.seed!(4280)
    N = 400
    x1 = reshape(rand(N) .* 10.0, 1, :)
    u = randn(1, N)
    θ = randn(N)
    w = 0.5 .+ rand(N)
    bins = [0.0; sort(rand(7)) .* 4.0]
    nb = length(bins) - 1
    dev = CB.GPUBackend(KA.CPU())
    for sf in LINE_OPS
        ref = SFC.calculate_structure_function(sf, x1, u, bins, RAW; backend = SERIAL)
        got = SFC.calculate_structure_function(sf, x1, u, bins, RAW; backend = dev)
        Test.@test (sf, got.counts == ref.counts, _close(got.sums, ref.sums)) == (sf, true, true)
        refw = SFC.calculate_structure_function(sf, x1, u, bins, Float64, RAW; backend = SERIAL, weights = w)
        gotw = SFC.calculate_structure_function(sf, x1, u, bins, Float64, RAW; backend = dev, weights = w)
        Test.@test (sf, _close(gotw.counts, refw.counts), _close(gotw.sums, refw.sums)) == (sf, true, true)
    end
    f = Fields(vectors = (u,), scalars = (θ,))
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{3}(), SFT.MixedSFType{1, 0, 1}())
        ref = SFC.calculate_structure_function(sf, x1, f, bins, RAW; backend = SERIAL)
        got = SFC.calculate_structure_function(sf, x1, f, bins, RAW; backend = dev)
        Test.@test (sf, got.counts == ref.counts, _close(got.sums, ref.sums)) == (sf, true, true)
    end
    # the mutating form adds, and a field on a large offset keeps its moments on the device too
    s, c = zeros(Float32, nb), zeros(UInt32, nb)
    x32, u32 = Float32.(x1), Float32.(1000 .+ 0.01 .* u)
    for _ in 1:2
        SFC.calculate_structure_function!(s, c, SFT.L3SFType(), x32, u32, Float32.(bins); backend = dev)
    end
    ref_s, ref_c = _line_pair_loop(SFT.L3SFType(), Float64.(vec(x32)), Float64.(u32), Float64.(Float32.(bins)),
                                   ones(N), Val(1), Val(1), Val(0))
    scale = _line_pair_abs(SFT.L3SFType(), Float64.(vec(x32)), Float64.(u32), Float64.(Float32.(bins)))
    Test.@test c == 2 .* UInt32.(ref_c)
    Test.@test all(abs.(s .- 2 .* ref_s) .<= 1e-3 .* 2 .* scale)
end

# The device sort behind the sorted line gives the stable order `sortperm` gives, NaNs last, at every key width and
# around the sort's tile of points.
Test.@testset "the device sort orders as sortperm does" begin
    GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    Random.seed!(4290)
    pool = [0.0, -0.0, NaN, -NaN, Inf, -Inf, 1.0, 1.0, -2.5, randn(8)...]
    for n in (0, 1, 2, 7, 2047, 2048, 2049, 5000)
        for v in (rand(pool, n), Float32.(rand(pool, n)), rand(Int32(-3):Int32(40), n),
                  rand(typemin(Int64):typemax(Int64), n))
            Test.@test (eltype(v), n, GE._gpu_sortperm(v) == sortperm(v)) == (eltype(v), n, true)
        end
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
        got = SFC.calculate_structure_function(cubic, x1, u, bins, RAW; backend)
        Test.@test got.counts == UInt32.(ref_c)
        Test.@test _close(got.sums, ref_s)
    end
    nb = length(bins) - 1
    Test.@test_throws ArgumentError SFC.sorted_line_sweep!(zeros(nb), zeros(Int, nb), cubic, x, u, bins,
                                                           Val(1), Val(1), Val(0))
    Test.@test_throws DimensionMismatch SFC.sorted_line_sweep!(zeros(nb), zeros(Int, nb), SFT.L2SFType(), x,
                                                               randn(2, N), bins, Val(1), Val(1), Val(0))
    # the sorted line forms no pair outside the bins, so every culling policy runs it
    for backend in (SERIAL, THREADED)
        always = SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins, RAW; backend,
            culling = SFC.AlwaysCulling())
        none = SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins, RAW; backend,
            culling = SFC.NoCulling())
        Test.@test always.counts == none.counts
        Test.@test backend === SERIAL ? always.sums == none.sums : isapprox(always.sums, none.sums; rtol = 1e-12)
    end
    # the serial entry on a one-dimensional list returns a result
    direct = SFC.serial_calculate_structure_function(SFT.L2SFType(), x1, u, bins, UInt32; geometry = SFH.FlatGeometry{1}())
    ref_s, ref_c = _line_pair_loop(SFT.L2SFType(), x, u, bins, ones_w, Val(1), Val(1), Val(0))
    Test.@test direct isa RAW
    Test.@test direct.counts == UInt32.(ref_c)
    Test.@test _close(direct.sums, ref_s)
end

Test.@testset "the sorted route is linear in the points" begin
    Random.seed!(4260)
    bins = collect(range(0.0, 0.5; length = 21))
    small = rand(1, 2000)
    SFC.calculate_structure_function(SFT.L2SFType(), small, randn(1, 2000), bins, Int64; backend = SERIAL)
    N = 100_000
    x = rand(1, N) .* 100.0
    u = randn(1, N)
    t = @elapsed res = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, Int64, RAW; backend = SERIAL)
    Test.@test sum(res.counts) > 0
    Test.@test t < 2.0
end
