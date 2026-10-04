using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    MultiFields as MF
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const RAW = SF.StructureFunctionSumsAndCounts
const SERIAL = CB.SerialBackend()
const THREADED = CB.ThreadedBackend()

# (operator, backend)
const LINE_CASES = (
    (SFT.S2SFType(), SERIAL), (SFT.L2SFType(), THREADED), (SFT.T2SFType(), SERIAL),
    (SFT.L3SFType(), SERIAL), (SFT.S3SFType(), SERIAL), (SFT.L1T2SFType(), SERIAL),
    (SFT.ProjectedStructureFunctionType{4, 0}(), SERIAL), (SFT.ProjectedStructureFunctionType{2, 2}(), SERIAL),
    (SFT.FullVectorStructureFunctionType{2}(), SERIAL), (SFT.FullVectorStructureFunctionType{4}(), SERIAL),
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
    for (sf, backend) in LINE_CASES
        ref_s, ref_c = _line_pair_loop(sf, x, u, bins, ones_w, Val(1), Val(1), Val(0))
        Test.@test sum(ref_c) > 0
        got = SFC.calculate_structure_function(sf, x1, u, bins, RAW; backend)
        Test.@test got.counts == UInt32.(ref_c)
        Test.@test _close(got.sums, ref_s)
    end
end

# (field: 1 a vector and a scalar, 2 two vectors, 3 two scalars; operator; backend)
const LINE_FIELD_CASES = (
    (1, SFT.MixedSFType{1, 0, 2}(), SERIAL), (1, SFT.ScalarSFType{2}(), THREADED), (1, SFT.ScalarSFType{3}(), SERIAL),
    (1, SFT.MixedSFType{1, 0, 1}(), SERIAL), (2, SFT.VectorDotSFType(1, 2), SERIAL), (2, SFT.L3SFType(), SERIAL),
    (3, SFT.ScalarDotSFType(1, 2), SERIAL), (3, SFT.ScalarSFType{4}(2), SERIAL),
)

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
    fields = (Fields(vectors = (u,), scalars = (θ,)), Fields(vectors = (u, a)), Fields(scalars = (θ, φ)))
    for (k, sf, backend) in LINE_FIELD_CASES
        f = fields[k]
        D, V, K = MF.field_dimension(f), MF.n_vector_fields(f), MF.n_scalar_fields(f)
        ref_s, ref_c = _line_pair_loop(sf, x, MF.packed(f), bins, ones_w, Val(D), Val(V), Val(K))
        Test.@test sum(ref_c) > 0
        got = SFC.calculate_structure_function(sf, x1, f, bins, RAW; backend)
        Test.@test got.counts == UInt32.(ref_c)
        Test.@test _close(got.sums, ref_s)
    end
end

const LINE_WEIGHT_FIELD_CASES = ((SFT.MixedSFType{1, 0, 2}(), THREADED), (SFT.ScalarSFType{3}(), SERIAL))

Test.@testset "pair weights on a line" begin
    Random.seed!(4230)
    N = 260
    x = rand(N) .* 8.0
    x1 = reshape(x, 1, :)
    u = randn(1, N)
    θ = randn(N)
    w = 0.5 .+ rand(N)
    bins = [0.0; sort(rand(7)) .* 3.5]
    ref_s, ref_c = _line_pair_loop(SFT.L3SFType(), x, u, bins, w, Val(1), Val(1), Val(0))
    got = SFC.calculate_structure_function(SFT.L3SFType(), x1, u, bins, Float64, RAW; backend = SERIAL, weights = w)
    Test.@test _close(got.counts, ref_c)
    Test.@test _close(got.sums, ref_s)
    f = Fields(vectors = (u,), scalars = (θ,))
    for (sf, backend) in LINE_WEIGHT_FIELD_CASES
        ref_s, ref_c = _line_pair_loop(sf, x, MF.packed(f), bins, w, Val(1), Val(1), Val(1))
        got = SFC.calculate_structure_function(sf, x1, f, bins, Float64, RAW; backend, weights = w)
        Test.@test _close(got.counts, ref_c)
        Test.@test _close(got.sums, ref_s)
    end
end

Test.@testset "the order of the points and coincident points change nothing" begin
    Random.seed!(4240)
    N = 320
    x = rand(1:60, N) .* 0.1
    u = randn(1, N)
    bins = [0.0; sort(rand(6)) .* 3.0]
    perm = Random.randperm(N)
    sf = SFT.L3SFType()
    ref_s, ref_c = _line_pair_loop(sf, x, u, bins, ones(N), Val(1), Val(1), Val(0))
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

Test.@testset "a field on a large offset keeps its moments on a line, in either precision" begin
    Random.seed!(4270)
    N = 40
    bins = collect(range(0.0, 1.0; length = 9))
    ones_w = ones(N)
    sf = SFT.ProjectedStructureFunctionType{4, 0}()
    for (T, rtol) in ((Float32, 1e-3), (Float64, 1e-9))
        x = T.(rand(N) .* 10.0)
        u = T.(1000 .+ 0.01 .* randn(1, N))
        xf, uf = Float64.(x), Float64.(u)
        ref_s, ref_c = _line_pair_loop(sf, xf, uf, bins, ones_w, Val(1), Val(1), Val(0))
        scale = _line_pair_abs(sf, xf, uf, bins)
        got = SFC.calculate_structure_function(sf, reshape(x, 1, :), u, bins, RAW; backend = SERIAL)
        Test.@test (T, got.counts == UInt32.(ref_c)) == (T, true)
        Test.@test (T, all(abs.(got.sums .- ref_s) .<= rtol .* scale)) == (T, true)
    end
end

# (operator, weighted)
const LINE_DEVICE_CASES = ((SFT.S2SFType(), false), (SFT.ProjectedStructureFunctionType{2, 2}(), true))

Test.@testset "the device sorted line gives the pair loop's answer" begin
    Random.seed!(4280)
    N = 400
    x = rand(N) .* 10.0
    x1 = reshape(x, 1, :)
    u = randn(1, N)
    θ = randn(N)
    w = 0.5 .+ rand(N)
    bins = [0.0; sort(rand(7)) .* 4.0]
    nb = length(bins) - 1
    dev = CB.GPUBackend(KA.CPU())
    for (sf, weighted) in LINE_DEVICE_CASES
        if weighted
            ref_s, ref_c = _line_pair_loop(sf, x, u, bins, w, Val(1), Val(1), Val(0))
            got = SFC.calculate_structure_function(sf, x1, u, bins, Float64, RAW; backend = dev, weights = w)
            Test.@test (sf, _close(got.counts, ref_c), _close(got.sums, ref_s)) == (sf, true, true)
        else
            ref_s, ref_c = _line_pair_loop(sf, x, u, bins, ones(N), Val(1), Val(1), Val(0))
            got = SFC.calculate_structure_function(sf, x1, u, bins, RAW; backend = dev)
            Test.@test (sf, got.counts == UInt32.(ref_c), _close(got.sums, ref_s)) == (sf, true, true)
        end
    end
    f = Fields(vectors = (u,), scalars = (θ,))
    ref_s, ref_c = _line_pair_loop(SFT.MixedSFType{1, 0, 1}(), x, MF.packed(f), bins, ones(N), Val(1), Val(1), Val(1))
    got = SFC.calculate_structure_function(SFT.MixedSFType{1, 0, 1}(), x1, f, bins, RAW; backend = dev)
    Test.@test got.counts == UInt32.(ref_c)
    Test.@test _close(got.sums, ref_s)
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

_sort_keys(::Type{Float64}, pool, n) = rand(pool, n)
_sort_keys(::Type{Float32}, pool, n) = Float32.(rand(pool, n))
_sort_keys(::Type{Int32}, pool, n) = rand(Int32(-3):Int32(40), n)
_sort_keys(::Type{Int64}, pool, n) = rand(typemin(Int64):typemax(Int64), n)

# (points, key type): every key type, and 2049 points to cross the sort's fixed 2048-point tile
const LINE_SORT_CASES = ((0, Float64), (7, Int32), (7, Int64), (2049, Float32), (2049, Float64))

Test.@testset "the device sort orders as sortperm does, within and across its 2048-point tile" begin
    GE = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
    Random.seed!(4290)
    pool = [0.0, -0.0, NaN, -NaN, Inf, -Inf, 1.0, 1.0, -2.5, randn(8)...]
    for (n, T) in LINE_SORT_CASES
        v = _sort_keys(T, pool, n)
        Test.@test (eltype(v), n, GE._gpu_sortperm(v) == sortperm(v)) == (eltype(v), n, true)
    end
end


Test.@testset "a non-polynomial operator on a line gives the pair loop's answer, and the sorted sweep refuses it" begin
    Random.seed!(4250)
    N = 150
    x = rand(N) .* 5.0
    x1 = reshape(x, 1, :)
    u = randn(1, N)
    bins = [0.0; sort(rand(5)) .* 2.5]
    ones_w = ones(N)
    cubic = SFT.FullVectorStructureFunctionType{3}()
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
    ref_s, ref_c = _line_pair_loop(SFT.L2SFType(), x, u, bins, ones_w, Val(1), Val(1), Val(0))
    # a policy that must cull still gives the pair loop's answer
    for backend in (SERIAL, THREADED)
        got = SFC.calculate_structure_function(SFT.L2SFType(), x1, u, bins, RAW; backend, culling = SFC.AlwaysCulling())
        Test.@test got.counts == UInt32.(ref_c)
        Test.@test _close(got.sums, ref_s)
    end
end
