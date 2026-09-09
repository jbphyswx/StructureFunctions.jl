using Test: Test
using StructureFunctions: StructureFunctions as SF, StructureFunctionTypes as SFT, Channels as CH
using StaticArrays: StaticArrays as SA
using LinearAlgebra: normalize
using Random: Random

Random.seed!(20260907)

# The rank-p moment tensor of a single pair, δu ⊗ … ⊗ δu, in the engine's symmetric storage.
function _outer(δu::SA.SVector{W, T}, ::Val{P}) where {W, T, P}
    js = SFT.symmetric_indices(Val(W), Val(P))
    data = SA.SVector{length(js), T}(ntuple(n -> prod(δu[j] for j in js[n]), length(js)))
    return SFT.SymmetricMoments{W, P}(data)
end

# The increment the operator itself consumes: the plain vector for one vector channel and no
# scalars, a ChannelIncrement otherwise.
function _increment(δu::SA.SVector{W, T}, ::Val{D}, ::Val{V}, ::Val{K}) where {W, T, D, V, K}
    V == 1 && K == 0 && return SA.SVector{D, T}(ntuple(d -> δu[d], Val(D)))
    vectors = ntuple(a -> SA.SVector{D, T}(ntuple(d -> δu[(a - 1) * D + d], Val(D))), Val(V))
    scalars = ntuple(k -> δu[V * D + k], Val(K))
    return CH.ChannelIncrement{D, V, K, T}(vectors, scalars)
end

function _check_operator(sf, D, V, K; n = 40)
    W = V * D + K
    p = SFT.order(sf)
    Test.@test SFT.is_polynomial_operator(sf)
    for _ in 1:n
        δu = SA.SVector{W, Float64}(randn(W))
        r̂ = D == 0 ? SA.SVector{0, Float64}() : SA.SVector{D, Float64}(normalize(randn(D)))
        want = sf(_increment(δu, Val(D), Val(V), Val(K)), r̂)
        got = SFT.moment_contract(sf, _outer(δu, Val(p)), r̂, Val(V), Val(K))
        Test.@test isapprox(got, want; atol = 1e-11, rtol = 1e-11)
    end
end

Test.@testset "symmetric storage: indices are sorted, complete and ranked by position" begin
    for (W, P) in ((2, 2), (3, 3), (4, 4), (3, 5), (5, 1))
        js = SFT.symmetric_indices(Val(W), Val(P))
        Test.@test length(js) == binomial(W + P - 1, P)
        Test.@test all(issorted, js)
        Test.@test allunique(js)
        for (n, j) in enumerate(js)
            Test.@test SFT.symmetric_rank(Val(W), Val(P), j) == n
            Test.@test SFT.symmetric_rank(Val(W), Val(P), reverse(j)) == n
        end
    end
    Test.@test_throws DimensionMismatch SFT.SymmetricMoments{3, 2}(SA.SVector{4, Float64}(1, 2, 3, 4))
end

Test.@testset "order is defined for every polynomial operator" begin
    Test.@test SFT.order(SFT.ScalarSFType{3}()) == 3
    Test.@test SFT.order(SFT.MixedSFType{1, 0, 2}()) == 3
    Test.@test SFT.order(SFT.MixedSFType{2, 2, 1}()) == 5
    Test.@test SFT.order(SFT.ScalarDotSFType(1, 2)) == 2
    Test.@test SFT.order(SFT.VectorDotSFType(1, 2)) == 2
    Test.@test SFT.order(SFT.ProjectedStructureFunctionType{0, 4}()) == 4
end

Test.@testset "single vector channel: contraction equals the operator, D = $D" for D in (2, 3)
    for sf in (
        SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType(), SFT.T2ComponentSFType(),
        SFT.L3SFType(), SFT.S3SFType(), SFT.L1T2SFType(), SFT.L1T2ComponentSFType(),
        SFT.L2T1SFType(), SFT.T3SFType(),
        SFT.ProjectedStructureFunctionType{4, 0}(), SFT.ProjectedStructureFunctionType{0, 4}(),
        SFT.ProjectedStructureFunctionType{2, 2}(), SFT.ProjectedStructureFunctionType{1, 4}(),
        SFT.FullVectorStructureFunctionType{2}(), SFT.FullVectorStructureFunctionType{4}(),
        SFT.VectorDotSFType(1, 1),
    )
        _check_operator(sf, D, 1, 0)
    end
end

Test.@testset "vector and scalar channels" begin
    for sf in (
        SFT.ScalarSFType{2}(), SFT.ScalarSFType{3}(), SFT.ScalarSFType{4}(),
        SFT.MixedSFType{1, 0, 2}(), SFT.MixedSFType{1, 0, 1}(), SFT.MixedSFType{1, 2, 1}(),
        SFT.MixedSFType{0, 2, 2}(), SFT.MixedSFType{0, 0, 3}(), SFT.L2SFType(), SFT.S3SFType(),
    )
        _check_operator(sf, 2, 1, 1)
        _check_operator(sf, 3, 1, 1)
    end
    _check_operator(SFT.MixedSFType{0, 4, 1}(), 2, 1, 1)      # two transverse pairs, W = 3, rank 5
    for sf in (SFT.VectorDotSFType(1, 2), SFT.VectorDotSFType(2, 2), SFT.MixedSFType{1, 2, 1}(2, 1))
        _check_operator(sf, 2, 2, 1)
    end
    for sf in (SFT.ScalarDotSFType(1, 2), SFT.ScalarSFType{3}(2), SFT.ScalarDotSFType(2, 2))
        _check_operator(sf, 0, 0, 2)
    end
end

struct AbsoluteIncrementOperator <: SFT.AbstractPairwiseStructureFunctionType end
(::AbsoluteIncrementOperator)(δu, r̂) = abs(SFT.SFC_channel_vector(δu, 1)[1])

Test.@testset "non-polynomial operators are refused by name" begin
    δu = SA.SVector{2, Float64}(0.3, -1.2)
    r̂ = SA.SVector{2, Float64}(1.0, 0.0)
    M3 = _outer(δu, Val(3))
    for sf in (SFT.FullVectorStructureFunctionType{3}(), SFT.MixedSFType{1, 1, 1}(),
               AbsoluteIncrementOperator())
        Test.@test !SFT.is_polynomial_operator(sf)
    end
    Test.@test_throws ArgumentError SFT.moment_contract(
        SFT.FullVectorStructureFunctionType{3}(), M3, r̂, Val(1), Val(0),
    )
    err = try
        SFT.moment_contract(AbsoluteIncrementOperator(), _outer(δu, Val(1)), r̂, Val(1), Val(0))
        nothing
    catch e
        e
    end
    Test.@test err isa ArgumentError
    Test.@test occursin("AbsoluteIncrementOperator", err.msg)
    Test.@test occursin("lag sweep", err.msg)
    Test.@test_throws ArgumentError SFT.moment_contract(
        SFT.MixedSFType{1, 1, 1}(), _outer(SA.SVector{3, Float64}(randn(3)), Val(3)), r̂, Val(1), Val(1),
    )
end

Test.@testset "a channel the field does not carry is refused" begin
    δu = SA.SVector{2, Float64}(randn(2))
    r̂ = SA.SVector{2, Float64}(1.0, 0.0)
    M2 = _outer(δu, Val(2))
    Test.@test_throws ArgumentError SFT.moment_contract(SFT.VectorDotSFType(1, 2), M2, r̂, Val(1), Val(0))
    Test.@test_throws ArgumentError SFT.moment_contract(SFT.ScalarSFType{2}(), M2, r̂, Val(1), Val(0))
    Test.@test_throws ArgumentError SFT.moment_contract(
        SFT.L2SFType(), M2, SA.SVector{0, Float64}(), Val(0), Val(2),
    )
    Test.@test_throws ArgumentError SFT.moment_contract(
        SFT.T2ComponentSFType(), _outer(SA.SVector{1, Float64}(1.0), Val(2)),
        SA.SVector{1, Float64}(1.0), Val(1), Val(0),
    )
end
