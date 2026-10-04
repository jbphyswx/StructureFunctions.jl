using Test: Test
using StructureFunctions: StructureFunctions as SF, StructureFunctionTypes as SFT, MultiFields as MF
using StaticArrays: StaticArrays as SA
using LinearAlgebra: normalize
using Random: Random

Random.seed!(20260907)

"""The rank-`P` moment tensor of a single pair, δu ⊗ … ⊗ δu, in symmetric storage."""
function _outer(δu::SA.SVector{W, T}, ::Val{P}) where {W, T, P}
    js = SFT.symmetric_indices(Val(W), Val(P))
    data = SA.SVector{length(js), T}(ntuple(n -> prod(δu[j] for j in js[n]), length(js)))
    return SFT.SymmetricMoments{W, P}(data)
end

"""The increment the operator consumes: the plain vector for one vector field and no scalars, else a FieldIncrement."""
function _increment(δu::SA.SVector{W, T}, ::Val{D}, ::Val{V}, ::Val{K}) where {W, T, D, V, K}
    V == 1 && K == 0 && return SA.SVector{D, T}(ntuple(d -> δu[d], Val(D)))
    vectors = ntuple(a -> SA.SVector{D, T}(ntuple(d -> δu[(a - 1) * D + d], Val(D))), Val(V))
    scalars = ntuple(k -> δu[V * D + k], Val(K))
    return MF.FieldIncrement{D, V, K, T}(vectors, scalars)
end

"""Whether contracting the moment tensor of each of `n` random pairs reproduces the operator on that pair."""
function _contracts(sf, D, V, K; n = 4)
    W = V * D + K
    return all(1:n) do _
        δu = SA.SVector{W, Float64}(randn(W))
        r̂ = D == 0 ? SA.SVector{0, Float64}() : SA.SVector{D, Float64}(normalize(randn(D)))
        want = sf(_increment(δu, Val(D), Val(V), Val(K)), r̂)
        got = SFT.moment_contract(sf, _outer(δu, Val(SFT.order(sf))), r̂, Val(V), Val(K))
        isapprox(got, want; atol = 1e-11, rtol = 1e-11)
    end
end

# symmetric_indices lists each sorted multi-index once, and symmetric_rank maps any ordering of one to its position.
Test.@testset "symmetric storage: indices are sorted, complete and ranked by position" begin
    shapes = ((2, 2), (3, 3), (4, 4), (3, 5), (5, 1))
    indices = [SFT.symmetric_indices(Val(W), Val(P)) for (W, P) in shapes]
    Test.@test all((((W, P), js),) -> length(js) == binomial(W + P - 1, P) && allunique(js) && all(issorted, js),
                   zip(shapes, indices))
    Test.@test all((((W, P), js),) -> all(((n, j),) -> SFT.symmetric_rank(Val(W), Val(P), j) == n ==
                                                    SFT.symmetric_rank(Val(W), Val(P), reverse(j)), enumerate(js)),
                   zip(shapes, indices))
    Test.@test_throws DimensionMismatch SFT.SymmetricMoments{3, 2}(SA.SVector{4, Float64}(1, 2, 3, 4))
end

# The contraction of a pair's moment tensor equals the operator on that pair.
Test.@testset "single vector field: contraction equals the operator, D = $D" for D in (2, 3)
    Test.@test all(sf -> _contracts(sf, D, 1, 0), (
        SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType(), SFT.T2ComponentSFType(),
        SFT.L3SFType(), SFT.S3SFType(), SFT.L1T2SFType(), SFT.L1T2ComponentSFType(),
        SFT.L2T1SFType(), SFT.T3SFType(),
        SFT.ProjectedStructureFunctionType{4, 0}(), SFT.ProjectedStructureFunctionType{0, 4}(),
        SFT.ProjectedStructureFunctionType{2, 2}(), SFT.ProjectedStructureFunctionType{1, 4}(),
        SFT.FullVectorStructureFunctionType{2}(), SFT.FullVectorStructureFunctionType{4}(),
        SFT.VectorDotSFType(1, 1),
    ))
end

# The same with scalar fields beside vector fields, and with several vector fields.
Test.@testset "vector and scalar fields" begin
    Test.@test all(sf -> _contracts(sf, 2, 1, 1) && _contracts(sf, 3, 1, 1), (
        SFT.ScalarSFType{2}(), SFT.ScalarSFType{3}(), SFT.ScalarSFType{4}(),
        SFT.MixedSFType{1, 0, 2}(), SFT.MixedSFType{1, 0, 1}(), SFT.MixedSFType{1, 2, 1}(),
        SFT.MixedSFType{0, 2, 2}(), SFT.MixedSFType{0, 0, 3}(), SFT.L2SFType(), SFT.S3SFType(),
    ))
    Test.@test _contracts(SFT.MixedSFType{0, 4, 1}(), 2, 1, 1)
    Test.@test all(sf -> _contracts(sf, 2, 2, 1),
                   (SFT.VectorDotSFType(1, 2), SFT.VectorDotSFType(2, 2), SFT.MixedSFType{1, 2, 1}(2, 1)))
    Test.@test all(sf -> _contracts(sf, 0, 0, 2),
                   (SFT.ScalarDotSFType(1, 2), SFT.ScalarSFType{3}(2), SFT.ScalarDotSFType(2, 2)))
end

struct AbsoluteIncrementOperator <: SFT.AbstractPairwiseStructureFunctionType end
(::AbsoluteIncrementOperator)(δu, r̂) = abs(SFT.SFC_field_vector(δu, 1)[1])

# A non-polynomial operator has no moment contraction and is refused with an ArgumentError naming it.
Test.@testset "non-polynomial operators are refused by name" begin
    δu = SA.SVector{2, Float64}(0.3, -1.2)
    r̂ = SA.SVector{2, Float64}(1.0, 0.0)
    Test.@test_throws ArgumentError SFT.moment_contract(
        SFT.FullVectorStructureFunctionType{3}(), _outer(δu, Val(3)), r̂, Val(1), Val(0),
    )
    err = try
        SFT.moment_contract(AbsoluteIncrementOperator(), _outer(δu, Val(1)), r̂, Val(1), Val(0))
        nothing
    catch e
        e
    end
    Test.@test err isa ArgumentError && occursin("AbsoluteIncrementOperator", err.msg)
    Test.@test_throws ArgumentError SFT.moment_contract(
        SFT.MixedSFType{1, 1, 1}(), _outer(SA.SVector{3, Float64}(randn(3)), Val(3)), r̂, Val(1), Val(1),
    )
end

# An operator reading a field the moment tensor does not carry is refused.
Test.@testset "a field the field does not carry is refused" begin
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
