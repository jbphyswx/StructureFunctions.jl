using StructureFunctions:
    StructureFunctions as SF, StructureFunctionTypes as SFT, Calculations as SFC, HelperFunctions as SFH
using Test: Test
using Random: Random
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using ComputationalBackends: ComputationalBackends as CB

Test.@testset "Core Correctness - Block A" begin
    # Three pairs with (δu⋅r̂)² = 1, 1 and 2, one of them along a diagonal, average to 4/3.
    Test.@testset "Numerical reference (Tiny case)" begin
        x3 = [0.0 1.0 0.0; 0.0 0.0 1.0]
        u3 = [0.0 1.0 0.0; 0.0 0.0 1.0]
        val3 = SFC.calculate_structure_function(SFT.LongitudinalSecondOrderStructureFunction, x3, u3,
                                                SA.SVector(0.0, 2.0))
        Test.@test val3[1] ≈ 4 / 3
    end

    # A transverse δu = (0, 1) on r̂ = (1, 0): L2, L3 and L1T2 vanish, T2 = 1, and T3 = +1 with n̂ = ẑ × r̂.
    Test.@testset "SF wiring and signed magnitude consistency" begin
        x = [0.0 1.0; 0.0 0.0]
        u = [0.0 0.0; 1.0 2.0]
        bins = SA.SVector(0.0, 2.0)
        value(sf) = SFC.calculate_structure_function(sf, x, u, bins; backend = CB.SerialBackend())[1][1]
        Test.@test value(SFT.LongitudinalSecondOrderStructureFunction) == 0.0
        Test.@test value(SFT.TransverseSecondOrderStructureFunction) == 1.0
        Test.@test value(SFT.DiagonalConsistentThirdOrderStructureFunction) == 0.0
        Test.@test value(SFT.OffDiagonalConsistentThirdOrderStructureFunction) == 1.0
        Test.@test value(SFT.OffDiagonalInconsistentThirdOrderStructureFunction) == 0.0
    end

    # δu = (2, 3, 4) on r̂ = x̂: S2 = L2 + T2, S3 = L3 + L1T2, component forms are 1/(D-1) of these, S3 ≠ |δu|³.
    Test.@testset "3D invariant transverse and S3 semantics" begin
        δu = SA.SVector(2.0, 3.0, 4.0)
        r̂ = SA.SVector(1.0, 0.0, 0.0)
        Test.@test SFT.S2SF(δu, r̂) ≈ SFT.L2SF(δu, r̂) + SFT.T2SF(δu, r̂)
        Test.@test SFT.L2SF(δu, r̂) ≈ 4.0
        Test.@test SFT.T2SF(δu, r̂) ≈ 25.0
        Test.@test SFT.T2ComponentSF(δu, r̂) ≈ 12.5
        Test.@test SFT.L3SF(δu, r̂) ≈ 8.0
        Test.@test SFT.L1T2SF(δu, r̂) ≈ 50.0
        Test.@test SFT.L1T2ComponentSF(δu, r̂) ≈ 25.0
        Test.@test SFT.S3SF(δu, r̂) ≈ SFT.L3SF(δu, r̂) + SFT.L1T2SF(δu, r̂)
        Test.@test SFT.S3SF(δu, r̂) != SFT.FullVectorStructureFunctionType(3)(δu, r̂)
        Test.@test SFT.FullVectorStructureFunctionType(3)(δu, r̂) ≈ sqrt(sum(abs2, δu))^3
    end

    # With n̂ = ẑ × r̂, δu_T = +3: a rotation leaves L2T1 and T3 unchanged and a reflection flips their sign; the operators
    # even in δu_T keep theirs.
    Test.@testset "Signed transverse operators use the documented 2D orientation" begin
        δu = SA.SVector(2.0, 3.0)
        r̂ = SA.SVector(1.0, 0.0)
        Test.@test SFT.L2T1SF(δu, r̂) ≈ 12.0
        Test.@test SFT.T3SF(δu, r̂) ≈ 27.0
        θ = 0.7
        Rot = SA.SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
        Test.@test SFT.L2T1SF(Rot * δu, Rot * r̂) ≈ SFT.L2T1SF(δu, r̂)
        Test.@test SFT.T3SF(Rot * δu, Rot * r̂) ≈ SFT.T3SF(δu, r̂)
        Flip = SA.SMatrix{2, 2}(1.0, 0.0, 0.0, -1.0)
        Test.@test SFT.L2T1SF(Flip * δu, Flip * r̂) ≈ -SFT.L2T1SF(δu, r̂)
        Test.@test SFT.T3SF(Flip * δu, Flip * r̂) ≈ -SFT.T3SF(δu, r̂)
        Test.@test all(sf -> sf(Flip * δu, Flip * r̂) ≈ sf(δu, r̂),
                       (SFT.L2SF, SFT.T2SF, SFT.S2SF, SFT.L3SF, SFT.S3SF, SFT.L1T2SF))
    end
end

const CC_CANONICAL = SFH.CanonicalTransverseBasis()
const CC_REFERENCE_AXIS = SFH.ReferenceAxisTransverseBasis(LA.normalize(SA.SVector(1.0, sqrt(2.0), sqrt(3.0))))
const CC_PROJECTED_CASES = (
    (NL = 0, NT = 3, basis = CC_CANONICAL, backend = CB.SerialBackend()),
    (NL = 2, NT = 1, basis = CC_REFERENCE_AXIS, backend = CB.SerialBackend()),
)

# The entry sums δu_L^NL (δu⋅e)^NT with e the first transverse vector of the operator's own convention.
Test.@testset "a projected operator's convention decides its signed transverse component through the entry" begin
    Random.seed!(1103)
    N = 24
    x = randn(3, N)
    u = randn(3, N)
    bins = [0.0, 100.0]
    function expected(NL, NT, basis)
        total = 0.0
        for i in 1:(N - 1), j in (i + 1):N
            dx = SA.SVector{3}(x[1, j] - x[1, i], x[2, j] - x[2, i], x[3, j] - x[3, i])
            r̂ = dx / LA.norm(dx)
            δu = SA.SVector{3}(u[1, j] - u[1, i], u[2, j] - u[2, i], u[3, j] - u[3, i])
            e = SFH.transverse_basis(basis, r̂)[1]
            total += LA.dot(δu, r̂)^NL * LA.dot(δu, e)^NT
        end
        return total
    end
    for (; NL, NT, basis, backend) in CC_PROJECTED_CASES
        res = SFC.calculate_structure_function(
            SFT.ProjectedStructureFunctionType{NL, NT}(basis), x, u, bins, SF.StructureFunctionSumsAndCounts;
            backend,
        )
        Test.@test res.counts == [N * (N - 1) ÷ 2] && isapprox(res.sums[1], expected(NL, NT, basis); rtol = 1e-12)
    end
end

# Auto-binned edges hold every pair of the data in either spacing and precision; the result keeps the input precision.
Test.@testset "auto-binned edges hold every pair" begin
    n = 12
    for spacing in (SF.LogBinEdges, SF.LinearBinEdges), FT in (Float64, Float32)
        held = map(1:40) do seed
            Random.seed!(seed)
            r = SFC.calculate_structure_function(SFT.L2SFType(), rand(FT, 2, n), randn(FT, 2, n), 8,
                                                 SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
                                                 bin_spacing = spacing)
            sum(r.counts)
        end
        Test.@test (spacing, FT, all(==(n * (n - 1) ÷ 2), held)) == (spacing, FT, true)
    end
    r = SFC.calculate_structure_function(SFT.L2SFType(), rand(Float32, 2, n), randn(Float32, 2, n), 8;
                                         backend = CB.SerialBackend())
    Test.@test eltype(r.distance) === Float32 && eltype(r.values) === Float32
end

# A nine-component field gives the pair sums of the definition.
Test.@testset "a nine-component field" begin
    x = zeros(9, 3); x[1, :] = [0, 1, 2]
    u = zeros(9, 3); u[end, :] = [0, 2, 5]
    r = SFC.calculate_structure_function(SFT.S2SFType(), x, u, [0.0, 3.0], SF.StructureFunctionSumsAndCounts;
                                         backend = CB.SerialBackend())
    Test.@test r.counts == [3] && r.sums == [38.0]
end
