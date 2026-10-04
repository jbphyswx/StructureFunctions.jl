using StructureFunctions: StructureFunctions as SF, HelperFunctions as SFH, StructureFunctionTypes as SFT
using Test: Test
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

struct FlippedTransverseBasis <: SFH.AbstractTransverseBasisConvention end
SFH.transverse_basis(::FlippedTransverseBasis, r_hat) = (-SFH.n̂(r_hat),)

Test.@testset "HelperFunctions.jl Unit Tests" begin
    # Bins are (a, b]: 0 below the first edge, length(bins) above the last.
    Test.@testset "digitize" begin
        bins = [0.0, 1.0, 2.0, 5.0]
        Test.@test SFH.digitize(0.5, bins) == 1
        Test.@test SFH.digitize(1.5, bins) == 2
        Test.@test SFH.digitize(4.0, bins) == 3
        Test.@test SFH.digitize(0.0, bins) == 0
        Test.@test SFH.digitize(5.0, bins) == 3
        Test.@test SFH.digitize(6.0, bins) == 4
        Test.@test SFH.digitize(1.5, SA.SVector{4, Float64}(0.0, 1.0, 2.0, 5.0)) == 2
        Test.@test SFH.digitize([0.5, 1.5, 4.0], bins) == [1, 2, 3]
    end

    # n̂ is the right-handed quarter turn ẑ × r̂ in 2D, and normalize(ẑ × r̂) in 3D.
    Test.@testset "Geometry: r̂ and n̂" begin
        x1 = [0.0, 0.0]
        x2 = [1.0, 0.0]
        Test.@test SFH.δr(x1, x2) == [1.0, 0.0]
        Test.@test SFH.r̂(x1, x2) == [1.0, 0.0]
        Test.@test SFH.n̂(x1, x2) == [-0.0, 1.0]
        let rh = SA.SVector(0.6, 0.8), nh = SFH.n̂(SA.SVector(0.6, 0.8))
            Test.@test rh[1] * nh[2] - rh[2] * nh[1] ≈ 1.0
        end
        Test.@test SFH.r̂([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]) ≈ [0.0, 1.0, 0.0]
        Test.@test SFH.n̂([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]) ≈ [-1.0, 0.0, 0.0]
        t1 = (0.0, 0.0)
        t2 = (1.0, 1.0)
        Test.@test SA.SVector(SFH.r̂(t1, t2)) ≈ SA.SVector(1 / sqrt(2), 1 / sqrt(2))
        Test.@test SA.SVector(SFH.n̂(t1, t2)) ≈ SA.SVector(-1 / sqrt(2), 1 / sqrt(2))
    end

    # δu = (2, 3) on r̂ = (1, 0): longitudinal 2 and signed transverse 3 along n̂ = (0, 1).
    Test.@testset "Projections: longitudinal and transverse" begin
        r_hat = SA.SVector(1.0, 0.0)
        δu = SA.SVector(2.0, 3.0)
        Test.@test SFH.mδu_l(δu, r_hat) == 2.0
        Test.@test SFH.δu_l(δu, r_hat) == [2.0, 0.0]
        Test.@test SFH.mδu_t(δu, r_hat) == 3.0
        Test.@test SFH.δu_t(δu, r_hat) == [0.0, 3.0]
        Test.@test LA.norm(SFH.δu_t(δu, r_hat))^2 ≈ SFH.mδu_t(δu, r_hat)^2
    end

    # A reference axis a gives the transverse vectors normalize(a × r̂) and r̂ × that, and is refused along r̂.
    Test.@testset "explicit transverse bases" begin
        r_hat = SA.SVector(1.0, 0.0, 0.0)
        δu = SA.SVector(2.0, 3.0, 4.0)
        z_basis = SFH.ReferenceAxisTransverseBasis(SA.SVector(0.0, 0.0, 1.0))
        e = SFH.transverse_basis_vector(r_hat, z_basis)
        Test.@test e ≈ SA.SVector(0.0, 1.0, 0.0)
        Test.@test e == SFH.n̂(r_hat)
        Test.@test SFH.transverse_component(δu, r_hat, z_basis) ≈ 3.0
        y_basis = SFH.ReferenceAxisTransverseBasis(SA.SVector(0.0, 1.0, 0.0))
        basis_vectors = SFH.transverse_basis(y_basis, r_hat)
        Test.@test length(basis_vectors) == 2
        Test.@test basis_vectors[1] ≈ SA.SVector(0.0, 0.0, -1.0)
        Test.@test basis_vectors[2] ≈ SA.SVector(0.0, 1.0, 0.0)
        Test.@test SFH.transverse_basis_vector(r_hat, y_basis, 2) ≈ SA.SVector(0.0, 1.0, 0.0)
        bad_basis = SFH.ReferenceAxisTransverseBasis(SA.SVector(1.0, 0.0, 0.0))
        Test.@test_throws ArgumentError SFH.transverse_basis(bad_basis, r_hat)
        Test.@test_throws ArgumentError SFH.transverse_basis(z_basis, SA.SVector(1.0, 0.0))
    end

    # n̂ is defined at and near ±ẑ; each convention's vectors are unit, normal and odd in r̂, and sign its operators.
    Test.@testset "the transverse direction is defined along ẑ and reaches the operators through their convention" begin
        ẑ = SA.SVector(0.0, 0.0, 1.0)
        Test.@test SFH.n̂(ẑ) == SA.SVector(0.0, -1.0, 0.0)
        Test.@test SFH.n̂(-ẑ) == SA.SVector(0.0, 1.0, 0.0)
        near = LA.normalize(SA.SVector(1e-9, 0.0, 1.0))
        Test.@test all(isfinite, SFH.n̂(near))
        Test.@test LA.norm(SFH.n̂(near)) ≈ 1.0
        Test.@test abs(LA.dot(SFH.n̂(near), near)) < 1e-12
        Test.@test SFH.n̂([0.0, 0.0, 1.0]) == SA.SVector(0.0, -1.0, 0.0)
        δu = SA.SVector(2.0, 3.0, 4.0)
        Test.@test SFT.T3SF(δu, ẑ) == (-3.0)^3
        Test.@test SFT.L2T1SF(δu, ẑ) == 4.0^2 * (-3.0)

        canonical = SFH.CanonicalTransverseBasis()
        Test.@test SFH.transverse_basis(canonical, ẑ) == (SFH.n̂(ẑ), LA.cross(ẑ, SFH.n̂(ẑ)))
        Test.@test SFH.transverse_basis(canonical, SA.SVector(1.0, 0.0)) == (SA.SVector(0.0, 1.0),)
        Test.@test SFH.transverse_basis(canonical, [0.0, 0.0, 1.0]) == SFH.transverse_basis(canonical, ẑ)
        Test.@test_throws ArgumentError SFH.transverse_basis(canonical, SA.SVector(1.0, 0.0, 0.0, 0.0))

        flipped = FlippedTransverseBasis()
        Test.@test SFT.ProjectedStructureFunctionType{0, 3}(flipped)(δu, ẑ) == -SFT.T3SF(δu, ẑ)
        Test.@test SFT.ProjectedStructureFunctionType{2, 1}(flipped)(δu, ẑ) == -SFT.L2T1SF(δu, ẑ)
        Test.@test SFT.ProjectedStructureFunctionType{0, 2}(flipped)(δu, ẑ) == SFT.T2SF(δu, ẑ)
        Test.@test SFT.ProjectedStructureFunctionType{0, 4}(flipped)(δu, ẑ) ==
                   SFT.ProjectedStructureFunctionType{0, 4}()(δu, ẑ)

        x̂ = SA.SVector(1.0, 0.0, 0.0)
        about_z = SFH.ReferenceAxisTransverseBasis(ẑ)
        Test.@test SFT.ProjectedStructureFunctionType{0, 3}(about_z)(δu, x̂) == SFT.T3SF(δu, x̂)
        Test.@test_throws ArgumentError SFT.ProjectedStructureFunctionType{0, 3}(about_z)(δu, ẑ)
        a = LA.normalize(SA.SVector(1.0, sqrt(2.0), sqrt(3.0)))
        about_a = SFH.ReferenceAxisTransverseBasis(a)
        r̂ = LA.normalize(SA.SVector(0.3, -1.1, 0.7))
        e1 = LA.normalize(LA.cross(a, r̂))
        Test.@test SFT.ProjectedStructureFunctionType{0, 3}(about_a)(δu, r̂) ≈ LA.dot(δu, e1)^3
        Test.@test !(SFT.ProjectedStructureFunctionType{0, 3}(about_a)(δu, r̂) ≈ SFT.T3SF(δu, r̂))

        cases = [(rule, r) for rule in (canonical, about_a, flipped) for r in (r̂, ẑ, x̂)]
        odd_ops(rule) = (SFT.ProjectedStructureFunctionType{0, 3}(rule), SFT.ProjectedStructureFunctionType{2, 1}(rule))
        Test.@test all(((rule, r),) -> all(v -> abs(LA.norm(v) - 1) < 1e-14, SFH.transverse_basis(rule, r)), cases)
        Test.@test all(((rule, r),) -> all(v -> abs(LA.dot(v, r)) < 1e-14, SFH.transverse_basis(rule, r)), cases)
        Test.@test all(((rule, r),) -> SFH.transverse_basis(rule, -r)[1] ≈ -SFH.transverse_basis(rule, r)[1], cases)
        Test.@test all(((rule, r),) -> all(op -> op(-δu, -r) ≈ op(δu, r), odd_ops(rule)), cases)
    end

    # |δu|² splits into the longitudinal and transverse parts, and the per-component transverse part is 1/(D-1) of it.
    Test.@testset "Projection identity in multiple dimensions" begin
        fixtures = (
            (SA.SVector(3.0, 4.0), SA.SVector(1.0, 0.0)),
            (SA.SVector(2.0, -3.0, 5.0), SA.SVector(0.0, 1.0, 0.0)),
            (SA.SVector(1.0, 2.0, 3.0, 4.0), SA.SVector(0.0, 0.0, 1.0, 0.0)),
        )
        Test.@test all(((δu, r),) -> sum(abs2, δu) ≈ SFH.mδu_l(δu, r)^2 + SFH.transverse_norm2(δu, r), fixtures)
        Test.@test all(((δu, r),) -> SFH.transverse_component_norm2(δu, r) ≈
                                     SFH.transverse_norm2(δu, r) / (length(δu) - 1), fixtures)
    end
end

# The KHM derivative is exact on quadratics and lines over nonuniform separations and refuses repeated or infinite ones.
Test.@testset "the KHM derivative on nonuniform separations" begin
    grids = ([1.0, 2.0, 4.0], [0.0, 0.2, 1.0, 3.0, 7.0])
    Test.@test all(r -> isapprox(SF.KHM.finite_difference(r, r .^ 2), 2 .* r; atol = 1e-14), grids)
    Test.@test all(r -> SF.KHM.finite_difference(r, 3 .* r .+ 7) ≈ fill(3.0, length(r)), grids)
    Test.@test SF.KHM.finite_difference([1, 3], [2, 8]) == [3.0, 3.0]
    Test.@test_throws ArgumentError SF.KHM.finite_difference([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    Test.@test_throws ArgumentError SF.KHM.finite_difference([1.0, Inf, 3.0], [1.0, 2.0, 3.0])
end
