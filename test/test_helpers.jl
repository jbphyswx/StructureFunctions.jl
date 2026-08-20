using StructureFunctions: StructureFunctions as SF, HelperFunctions as SFH
using Test: Test
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

Test.@testset "HelperFunctions.jl Unit Tests" begin
    Test.@testset "digitize" begin
        bins = [0.0, 1.0, 2.0, 5.0]
        # (a, b] inclusive. searchsortedfirst - 1
        # x = 0.5 -> searchsortedfirst(0.5) = 2 -> index 1
        Test.@test SFH.digitize(0.5, bins) == 1
        Test.@test SFH.digitize(1.5, bins) == 2
        Test.@test SFH.digitize(4.0, bins) == 3

        # Edge cases
        Test.@test SFH.digitize(0.0, bins) == 0 # Out of bounds (below)
        Test.@test SFH.digitize(5.0, bins) == 3 # Right edge
        Test.@test SFH.digitize(6.0, bins) == 4 # Out of bounds (above)

        # SVector bins
        sv_bins = SA.SVector{4, Float64}(0.0, 1.0, 2.0, 5.0)
        Test.@test SFH.digitize(1.5, sv_bins) == 2

        # Vectorized digitize
        x = [0.5, 1.5, 4.0]
        Test.@test SFH.digitize(x, bins) == [1, 2, 3]
    end

    Test.@testset "Geometry: r̂ and n̂" begin
        x1 = [0.0, 0.0]
        x2 = [1.0, 0.0]

        Test.@test SFH.δr(x1, x2) == [1.0, 0.0]
        Test.@test SFH.r̂(x1, x2) == [1.0, 0.0]

        # n̂ is the right-handed quarter turn ẑ × r̂ -> [-r_hat[2], r_hat[1]], so (r̂, n̂, ẑ) is
        # right-handed. If r_hat = [1, 0], n̂ = [0, 1].
        Test.@test SFH.n̂(x1, x2) == [-0.0, 1.0]
        # pin the orientation itself, not just this instance: det[r̂ n̂] == +1
        let rh = SA.SVector(0.6, 0.8), nh = SFH.n̂(SA.SVector(0.6, 0.8))
            Test.@test rh[1] * nh[2] - rh[2] * nh[1] ≈ 1.0
        end

        # 3D case
        x1_3d = [0.0, 0.0, 0.0]
        x2_3d = [0.0, 1.0, 0.0] # y-axis
        # r̂ = [0, 1, 0]
        # k̂ = [0, 0, 1]
        # n̂ = normalize(cross(k̂, r̂)) = normalize([-1, 0, 0]) = [-1, 0, 0]
        Test.@test SFH.r̂(x1_3d, x2_3d) ≈ [0.0, 1.0, 0.0]
        Test.@test SFH.n̂(x1_3d, x2_3d) ≈ [-1.0, 0.0, 0.0]

        # Tuples and StaticArrays
        t1 = (0.0, 0.0)
        t2 = (1.0, 1.0)
        Test.@test SA.SVector(SFH.r̂(t1, t2)) ≈ SA.SVector(1 / sqrt(2), 1 / sqrt(2))
        Test.@test SA.SVector(SFH.n̂(t1, t2)) ≈ SA.SVector(-1 / sqrt(2), 1 / sqrt(2))
    end

    Test.@testset "Projections: longitudinal and transverse" begin
        # r̂ = [1, 0], n̂ = [0, 1]
        r_hat = SA.SVector(1.0, 0.0)
        δu = SA.SVector(2.0, 3.0)

        # Longitudinal: dot([2,3], [1,0]) = 2.0
        Test.@test SFH.mδu_l(δu, r_hat) == 2.0
        Test.@test SFH.δu_l(δu, r_hat) == [2.0, 0.0]

        # Transverse: dot([2,3], n̂([1,0])) = dot([2,3], [0, 1]) = 3.0
        Test.@test SFH.mδu_t(δu, r_hat) == 3.0
        Test.@test SFH.δu_t(δu, r_hat) == [0.0, 3.0] # δu - δu_l = [2,3] - [2,0] = [0,3]

        # Consistency check: magnitude squared
        # magnitude_δu_transverse is SIGNED relative to n̂
        # δu_transverse is the VECTOR difference δu - δu_l
        # norm(δu_t)^2 should be mδu_t^2
        Test.@test LA.norm(SFH.δu_t(δu, r_hat))^2 ≈ SFH.mδu_t(δu, r_hat)^2
    end

    Test.@testset "Invariant transverse energy and explicit bases" begin
        r_hat = SA.SVector(1.0, 0.0, 0.0)
        δu = SA.SVector(2.0, 3.0, 4.0)

        Test.@test SFH.transverse_norm2(δu, r_hat) ≈ 25.0
        Test.@test SFH.transverse_component_norm2(δu, r_hat) ≈ 12.5

        z_basis = SFH.ReferenceAxisTransverseBasis(SA.SVector(0.0, 0.0, 1.0))
        e = SFH.transverse_basis_vector(r_hat, z_basis)
        Test.@test e ≈ SA.SVector(0.0, 0.0, 1.0)
        Test.@test SFH.transverse_component(δu, r_hat, z_basis) ≈ 4.0

        y_basis = SFH.ReferenceAxisTransverseBasis(SA.SVector(0.0, 1.0, 0.0))
        basis_vectors = SFH.transverse_basis(y_basis, r_hat)
        Test.@test length(basis_vectors) == 2
        Test.@test basis_vectors[1] ≈ SA.SVector(0.0, 1.0, 0.0)
        Test.@test basis_vectors[2] ≈ SA.SVector(0.0, 0.0, 1.0)

        bad_basis = SFH.ReferenceAxisTransverseBasis(SA.SVector(1.0, 0.0, 0.0))
        Test.@test_throws ArgumentError SFH.transverse_basis(bad_basis, r_hat)

        gauge_basis = SFH.transverse_basis(SFH.CoordinateGaugeTransverseBasis(), r_hat)
        Test.@test length(gauge_basis) == 2
        Test.@test LA.dot(gauge_basis[1], r_hat) ≈ 0.0
        Test.@test LA.dot(gauge_basis[2], r_hat) ≈ 0.0
    end

    Test.@testset "Projection identity in multiple dimensions" begin
        fixtures = (
            (SA.SVector(3.0, 4.0), SA.SVector(1.0, 0.0)),
            (SA.SVector(2.0, -3.0, 5.0), SA.SVector(0.0, 1.0, 0.0)),
            (SA.SVector(1.0, 2.0, 3.0, 4.0), SA.SVector(0.0, 0.0, 1.0, 0.0)),
        )

        for (δu, r_hat) in fixtures
            du_L = SFH.mδu_l(δu, r_hat)
            Test.@test sum(abs2, δu) ≈ du_L^2 + SFH.transverse_norm2(δu, r_hat)
            Test.@test SFH.transverse_component_norm2(δu, r_hat) ≈
                SFH.transverse_norm2(δu, r_hat) / (length(δu) - 1)
        end
    end
end
