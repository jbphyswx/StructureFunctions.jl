using StructureFunctions.HelperFunctions
using Test
using LinearAlgebra
using StaticArrays

@testset "HelperFunctions.jl Unit Tests" begin
    @testset "digitize" begin
        bins = [0.0, 1.0, 2.0, 5.0]
        # (a, b] inclusive. searchsortedfirst - 1
        # x = 0.5 -> searchsortedfirst(0.5) = 2 -> index 1
        @test digitize(0.5, bins) == 1
        @test digitize(1.5, bins) == 2
        @test digitize(4.0, bins) == 3
        
        # Edge cases
        @test digitize(0.0, bins) == 0 # Out of bounds (below)
        @test digitize(5.0, bins) == 3 # Right edge
        @test digitize(6.0, bins) == 4 # Out of bounds (above)
        
        # SVector bins
        sv_bins = SVector{4, Float64}(0.0, 1.0, 2.0, 5.0)
        @test digitize(1.5, sv_bins) == 2
        
        # Vectorized digitize
        x = [0.5, 1.5, 4.0]
        @test digitize(x, bins) == [1, 2, 3]
    end

    @testset "Geometry: r̂ and n̂" begin
        x1 = [0.0, 0.0]
        x2 = [1.0, 0.0]
        
        @test δr(x1, x2) == [1.0, 0.0]
        @test r̂(x1, x2) == [1.0, 0.0]
        
        # n̂ in 2D (90 deg clockwise) -> [r_hat[2], -r_hat[1]]
        # If r_hat = [1, 0], n̂ = [0, -1]
        @test n̂(x1, x2) == [0.0, -1.0]
        
        # 3D case
        x1_3d = [0.0, 0.0, 0.0]
        x2_3d = [0.0, 1.0, 0.0] # y-axis
        # r̂ = [0, 1, 0]
        # k̂ = [0, 0, 1]
        # n̂ = normalize(cross(r̂, k̂)) = normalize([1, 0, 0]) = [1, 0, 0]
        @test r̂(x1_3d, x2_3d) ≈ [0.0, 1.0, 0.0]
        @test n̂(x1_3d, x2_3d) ≈ [1.0, 0.0, 0.0]
        
        # Tuples and StaticArrays
        t1 = (0.0, 0.0)
        t2 = (1.0, 1.0)
        @test SVector(r̂(t1, t2)) ≈ SVector(1/sqrt(2), 1/sqrt(2))
        @test SVector(n̂(t1, t2)) ≈ SVector(1/sqrt(2), -1/sqrt(2))
    end

    @testset "Projections: longitudinal and transverse" begin
        # r̂ = [1, 0], n̂ = [0, -1]
        r_hat = SVector(1.0, 0.0)
        δu = SVector(2.0, 3.0)
        
        # Longitudinal: dot([2,3], [1,0]) = 2.0
        @test mδu_l(δu, r_hat) == 2.0
        @test δu_l(δu, r_hat) == [2.0, 0.0]
        
        # Transverse: dot([2,3], n̂([1,0])) = dot([2,3], [0, -1]) = -3.0
        @test mδu_t(δu, r_hat) == -3.0
        @test δu_t(δu, r_hat) == [0.0, 3.0] # δu - δu_l = [2,3] - [2,0] = [0,3]
        
        # Consistency check: magnitude squared
        # magnitude_δu_transverse is SIGNED relative to n̂
        # δu_transverse is the VECTOR difference δu - δu_l
        # norm(δu_t)^2 should be mδu_t^2
        @test norm(δu_t(δu, r_hat))^2 ≈ mδu_t(δu, r_hat)^2
    end
end
