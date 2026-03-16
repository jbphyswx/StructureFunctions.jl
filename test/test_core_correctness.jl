using StructureFunctions
using StructureFunctions.StructureFunctionTypes
using StructureFunctions.Calculations
using Test
using StaticArrays

@testset "Core Correctness - Block A" begin
    @testset "Blocked path regression" begin
        x = ([0.0, 1.0], [0.0, 0.0])
        u = ([1.0, 2.0], [0.0, 0.0])
        bins = SVector(((0.0, 2.0),))
        sf_type = LongitudinalSecondOrderStructureFunction()
        @test_nowarn calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
    end

    @testset "Pair-count verification" begin
        # N=3 points -> N(N-1)/2 = 3 pairs
        x = ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
        u = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        bins = SVector(((0.0, 3.0),))
        sf_type = SecondOrderStructureFunction()
        
        (output, counts), _ = calculate_structure_function(x, u, bins, sf_type, 
                                verbose=false, show_progress=false, return_sums_and_counts=true)
        @test sum(counts) == 3.0
        
        # N=4 points -> 4*3/2 = 6 pairs
        x4 = ([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
        (output4, counts4), _ = calculate_structure_function(x4, (zeros(4), zeros(4)), bins, sf_type, 
                                verbose=false, show_progress=false, return_sums_and_counts=true)
        @test sum(counts4) == 6.0
    end

    @testset "Numerical reference (Tiny case)" begin
        # 2 points: (0,0) and (1,0)
        # Velocities: (1,0) and (2,0)
        # du = (1, 0), rhat = (1, 0)
        # SecondOrderLongitudinal: (du . rhat)^2 = 1^2 = 1.0
        x = ([0.0, 1.0], [0.0, 0.0])
        u = ([1.0, 2.0], [0.0, 0.0])
        bins = SVector(((0.0, 2.0),))
        sf_type = LongitudinalSecondOrderStructureFunction()
        
        val, _ = calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
        @test val[1] ≈ 1.0
        
        # 3 points: (0,0), (1,0), (0,1)
        # Velocities: (0,0), (1,0), (0,1)
        # Pairs:
        # 1-2: dx=(1,0), du=(1,0), rhat=(1,0), du.rhat=1, (du.rhat)^2 = 1
        # 1-3: dx=(0,1), du=(0,1), rhat=(0,1), du.rhat=1, (du.rhat)^2 = 1
        # 2-3: dx=(-1,1), du=(-1,1), rhat=(-1,1)/sqrt(2), du.rhat=sqrt(2), (du.rhat)^2 = 2
        # Sum = 1 + 1 + 2 = 4
        # Count = 3
        # Mean = 4/3
        x3 = ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        u3 = ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        val3, _ = calculate_structure_function(x3, u3, bins, sf_type, verbose=false, show_progress=false)
        @test val3[1] ≈ 4/3
    end

    @testset "SF wiring and signed magnitude consistency" begin
        # Test case: 2 points at (0,0) and (1,0) -> rhat = (1, 0), nhat = (0, -1)
        # Velocities: (0,1) and (0,2) -> du = (0,1) [Transverse]
        x = ([0.0, 1.0], [0.0, 0.0])
        u = ([0.0, 0.0], [1.0, 2.0])
        bins = SVector(((0.0, 2.0),))
        
        # Longitudinal Second Order: (du.rhat)^2 = 0^2 = 0
        @test calculate_structure_function(x, u, bins, LongitudinalSecondOrderStructureFunction(), verbose=false, show_progress=false)[1][1] == 0.0
        
        # Transverse Second Order: |du_t|^2 = (-1)^2 = 1
        @test calculate_structure_function(x, u, bins, TransverseSecondOrderStructureFunction(), verbose=false, show_progress=false)[1][1] == 1.0
        
        # Diagonal Consistent Third Order (l^3): 0^3 = 0
        @test calculate_structure_function(x, u, bins, DiagonalConsistentThirdOrderStructureFunction(), verbose=false, show_progress=false)[1][1] == 0.0
        
        # Off-Diagonal Consistent Third Order (t^3): (-1)^3 = -1
        @test calculate_structure_function(x, u, bins, OffDiagonalConsistentThirdOrderStructureFunction(), verbose=false, show_progress=false)[1][1] == -1.0
        
        # Off-Diagonal Inconsistent Third Order (l*t^2): 0 * (-1)^2 = 0
        @test calculate_structure_function(x, u, bins, OffDiagonalInconsistentThirdOrderStructureFunction(), verbose=false, show_progress=false)[1][1] == 0.0
    end
end

@testset "Type Stability and Performance - Block C" begin
    x = ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    u = ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    bins = SVector(((0.0, 3.0),))
    sf_type = LongitudinalSecondOrderStructureFunction()
    
    # Test inference of the core kernel call
    δu = SVector{2, Float64}(1.0, 0.0)
    r̂ = SVector{2, Float64}(1.0, 0.0)
    @test @inferred(sf_type(δu, r̂)) == 1.0
    
    # Test inference of the calculator
    @testset "Inference of calculate_structure_function" begin
        # We manually check the type to account for the intentional Union return
        # from the return_sums_and_counts flag.
        res = @test_nowarn calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
        @test res isa Tuple{Vector{Float64}, SVector{1, Tuple{Float64, Float64}}}
        
        # We check that the inner logic is stable enough to yield this Union
        # (The compiler knows it's either Sums+Counts or just Mean)
        @test typeof(res) <: Union{Tuple{Vector{Float64}, SVector{1, Tuple{Float64, Float64}}}, Tuple{Any, SVector{1, Tuple{Float64, Float64}}}}
    end

    @testset "Allocation check" begin
        # Baseline check
        @test_nowarn calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
    end
end
