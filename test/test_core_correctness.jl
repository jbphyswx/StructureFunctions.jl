using StructureFunctions
using StructureFunctions.StructureFunctionTypes
using StructureFunctions.Calculations
using Test
using StaticArrays

@testset "Core Correctness - Block A" begin
    @testset "Commit A1: Blocked Path Regression" begin
        x = ([0.0, 1.0], [0.0, 0.0])
        u = ([1.0, 2.0], [0.0, 0.0])
        bins = SVector(((0.0, 2.0),))
        sf_type = LongitudinalSecondOrderStructureFunction()
        @test_nowarn calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
    end

    @testset "Commit A3: Pair-count Verification" begin
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

    @testset "Commit A4: Numerical Reference (Tiny Case)" begin
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
end
