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
end
