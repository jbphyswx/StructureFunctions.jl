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
end
