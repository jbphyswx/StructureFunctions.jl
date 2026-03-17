using StructureFunctions
using StructureFunctions.StructureFunctionTypes
using StructureFunctions.Calculations
using Test
using StaticArrays

@testset "Baseline Correctness" begin
    # Test case: 2 points, 1 bin
    # If the blocker is present, this should throw an ErrorException
    x = ([0.0, 1.0], [0.0, 0.0]) # NTuple of vectors
    u = ([1.0, 2.0], [0.0, 0.0])
    bins = SVector(((0.0, 2.0),))
    sf_type = LongitudinalSecondOrderStructureFunction

    @test_throws ErrorException calculate_structure_function(sf_type, x, u, bins)
end
