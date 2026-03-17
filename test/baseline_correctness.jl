using StructureFunctions: StructureFunctions as SF, StructureFunctionTypes as SFT, Calculations as SFC
using Test: Test
using StaticArrays: StaticArrays as SA

Test.@testset "Baseline Correctness" begin
    # Test case: 2 points, 1 bin
    # If the blocker is present, this should throw an ErrorException
    x = ([0.0, 1.0], [0.0, 0.0]) # NTuple of vectors
    u = ([1.0, 2.0], [0.0, 0.0])
    bins = SA.SVector(((0.0, 2.0),))
    sf_type = SFT.LongitudinalSecondOrderStructureFunction

    Test.@test_throws ErrorException SFC.calculate_structure_function(sf_type, x, u, bins)
end
