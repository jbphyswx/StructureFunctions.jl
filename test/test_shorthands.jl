using Test: Test
using StructureFunctions: StructureFunctionTypes as SFT, Calculations as SFC
using StaticArrays: StaticArrays as SA

# Names, orders and categories resolve to their operators, and a derived operator is refused by the pair entry.
Test.@testset "Structure function resolver API" begin
    x, u = rand(2, 10), rand(2, 10)
    Test.@test SFT.get_structure_function_type(:L2SF) === SFT.L2SF
    Test.@test SFT.get_structure_function_type(Val(:L2SF)) === SFT.L2SF
    Test.@test SFT.get_structure_function_type(2, :longitudinal) === SFT.L2SF
    Test.@test SFT.get_structure_function_type(Val(2), Val(:long)) === SFT.L2SF
    Test.@test SFT.get_structure_function_type(2, :rotational) === SFT.RotationalSecondOrderStructureFunction
    Test.@test SFT.get_structure_function_type(2, :divergent) === SFT.DivergentSecondOrderStructureFunction
    Test.@test_throws ArgumentError SFC.calculate_structure_function(SFT.RotationalSecondOrderStructureFunction, x, u,
                                                                     SA.SVector(0.0, 1.4))
end
