using Test: Test
using StructureFunctions:
    StructureFunctions as SF, StructureFunctionObjects as SFO, StructureFunctionTypes as SFT
using StaticArrays: StaticArrays as SA

Test.@testset "Structure function resolver API" begin
    N = 10
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bins = SA.SVector(0.0, 1.4)

    Test.@test SF.get_structure_function_type(:L2SF) === SF.L2SF
    Test.@test SF.get_structure_function_type(Val(:L2SF)) === SF.L2SF
    Test.@test SF.get_structure_function_type(2, :longitudinal) === SF.L2SF
    Test.@test SF.get_structure_function_type(Val(2), Val(:long)) === SF.L2SF

    op = SF.get_structure_function_type(:L2SF)
    res = SF.calculate_structure_function(op, x, u, bins; verbose = false, show_progress = false)
    Test.@test res isa SF.StructureFunction
    Test.@test res.operator === SF.L2SF

    Test.@test_throws MethodError SF.calculate_structure_function(
        :L2SF,
        x,
        u,
        bins;
        verbose = false,
        show_progress = false,
    )
    Test.@test_throws MethodError SF.calculate_structure_function(
        2,
        :longitudinal,
        x,
        u,
        bins;
        verbose = false,
        show_progress = false,
    )

    x_tuple = (vec(x[1, :]), vec(x[2, :]))
    u_tuple = (vec(u[1, :]), vec(u[2, :]))
    Test.@test_throws ArgumentError SF.calculate_structure_function(
        SF.L2SF,
        x_tuple,
        u_tuple,
        bins;
        verbose = false,
        show_progress = false,
    )

    Test.@test SF.S2SF === SFT.SecondOrderStructureFunction
    Test.@test SF.S3SF === SFT.ThirdOrderStructureFunction
    Test.@test SF.T3SF === SFT.OffDiagonalConsistentThirdOrderStructureFunction
    Test.@test SF.L2T1SF === SFT.DiagonalInconsistentThirdOrderStructureFunction
    Test.@test SF.L1T2SF === SFT.OffDiagonalInconsistentThirdOrderStructureFunction

    Test.@test SF.get_structure_function_type(2, :rotational) ===
        SF.RotationalSecondOrderStructureFunction
    Test.@test SF.get_structure_function_type(2, :divergent) ===
        SF.DivergentSecondOrderStructureFunction
    Test.@test SF.RotationalSecondOrderStructureFunction isa SF.AbstractDerivedStructureFunctionType
    Test.@test SF.DivergentSecondOrderStructureFunction isa SF.AbstractDerivedStructureFunctionType
    Test.@test_throws ArgumentError SF.calculate_structure_function(
        SF.RotationalSecondOrderStructureFunction,
        x,
        u,
        bins;
        verbose = false,
        show_progress = false,
    )
end
