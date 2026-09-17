using Test: Test
using StructureFunctions:
    StructureFunctions as SF, StructureFunctionObjects as SFO, StructureFunctionTypes as SFT,
    Calculations as SFC
using StaticArrays: StaticArrays as SA

Test.@testset "Structure function resolver API" begin
    N = 10
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bins = SA.SVector(0.0, 1.4)

    Test.@test SFT.get_structure_function_type(:L2SF) === SFT.L2SF
    Test.@test SFT.get_structure_function_type(Val(:L2SF)) === SFT.L2SF
    Test.@test SFT.get_structure_function_type(2, :longitudinal) === SFT.L2SF
    Test.@test SFT.get_structure_function_type(Val(2), Val(:long)) === SFT.L2SF

    op = SFT.get_structure_function_type(:L2SF)
    res = SFC.calculate_structure_function(op, x, u, bins; verbose = false, show_progress = false)
    Test.@test res isa SF.StructureFunction
    Test.@test res.operator === SFT.L2SF

    Test.@test_throws MethodError SFC.calculate_structure_function(
        :L2SF,
        x,
        u,
        bins;
        verbose = false,
        show_progress = false,
    )
    Test.@test_throws MethodError SFC.calculate_structure_function(
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
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        SFT.L2SF,
        x_tuple,
        u_tuple,
        bins;
        verbose = false,
        show_progress = false,
    )

    Test.@test SFT.S2SF === SFT.SecondOrderStructureFunction
    Test.@test SFT.S3SF === SFT.ThirdOrderStructureFunction
    Test.@test SFT.T3SF === SFT.OffDiagonalConsistentThirdOrderStructureFunction
    Test.@test SFT.L2T1SF === SFT.DiagonalInconsistentThirdOrderStructureFunction
    Test.@test SFT.L1T2SF === SFT.OffDiagonalInconsistentThirdOrderStructureFunction

    Test.@test SFT.get_structure_function_type(2, :rotational) ===
        SFT.RotationalSecondOrderStructureFunction
    Test.@test SFT.get_structure_function_type(2, :divergent) ===
        SFT.DivergentSecondOrderStructureFunction
    Test.@test SFT.RotationalSecondOrderStructureFunction isa SFT.AbstractDerivedStructureFunctionType
    Test.@test SFT.DivergentSecondOrderStructureFunction isa SFT.AbstractDerivedStructureFunctionType
    Test.@test_throws ArgumentError SFC.calculate_structure_function(
        SFT.RotationalSecondOrderStructureFunction,
        x,
        u,
        bins;
        verbose = false,
        show_progress = false,
    )
end
