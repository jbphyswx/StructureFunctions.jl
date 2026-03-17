using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using Test: Test


Test.@testset "StructureFunctions" begin
    a = SF.SA.SVector([1, 2, 3]...)
    b = SF.SA.SVector([0, 0, 0]...)
    Test.@test SFC.calculate_structure_function(
        SFT.LongitudinalSecondOrderStructureFunction,
        (a, a),
        (b, b),
        1;
        show_progress = true,
        verbose = true,
        bin_spacing = :linear,
    ).values == [0.0]

    include("test_core_correctness.jl")
end
