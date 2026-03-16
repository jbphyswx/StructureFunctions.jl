using StructureFunctions
using Test


@testset "StructureFunctions" begin
    a = StructureFunctions.SA.SVector([1, 2, 3]...)
    b = StructureFunctions.SA.SVector([0, 0, 0]...)
    @test StructureFunctions.Calculations.calculate_structure_function(
        (a, a),
        (b, b),
        1,
        StructureFunctions.StructureFunctionTypes.LongitudinalSecondOrderStructureFunction();
        show_progress = true,
        verbose = true,
        bin_spacing = :linear,
    )[1] == [0.0]

    include("test_core_correctness.jl")
end
