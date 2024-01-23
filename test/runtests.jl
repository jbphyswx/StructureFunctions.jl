using StructureFunctions
using Test


@testset "StructureFunctions" begin
    a = StructureFunctions.StaticArrays.SVector([1, 2, 3]...)
    b = StructureFunctions.StaticArrays.SVector([0, 0, 0]...)
    @assert StructureFunctions.Calculations.calculate_structure_function(
        (a, a),
        (b, b),
        1,
        Main.StructureFunctions.StructureFunctionTypes.LongitudinalSecondOrderStructureFunction();
        show_progress = true,
        verbose = true,
        bin_spacing = :linear,
    )[1] == [0.0]
end
