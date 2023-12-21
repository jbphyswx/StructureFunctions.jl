using StructureFunctions
using Test


@testset "StructureFunctions" begin
    @assert StructureFunctions.Calculations.calculate_structure_function(
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        3,
        order = 1,
    )[1] == [0.0, 0.0, 0.0]
end
