using Test: Test
using StructureFunctions: StructureFunctions as SF

# Uniform and logarithmic edges hit their endpoints exactly, constructors agree, and unresolvable grids are refused.
Test.@testset "Bin constructor contracts" begin
    lin = SF.LinearBinEdges(0.25f0, 4.0f0, 17)
    Test.@test SF.LinearBinEdges(range(0.25f0, 4.0f0; length = 17)) == lin
    Test.@test (first(lin), last(lin), length(lin)) == (0.25f0, 4.0f0, 17)
    Test.@test_throws ArgumentError SF.LinearBinEdges(collect(lin))
    irregular = collect(SF.LinearBinEdges(0.25, 4.0, 17))
    irregular[8] = nextfloat(irregular[8])
    Test.@test collect(SF.BinEdges(irregular)) == irregular
    Test.@test_throws ArgumentError SF.LinearBinEdges(0.0, 1.0, 1)
    Test.@test_throws ArgumentError SF.LinearBinEdges(1.0f0, 1.0f0, 3)
    Test.@test_throws ArgumentError SF.LinearBinEdges(2.0, 1.0, 3)
    Test.@test_throws ArgumentError SF.LinearBinEdges(0.0f0, Inf32, 3)
    Test.@test_throws ArgumentError SF.LinearBinEdges(NaN, 1.0, 3)
    Test.@test_throws ArgumentError SF.LinearBinEdges(-floatmax(Float32), floatmax(Float32), 3)
    Test.@test_throws ArgumentError SF.LinearBinEdges(1.0, nextfloat(1.0, 2), 3)

    logb = SF.LogBinEdges(0.25, 4.0, 17)
    Test.@test (first(logb), last(logb), length(logb)) == (0.25, 4.0, 17)
    Test.@test_throws ArgumentError SF.LogBinEdges(collect(logb))
    Test.@test_throws ArgumentError SF.LogBinEdges(0.0f0, 1.0f0, 3)
    Test.@test_throws ArgumentError SF.LogBinEdges(-1.0, 1.0, 3)
    Test.@test_throws ArgumentError SF.LogBinEdges(1.0f0, nextfloat(1.0f0), 3)

    log_grid = range(-2.0f0, 2.0f0; length = 17)
    Test.@test collect(SF.LogBinEdges_from_log_edges(log_grid)) ≈ exp.(log_grid)
    Test.@test_throws ArgumentError SF.LogBinEdges_from_log_edges(collect(range(-2.0, 2.0; length = 17)))
    Test.@test_throws ArgumentError SF.LogBinEdges_from_log_edges(range(9999.0, 10000.0; length = 2))

    Test.@test SF.midpoints(SF.LinearBinEdges(1.0, 3.0, 2)) == [2.0]
    Test.@test SF.midpoints(SF.LogBinEdges(1.0f0, 4.0f0, 2)) ≈ Float32[2]
end
