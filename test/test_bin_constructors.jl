using Test: Test
using StructureFunctions: StructureFunctions as SF

Test.@testset "Bin constructor contracts" begin
    for T in (Float32, Float64)
        # A uniform grid is endpoints and a count, or a range; a vector of edges is a BinEdges.
        lin = SF.LinearBinEdges(T(0.25), T(4), 17)
        Test.@test SF.LinearBinEdges(range(T(0.25), T(4); length = 17)) == lin
        Test.@test (first(lin), last(lin), length(lin)) == (T(0.25), T(4), 17)
        Test.@test_throws ArgumentError SF.LinearBinEdges(collect(lin))
        irregular = collect(lin)
        irregular[8] = nextfloat(irregular[8])
        Test.@test collect(SF.BinEdges(irregular)) == irregular

        Test.@test_throws ArgumentError SF.LinearBinEdges(T(0), T(1), 1)
        Test.@test_throws ArgumentError SF.LinearBinEdges(T(1), T(1), 3)
        Test.@test_throws ArgumentError SF.LinearBinEdges(T(2), T(1), 3)
        Test.@test_throws ArgumentError SF.LinearBinEdges(T(0), T(Inf), 3)
        Test.@test_throws ArgumentError SF.LinearBinEdges(T(NaN), T(1), 3)
        Test.@test_throws ArgumentError SF.LinearBinEdges(-floatmax(T), floatmax(T), 3)
        # A step of at most two ulps of the larger endpoint is refused.
        Test.@test_throws ArgumentError SF.LinearBinEdges(T(1), nextfloat(T(1), 2), 3)

        logb = SF.LogBinEdges(T(0.25), T(4), 17)
        Test.@test (first(logb), last(logb), length(logb)) == (T(0.25), T(4), 17)
        Test.@test_throws ArgumentError SF.LogBinEdges(collect(logb))
        Test.@test_throws ArgumentError SF.LogBinEdges(T(0), T(1), 3)
        Test.@test_throws ArgumentError SF.LogBinEdges(T(-1), T(1), 3)
        # Three strictly increasing edges need three floats; [1, nextfloat(1)] holds two.
        Test.@test_throws ArgumentError SF.LogBinEdges(T(1), nextfloat(T(1)), 3)

        log_grid = range(T(-2), T(2); length = 17)
        from_log = SF.LogBinEdges_from_log_edges(log_grid)
        Test.@test collect(from_log) ≈ exp.(log_grid)
        Test.@test_throws ArgumentError SF.LogBinEdges_from_log_edges(collect(log_grid))
        Test.@test_throws ArgumentError SF.LogBinEdges_from_log_edges(range(T(9999), T(10000); length = 2))

        Test.@test SF.midpoints(SF.LinearBinEdges(T(1), T(3), 2)) == T[2]
        Test.@test SF.midpoints(SF.LogBinEdges(T(1), T(4), 2)) ≈ T[2]
    end
end
