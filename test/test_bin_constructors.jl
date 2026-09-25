using Test
using StructureFunctions: StructureFunctions as SF

@testset "Bin constructor contracts" begin
    for T in (Float32, Float64)
        # A uniform grid is endpoints and a count, or a range; a vector of edges is a BinEdges.
        lin = SF.LinearBinEdges(T(0.25), T(4), 17)
        @test SF.LinearBinEdges(range(T(0.25), T(4); length = 17)) == lin
        @test (first(lin), last(lin), length(lin)) == (T(0.25), T(4), 17)
        @test_throws ArgumentError SF.LinearBinEdges(collect(lin))
        irregular = collect(lin)
        irregular[8] = nextfloat(irregular[8])
        @test collect(SF.BinEdges(irregular)) == irregular

        @test_throws ArgumentError SF.LinearBinEdges(T(0), T(1), 1)
        @test_throws ArgumentError SF.LinearBinEdges(T(1), T(1), 3)
        @test_throws ArgumentError SF.LinearBinEdges(T(2), T(1), 3)
        @test_throws ArgumentError SF.LinearBinEdges(T(0), T(Inf), 3)
        @test_throws ArgumentError SF.LinearBinEdges(T(NaN), T(1), 3)
        @test_throws ArgumentError SF.LinearBinEdges(-floatmax(T), floatmax(T), 3)
        # A step of at most two ulps of the larger endpoint is refused.
        @test_throws ArgumentError SF.LinearBinEdges(T(1), nextfloat(T(1), 2), 3)

        logb = SF.LogBinEdges(T(0.25), T(4), 17)
        @test (first(logb), last(logb), length(logb)) == (T(0.25), T(4), 17)
        @test_throws ArgumentError SF.LogBinEdges(collect(logb))
        @test_throws ArgumentError SF.LogBinEdges(T(0), T(1), 3)
        @test_throws ArgumentError SF.LogBinEdges(T(-1), T(1), 3)
        # Three strictly increasing edges need three floats; [1, nextfloat(1)] holds two.
        @test_throws ArgumentError SF.LogBinEdges(T(1), nextfloat(T(1)), 3)

        log_grid = range(T(-2), T(2); length = 17)
        from_log = SF.LogBinEdges_from_log_edges(log_grid)
        @test collect(from_log) ≈ exp.(log_grid)
        @test_throws ArgumentError SF.LogBinEdges_from_log_edges(collect(log_grid))
        @test_throws ArgumentError SF.LogBinEdges_from_log_edges(range(T(9999), T(10000); length = 2))

        @test SF.midpoints(SF.LinearBinEdges(T(1), T(3), 2)) == T[2]
        @test SF.midpoints(SF.LogBinEdges(T(1), T(4), 2)) ≈ T[2]
    end
end
