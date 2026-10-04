using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionObjects as SFO, StructureFunctionTypes as SFT

Test.@testset "2D Joint-Probability Binning Tests" begin
    Random.seed!(1234)
    n_points = 50
    x_mat = permutedims(hcat(rand(n_points) .* 50000.0, rand(n_points) .* 50000.0))
    u_mat = permutedims(hcat(randn(n_points) .* 0.5, randn(n_points) .* 0.5))

    distance_bins = [0.0, 10000.0, 20000.0, 30000.0, 50000.0]
    n_dist = length(distance_bins) - 1
    l2_value_bins = range(0.0, 1000.0, length = 11)

    sf1d = SFC.calculate_structure_function(SFT.L2SF, x_mat, u_mat, distance_bins, SF.StructureFunctionSumsAndCounts;
                                            backend = CB.SerialBackend())
    sf2d = SFC.calculate_structure_function(SFT.L2SF, x_mat, u_mat, distance_bins, l2_value_bins;
                                            backend = CB.SerialBackend())

    # Value bins wide enough to clip nothing: summing the L2SF joint histogram over values gives the 1D histogram.
    Test.@testset "L2SF Joint-Probability Binning & Mass Conservation" begin
        Test.@test sf2d isa SFO.StructureFunction2DSumsAndCounts
        Test.@test size(sf2d.sums) == size(sf2d.counts) == (n_dist, 10)
        Test.@test vec(sum(sf2d.counts; dims = 2)) == sf1d.counts
        Test.@test vec(sum(sf2d.sums; dims = 2)) ≈ sf1d.sums
    end

    # Adding two joint histograms adds their sums and counts.
    Test.@testset "Base algebraic addition (+)" begin
        combined = sf2d + sf2d
        Test.@test combined.sums == sf2d.sums .+ sf2d.sums
        Test.@test combined.counts == sf2d.counts .+ sf2d.counts
    end
end
