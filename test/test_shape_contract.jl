using Test
using Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT

Random.seed!(42)

@testset "Array shape contract" begin
    sf = SFT.L2SF
    bins = collect(range(0.0, 2.0; length = 6))
    n_bins = length(bins) - 1

    @testset "2D and 3D point fields" begin
        x2 = rand(2, 8)
        u2 = rand(2, 8)
        x3 = rand(3, 8)
        u3 = rand(3, 8)

        r2 = SFC.calculate_structure_function(
            sf, x2, u2, bins; backend = SFC.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
        )
        r3 = SFC.calculate_structure_function(
            sf, x3, u3, bins; backend = SFC.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
        )

        @test size(r2.sums) == (n_bins,)
        @test size(r3.sums) == (n_bins,)
    end

    @testset "shared-position auxiliary axes" begin
        x = rand(2, 9)
        u = rand(2, 9, 3)
        r = SFC.calculate_structure_function(
            sf, x, u, bins; backend = SFC.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
        )
        ref_sums = zeros(eltype(r.sums), n_bins, 3)
        ref_counts = zeros(eltype(r.counts), n_bins, 3)
        for t in 1:3
            rt = SFC.calculate_structure_function(
                sf, x, @view(u[:, :, t]), bins; backend = SFC.SerialBackend(),
                output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
            )
            ref_sums[:, t] .= rt.sums
            ref_counts[:, t] .= rt.counts
        end
        @test r.sums ≈ ref_sums
        @test r.counts == ref_counts
    end

    @testset "shared-position multiple auxiliary axes" begin
        x = rand(2, 7)
        u = rand(2, 7, 2, 3)
        r = SFC.calculate_structure_function(
            sf, x, u, bins; backend = SFC.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
        )
        @test size(r.sums) == (n_bins, 2, 3)
        @test size(r.counts) == (n_bins, 2, 3)
    end

    @testset "varying-position auxiliary axes" begin
        x = rand(2, 8, 3)
        u = rand(2, 8, 3)
        r = SFC.calculate_structure_function(
            sf, x, u, bins; backend = SFC.SerialBackend(),
            output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
        )
        ref_sums = zeros(eltype(r.sums), n_bins, 3)
        ref_counts = zeros(eltype(r.counts), n_bins, 3)
        for t in 1:3
            rt = SFC.calculate_structure_function(
                sf, @view(x[:, :, t]), @view(u[:, :, t]), bins; backend = SFC.SerialBackend(),
                output_type = SF.StructureFunctionSumsAndCounts, verbose = false, show_progress = false,
            )
            ref_sums[:, t] .= rt.sums
            ref_counts[:, t] .= rt.counts
        end
        @test r.sums ≈ ref_sums
        @test r.counts == ref_counts
    end

    @testset "invalid shapes" begin
        @test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(1, 5), rand(1, 5), bins; verbose = false, show_progress = false,
        )
        @test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(2, 5), rand(3, 5), bins; verbose = false, show_progress = false,
        )
        @test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(2, 5, 2), rand(2, 5, 3), bins; verbose = false, show_progress = false,
        )
        @test_throws ArgumentError SFC.calculate_structure_function(
            sf, (rand(5), rand(5)), (rand(5), rand(5)), bins; verbose = false, show_progress = false,
        )
    end
end
