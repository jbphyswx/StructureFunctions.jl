using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT

Random.seed!(42)

Test.@testset "Array shape contract" begin
    sf = SFT.L2SF
    bins = collect(range(0.0, 2.0; length = 6))
    n_bins = length(bins) - 1

    Test.@testset "2D and 3D point fields" begin
        x2 = rand(2, 8)
        u2 = rand(2, 8)
        x3 = rand(3, 8)
        u3 = rand(3, 8)

        r2 = SFC.calculate_structure_function(
            sf, x2, u2, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
        )
        r3 = SFC.calculate_structure_function(
            sf, x3, u3, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
        )

        Test.@test size(r2.sums) == (n_bins,)
        Test.@test size(r3.sums) == (n_bins,)
    end

    Test.@testset "shared-position auxiliary axes" begin
        x = rand(2, 9)
        u = rand(2, 9, 3)
        r = SFC.calculate_structure_function(
            sf, x, u, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
        )
        ref_sums = zeros(eltype(r.sums), n_bins, 3)
        ref_counts = zeros(eltype(r.counts), n_bins, 3)
        for t in 1:3
            rt = SFC.calculate_structure_function(
                sf, x, @view(u[:, :, t]), bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
            )
            ref_sums[:, t] .= rt.sums
            ref_counts[:, t] .= rt.counts
        end
        Test.@test r.sums ≈ ref_sums
        Test.@test r.counts == ref_counts
    end

    Test.@testset "shared-position multiple auxiliary axes" begin
        x = rand(2, 7)
        u = rand(2, 7, 2, 3)
        r = SFC.calculate_structure_function(
            sf, x, u, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
        )
        Test.@test size(r.sums) == (n_bins, 2, 3)
        Test.@test size(r.counts) == (n_bins, 2, 3)
    end

    Test.@testset "varying-position auxiliary axes" begin
        x = rand(2, 8, 3)
        u = rand(2, 8, 3)
        r = SFC.calculate_structure_function(
            sf, x, u, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
        )
        ref_sums = zeros(eltype(r.sums), n_bins, 3)
        ref_counts = zeros(eltype(r.counts), n_bins, 3)
        for t in 1:3
            rt = SFC.calculate_structure_function(
                sf, @view(x[:, :, t]), @view(u[:, :, t]), bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
            )
            ref_sums[:, t] .= rt.sums
            ref_counts[:, t] .= rt.counts
        end
        Test.@test r.sums ≈ ref_sums
        Test.@test r.counts == ref_counts
    end

    Test.@testset "one-dimensional fields" begin
        # A single velocity component on a line: the separation direction is ±1, so the longitudinal
        # increment is the whole increment and L2SF is the mean squared difference.
        x1 = reshape(collect(0.0:0.2:1.8), 1, :)
        u1 = reshape(randn(10), 1, :)
        r = SFC.calculate_structure_function(
            sf, x1, u1, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
        )
        ref_sums = zeros(eltype(r.sums), n_bins)
        ref_counts = zeros(eltype(r.counts), n_bins)
        for i in 1:9, j in (i + 1):10
            b = searchsortedfirst(bins, abs(x1[1, j] - x1[1, i])) - 1
            1 <= b <= n_bins || continue
            ref_sums[b] += (u1[1, j] - u1[1, i])^2
            ref_counts[b] += 1
        end
        Test.@test r.counts == ref_counts
        Test.@test r.sums ≈ ref_sums
        Test.@test sum(r.counts) > 0
    end

    Test.@testset "invalid shapes" begin
        Test.@test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(2, 5), rand(3, 5), bins,
        )
        Test.@test_throws DimensionMismatch SFC.calculate_structure_function(
            sf, rand(2, 5, 2), rand(2, 5, 3), bins,
        )
        Test.@test_throws ArgumentError SFC.calculate_structure_function(
            sf, (rand(5), rand(5)), (rand(5), rand(5)), bins,
        )
    end
end

Test.@testset "dimension support follows what an operator needs" begin
    # Operators built from δu_L and ‖δu‖² run at every D; operators odd in the transverse component are refused outside D = 2 and 3.
    Random.seed!(1919)
    bins = collect(range(0.0, 2.0; length = 5))
    unrestricted = (SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.L1T2SFType())
    for D in (1, 2, 3, 4, 5)
        x = rand(D, 30)
        u = randn(D, 30)
        for sf in unrestricted
            s = zeros(4); c = zeros(Int, 4)
            SFC.calculate_structure_function!(s, c, sf, x, u, bins; backend = CB.SerialBackend())
            Test.@test sum(c) == 30 * 29 ÷ 2
        end
        # odd in the transverse component: needs an orientation, which exists only at D = 2 and 3
        # (these entries accumulate, so every call gets its own buffers)
        if D == 2 || D == 3
            s3 = zeros(4); c3 = zeros(Int, 4)
            SFC.calculate_structure_function!(s3, c3, SFT.T3SFType(), x, u, bins;
                                              backend = CB.SerialBackend())
            Test.@test sum(c3) == 30 * 29 ÷ 2
        else
            Test.@test_throws ArgumentError SFC.calculate_structure_function!(
                zeros(4), zeros(Int, 4), SFT.T3SFType(), x, u, bins; backend = CB.SerialBackend())
        end
        # averaging over transverse directions needs at least one of them
        if D == 1
            Test.@test_throws ArgumentError SFC.calculate_structure_function!(
                zeros(4), zeros(Int, 4), SFT.T2ComponentSFType(), x, u, bins;
                backend = CB.SerialBackend())
        else
            sc = zeros(4); cc = zeros(Int, 4)
            SFC.calculate_structure_function!(sc, cc, SFT.T2ComponentSFType(), x, u, bins;
                                              backend = CB.SerialBackend())
            Test.@test sum(cc) == 30 * 29 ÷ 2
        end
    end
end
