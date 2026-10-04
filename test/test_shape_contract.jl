using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT

Random.seed!(42)

const SC_SF = SFT.L2SF
const SC_BINS = collect(range(0.0, 2.0; length = 6))

"""Raw sums and counts of one serial call."""
sc_raw(x, u) = SFC.calculate_structure_function(SC_SF, x, u, SC_BINS, SF.StructureFunctionSumsAndCounts;
                                                backend = CB.SerialBackend())

Test.@testset "Array shape contract" begin
    # Each slice of a result over auxiliary axes is the point-field result of that slice.
    Test.@testset "auxiliary axes, shared and varying positions" begin
        x = rand(2, 7)
        u = rand(2, 7, 2, 3)
        r = sc_raw(x, u)
        slices = [sc_raw(x, u[:, :, a, b]) for a in 1:2, b in 1:3]
        Test.@test r.sums ≈ stack(s.sums for s in slices) && r.counts == stack(s.counts for s in slices)

        xv = rand(2, 8, 3)
        uv = rand(2, 8, 3)
        rv = sc_raw(xv, uv)
        slices_v = [sc_raw(xv[:, :, t], uv[:, :, t]) for t in 1:3]
        Test.@test rv.sums ≈ stack(s.sums for s in slices_v) && rv.counts == stack(s.counts for s in slices_v)
    end

    # On a line, L2SF sums each pair's squared increment in the bin of its separation.
    Test.@testset "one-dimensional fields" begin
        x1 = reshape(collect(0.0:0.2:1.8), 1, :)
        u1 = reshape(randn(10), 1, :)
        r = sc_raw(x1, u1)
        ref_sums = zeros(eltype(r.sums), length(SC_BINS) - 1)
        ref_counts = zeros(eltype(r.counts), length(SC_BINS) - 1)
        for i in 1:9, j in (i + 1):10
            b = searchsortedfirst(SC_BINS, abs(x1[1, j] - x1[1, i])) - 1
            1 <= b <= length(ref_sums) || continue
            ref_sums[b] += (u1[1, j] - u1[1, i])^2
            ref_counts[b] += 1
        end
        Test.@test r.counts == ref_counts
        Test.@test r.sums ≈ ref_sums
    end

    # Mismatched widths or auxiliary axes are a DimensionMismatch; tuple inputs an ArgumentError.
    Test.@testset "invalid shapes" begin
        Test.@test_throws DimensionMismatch SFC.calculate_structure_function(SC_SF, rand(2, 5), rand(3, 5), SC_BINS)
        Test.@test_throws DimensionMismatch SFC.calculate_structure_function(SC_SF, rand(2, 5, 2), rand(2, 5, 3),
                                                                             SC_BINS)
        Test.@test_throws ArgumentError SFC.calculate_structure_function(SC_SF, (rand(5), rand(5)), (rand(5), rand(5)),
                                                                         SC_BINS)
    end
end

const DIMENSION_CASES = (
    (1, SFT.T2SFType(), true), (1, SFT.T3SFType(), false), (1, SFT.T2ComponentSFType(), false),
    (2, SFT.L1T2SFType(), true), (2, SFT.T3SFType(), true), (3, SFT.T3SFType(), true),
    (3, SFT.T2ComponentSFType(), true), (4, SFT.S2SFType(), true), (4, SFT.T3SFType(), false),
    (5, SFT.L3SFType(), true),
)

# Each case counts every pair, or throws ArgumentError where D lacks the transverse direction or orientation needed.
Test.@testset "dimension support follows what an operator needs" begin
    Random.seed!(1919)
    bins = collect(range(0.0, 3.0; length = 5))
    n = 12
    for (D, sf, supported) in DIMENSION_CASES
        x = rand(D, n)
        u = randn(D, n)
        s = zeros(4); c = zeros(Int, 4)
        if supported
            SFC.calculate_structure_function!(s, c, sf, x, u, bins; backend = CB.SerialBackend())
            Test.@test sum(c) == n * (n - 1) ÷ 2
        else
            Test.@test_throws ArgumentError SFC.calculate_structure_function!(
                s, c, sf, x, u, bins; backend = CB.SerialBackend())
        end
    end
end
