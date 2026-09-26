using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as C, StructureFunctionTypes as T

Test.@testset "Count-bound arithmetic" begin
    for (n, pairs) in ((0, 0), (1, 0), (2, 1), (24, 276))
        Test.@test C._pair_count_bound(n) === UInt128(pairs)
    end
    Test.@test C._pair_count_bound(typemax(Int64)) === 0x1fffffffffffffff4000000000000001
    Test.@test_throws ArgumentError C._pair_count_bound(-1)
    Test.@test C._assert_counts_representable(UInt128, typemax(Int64)) === nothing
    Test.@test_throws ArgumentError C._assert_counts_representable(UInt64, typemax(Int64))
    Test.@test_throws ArgumentError C._assert_counts_can_accumulate(Int16[-1, 0], 2, C.NoWeights())
end

Test.@testset "Host result conversion and normalization" begin
    bins = [0.0, 1.0, 2.0]
    raw = SF.StructureFunctionSumsAndCounts(T.S2SFType(), bins, [6.0, 0.0], UInt64[3, 0])
    host = SF.to_host(raw)
    Test.@test host.sums == raw.sums
    Test.@test host.counts == raw.counts
    Test.@test host.distance === bins
    Test.@test host.sums !== raw.sums
    Test.@test host.counts !== raw.counts
    Test.@test SF.to_host((S2=raw,)).S2.sums == raw.sums
    Test.@test isequal(C._bin_average(raw.sums, raw.counts), [2.0, NaN])
    tensor = SF.StructureFunctionTensorSumsAndCounts(Val(2), bins, fill(6.0, 2, 2, 2), UInt64[3, 0])
    values = C._tensor_bin_average(tensor.sums, tensor.counts, Val(2))
    Test.@test values[:, :, 1] == fill(2.0, 2, 2)
    Test.@test all(isnan, values[:, :, 2])
    Test.@test SF.to_host(tensor).sums == tensor.sums
    Test.@test_throws DimensionMismatch C._bin_average!(zeros(3), raw.sums, raw.counts)
end
