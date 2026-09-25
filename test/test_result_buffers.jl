using Test
using StructureFunctions: StructureFunctions as SF, Calculations as C, StructureFunctionTypes as T

@testset "Count-bound arithmetic" begin
    for n in (0, 1, 24, typemax(Int64), typemax(UInt64), big(2)^100)
        @test C._pair_count_bound(n) == big(n) * (big(n) - 1) ÷ 2
    end
    @test_throws ArgumentError C._pair_count_bound(-1)
    @test C._assert_counts_representable(UInt128, typemax(UInt64)) === nothing
    @test_throws ArgumentError C._assert_counts_representable(UInt64, typemax(UInt64))
    @test_throws ArgumentError C._assert_counts_can_accumulate(Int16[-1, 0], 2)
end

@testset "Host result conversion and normalization" begin
    bins = [0.0, 1.0, 2.0]
    raw = SF.StructureFunctionSumsAndCounts(T.S2SFType(), bins, [6.0, 0.0], UInt64[3, 0])
    host = SF.to_host(raw)
    @test host.sums == raw.sums
    @test host.counts == raw.counts
    @test host.distance === bins
    @test host.sums !== raw.sums
    @test host.counts !== raw.counts
    @test SF.to_host((S2=raw,)).S2.sums == raw.sums
    @test isequal(C._bin_average(raw.sums, raw.counts), [2.0, NaN])
    tensor = SF.StructureFunctionTensorSumsAndCounts(Val(2), bins, fill(6.0, 2, 2, 2), UInt64[3, 0])
    values = C._tensor_bin_average(tensor.sums, tensor.counts, Val(2))
    @test values[:, :, 1] == fill(2.0, 2, 2)
    @test all(isnan, values[:, :, 2])
    @test SF.to_host(tensor).sums == tensor.sums
    @test_throws DimensionMismatch C._bin_average!(zeros(3), raw.sums, raw.counts)
end
