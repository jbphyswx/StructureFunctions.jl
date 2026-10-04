using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as C, StructureFunctionTypes as T
using ComputationalBackends: ComputationalBackends as CB

# The worst-case pair count is exact for every Int point count, and a count type is refused exactly past it.
Test.@testset "pair-count bound at its boundaries" begin
    Test.@test all(n -> C._pair_count_bound(n) == binomial(big(n), 2), (0, 1, 2, 24, 92682, typemax(Int64)))
    Test.@test_throws ArgumentError C._pair_count_bound(-1)
    Test.@test C._assert_counts_representable(UInt32, 92682) === nothing
    Test.@test_throws ArgumentError C._assert_counts_representable(UInt32, 92683)
    Test.@test C._assert_counts_representable(UInt128, typemax(Int64)) === nothing
    Test.@test_throws ArgumentError C._assert_counts_representable(UInt64, typemax(Int64))
end

# to_host returns equal values in fresh arrays, through named tuples of results.
Test.@testset "to_host copies a result's buffers" begin
    bins = [0.0, 1.0, 2.0]
    raw = SF.StructureFunctionSumsAndCounts(T.S2SFType(), bins, [6.0, 0.0], UInt64[3, 0])
    tensor = SF.StructureFunctionTensorSumsAndCounts(Val(2), bins, fill(6.0, 2, 2, 2), UInt64[3, 0])
    host = SF.to_host((S2 = raw, tensor = tensor))
    Test.@test (host.S2.sums, host.S2.counts, host.S2.distance) == (raw.sums, raw.counts, bins)
    Test.@test (host.tensor.sums, host.tensor.counts) == (tensor.sums, tensor.counts)
    Test.@test host.S2.sums !== raw.sums && host.S2.counts !== raw.counts
end

# Pairs at r = 1 with δu_L = 1, 2 and at r = 2 with δu_L = 3: bin means 5/2 and 9, and NaN in the empty bin.
Test.@testset "a result's values are its bin means, NaN in an empty bin" begin
    x = [0.0 1.0 2.0; 0.0 0.0 0.0]
    u = [0.0 1.0 3.0; 0.0 0.0 0.0]
    bins = [0.0, 1.5, 2.5, 3.5]
    ser = CB.SerialBackend()
    Test.@test isequal(C.calculate_structure_function(T.L2SFType(), x, u, bins; backend = ser).values, [2.5, 9.0, NaN])
    t = C.calculate_structure_function_tensor(Val(2), x, u, bins; backend = ser).values
    Test.@test isequal(t, cat([2.5 0.0; 0.0 0.0], [9.0 0.0; 0.0 0.0], fill(NaN, 2, 2); dims = 3))
end

# A count type or buffer that cannot take every pair is refused before anything is written.
Test.@testset "count overflow is refused before anything is written" begin
    bins = [0.0, 2.0]
    a = SF.StructureFunctionSumsAndCounts(T.S2SFType(), bins, [1.0], UInt8[250])
    b = SF.StructureFunctionSumsAndCounts(T.S2SFType(), bins, [1.0], UInt8[6])
    Test.@test_throws ArgumentError a + b
    x = [0.0 0.1 0.2 0.3; 0.0 0.0 0.0 0.0]
    u = [0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0]
    s, c = [0.0], UInt8[250]
    Test.@test_throws ArgumentError C.calculate_structure_function!(s, c, T.S2SFType(), x, u, bins)
    Test.@test s == [0.0] && c == UInt8[250]
    ts, tc = zeros(2, 2, 1), UInt8[250]
    Test.@test_throws ArgumentError C.calculate_structure_function_tensor!(ts, tc, Val(2), x, u, bins)
    Test.@test all(iszero, ts) && tc == UInt8[250]
    Test.@test_throws ArgumentError C.calculate_structure_function!(zeros(1), Int16[-1], T.S2SFType(), x, u, bins)
    x24 = rand(2, 24)
    Test.@test_throws ArgumentError C.calculate_structure_function(T.S2SFType(), x24, x24, bins, UInt8)
end
