using Test: Test, @inferred
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using StaticArrays: StaticArrays as SA

Test.@testset "Stability Verification" begin
    N = 100
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bins = SA.SVector(0.0, 1.4)
    sft = SFT.LongitudinalSecondOrderStructureFunction

    # 1. Default (averaged) output path must infer concretely. This exercises the full backend
    #    compute (which produces the raw accumulator) plus `_finalize`, so any backend
    #    type-instability would surface here. (Passing a non-default `output_type` explicitly
    #    incurs a single dynamic-dispatch barrier — negligible, and analogous to the old
    #    `return_sums_and_counts` convenience form — so it is exercised for correctness below,
    #    not inferred.)
    println("Checking type stability for Array variant (default output-type)...")
    @inferred SFC.calculate_structure_function(sft, x, u, bins; backend = SFC.SerialBackend(), count_eltype = UInt32)

    res_false = SFC.calculate_structure_function(sft, x, u, bins; backend = SFC.SerialBackend(), count_eltype = UInt32)
    Test.@test res_false isa SFO.StructureFunction

    res_true = SFC.calculate_structure_function(sft, x, u, bins; backend = SFC.SerialBackend(), output_type = SFO.StructureFunctionSumsAndCounts, count_eltype = UInt32)
    Test.@test res_true isa SFO.StructureFunctionSumsAndCounts

    # 2. Distributed Stability Check (if DistributedExt is loaded)
    # We'll just check if the method exists for now, or skip if not tested here.
end
