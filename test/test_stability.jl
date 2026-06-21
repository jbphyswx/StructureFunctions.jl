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

    # 1. Core positional Val stability check.
    println("Checking type stability for Array variant (positional Val)...")
    @inferred SFC.calculate_structure_function(sft, x, u, bins, Val(false); backend = SFC.SerialBackend(), count_eltype = UInt32)

    res_false = SFC.calculate_structure_function(sft, x, u, bins, Val(false); backend = SFC.SerialBackend(), count_eltype = UInt32)
    Test.@test res_false isa SFO.StructureFunction

    res_true = SFC.calculate_structure_function(sft, x, u, bins, Val(true); backend = SFC.SerialBackend(), count_eltype = UInt32)
    Test.@test res_true isa SFO.StructureFunctionSumsAndCounts

    # 2. Distributed Stability Check (if DistributedExt is loaded)
    # We'll just check if the method exists for now, or skip if not tested here.
end
