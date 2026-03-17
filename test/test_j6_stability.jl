using Test: Test, @inferred
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using StaticArrays: StaticArrays as SA

Test.@testset "Phase J6 Stability Verification" begin
    N = 100
    FT = Float64
    x = ([rand(FT) for _ in 1:N], [rand(FT) for _ in 1:N])
    u = ([rand(FT) for _ in 1:N], [rand(FT) for _ in 1:N])
    bins = SA.SVector{1}(((0.0, 1.4),))
    sft = SFT.LongitudinalSecondOrderStructureFunction

    # 1. Core Positional Stability Check (Tuple variant)
    println("Checking type stability for Tuple variant (positional Val)...")
    @inferred SFC.calculate_structure_function(sft, x, u, bins, Val(false))
    @inferred SFC.calculate_structure_function(sft, x, u, bins, Val(true))
    
    res_false = SFC.calculate_structure_function(sft, x, u, bins, Val(false))
    Test.@test res_false isa SFO.StructureFunction
    
    res_true = SFC.calculate_structure_function(sft, x, u, bins, Val(true))
    Test.@test res_true isa SFO.StructureFunctionSumsAndCounts

    # 2. Array variant Stability Check (positional Val)
    println("Checking type stability for Array variant (positional Val)...")
    x_arr = rand(FT, 2, N)
    u_arr = rand(FT, 2, N)
    @inferred SFC.calculate_structure_function(sft, x_arr, u_arr, bins, Val(false))
    @inferred SFC.calculate_structure_function(sft, x_arr, u_arr, bins, Val(true))

    # 3. Parallel Stability Check (if ParallelCalculationsExt is loaded)
    # We'll just check if the method exists for now, or skip if not tested here.
end
