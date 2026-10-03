using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using StaticArrays: StaticArrays as SA

Test.@testset "Stability Verification" begin
    N = 100
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bins = SA.SVector(0.0, 1.4)
    vbins = collect(range(-1.0, 1.0; length = 5))
    sft = SFT.LongitudinalSecondOrderStructureFunction
    ser = CB.SerialBackend()

    # Every public entry infers concretely, with the defaults and with an explicit count and result type.
    println("Checking type stability for Array variant...")
    Test.@inferred SFC.calculate_structure_function(sft, x, u, bins; backend = ser)
    res = Test.@inferred SFC.calculate_structure_function(sft, x, u, bins, UInt32; backend = ser)
    Test.@test res isa SFO.StructureFunction
    raw = Test.@inferred SFC.calculate_structure_function(sft, x, u, bins, UInt64, SFO.StructureFunctionSumsAndCounts;
                                                     backend = ser)
    Test.@test raw isa SFO.StructureFunctionSumsAndCounts
    Test.@test eltype(raw.counts) === UInt64

    println("Checking type stability for the joint histogram...")
    Test.@inferred SFC.calculate_structure_function(sft, x, u, bins, vbins; backend = ser)
    joint = Test.@inferred SFC.calculate_structure_function(sft, x, u, bins, vbins, Float64; backend = ser)
    Test.@test eltype(joint.counts) === Float64

    # The keyed single-pass result is a single concrete NamedTuple per rank, for point-field and
    # batched (auxiliary-axis) input.
    println("Checking type stability for single-pass (point-field + batched)...")
    Test.@inferred SFC.calculate_structure_functions_single_pass(x, u, bins; backend = ser)
    Test.@inferred SFC.calculate_structure_functions_single_pass(x, u, bins, UInt64, SFO.StructureFunction; backend = ser)
    u_batched = rand(FT, 2, N, 2)
    Test.@inferred SFC.calculate_structure_functions_single_pass(x, u_batched, bins; backend = ser)
    Test.@inferred SFC.calculate_structure_functions_single_pass_2d(x, u, bins, vbins; backend = ser)
    Test.@inferred SFC.calculate_structure_functions_single_pass_2d(x, u, bins, vbins, Float64; backend = ser)

    println("Checking type stability for the moment tensor...")
    Test.@inferred SFC.calculate_structure_function_tensor(Val(2), x, u, bins; backend = ser)
    Test.@inferred SFC.calculate_structure_function_tensor(Val(2), x, u, bins, UInt64; backend = ser)
end
