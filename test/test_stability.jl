using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO
using StaticArrays: StaticArrays as SA

# Operators and every public entry infer concretely, and honour an explicit count and result type.
Test.@testset "Stability Verification" begin
    N = 10
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bins = SA.SVector(0.0, 1.4)
    vbins = collect(range(-1.0, 1.0; length = 5))
    sft = SFT.LongitudinalSecondOrderStructureFunction
    ser = CB.SerialBackend()

    Test.@test Test.@inferred(sft(SA.SVector(1.0, 0.0), SA.SVector(1.0, 0.0))) == 1.0
    raw = Test.@inferred SFC.calculate_structure_function(sft, x, u, bins, UInt64, SFO.StructureFunctionSumsAndCounts;
                                                          backend = ser)
    Test.@test raw isa SFO.StructureFunctionSumsAndCounts && eltype(raw.counts) === UInt64
    Test.@inferred SFC.calculate_structure_function(sft, x, u, bins; backend = ser)
    joint = Test.@inferred SFC.calculate_structure_function(sft, x, u, bins, vbins, Float64; backend = ser)
    Test.@test eltype(joint.counts) === Float64
    Test.@inferred SFC.calculate_structure_functions_single_pass(x, u, bins, UInt64, SFO.StructureFunction;
                                                                 backend = ser)
    Test.@inferred SFC.calculate_structure_functions_single_pass(x, rand(FT, 2, N, 2), bins; backend = ser)
    Test.@inferred SFC.calculate_structure_functions_single_pass_2d(x, u, bins, vbins; backend = ser)
    Test.@inferred SFC.calculate_structure_function_tensor(Val(2), x, u, bins, UInt64; backend = ser)
end
