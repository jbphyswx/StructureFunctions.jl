using Test: Test, @inferred
using StructureFunctions: StructureFunctions as SF, StructureFunctionObjects as SFO, StructureFunctionTypes as SFT
using StaticArrays: StaticArrays as SA

Test.@testset "Phase J7 Shorthands Verification" begin
    N = 10
    FT = Float64
    x = ([rand(FT) for _ in 1:N], [rand(FT) for _ in 1:N])
    u = ([rand(FT) for _ in 1:N], [rand(FT) for _ in 1:N])
    bins = SA.SVector{1}(((0.0, 1.4),))

    # 1. Functor call shorthand
    println("Testing functor call shorthand (L2SF)...")
    res1 = SF.L2SF(x, u, bins; verbose=false, show_progress=false)
    Test.@test res1 isa SF.StructureFunction
    Test.@test res1.operator === SF.L2SF

    # 2. Symbol dispatch shorthand
    println("Testing Symbol dispatch shorthand (:L2SF)...")
    res2 = SF.calculate_structure_function(:L2SF, x, u, bins; verbose=false, show_progress=false)
    Test.@test res2 isa SF.StructureFunction
    Test.@test res2.operator === SF.L2SF

    # 3. Val dispatch shorthand (stable)
    println("Testing Val dispatch shorthand (Val(:L2SF))...")
    @inferred SF.calculate_structure_function(Val(:L2SF), x, u, bins; verbose=false, show_progress=false)
    res2v = SF.calculate_structure_function(Val(:L2SF), x, u, bins; verbose=false, show_progress=false)
    Test.@test res2v.operator === SF.L2SF

    # 4. Order/Mode dispatch shorthand
    println("Testing Order/Mode dispatch shorthand (2, :longitudinal)...")
    res3 = SF.calculate_structure_function(2, :longitudinal, x, u, bins; verbose=false, show_progress=false)
    Test.@test res3 isa SF.StructureFunction
    Test.@test res3.operator === SF.L2SF

    # 5. Order/Mode Val dispatch (stable)
    println("Testing Order/Mode Val dispatch (Val(2), Val(:long))...")
    @inferred SF.calculate_structure_function(Val(2), Val(:long), x, u, bins; verbose=false, show_progress=false)
    res3v = SF.calculate_structure_function(Val(2), Val(:long), x, u, bins; verbose=false, show_progress=false)
    Test.@test res3v.operator === SF.L2SF

    # 6. Factory constructor shorthand
    println("Testing Factory constructor shorthand (StructureFunction(2, :long))...")
    res4 = SF.StructureFunction(2, :long, x, u, bins; verbose=false, show_progress=false)
    Test.@test res4 isa SF.StructureFunction
    Test.@test res4.operator === SF.L2SF

    # 7. Factory Val constructor (stable)
    println("Testing Factory Val constructor (StructureFunction(Val(2), Val(:long)))...")
    @inferred SF.StructureFunction(Val(2), Val(:long), x, u, bins; verbose=false, show_progress=false)

    # 8. Check other shorthands presence
    println("Checking other shorthands presence...")
    Test.@test SF.S2SF === SFT.SecondOrderStructureFunction
    Test.@test SF.S3SF === SFT.ThirdOrderStructureFunction
    Test.@test SF.T3SF === SFT.OffDiagonalConsistentThirdOrderStructureFunction
    Test.@test SF.L2T1SF === SFT.DiagonalInconsistentThirdOrderStructureFunction
    Test.@test SF.L1T2SF === SFT.OffDiagonalInconsistentThirdOrderStructureFunction

    # 9. Verify they all return StructureFunction objects
    Test.@test SF.S2SF(x, u, bins; verbose=false, show_progress=false) isa SF.StructureFunction
    Test.@test SF.T3SF(x, u, bins; verbose=false, show_progress=false) isa SF.StructureFunction
end
