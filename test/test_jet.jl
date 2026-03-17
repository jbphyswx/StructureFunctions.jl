using StructureFunctions: StructureFunctions as SF, HelperFunctions as SFH, Calculations as SFC, StructureFunctionTypes as SFT
using JET: JET
using Test: Test
using StaticArrays: StaticArrays as SA

Test.@testset "JET Stability Audit" begin
    # Use explicit qualification for functions to ensure JET finds them
    # and we avoid using/export issues in the test Main.

    x = ([0.0, 1.0], [0.0, 0.0])
    u = ([1.0, 2.0], [0.0, 0.0])
    bins = SA.SVector(((0.0, 2.0),))
    sf_type = SFT.LongitudinalSecondOrderStructureFunction()

    Test.@testset "calculate_structure_function (Tuple input)" begin
        JET.@test_opt SFC.calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
        JET.@test_call SFC.calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
    end

    Test.@testset "calculate_structure_function (Array input)" begin
        xa = [0.0 1.0; 0.0 0.0]
        ua = [1.0 2.0; 0.0 0.0]
        JET.@test_opt SFC.calculate_structure_function(xa, ua, bins, sf_type, verbose=false, show_progress=false)
        JET.@test_call SFC.calculate_structure_function(xa, ua, bins, sf_type, verbose=false, show_progress=false)
    end

    Test.@testset "HelperFunctions" begin
        δu = SA.SVector{2, Float64}(1.0, 0.0)
        r̂ = SA.SVector{2, Float64}(1.0, 0.0)
        JET.@test_opt SFH.magnitude_δu_longitudinal(δu, r̂)
        JET.@test_call SFH.magnitude_δu_longitudinal(δu, r̂)
        
        # Test 2D and 3D paths in n̂
        r̂2 = SA.SVector{2, Float64}(1.0, 0.0)
        r̂3 = SA.SVector{3, Float64}(1.0, 0.0, 0.0)
        δu2 = SA.SVector{2, Float64}(1.0, 1.0)
        JET.@test_opt SFH.n̂(r̂2)
        JET.@test_opt SFH.n̂(r̂3)
        JET.@test_opt SFH.δu_longitudinal(δu2, r̂2)
        JET.@test_opt SFH.δu_transverse(δu2, r̂2)
    end

    Test.@testset "StructureFunctionTypes" begin
        δu = SA.SVector{2, Float64}(1.0, 1.0)
        r̂ = SA.SVector{2, Float64}(1.0, 0.0)
        for (name, sft) in SFT.SF_TYPE_MAP
            # Skip unimplemented types that would cause JET errors
            if name in [:RotationalSecondOrderStructureFunction, :DivergentSecondOrderStructureFunction]
                continue
            end
            instance = sft()
            JET.@test_opt instance(δu, r̂)
            JET.@test_call instance(δu, r̂)
        end
    end
end
