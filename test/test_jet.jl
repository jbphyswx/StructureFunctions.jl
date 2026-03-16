using StructureFunctions
using StructureFunctions.HelperFunctions
using StructureFunctions.Calculations
using StructureFunctions.StructureFunctionTypes
using JET
using Test
using StaticArrays

@testset "JET Stability Audit" begin
    # Use explicit qualification for functions to ensure JET finds them
    # and we avoid using/export issues in the test Main.
    HF = StructureFunctions.HelperFunctions
    CALC = StructureFunctions.Calculations
    SFT = StructureFunctions.StructureFunctionTypes

    x = ([0.0, 1.0], [0.0, 0.0])
    u = ([1.0, 2.0], [0.0, 0.0])
    bins = SVector(((0.0, 2.0),))
    sf_type = SFT.LongitudinalSecondOrderStructureFunction()

    @testset "calculate_structure_function (Tuple input)" begin
        @test_opt CALC.calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
        @test_call CALC.calculate_structure_function(x, u, bins, sf_type, verbose=false, show_progress=false)
    end

    @testset "calculate_structure_function (Array input)" begin
        xa = [0.0 1.0; 0.0 0.0]
        ua = [1.0 2.0; 0.0 0.0]
        @test_opt CALC.calculate_structure_function(xa, ua, bins, sf_type, verbose=false, show_progress=false)
        @test_call CALC.calculate_structure_function(xa, ua, bins, sf_type, verbose=false, show_progress=false)
    end

    @testset "HelperFunctions" begin
        δu = SVector{2, Float64}(1.0, 0.0)
        r̂ = SVector{2, Float64}(1.0, 0.0)
        @test_opt HF.magnitude_δu_longitudinal(δu, r̂)
        @test_call HF.magnitude_δu_longitudinal(δu, r̂)
        
        # Test 2D and 3D paths in n̂
        r̂2 = SVector{2, Float64}(1.0, 0.0)
        r̂3 = SVector{3, Float64}(1.0, 0.0, 0.0)
        δu2 = SVector{2, Float64}(1.0, 1.0)
        @test_opt HF.n̂(r̂2)
        @test_opt HF.n̂(r̂3)
        @test_opt HF.δu_longitudinal(δu2, r̂2)
        @test_opt HF.δu_transverse(δu2, r̂2)
    end

    @testset "StructureFunctionTypes" begin
        δu = SVector{2, Float64}(1.0, 1.0)
        r̂ = SVector{2, Float64}(1.0, 0.0)
        for (name, sft) in SFT.SF_TYPE_MAP
            # Skip unimplemented types that would cause JET errors
            if name in [:RotationalSecondOrderStructureFunction, :DivergentSecondOrderStructureFunction]
                continue
            end
            instance = sft()
            @test_opt instance(δu, r̂)
            @test_call instance(δu, r̂)
        end
    end
end
