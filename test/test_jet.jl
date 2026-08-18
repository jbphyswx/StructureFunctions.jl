using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, HelperFunctions as SFH,
    Calculations as SFC, StructureFunctionTypes as SFT
using JET: JET
using Test: Test
using StaticArrays: StaticArrays as SA

Test.@testset "JET Stability Audit" begin
    # Use explicit qualification for functions to ensure JET finds them
    # and we avoid using/export issues in the test Main.

    x = [0.0 1.0; 0.0 0.0]
    u = [1.0 2.0; 0.0 0.0]
    bins = SA.SVector(0.0, 2.0)
    sf_type = SFT.LongitudinalSecondOrderStructureFunction

    Test.@testset "calculate_structure_function (Array input)" begin
        # Audit type stability of the compute kernel on a concrete backend. The default (averaged)
        # path exercises the full backend compute (which produces the raw
        # `StructureFunctionSumsAndCounts`) plus `_finalize`, so any kernel instability surfaces
        # here. We audit `SerialBackend` (concrete) rather than `AutoBackend` because AutoBackend's
        # runtime backend selection is an intended runtime branch, not a fixable instability. We
        # only audit the SF module to ignore internal Base.Threads dispatches. (Passing a
        # non-default `output_type` explicitly incurs a single by-design dynamic-dispatch barrier
        # in `_finalize`; it is checked for error-freedom via @test_call below, not @test_opt.)
        JET.@test_opt target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins;
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false,
        )
        # Error-freedom of the default and explicit-output_type convenience entries.
        # Only analyze StructureFunctions module code, not external packages like ProgressMeter
        # which do compile-time checks for Main.IJulia that may not be present.
        # See: https://github.com/timholy/ProgressMeter.jl/issues/348
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins;
            verbose = false,
            show_progress = false,
        )
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins;
            output_type = SF.StructureFunctionSumsAndCounts,
            verbose = false,
            show_progress = false,
        )
    end
    Test.@testset "calculate_structure_function (3D Array input)" begin
        xa = [0.0 1.0; 0.0 0.0; 0.0 0.0]
        ua = [1.0 2.0; 0.0 0.0; 0.0 0.0]
        # Audit type stability of the compute kernel on a concrete backend (see note above).
        JET.@test_opt target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            xa,
            ua,
            bins;
            backend = CB.SerialBackend(),
            verbose = false,
            show_progress = false,
        )
        # Error-freedom of the default and explicit-output_type convenience entries.
        # Only analyze StructureFunctions module code, not external packages like ProgressMeter
        # which do compile-time checks for Main.IJulia that may not be present.
        # See: https://github.com/timholy/ProgressMeter.jl/issues/348
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            xa,
            ua,
            bins;
            verbose = false,
            show_progress = false,
        )
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            xa,
            ua,
            bins;
            output_type = SF.StructureFunctionSumsAndCounts,
            verbose = false,
            show_progress = false,
        )
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
            instance = sft()
            instance isa SFT.AbstractPairwiseStructureFunctionType || continue
            JET.@test_opt instance(δu, r̂)
            JET.@test_call instance(δu, r̂)
        end
    end
end
