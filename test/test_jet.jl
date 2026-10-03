using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, HelperFunctions as SFH,
    Calculations as SFC, StructureFunctionTypes as SFT
using JET: JET
using Test: Test
using StaticArrays: StaticArrays as SA
using Random: Random

# Reports outside the width boundary `_at_width(g, Val(D))`, which an entry reaches with an abstract `Val` only for
# widths past `_by_width`'s explicit branches, and which dispatches at run time to the concrete width.
_explicit_width_reports(r) = filter(JET.get_reports(r)) do rep
    !any(v -> v.linfo.def.name === :_at_width && !isconcretetype(v.linfo.specTypes.parameters[3]), rep.vst)
end

Test.@testset "JET Stability Audit" begin
    # Use explicit qualification for functions to ensure JET finds them
    # and we avoid using/export issues in the test Main.

    x = [0.0 1.0; 0.0 0.0]
    u = [1.0 2.0; 0.0 0.0]
    bins = SA.SVector(0.0, 2.0)
    sf_type = SFT.LongitudinalSecondOrderStructureFunction

    Test.@testset "calculate_structure_function (Array input)" begin
        Test.@test isempty(_explicit_width_reports(JET.report_opt(
            (s, x, u, b) -> SFC.calculate_structure_function(s, x, u, b; backend = CB.SerialBackend()),
            (typeof(sf_type), typeof(x), typeof(u), typeof(bins)); target_modules = (SF,))))
        # Error-freedom of the default and explicit result-type convenience entries.
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins,
        )
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            x,
            u,
            bins,
            SF.StructureFunctionSumsAndCounts,
        )
    end
    Test.@testset "calculate_structure_function (3D Array input)" begin
        xa = [0.0 1.0; 0.0 0.0; 0.0 0.0]
        ua = [1.0 2.0; 0.0 0.0; 0.0 0.0]
        Test.@test isempty(_explicit_width_reports(JET.report_opt(
            (s, x, u, b) -> SFC.calculate_structure_function(s, x, u, b; backend = CB.SerialBackend()),
            (typeof(sf_type), typeof(xa), typeof(ua), typeof(bins)); target_modules = (SF,))))
        # Error-freedom of the default and explicit result-type convenience entries.
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            xa,
            ua,
            bins,
        )
        JET.@test_call target_modules = (SF,) SFC.calculate_structure_function(
            sf_type,
            xa,
            ua,
            bins,
            SF.StructureFunctionSumsAndCounts,
        )
    end

    Test.@testset "the per-worker reduction kernels carry no runtime dispatch" begin
        # These are the hot loops every backend calls with concrete arguments, so zero is the
        # right assertion — unlike a whole entry, which has by-design dispatch barriers at
        # `_finalize` and at backend selection and whose count would only be a number to pin.
        # This is the check for the defect class that has cost the most here: a captured and
        # reassigned variable boxes to `Any`, which is merely slow on the host and a compile
        # failure on a device.
        Random.seed!(3)
        Np = 60
        xp = rand(2, Np)
        up = randn(2, Np)
        xv = (collect(view(xp, 1, :)), collect(view(xp, 2, :)))
        uv = (collect(view(up, 1, :)), collect(view(up, 2, :)))
        dbins = collect(range(0.0, 1.0; length = 7))
        vbins = collect(range(-3.0, 3.0; length = 6))
        op = SFT.L2SFType()

        JET.@test_opt target_modules = (SF,) SFC._partial_sums_counts(
            CB.SerialBackend(), op, xv, uv, dbins, (1, 2), UInt32)
        JET.@test_opt target_modules = (SF,) SFC._partial_2d_sums_counts(
            CB.SerialBackend(), op, xv, uv, dbins, vbins, (1, 2), UInt32)
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
