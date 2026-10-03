using ComputationalBackends: ComputationalBackends as CB
using StructureFunctions: StructureFunctions as SF, HelperFunctions as SFH,
    Calculations as SFC, StructureFunctionTypes as SFT
using JET: JET
using Test: Test
using StaticArrays: StaticArrays as SA
using Random: Random
using SpectralBackends: SpectralBackends as SB
using OhMyThreads: OhMyThreads
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs

# Reports outside the two dynamic boundaries: `_by_width`'s call of `_at_width(g, Val(D))`, made for widths past its
# explicit branches with a width known only at run time, and what `_at_width` analyses at that abstract width; and
# `_release_held`, which releases a value a transform workspace kept, of a type known only at run time.
_reports_past_boundaries(r) = filter(JET.get_reports(r)) do rep
    rep.vst[end].linfo.def.name in (:_by_width, :_release_held) && return false
    !any(v -> v.linfo.def.name === :_at_width && !isconcretetype(v.linfo.specTypes.parameters[3]), rep.vst)
end

"""Data of every route, passed to it as an argument so JET analyzes concrete types."""
function _jet_route_data()
    Random.seed!(11)
    N, T, nb, nv = 40, 3, 6, 5
    abins = collect(range(prevfloat(0.0), π; length = 4))
    xp = rand(2, N)
    return (;
        N, T, nb, nv, na = length(abins) - 1,
        op = SFT.L2SFType(),
        xp, up = randn(2, N),
        xb = rand(2, N, T), ub = randn(2, N, T),
        x1 = reshape(sort(rand(N)), 1, N), u1 = randn(1, N),
        bins = collect(range(0.0, 1.0; length = nb + 1)),
        vbins = collect(range(-3.0, 3.0; length = nv + 1)),
        abins,
        ax = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0)),
        tbins = collect(range(0.0, 0.25; length = nb + 1)),
        gs = SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true)),
        gu = reshape(Float64[sin(d + 3i + 7j) for d in 1:2, i in 1:8, j in 1:8], 2, :),
        gub = reshape(Float64[sin(d + 3i + 7j + 2t) for d in 1:2, i in 1:8, j in 1:8, t in 1:2], 2, 64, 2),
        gb = collect(range(0.0, 0.5; length = nb + 1)),
        hx = permutedims(hcat([2π * (i * 0.6180339887498949 % 1) for i in 1:N],
            π / 2 .- acos.(clamp.(range(-0.95, 0.95; length = N), -1, 1)))),
        hu = Float64[sin(d + 2i) for d in 1:2, i in 1:N],
        nodes = SF.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), 16),
        sm = SFC.ScatteredModesSchedule(xp, 0.5, (24, 16); taper = SF.GaussianTaper(0.03)),
        fields = SF.MultiFields.Fields(vectors = (randn(2, N),)),
    )
end

"""Every capability-matrix route as the entry a user calls, with backend `be` and the data `d` of
[`_jet_route_data`](@ref)."""
_jet_routes() = (

    ("point 1D", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xp, d.up, d.bins, SF.StructureFunctionSumsAndCounts;
        backend = be); (r.sums, r.counts)))),
    ("point 1D in-place", ((be, d) -> (s = zeros(d.nb); c = zeros(Int, d.nb);
        SFC.calculate_structure_function!(s, c, d.op, d.xp, d.up, d.bins; backend = be);
            (s, c)))),
    ("point joint value", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xp, d.up,
        d.bins, d.vbins; backend = be); (r.sums, r.counts)))),
    ("point joint angle", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xp, d.up,
        d.bins, d.abins; backend = be, second_axis = d.ax);
        (r.sums, r.counts)))),
    ("point sorted line", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.x1, d.u1,
        d.bins, SF.StructureFunctionSumsAndCounts; backend = be); (r.sums, r.counts)))),
    ("point multi-field", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xp,
        d.fields, d.bins, SF.StructureFunctionSumsAndCounts; backend = be);
        (r.sums, r.counts)))),
    ("moment tensor", ((be, d) -> (r = SFC.calculate_structure_function_tensor(Val(2), d.xp, d.up,
        d.bins, SF.StructureFunctionObjects.StructureFunctionTensorSumsAndCounts; backend = be); (r.sums, r.counts)))),
    ("moment tensor joint", ((be, d) -> (r = SFC.calculate_structure_function_tensor(Val(2), d.xp,
        d.up, d.bins, d.abins; second_axis = d.ax, backend = be); (r.sums, r.counts)))),
    ("single-pass 1D", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb);
        SFC.calculate_structure_functions_single_pass!(s, c, d.xp, d.up, d.bins; backend = be);
        (s, c)))),
    ("single-pass 2D", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb, d.nv);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb, d.nv);
        SFC.calculate_structure_functions_single_pass_2d!(s, c, d.xp, d.up, d.bins, d.vbins;
            backend = be); (s, c)))),
    ("aux axes 1D", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xb, d.ub, d.bins, SF.StructureFunctionSumsAndCounts;
        backend = be); (r.sums, r.counts)))),
    ("aux axes joint", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xb, d.ub, d.bins,
        d.vbins; backend = be); (r.sums, r.counts)))),
    ("aux axes joint angle", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xb, d.ub, d.bins,
        d.abins; backend = be, second_axis = d.ax); (r.sums, r.counts)))),
    ("aux axes joint angle shared", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xp, d.ub,
        d.bins, d.abins; backend = be, second_axis = d.ax); (r.sums, r.counts)))),
    ("aux axes 1D culled", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xb, d.ub, d.tbins,
        SF.StructureFunctionSumsAndCounts; backend = be, culling = SFC.AlwaysCulling()); (r.sums, r.counts)))),
    ("aux axes 1D culled shared", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xp, d.ub,
        d.tbins, SF.StructureFunctionSumsAndCounts; backend = be, culling = SFC.AlwaysCulling()); (r.sums, r.counts)))),
    ("slice batch 1D", ((be, d) -> (s = zeros(d.nb, d.T); c = zeros(Int, d.nb, d.T);
        SFC.calculate_structure_function_batch!(s, c, d.op, d.xb, d.ub, d.bins; backend = be);
        (s, c)))),
    ("slice batch joint", ((be, d) -> (s = zeros(d.nb, d.nv, d.T);
        c = zeros(Int, d.nb, d.nv, d.T);
        SFC.calculate_structure_function_2d_batch!(s, c, d.op, d.xb, d.ub, d.bins, d.vbins;
            backend = be); (s, c)))),
    ("slice batch joint angle", ((be, d) -> (s = zeros(d.nb, d.na, d.T);
        c = zeros(Int, d.nb, d.na, d.T);
        SFC.calculate_structure_function_2d_batch!(s, c, d.op, d.xb, d.ub, d.bins, d.abins;
            backend = be, second_axis = d.ax); (s, c)))),
    ("slice batch sp1d", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb, d.T);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb, d.T);
        SFC.calculate_structure_functions_single_pass_batch!(s, c, d.xb, d.ub, d.bins;
            backend = be); (s, c)))),
    ("slice batch sp1d culled shared", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb, d.T);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb, d.T);
        SFC.calculate_structure_functions_single_pass_batch!(s, c, d.xp, d.ub, d.tbins;
            backend = be, culling = SFC.AlwaysCulling()); (s, c)))),
    ("slice batch sp2d", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb, d.nv, d.T);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb, d.nv, d.T);
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, d.xb, d.ub, d.bins,
            d.vbins; backend = be); (s, c)))),
    ("gridded lag sweep", ((be, d) -> (s = zeros(d.nb); c = zeros(Int, d.nb);
        SFC.gridded_lag_sweep!(s, c, d.op, d.gu, d.gs, d.gb, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded lag sweep joint value", ((be, d) -> (s = zeros(d.nb, d.nv); c = zeros(d.nb, d.nv);
        SFC.gridded_lag_sweep!(s, c, d.op, d.gu, d.gs, d.gb, d.vbins, Val(2), Val(1), Val(0);
            backend = be, second_axis = SFC.InvariantValueAxis()); (s, c)))),
    ("gridded lag sweep joint angle", ((be, d) -> (s = zeros(d.nb, d.na); c = zeros(d.nb, d.na);
        SFC.gridded_lag_sweep!(s, c, d.op, d.gu, d.gs, d.gb, d.abins, Val(2), Val(1), Val(0);
            backend = be, second_axis = d.ax); (s, c)))),
    ("gridded lag sweep batch", ((be, d) -> (s = zeros(d.nb, 2); c = zeros(Int, d.nb, 2);
        SFC.gridded_lag_sweep_batch!(s, c, d.op, d.gub, d.gs, d.gb, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded lag sweep batch joint value", ((be, d) -> (s = zeros(d.nb, d.nv, 2); c = zeros(d.nb, d.nv, 2);
        SFC.gridded_lag_sweep_batch!(s, c, d.op, d.gub, d.gs, d.gb, d.vbins, Val(2), Val(1), Val(0);
            backend = be, second_axis = SFC.InvariantValueAxis()); (s, c)))),
    ("gridded transform", ((be, d) -> (s = zeros(d.nb); c = zeros(Int, d.nb);
        SFC.gridded_sweep!(s, c, d.op, d.gu, d.gs, d.gb, Val(2), Val(1), Val(0), SB.FastFourierTransformSpectralBackend();
            backend = be); (s, c)))),
    ("gridded single pass", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb); c = zeros(Int, SFC.SINGLE_PASS_N, d.nb);
        SFC.gridded_lag_sweep!(s, c, SFT.SinglePassInvariants(), d.gu, d.gs, d.gb, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded single pass transform", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb);
        SFC.gridded_sweep!(s, c, SFT.SinglePassInvariants(), d.gu, d.gs, d.gb, Val(2), Val(1), Val(0), SB.FastFourierTransformSpectralBackend();
            backend = be); (s, c)))),
    ("gridded transform batch", ((be, d) -> (s = zeros(d.nb, 2); c = zeros(Int, d.nb, 2);
        SFC.gridded_sweep_batch!(s, c, d.op, d.gub, d.gs, d.gb, Val(2), Val(1), Val(0), SB.FastFourierTransformSpectralBackend();
            backend = be); (s, c)))),
    ("gridded tensor", ((be, d) -> (s = zeros(2, 2, d.nb); c = zeros(Int, d.nb);
        SFC.gridded_tensor_sweep!(s, c, Val(2), d.gu, d.gs, d.gb, Val(2), SB.FastFourierTransformSpectralBackend();
            backend = be); (s, c)))),
    ("gridded tensor joint angle", ((be, d) -> (s = zeros(2, 2, d.nb, d.na); c = zeros(d.nb, d.na);
        SFC.gridded_tensor_sweep!(s, c, Val(2), d.gu, d.gs, d.gb, d.abins, Val(2), SB.FastFourierTransformSpectralBackend();
            backend = be, second_axis = d.ax); (s, c)))),
    ("harmonic direct sum", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.hx, d.hu,
        d.nodes, SB.DirectSumSpectralBackend(), SF.StructureFunctionSumsAndCounts; backend = be);
        (r.sums, r.counts)))),
    ("scattered modes NUFFT", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.sm, d.up,
        d.bins, SFC.NonuniformFFTsSpectralBackend(), SF.StructureFunctionSumsAndCounts; backend = be);
        (r.sums, r.counts)))),
)

Test.@testset "JET Stability Audit" begin
    # Use explicit qualification for functions to ensure JET finds them
    # and we avoid using/export issues in the test Main.

    x = [0.0 1.0; 0.0 0.0]
    u = [1.0 2.0; 0.0 0.0]
    bins = SA.SVector(0.0, 2.0)
    sf_type = SFT.LongitudinalSecondOrderStructureFunction

    Test.@testset "calculate_structure_function (Array input)" begin
        Test.@test isempty(_reports_past_boundaries(JET.report_opt(
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
        Test.@test isempty(_reports_past_boundaries(JET.report_opt(
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

    Test.@testset "every route has no runtime dispatch outside the width and workspace boundaries" begin
        d = _jet_route_data()
        for (name, run) in _jet_routes(), be in (CB.SerialBackend(), CB.ThreadedBackend())
            reps = _reports_past_boundaries(JET.report_opt(run, (typeof(be), typeof(d)); target_modules = (SF,)))
            Test.@test (name, nameof(typeof(be)), length(reps)) == (name, nameof(typeof(be)), 0)
        end
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
