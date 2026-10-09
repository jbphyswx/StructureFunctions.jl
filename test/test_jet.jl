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

"""The package and the extensions of it the routes below run, whose code JET analyzes."""
const JET_MODULES = (SF, map((:StructureFunctionsOhMyThreadsExt, :StructureFunctionsAbstractFFTsExt,
                              :StructureFunctionsFFTWExt, :StructureFunctionsNonuniformFFTsExt)) do name
    ext = Base.get_extension(SF, name)
    ext === nothing && error("extension $name is not loaded")
    ext
end...)

# Reports outside `_by_width`'s run-time call of `_at_width`, `_at_width` at an abstract width, and `_release_held`.
_reports_past_boundaries(r) = filter(JET.get_reports(r)) do rep
    rep.vst[end].linfo.def.name in (:_by_width, :_release_held) && return false
    !any(v -> v.linfo.def.name === :_at_width && !isconcretetype(v.linfo.specTypes.parameters[3]), rep.vst)
end

# The closures `_by_width` calls at run time in the optimization analysis `r`.
_width_closures(r) = unique([rep.vst[end].linfo.specTypes.parameters[2] for rep in JET.get_reports(r)
                             if rep.vst[end].linfo.def.name === :_by_width])

"""`analyze(f, types)` of JET (`JET.report_opt` or `JET.report_call`) past the boundaries, and of the code each
run-time width call of it runs at width `W`."""
function _reports_through_width(analyze, f, types, W)
    r = analyze(f, types; target_modules = JET_MODULES)
    reps = _reports_past_boundaries(r)
    opt = analyze === JET.report_opt ? r : JET.report_opt(f, types; target_modules = JET_MODULES)
    for G in _width_closures(opt)
        append!(reps, _reports_past_boundaries(analyze(Tuple{typeof(SFC._at_width), G, Val{W}};
                                                       target_modules = JET_MODULES)))
    end
    return reps
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

"""Routes as the entry a user calls, on `be` with the data `d` of [`_jet_route_data`](@ref): each entry form and
family once, and each argument type that changes the call chain (second axis, shared or per-slice positions, culling,
batch) at least once."""
_jet_routes() = (
    ("point 1D", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xp, d.up, d.bins, SF.StructureFunctionSumsAndCounts;
        backend = be); (r.sums, r.counts)))),
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
    ("aux axes 1D culled", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.xb, d.ub, d.tbins,
        SF.StructureFunctionSumsAndCounts; backend = be, culling = SFC.AlwaysCulling()); (r.sums, r.counts)))),
    ("slice batch joint", ((be, d) -> (s = zeros(d.nb, d.nv, d.T);
        c = zeros(Int, d.nb, d.nv, d.T);
        SFC.calculate_structure_function_2d_batch!(s, c, d.op, d.xb, d.ub, d.bins, d.vbins;
            backend = be); (s, c)))),
    ("slice batch sp2d", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb, d.nv, d.T);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb, d.nv, d.T);
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, d.xb, d.ub, d.bins,
            d.vbins; backend = be); (s, c)))),
    ("gridded lag sweep joint angle", ((be, d) -> (s = zeros(d.nb, d.na); c = zeros(d.nb, d.na);
        SFC.gridded_lag_sweep!(s, c, d.op, d.gu, d.gs, d.gb, d.abins, Val(2), Val(1), Val(0);
            backend = be, second_axis = d.ax); (s, c)))),
    ("gridded single pass transform", ((be, d) -> (s = zeros(SFC.SINGLE_PASS_N, d.nb);
        c = zeros(Int, SFC.SINGLE_PASS_N, d.nb);
        SFC.gridded_sweep!(s, c, SFT.SinglePassInvariants(), d.gu, d.gs, d.gb, Val(2), Val(1), Val(0), SB.FastFourierTransformSpectralBackend();
            backend = be); (s, c)))),
    ("gridded transform batch", ((be, d) -> (s = zeros(d.nb, 2); c = zeros(Int, d.nb, 2);
        SFC.gridded_sweep_batch!(s, c, d.op, d.gub, d.gs, d.gb, Val(2), Val(1), Val(0), SB.FastFourierTransformSpectralBackend();
            backend = be); (s, c)))),
    ("harmonic direct sum", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.hx, d.hu,
        d.nodes, SB.DirectSumSpectralBackend(), SF.StructureFunctionSumsAndCounts; backend = be);
        (r.sums, r.counts)))),
    ("scattered modes NUFFT", ((be, d) -> (r = SFC.calculate_structure_function(d.op, d.sm, d.up,
        d.bins, SFC.NonuniformFFTsSpectralBackend(), SF.StructureFunctionSumsAndCounts; backend = be);
        (r.sums, r.counts)))),
)

"""Routes whose data is one coordinate wide; every other route's is two."""
const _JET_LINE_ROUTES = ("point sorted line",)

"""`(name, count)` of each route in `names` whose analysis on `be`, through the width boundary at its data's width,
reports runtime dispatch past the boundaries."""
function _dispatching_routes(be, names)
    issubset(names, first.(_jet_routes())) || error("not routes: $(setdiff(names, first.(_jet_routes())))")
    d = _jet_route_data()
    found = Tuple{String, Int}[]
    for (name, run) in _jet_routes()
        name in names || continue
        W = name in _JET_LINE_ROUTES ? 1 : 2
        n = length(_reports_through_width(JET.report_opt, run, (typeof(be), typeof(d)), W))
        n == 0 || push!(found, (name, n))
    end
    return found
end

# The default point entry has no runtime dispatch past the boundaries and no possible error, in 2 and 3 dimensions.
Test.@testset "the default point entry, in two and three dimensions" begin
    sf, bins = SFT.LongitudinalSecondOrderStructureFunction, SA.SVector(0.0, 2.0)
    x2, u2 = [0.0 1.0; 0.0 0.0], [1.0 2.0; 0.0 0.0]
    x3, u3 = [0.0 1.0; 0.0 0.0; 0.0 0.0], [1.0 2.0; 0.0 0.0; 0.0 0.0]
    serial = (s, x, u, b) -> SFC.calculate_structure_function(s, x, u, b; backend = CB.SerialBackend())
    default = (s, x, u, b) -> SFC.calculate_structure_function(s, x, u, b)
    Test.@test !isempty(_width_closures(JET.report_opt(serial, typeof.((sf, x2, u2, bins)); target_modules = JET_MODULES)))
    Test.@test isempty(_reports_through_width(JET.report_opt, serial, typeof.((sf, x2, u2, bins)), 2))
    Test.@test isempty(_reports_through_width(JET.report_opt, serial, typeof.((sf, x3, u3, bins)), 3))
    Test.@test isempty(_reports_through_width(JET.report_call, default, typeof.((sf, x2, u2, bins)), 2))
    Test.@test isempty(_reports_through_width(JET.report_call, default, typeof.((sf, x3, u3, bins)), 3))
end

# Every route has no runtime dispatch past the boundaries on the serial backend, and the threaded subset on threads.
Test.@testset "every route has no runtime dispatch outside the width and workspace boundaries" begin
    Test.@test isempty(_dispatching_routes(CB.SerialBackend(), first.(_jet_routes())))
    Test.@test isempty(_dispatching_routes(CB.ThreadedBackend(),
        ("point 1D", "point joint angle", "single-pass 1D", "single-pass 2D", "slice batch joint",
         "gridded transform batch", "harmonic direct sum")))
end

# The kernels a distributed worker runs, at the argument types the Distributed extension passes, do not dispatch.
Test.@testset "the per-worker reduction kernels have no runtime dispatch" begin
    d = _jet_route_data()
    xv, uv = SFC._prepared_tuples(SFH.FlatGeometry{2}(), d.xp, d.up)
    JET.@test_opt target_modules = JET_MODULES SFC._partial_sums_counts(CB.SerialBackend(), d.op, xv, uv, d.bins,
        (1, 2), UInt32; geometry = SFH.FlatGeometry{2}(), culling = SFC.AutoCulling(), weights = SFC.NoWeights())
    JET.@test_opt target_modules = JET_MODULES SFC._partial_2d_sums_counts(CB.SerialBackend(), d.op, xv, uv, d.bins,
        d.vbins, (1, 2), UInt32; geometry = SFH.FlatGeometry{2}(), culling = SFC.AutoCulling(),
        weights = SFC.NoWeights(), second_axis = SFC.InvariantValueAxis())
end

# Every pairwise operator evaluates one pair with no runtime dispatch and no possible error.
Test.@testset "every pairwise operator evaluates a pair without runtime dispatch or possible error" begin
    args = (SA.SVector{2, Float64}, SA.SVector{2, Float64})
    ops = unique(filter(op -> op isa SFT.AbstractPairwiseStructureFunctionType, collect(values(SFT.SF_TYPE_MAP))))
    Test.@test isempty(filter(op -> !isempty(JET.get_reports(JET.report_opt(op, args))), ops))
    Test.@test isempty(filter(op -> !isempty(JET.get_reports(JET.report_call(op, args))), ops))
end

# The three-dimensional transverse direction and the vector transverse increment have no runtime dispatch.
Test.@testset "the helpers no operator calls have no runtime dispatch" begin
    JET.@test_opt SFH.n̂(SA.SVector(1.0, 0.0, 0.0))
    JET.@test_opt SFH.δu_transverse(SA.SVector(1.0, 1.0), SA.SVector(1.0, 0.0))
end
