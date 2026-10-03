using ComputationalBackends: SerialBackend
using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using AbstractFFTs: AbstractFFTs
using Random: Random

const QUADRATIC = (SFT.S2SFType(), SFT.L2SFType(), SFT.T2SFType(), SFT.T2ComponentSFType())

function _run(sf, u, dims, spacing, periodic, bins, D, backend)
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    sched = SFC.UniformLagSchedule(dims, spacing, periodic)
    if backend === nothing
        SFC.gridded_lag_sweep!(s, c, sf, u, sched, bins, Val(D); backend=SerialBackend())
    else
        SFC.gridded_sweep!(s, c, sf, u, sched, bins, Val(D), backend; backend=SerialBackend())
    end
    return s, c
end

# The transform and the lag sweep are two algorithms for one definition, so they must agree to
# round-off on the same data. The sweep is the reference because it is checked against the
# unstructured pair loop and against brute force in test_gridded.jl.
Test.@testset "the transform agrees with the lag sweep" begin
    T = Float64
    for (dims, spacing, periodic) in (((9, 6), (0.1, 0.2), (false, false)),
                                      ((8, 8), (0.25, 0.25), (true, true)),
                                      ((10, 7), (0.15, 0.15), (true, false)),
                                      ((12,), (0.3,), (false,)),
                                      ((6, 5, 4), (0.2, 0.2, 0.3), (false, false, false)))
        Dg = length(dims)
        Random.seed!(6100 + prod(dims) + Dg)
        u = randn(T, Dg, dims...)
        r_max = 0.7 * sum(d -> spacing[d] * dims[d], 1:Dg)
        bins = collect(range(0.0, r_max; length = 9))
        for sf in QUADRATIC
            Dg == 1 && sf === SFT.T2ComponentSFType() && continue   # no transverse subspace on a line
            ref_s, ref_c = _run(sf, u, dims, spacing, periodic, bins, Dg, nothing)
            got_s, got_c = _run(sf, u, dims, spacing, periodic, bins, Dg,
                                SB.FastFourierTransformSpectralBackend())
            Test.@test got_c == ref_c
            Test.@test isapprox(got_s, ref_s; rtol = 1e-8, atol = 1e-10)
            Test.@test sum(got_c) > 0
        end
    end
end

Test.@testset "the transform counts every pair once" begin
    T = Float64
    for (dims, periodic) in (((6, 4), (false, false)), ((6, 4), (true, true)), ((5, 5), (true, false)))
        Dg = length(dims)
        spacing = ntuple(_ -> T(0.5), Dg)
        N = prod(dims)
        Random.seed!(6200 + N)
        u = randn(T, Dg, dims...)
        bins = collect(range(0.0, 1e3; length = 4))
        _, c = _run(SFT.S2SFType(), u, dims, spacing, periodic, bins, Dg,
                    SB.FastFourierTransformSpectralBackend())
        Test.@test sum(c) == N * (N - 1) ÷ 2
    end
end

Test.@testset "the transform refuses what it cannot express" begin
    T = Float64
    dims = (6, 5)
    spacing = (0.2, 0.2)
    periodic = (false, false)
    u = randn(T, 2, dims...)
    bins = collect(range(0.0, 1.0; length = 5))
    # an odd power of a norm is not a polynomial in the increment, so no moment tensor produces it
    sf = SFT.FullVectorStructureFunctionType{3}()
    Test.@test_throws ArgumentError _run(sf, u, dims, spacing, periodic, bins, 2,
                                         SB.FastFourierTransformSpectralBackend())
    # ...and the lag sweep still does it
    _, c = _run(sf, u, dims, spacing, periodic, bins, 2, nothing)
    Test.@test sum(c) > 0
    # the third-order operators are polynomials and the transform computes them
    for sf3 in (SFT.L3SFType(), SFT.S3SFType())
        ref_s, ref_c = _run(sf3, u, dims, spacing, periodic, bins, 2, nothing)
        got_s, got_c = _run(sf3, u, dims, spacing, periodic, bins, 2, SB.FastFourierTransformSpectralBackend())
        Test.@test got_c == ref_c
        Test.@test isapprox(got_s, ref_s; rtol = 1e-9, atol = 1e-10)
    end
end

Test.@testset "the algorithm tags select as documented" begin
    T = Float64
    dims = (8, 6)
    spacing = (0.2, 0.25)
    periodic = (false, false)
    Random.seed!(6300)
    u = randn(T, 2, dims...)
    bins = collect(range(0.0, 1.2; length = 7))
    ref_s, ref_c = _run(SFT.L2SFType(), u, dims, spacing, periodic, bins, 2, nothing)

    # Both calls use the same serial lag reduction and preserve its summation order.
    ds_s, ds_c = _run(SFT.L2SFType(), u, dims, spacing, periodic, bins, 2,
                      SB.DirectSumSpectralBackend())
    Test.@test ds_c == ref_c
    Test.@test ds_s == ref_s

    # auto picks one of the two exact algorithms, so it must agree with both
    au_s, au_c = _run(SFT.L2SFType(), u, dims, spacing, periodic, bins, 2, SB.AutoSpectralBackend())
    Test.@test au_c == ref_c
    Test.@test isapprox(au_s, ref_s; rtol = 1e-8, atol = 1e-10)

    # A nonpolynomial operator selects the serial direct reduction.
    l3_s, l3_c = _run(SFT.FullVectorStructureFunctionType{3}(), u, dims, spacing, periodic, bins, 2,
                      SB.AutoSpectralBackend())
    ref3_s, ref3_c = _run(SFT.FullVectorStructureFunctionType{3}(), u, dims, spacing, periodic, bins,
                          2, nothing)
    Test.@test l3_c == ref3_c
    Test.@test l3_s == ref3_s
    # auto on a third-order polynomial picks one of two exact algorithms
    a3_s, a3_c = _run(SFT.L3SFType(), u, dims, spacing, periodic, bins, 2, SB.AutoSpectralBackend())
    r3_s, r3_c = _run(SFT.L3SFType(), u, dims, spacing, periodic, bins, 2, nothing)
    Test.@test a3_c == r3_c
    Test.@test isapprox(a3_s, r3_s; rtol = 1e-9, atol = 1e-10)
end

Test.@testset "the grid entry takes the tag positionally" begin
    geo = FG.Geometry.CartesianGeometry()
    nx, ny = 9, 7
    grid = FG.Grids.StructuredGrid(geo, range(0.0, step = 0.2, length = nx),
                                   range(0.0, step = 0.2, length = ny))
    Random.seed!(6400)
    u = randn(2, nx, ny)
    bins = collect(range(0.0, 1.4; length = 8))
    swept = SFC.calculate_structure_function(
        SFT.L2SFType(), grid, u, bins, SF.StructureFunctionSumsAndCounts)
    transformed = SFC.calculate_structure_function(
        SFT.L2SFType(), grid, u, bins, SB.FastFourierTransformSpectralBackend(), UInt32,
        SF.StructureFunctionSumsAndCounts)
    Test.@test transformed.counts == swept.counts
    Test.@test isapprox(transformed.sums, swept.sums; rtol = 1e-8, atol = 1e-10)
    Test.@test sum(swept.counts) > 0
end

Test.@testset "a transform workspace keeps its buffers and changes no answer" begin
    Random.seed!(6500)
    fft = SB.FastFourierTransformSpectralBackend()
    backends = (CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
    bins = collect(range(0.0, 1.2; length = 7)) .+ 0.0137
    nb = length(bins) - 1
    lats = collect(range(-1.0, 1.0; length = 7))
    for (s, dims) in ((SFC.UniformLagSchedule((12, 10), (0.1, 0.12), (true, false)), (12, 10)),
                      (SFC.ZonalLagSchedule(lats, 12, 2π / 12, 1.0, true), (12, 7)))
        u = randn(2, dims...)
        ref_s, ref_c = zeros(nb), zeros(Int, nb)
        SFC.gridded_sweep!(ref_s, ref_c, SFT.L3SFType(), u, s, bins, Val(2), fft; backend = CB.SerialBackend())
        Test.@test sum(ref_c) > 0
        ws = SFC.TransformWorkspace()
        for backend in backends, _ in 1:2
            gs, gc = zeros(nb), zeros(Int, nb)
            SFC.gridded_sweep!(gs, gc, SFT.L3SFType(), u, s, bins, Val(2), fft; backend, workspace = ws)
            Test.@test gc == ref_c
            Test.@test isapprox(gs, ref_s; rtol = 1e-10, atol = 1e-12)
        end
        # a call of the same sizes writes into the spectra the workspace keeps, and returns the scratch it borrowed
        spectra = last(ws.kept[:spectra])
        SFC.gridded_sweep!(zeros(nb), zeros(Int, nb), SFT.L3SFType(), randn(2, dims...), s, bins, Val(2), fft;
                           workspace = ws)
        Test.@test last(ws.kept[:spectra]) === spectra
        Test.@test !isempty(ws.pool)
        ub = randn(2, dims..., 3)
        rb, rcb = zeros(nb, 3), zeros(Int, nb, 3)
        SFC.gridded_sweep_batch!(rb, rcb, SFT.L3SFType(), ub, s, bins, Val(2), fft; backend = CB.SerialBackend())
        for backend in backends, _ in 1:2
            gb, gcb = zeros(nb, 3), zeros(Int, nb, 3)
            SFC.gridded_sweep_batch!(gb, gcb, SFT.L3SFType(), ub, s, bins, Val(2), fft; backend, workspace = ws)
            Test.@test gcb == rcb
            Test.@test isapprox(gb, rb; rtol = 1e-10, atol = 1e-12)
        end
        # every kind of scratch a slice borrows comes back for the next slice and the next call
        pooled() = Dict(first(first(e)) => last(e) for e in ws.pool)
        SFC.gridded_sweep_batch!(zeros(nb, 3), zeros(Int, nb, 3), SFT.L3SFType(), ub, s, bins, Val(2), fft;
                                 backend = CB.SerialBackend(), workspace = ws)
        before = pooled()
        SFC.gridded_sweep_batch!(zeros(nb, 3), zeros(Int, nb, 3), SFT.L3SFType(), ub, s, bins, Val(2), fft;
                                 backend = CB.SerialBackend(), workspace = ws)
        Test.@test !isempty(before) && all(k -> pooled()[k] === before[k], keys(before))
    end
    # another grid's sizes rebuild what the workspace keeps; the tensor and the grid entry take it too
    ws = SFC.TransformWorkspace()
    small = SFC.UniformLagSchedule((12, 10), (0.1, 0.12), (true, false))
    large = SFC.UniformLagSchedule((16, 10), (0.1, 0.12), (true, false))
    SFC.gridded_sweep!(zeros(nb), zeros(Int, nb), SFT.L2SFType(), randn(2, 12, 10), small, bins, Val(2), fft;
                       workspace = ws)
    kept_small = last(ws.kept[:spectra])
    rl, rcl = zeros(nb), zeros(Int, nb)
    ul = randn(2, 16, 10)
    SFC.gridded_sweep!(rl, rcl, SFT.L2SFType(), ul, large, bins, Val(2), fft)
    gl, gcl = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(gl, gcl, SFT.L2SFType(), ul, large, bins, Val(2), fft; workspace = ws)
    Test.@test size(last(ws.kept[:spectra])) != size(kept_small)
    Test.@test gcl == rcl
    Test.@test isapprox(gl, rl; rtol = 1e-10, atol = 1e-12)
    ut = randn(2, 12, 10)
    rt, rct = zeros(2, 2, nb), zeros(Int, nb)
    SFC.gridded_tensor_sweep!(rt, rct, Val(2), reshape(ut, 2, :), small, bins, Val(2), fft)
    gt, gct = zeros(2, 2, nb), zeros(Int, nb)
    SFC.gridded_tensor_sweep!(gt, gct, Val(2), reshape(ut, 2, :), small, bins, Val(2), fft; workspace = ws)
    Test.@test gct == rct
    Test.@test isapprox(gt, rt; rtol = 1e-10, atol = 1e-12)
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 12),
                                   range(0.0, step = 0.12, length = 10))
    reference = SFC.calculate_structure_function(SFT.L2SFType(), grid, ut, bins, fft, UInt32, SF.StructureFunctionSumsAndCounts)
    kept = SFC.calculate_structure_function(SFT.L2SFType(), grid, ut, bins, fft, UInt32, SF.StructureFunctionSumsAndCounts;
                                            workspace = ws)
    Test.@test kept.counts == reference.counts
    Test.@test isapprox(kept.sums, reference.sums; rtol = 1e-10, atol = 1e-12)
end

mutable struct _FlagPlan <: AbstractFFTs.Plan{Float64}
    freed::Base.RefValue{Bool}
    _FlagPlan(flag) = finalizer(p -> (p.freed[] = true), new(flag))
end

# A call without a workspace owns one, and on return finalizes every plan it holds — wrapped, as an inverse
# plan is, or nested in the buffers' named tuples — and keeps nothing.
Test.@testset "a call's own workspace finalizes its plans on return" begin
    flags = [Ref(false) for _ in 1:3]
    plans = map(_FlagPlan, flags)
    ws = SFC.TransformWorkspace()
    ws.kept[:stage] = (1,) => (plan = plans[1], plan_last = plans[1], held = zeros(2))
    push!(ws.pool, (2,) => (iplan = AbstractFFTs.ScaledPlan(plans[2], 0.5), spec = zeros(2)))
    push!(ws.pool, (3,) => (B = 1, buffers = (out = zeros(1), iplan = plans[3])))
    SFC._release_plans!(ws)
    Test.@test all(f -> f[], flags)
    Test.@test isempty(ws.kept) && isempty(ws.pool)
    s = SFC.UniformLagSchedule((12, 10), (0.1, 0.12), (true, false))
    bins = collect(range(0.0, 1.2; length = 7)) .+ 0.0137
    u = randn(2, 12, 10)
    rs, rc = zeros(6), zeros(Int, 6)
    SFC.gridded_lag_sweep!(rs, rc, SFT.L2SFType(), u, s, bins, Val(2))
    for backend in (CB.SerialBackend(), CB.GPUBackend(KA.CPU()))
        gs, gc = zeros(6), zeros(Int, 6)
        SFC.gridded_sweep!(gs, gc, SFT.L2SFType(), u, s, bins, Val(2), SB.FastFourierTransformSpectralBackend(); backend)
        Test.@test (backend, gc == rc, isapprox(gs, rs; rtol = 1e-10, atol = 1e-12)) == (backend, true, true)
    end
end
