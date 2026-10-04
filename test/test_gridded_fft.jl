using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using FFTW: FFTW
using SpectralBackends: SpectralBackends as SB
using FlowGeometries: FlowGeometries as FG
using KernelAbstractions: KernelAbstractions as KA
using OhMyThreads: OhMyThreads
using Random: Random

function _run(sf, u, dims, spacing, periodic, bins, D, tag)
    plan = SFC.squared_digitize_plan(bins)
    nb = SFC.n_histogram_bins(plan)
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    sched = SFC.UniformLagSchedule(dims, spacing, periodic)
    if tag === nothing
        SFC.gridded_lag_sweep!(s, c, sf, u, sched, bins, Val(D); backend = CB.SerialBackend())
    else
        SFC.gridded_sweep!(s, c, sf, u, sched, bins, Val(D), tag; backend = CB.SerialBackend())
    end
    return s, c
end

Test.@testset "the transform refuses an operator that is not a polynomial in the increment" begin
    u = randn(2, 6, 5)
    bins = collect(range(0.0, 1.0; length = 5))
    Test.@test_throws ArgumentError _run(SFT.FullVectorStructureFunctionType{3}(), u, (6, 5), (0.2, 0.2),
                                         (false, false), bins, 2, SB.FastFourierTransformSpectralBackend())
end

Test.@testset "every algorithm tag gives the lag sweep's answer" begin
    # The direct sum, and the automatic choice on a polynomial and on a non-polynomial operator.
    dims, spacing, periodic = (8, 6), (0.2, 0.25), (false, false)
    Random.seed!(6300)
    u = randn(2, dims...)
    bins = collect(range(0.0, 1.2; length = 7))
    cases = ((SFT.L2SFType(), SB.DirectSumSpectralBackend()), (SFT.L3SFType(), SB.AutoSpectralBackend()),
             (SFT.FullVectorStructureFunctionType{3}(), SB.AutoSpectralBackend()))
    counts_ok, sums_ok = Bool[], Bool[]
    for (sf, tag) in cases
        ref_s, ref_c = _run(sf, u, dims, spacing, periodic, bins, 2, nothing)
        got_s, got_c = _run(sf, u, dims, spacing, periodic, bins, 2, tag)
        push!(counts_ok, got_c == ref_c && sum(ref_c) > 0)
        push!(sums_ok, isapprox(got_s, ref_s; rtol = 1e-9, atol = 1e-10))
    end
    Test.@test all(counts_ok)
    Test.@test all(sums_ok)
end

Test.@testset "a transform workspace changes no answer, reused across backends, batches, sizes and entries" begin
    Random.seed!(6500)
    fft = SB.FastFourierTransformSpectralBackend()
    bins = collect(range(0.0, 1.2; length = 7)) .+ 0.0137
    nb = length(bins) - 1
    lats = collect(range(-1.0, 1.0; length = 7))
    close(g, r) = isapprox(g, r; rtol = 1e-10, atol = 1e-12)
    cases = ((SFC.UniformLagSchedule((12, 10), (0.1, 0.12), (true, false)), (12, 10),
              (CB.SerialBackend(), CB.GPUBackend(KA.CPU()))),
             (SFC.ZonalLagSchedule(lats, 12, 2π / 12, 1.0, true), (12, 7),
              (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))))
    agree = Bool[]
    for (s, dims, backends) in cases
        u = randn(2, dims...)
        ref_s, ref_c = zeros(nb), zeros(Int, nb)
        SFC.gridded_sweep!(ref_s, ref_c, SFT.L3SFType(), u, s, bins, Val(2), fft; backend = CB.SerialBackend())
        ws = SFC.TransformWorkspace()
        for backend in backends, workspace in (nothing, ws, ws)
            gs, gc = zeros(nb), zeros(Int, nb)
            SFC.gridded_sweep!(gs, gc, SFT.L3SFType(), u, s, bins, Val(2), fft; backend, workspace)
            push!(agree, gc == ref_c && sum(ref_c) > 0 && close(gs, ref_s))
        end
        ub = randn(2, dims..., 3)
        rb, rcb = zeros(nb, 3), zeros(Int, nb, 3)
        SFC.gridded_sweep_batch!(rb, rcb, SFT.L3SFType(), ub, s, bins, Val(2), fft; backend = CB.SerialBackend())
        for backend in backends, _ in 1:2
            gb, gcb = zeros(nb, 3), zeros(Int, nb, 3)
            SFC.gridded_sweep_batch!(gb, gcb, SFT.L3SFType(), ub, s, bins, Val(2), fft; backend, workspace = ws)
            push!(agree, gcb == rcb && close(gb, rb))
        end
    end
    # one workspace taken by grids of two sizes, by the tensor entry and by the grid entry
    ws = SFC.TransformWorkspace()
    small = SFC.UniformLagSchedule((12, 10), (0.1, 0.12), (true, false))
    large = SFC.UniformLagSchedule((16, 10), (0.1, 0.12), (true, false))
    SFC.gridded_sweep!(zeros(nb), zeros(Int, nb), SFT.L2SFType(), randn(2, 12, 10), small, bins, Val(2), fft;
                       workspace = ws)
    ul = randn(2, 16, 10)
    rl, rcl = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(rl, rcl, SFT.L2SFType(), ul, large, bins, Val(2), fft)
    gl, gcl = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(gl, gcl, SFT.L2SFType(), ul, large, bins, Val(2), fft; workspace = ws)
    push!(agree, gcl == rcl && close(gl, rl))
    ut = randn(2, 12, 10)
    rt, rct = zeros(2, 2, nb), zeros(Int, nb)
    SFC.gridded_tensor_sweep!(rt, rct, Val(2), reshape(ut, 2, :), small, bins, Val(2), fft)
    gt, gct = zeros(2, 2, nb), zeros(Int, nb)
    SFC.gridded_tensor_sweep!(gt, gct, Val(2), reshape(ut, 2, :), small, bins, Val(2), fft; workspace = ws)
    push!(agree, gct == rct && close(gt, rt))
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, step = 0.1, length = 12),
                                   range(0.0, step = 0.12, length = 10))
    reference = SFC.calculate_structure_function(SFT.L2SFType(), grid, ut, bins, fft, UInt32,
                                                 SF.StructureFunctionSumsAndCounts)
    kept = SFC.calculate_structure_function(SFT.L2SFType(), grid, ut, bins, fft, UInt32,
                                            SF.StructureFunctionSumsAndCounts; workspace = ws)
    push!(agree, kept.counts == reference.counts && close(kept.sums, reference.sums))
    Test.@test all(agree)
end

Test.@testset "the threaded transform gives the lag sweep's answer under any FFTW thread count" begin
    # Per grid: each operator, backend and workspace once, a batch of fewer or more slices than threads, the tensor.
    tag = SB.FastFourierTransformSpectralBackend()
    Random.seed!(17)
    close(a, b) = isapprox(a, b; rtol = 1e-10, atol = 1e-10 * maximum(abs, b; init = 0.0))
    grids = (((24, 16), (true, true), ((SFT.L2SFType(), CB.SerialBackend(), nothing),
                                       (SFT.S3SFType(), CB.ThreadedBackend(), SFC.TransformWorkspace())), 2),
             ((20, 15), (false, true), ((SFT.S3SFType(), CB.SerialBackend(), SFC.TransformWorkspace()),
                                        (SFT.L2SFType(), CB.ThreadedBackend(), nothing)), 2 * Threads.nthreads() + 1))
    agree = Bool[]
    before = FFTW.get_num_threads()
    try
        for fftw in unique((1, Threads.nthreads())), (dims, periodic, runs, nt) in grids
            FFTW.set_num_threads(fftw)
            s = SFC.UniformLagSchedule(dims, 1 ./ dims, periodic)
            bins = collect(range(0.0, 0.3; length = 9))
            nb = length(bins) - 1
            u = randn(2, prod(dims))
            for (sf, backend, ws) in runs
                ref = (zeros(nb), zeros(Int, nb))
                SFC.gridded_lag_sweep!(ref..., sf, u, s, bins, Val(2))
                got = (zeros(nb), zeros(Int, nb))
                SFC.gridded_sweep!(got..., sf, u, s, bins, Val(2), tag; backend, workspace = ws)
                push!(agree, got[2] == ref[2] && sum(ref[2]) > 0 && close(got[1], ref[1]))
            end
            ub = randn(2, prod(dims), nt)
            ref = (zeros(nb, nt), zeros(Int, nb, nt))
            SFC.gridded_lag_sweep_batch!(ref..., SFT.L2SFType(), ub, s, bins, Val(2), Val(1), Val(0))
            got = (zeros(nb, nt), zeros(Int, nb, nt))
            SFC.gridded_sweep_batch!(got..., SFT.L2SFType(), ub, s, bins, Val(2), Val(1), Val(0), tag;
                                     backend = CB.ThreadedBackend())
            push!(agree, got[2] == ref[2] && close(got[1], ref[1]))
            serial = (zeros(2, 2, nb), zeros(Int, nb))
            SFC.gridded_tensor_sweep!(serial..., Val(2), u, s, bins, Val(2), tag; backend = CB.SerialBackend())
            threaded = (zeros(2, 2, nb), zeros(Int, nb))
            SFC.gridded_tensor_sweep!(threaded..., Val(2), u, s, bins, Val(2), tag; backend = CB.ThreadedBackend())
            push!(agree, threaded[2] == serial[2] && close(threaded[1], serial[1]))
        end
    finally
        FFTW.set_num_threads(before)
    end
    Test.@test all(agree)
end

Test.@testset "Float32 zonal transforms in forward blocks equal the lag sweep" begin
    # A forward budget of 4 latitude slabs splits 9 into blocks of 4, 4 and 1; a budget of all 9 is one block.
    ext = Base.get_extension(SF, :StructureFunctionsAbstractFFTsExt)
    old_budget = ext.FORWARD_BATCH_BYTES[]
    nlon, nlat = 24, 9
    u = randn(Random.MersenneTwister(14), Float32, 2, nlon, nlat)
    lats = collect(range(-0.6f0, 0.5f0; length = nlat))
    schedule = SFC.ZonalLagSchedule(lats, nlon, Float32(2π / nlon), 1.0f0, true)
    bins = Float32[0, 0.15, 0.35]
    reference, refcounts = zeros(Float32, 2), zeros(UInt64, 2)
    SFC.gridded_lag_sweep!(reference, refcounts, SFT.S2SFType(), u, schedule, bins, Val(2))
    agree = Bool[]
    try
        nkeys = length(SFC._monomial_keys(Val(2), Val(2)))
        for slabs in (4, nlat)
            ext.FORWARD_BATCH_BYTES[] = slabs * nkeys * (nlon * sizeof(Float32) + (nlon ÷ 2 + 1) * sizeof(ComplexF32))
            sums, counts = zeros(Float32, 2), zeros(UInt64, 2)
            SFC.gridded_sweep!(sums, counts, SFT.S2SFType(), u, schedule, bins, Val(2),
                               SB.FastFourierTransformSpectralBackend())
            push!(agree, counts == refcounts && all(>(0), refcounts) && isapprox(sums, reference; rtol = 5e-5))
        end
    finally
        ext.FORWARD_BATCH_BYTES[] = old_budget
    end
    Test.@test all(agree)
end
