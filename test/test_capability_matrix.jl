using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using OhMyThreads: OhMyThreads
using FFTW: FFTW
using AbstractFFTs: AbstractFFTs
using NonuniformFFTs: NonuniformFFTs
using Distributed: Distributed
using Random: Random

# What runs where, as data. Every route × backend is a cell, and a cell has exactly one admitted
# outcome: it runs and reproduces the serial reference, or it raises an `ArgumentError` whose
# message and reason are listed in `CM_REFUSED` below. A `MethodError`, a crash, or an answer that
# differs from serial fails here, and so does a cell whose outcome changes without this file
# changing — in either direction, so a refusal that becomes an implementation must be deleted from
# the list.

Random.seed!(11)
const CM_N, CM_T, CM_NB, CM_NV = 40, 3, 6, 5
const CM_OP = SFT.L2SFType()
const CM_XP, CM_UP = rand(2, CM_N), randn(2, CM_N)
const CM_XB, CM_UB = rand(2, CM_N, CM_T), randn(2, CM_N, CM_T)
const CM_X1, CM_U1 = reshape(sort(rand(CM_N)), 1, CM_N), randn(1, CM_N)
const CM_BINS = collect(range(0.0, 1.0; length = CM_NB + 1))
const CM_VBINS = collect(range(-3.0, 3.0; length = CM_NV + 1))
const CM_ABINS = collect(range(prevfloat(0.0), π; length = 4))
const CM_NA = length(CM_ABINS) - 1
const CM_AX = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
# Bins short enough for a cull grid to exclude pairs on the unit square.
const CM_TBINS = collect(range(0.0, 0.25; length = CM_NB + 1))
const CM_CULL = SFC.AlwaysCulling()
const CM_RAW = SF.StructureFunctionSumsAndCounts
const CM_TRAW = SFO.StructureFunctionTensorSumsAndCounts
const CM_FFT = SB.FastFourierTransformSpectralBackend()
const CM_GS = SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true))
const CM_GU = reshape(Float64[sin(d + 3i + 7j) for d in 1:2, i in 1:8, j in 1:8], 2, :)
const CM_GUB =
    reshape(Float64[sin(d + 3i + 7j + 2t) for d in 1:2, i in 1:8, j in 1:8, t in 1:2], 2, 64, 2)
const CM_GB = collect(range(0.0, 0.5; length = CM_NB + 1))
const CM_HX = permutedims(hcat([2π * (i * 0.6180339887498949 % 1) for i in 1:CM_N],
    π / 2 .- acos.(clamp.(range(-0.95, 0.95; length = CM_N), -1, 1))))
const CM_HU = Float64[sin(d + 2i) for d in 1:2, i in 1:CM_N]
const CM_NODES = SF.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), 16)
const CM_SM = SFC.ScatteredModesSchedule(CM_XP, 0.5, (24, 16); taper = SF.GaussianTaper(0.03))
const CM_NUTAG = SFC.NonuniformFFTsSpectralBackend()

"""Every route, as the entry a user calls, returning `(sums, counts)`."""
const CM_ROUTES = (
    ("point 1D", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UP, CM_BINS, CM_RAW;
        backend = be); (r.sums, r.counts)))),
    ("point 1D in-place", (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.calculate_structure_function!(s, c, CM_OP, CM_XP, CM_UP, CM_BINS; backend = be);
            (s, c)))),
    ("point joint value", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UP,
        CM_BINS, CM_VBINS; backend = be); (r.sums, r.counts)))),
    ("point joint angle", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UP,
        CM_BINS, CM_ABINS; backend = be, second_axis = CM_AX);
        (r.sums, r.counts)))),
    ("point sorted line", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_X1, CM_U1,
        CM_BINS, CM_RAW; backend = be); (r.sums, r.counts)))),
    ("point multi-field", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP,
        MF.Fields(vectors = (CM_UP,)), CM_BINS, CM_RAW; backend = be);
        (r.sums, r.counts)))),
    ("moment tensor", (be -> (r = SFC.calculate_structure_function_tensor(Val(2), CM_XP, CM_UP,
        CM_BINS, CM_TRAW; backend = be); (r.sums, r.counts)))),
    ("moment tensor joint", (be -> (r = SFC.calculate_structure_function_tensor(Val(2), CM_XP,
        CM_UP, CM_BINS, CM_ABINS; second_axis = CM_AX, backend = be); (r.sums, r.counts)))),
    ("single-pass 1D", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB);
        SFC.calculate_structure_functions_single_pass!(s, c, CM_XP, CM_UP, CM_BINS; backend = be);
        (s, c)))),
    ("single-pass 2D", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_NV);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_NV);
        SFC.calculate_structure_functions_single_pass_2d!(s, c, CM_XP, CM_UP, CM_BINS, CM_VBINS;
            backend = be); (s, c)))),
    ("aux axes 1D", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_BINS, CM_RAW;
        backend = be); (r.sums, r.counts)))),
    ("aux axes joint", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_BINS,
        CM_VBINS; backend = be); (r.sums, r.counts)))),
    ("aux axes joint angle", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_BINS,
        CM_ABINS; backend = be, second_axis = CM_AX); (r.sums, r.counts)))),
    ("aux axes joint angle shared", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UB,
        CM_BINS, CM_ABINS; backend = be, second_axis = CM_AX); (r.sums, r.counts)))),
    ("aux axes 1D culled", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_TBINS,
        CM_RAW; backend = be, culling = CM_CULL); (r.sums, r.counts)))),
    ("aux axes 1D culled shared", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UB,
        CM_TBINS, CM_RAW; backend = be, culling = CM_CULL); (r.sums, r.counts)))),
    ("slice batch 1D", (be -> (s = zeros(CM_NB, CM_T); c = zeros(Int, CM_NB, CM_T);
        SFC.calculate_structure_function_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS; backend = be);
        (s, c)))),
    ("slice batch joint", (be -> (s = zeros(CM_NB, CM_NV, CM_T);
        c = zeros(Int, CM_NB, CM_NV, CM_T);
        SFC.calculate_structure_function_2d_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS, CM_VBINS;
            backend = be); (s, c)))),
    ("slice batch joint angle", (be -> (s = zeros(CM_NB, CM_NA, CM_T);
        c = zeros(Int, CM_NB, CM_NA, CM_T);
        SFC.calculate_structure_function_2d_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS, CM_ABINS;
            backend = be, second_axis = CM_AX); (s, c)))),
    ("slice batch sp1d", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_T);
        SFC.calculate_structure_functions_single_pass_batch!(s, c, CM_XB, CM_UB, CM_BINS;
            backend = be); (s, c)))),
    ("slice batch sp1d culled shared", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_T);
        SFC.calculate_structure_functions_single_pass_batch!(s, c, CM_XP, CM_UB, CM_TBINS;
            backend = be, culling = CM_CULL); (s, c)))),
    ("slice batch sp2d", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_NV, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_NV, CM_T);
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, CM_XB, CM_UB, CM_BINS,
            CM_VBINS; backend = be); (s, c)))),
    ("gridded lag sweep", (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded lag sweep joint value", (be -> (s = zeros(CM_NB, CM_NV); c = zeros(CM_NB, CM_NV);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, CM_VBINS, Val(2), Val(1), Val(0);
            backend = be, second_axis = SFC.InvariantValueAxis()); (s, c)))),
    ("gridded lag sweep joint angle", (be -> (s = zeros(CM_NB, CM_NA); c = zeros(CM_NB, CM_NA);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, CM_ABINS, Val(2), Val(1), Val(0);
            backend = be, second_axis = CM_AX); (s, c)))),
    ("gridded lag sweep batch", (be -> (s = zeros(CM_NB, 2); c = zeros(Int, CM_NB, 2);
        SFC.gridded_lag_sweep_batch!(s, c, CM_OP, CM_GUB, CM_GS, CM_GB, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded lag sweep batch joint value", (be -> (s = zeros(CM_NB, CM_NV, 2); c = zeros(CM_NB, CM_NV, 2);
        SFC.gridded_lag_sweep_batch!(s, c, CM_OP, CM_GUB, CM_GS, CM_GB, CM_VBINS, Val(2), Val(1), Val(0);
            backend = be, second_axis = SFC.InvariantValueAxis()); (s, c)))),
    ("gridded transform", (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded single pass", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB); c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB);
        SFC.gridded_lag_sweep!(s, c, SFT.SinglePassInvariants(), CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded single pass transform", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB);
        SFC.gridded_sweep!(s, c, SFT.SinglePassInvariants(), CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded transform batch", (be -> (s = zeros(CM_NB, 2); c = zeros(Int, CM_NB, 2);
        SFC.gridded_sweep_batch!(s, c, CM_OP, CM_GUB, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded tensor", (be -> (s = zeros(2, 2, CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_tensor_sweep!(s, c, Val(2), CM_GU, CM_GS, CM_GB, Val(2), CM_FFT;
            backend = be); (s, c)))),
    ("gridded tensor joint angle", (be -> (s = zeros(2, 2, CM_NB, CM_NA); c = zeros(CM_NB, CM_NA);
        SFC.gridded_tensor_sweep!(s, c, Val(2), CM_GU, CM_GS, CM_GB, CM_ABINS, Val(2), CM_FFT;
            backend = be, second_axis = CM_AX); (s, c)))),
    ("harmonic direct sum", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_HX, CM_HU,
        CM_NODES, SB.DirectSumSpectralBackend(), CM_RAW; backend = be);
        (r.sums, r.counts)))),
    ("scattered modes NUFFT", (be -> (r = SFC.calculate_structure_function(CM_OP, CM_SM, CM_UP,
        CM_BINS, CM_NUTAG, CM_RAW; backend = be);
        (r.sums, r.counts)))),
)

const CM_BACKENDS = (
    ("serial", CB.SerialBackend()),
    ("threaded", CB.ThreadedBackend()),
    ("distributed", CB.DistributedBackend()),
    ("dist+threaded", CB.DistributedBackend(CB.ThreadedBackend())),
    ("gpu", CB.GPUBackend(KA.CPU())),
)

"""
The cells that do not run, each with the message its `ArgumentError` must carry and the reason the
cell is refused. Every other cell must run and reproduce the serial reference. Emptying this is
what "every route on every backend" means.

A cell added here must carry a reason, and a cell whose refusal is later implemented must be
removed, or the matrix stops asserting anything about it.
"""
const CM_REFUSED = Dict{Tuple{String, String}, @NamedTuple{message::String, reason::String}}()

"""Counts exactly when both are integer; a kernel-weighted count is a mass, compared like a sum."""
function cm_agrees(got, ref)
    size(got) == size(ref) || return false
    if eltype(got) <: Integer && eltype(ref) <: Integer
        return got == ref
    end
    scale = max(maximum(abs, ref), 1e-12)
    return all(p -> (isnan(p[1]) && isnan(p[2])) || abs(p[1] - p[2]) <= 1e-9 * scale,
               zip(got, ref))
end

const CM_WU_S = SFC.UniformLagSchedule((16, 16), (1 / 16, 1 / 16), (true, true))

const CM_WORKERS_ADDED_HERE =
    Distributed.nprocs() == 1 ?
    Distributed.addprocs(2; exeflags = ["--project=$(Base.active_project())", "-t", "1"]) : Int[]
try

Distributed.@everywhere using StructureFunctions: Calculations as SFC,
    StructureFunctionTypes as SFT, MultiFields as MF
Distributed.@everywhere using OhMyThreads: OhMyThreads
Distributed.@everywhere using FFTW: FFTW
Distributed.@everywhere using NonuniformFFTs: NonuniformFFTs

Test.@testset "the capability matrix: every route on every backend" begin
    Test.@test Distributed.nworkers() > 1
    n_workers = Distributed.nworkers()

    for (rname, run) in CM_ROUTES
        Test.@testset "$rname" begin
            ref_s, ref_c = run(CB.SerialBackend())
            # A route whose reference is all zeros would make every cell below pass vacuously.
            Test.@test any(!iszero, ref_c)

            for (bname, be) in CM_BACKENDS
                refusal = get(CM_REFUSED, (rname, bname), nothing)
                outcome = try
                    run(be)
                catch e
                    e
                end

                if refusal === nothing
                    if outcome isa Exception
                        # Naming the exception is what makes the failure readable: an unlisted
                        # refusal, a MethodError and a crashed worker are three different defects.
                        Test.@test (bname, nameof(typeof(outcome))) == (bname, :NoException)
                    else
                        Test.@test cm_agrees(outcome[1], ref_s)
                        Test.@test cm_agrees(outcome[2], ref_c)
                    end
                else
                    Test.@test outcome isa ArgumentError
                    Test.@test outcome isa ArgumentError &&
                               occursin(refusal.message, outcome.msg)
                end
            end

            # A cell that kills a worker leaves `nworkers()` at 1, and every later distributed cell
            # would then run on the driver and report a result that was never distributed.
            Test.@test Distributed.nworkers() == n_workers
        end
    end
end

Test.@testset "an angle cell bins the angle" begin
    # Over the same edges the value histogram differs, so an angle cell that binned the value would
    # disagree with its serial reference.
    ser = CB.SerialBackend()
    for (name, x, u) in (("point", CM_XP, CM_UP), ("aux axes", CM_XB, CM_UB), ("aux axes shared", CM_XP, CM_UB))
        angle = SFC.calculate_structure_function(CM_OP, x, u, CM_BINS, CM_ABINS; backend = ser, second_axis = CM_AX)
        value = SFC.calculate_structure_function(CM_OP, x, u, CM_BINS, CM_ABINS; backend = ser)
        Test.@test (name, angle.counts != value.counts) == (name, true)
    end
end

Test.@testset "a schedule of one slab pair splits its lags across every task" begin
    n_tasks = SFC.sweep_tasks(CB.ThreadedBackend())
    Test.@test n_tasks == Threads.nthreads()
    Test.@test length(collect(SFC.enumerated_pairs(CM_WU_S, 0.4))) == 1
    items = SFC.sweep_items(CM_WU_S, 0.4, n_tasks, true)
    Test.@test length(items) >= n_tasks
    Test.@test sort([it[3] for it in items]) == 1:items[1][4]
end

Test.@testset "the matrix table names only cells that exist" begin
    # A refusal that has been implemented must be deleted from the table, not left behind: a stale
    # row would quietly stop asserting anything.
    cells = Set((r[1], b[1]) for r in CM_ROUTES, b in CM_BACKENDS)
    for key in keys(CM_REFUSED)
        Test.@test key in cells
    end
    for (_, spec) in CM_REFUSED
        Test.@test !isempty(spec.reason)
    end
end

finally
    isempty(CM_WORKERS_ADDED_HERE) || Distributed.rmprocs(CM_WORKERS_ADDED_HERE; waitfor = 30)
end
