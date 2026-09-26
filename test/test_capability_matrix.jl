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

const CM_WORKERS_ADDED_HERE =
    Distributed.nprocs() == 1 ?
    Distributed.addprocs(2; exeflags = ["--project=$(Base.active_project())", "-t", "1"]) : Int[]

Distributed.@everywhere using StructureFunctions: Calculations as SFC,
    StructureFunctionTypes as SFT, MultiFields as MF
Distributed.@everywhere using OhMyThreads: OhMyThreads
Distributed.@everywhere using FFTW: FFTW
Distributed.@everywhere using NonuniformFFTs: NonuniformFFTs

Random.seed!(11)
const CM_N, CM_T, CM_NB, CM_NV = 40, 3, 6, 5
const CM_OP = SFT.L2SFType()
const CM_XP, CM_UP = rand(2, CM_N), randn(2, CM_N)
const CM_XB, CM_UB = rand(2, CM_N, CM_T), randn(2, CM_N, CM_T)
const CM_X1, CM_U1 = reshape(sort(rand(CM_N)), 1, CM_N), randn(1, CM_N)
const CM_BINS = collect(range(0.0, 1.0; length = CM_NB + 1))
const CM_VBINS = collect(range(-3.0, 3.0; length = CM_NV + 1))
const CM_ABINS = collect(range(prevfloat(0.0), π; length = 4))
const CM_AX = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
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
    ("slice batch 1D", (be -> (s = zeros(CM_NB, CM_T); c = zeros(Int, CM_NB, CM_T);
        SFC.calculate_structure_function_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS; backend = be);
        (s, c)))),
    ("slice batch joint", (be -> (s = zeros(CM_NB, CM_NV, CM_T);
        c = zeros(Int, CM_NB, CM_NV, CM_T);
        SFC.calculate_structure_function_2d_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS, CM_VBINS;
            backend = be); (s, c)))),
    ("slice batch sp1d", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_T);
        SFC.calculate_structure_functions_single_pass_batch!(s, c, CM_XB, CM_UB, CM_BINS;
            backend = be); (s, c)))),
    ("slice batch sp2d", (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_NV, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_NV, CM_T);
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, CM_XB, CM_UB, CM_BINS,
            CM_VBINS; backend = be); (s, c)))),
    ("gridded lag sweep", (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded transform", (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded transform batch", (be -> (s = zeros(CM_NB, 2); c = zeros(Int, CM_NB, 2);
        SFC.gridded_sweep_batch!(s, c, CM_OP, CM_GUB, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded tensor", (be -> (s = zeros(2, 2, CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_tensor_sweep!(s, c, Val(2), CM_GU, CM_GS, CM_GB, Val(2), CM_FFT;
            backend = be); (s, c)))),
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
what "every route on every backend" means, and it is empty: all 105 cells run.

A cell added here must carry a reason, and a cell whose refusal is later implemented must be
removed, or the matrix stops asserting anything about it.
"""
const CM_REFUSED = Dict{Tuple{String, String},
                        @NamedTuple{message::String, reason::String}}(
    ("gridded tensor", "gpu") => (
        message = "supplies no `sweep_reduce!`",
        reason = "the gridded tensor has no device kernel. Every other gridded route reaches a " *
                 "device through its own hook — `device_transform_sweep!` for the transform, " *
                 "`device_lag_sweep!` for the direct sweep — and no `device_tensor_sweep!` is " *
                 "written, so the tensor falls through to the `sweep_reduce!` catch-all. The " *
                 "kernel differs from the transform's only in accumulating the symmetric moment " *
                 "store instead of contracting it, so this is unwritten work, not an impossibility.",
    ),
)

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

# How many work items a route actually creates, observed on the route itself rather than on a copy
# of its splitter: this backend records the item count `sweep_reduce!` is handed and then runs the
# sweep on the backend it wraps, so the answer is still checked.
struct CMCountingBackend{B} <: CB.AbstractExecutionBackend
    inner::B
    n::Base.RefValue{Int}
end
CMCountingBackend(inner) = CMCountingBackend(inner, Ref(0))
SFC.sweep_tasks(b::CMCountingBackend) = SFC.sweep_tasks(b.inner)
function SFC.sweep_reduce!(sums, counts, b::CMCountingBackend, items, make_scratch, body!)
    b.n[] = length(items)
    return SFC.sweep_reduce!(sums, counts, b.inner, items, make_scratch, body!)
end

const CM_WU_S = SFC.UniformLagSchedule((16, 16), (1 / 16, 1 / 16), (true, true))
const CM_WU_U = reshape(Float64[sin(d + 3i + 7j) for d in 1:2, i in 1:16, j in 1:16], 2, :)
const CM_WU_B = collect(range(0.0, 0.4; length = 5))

"""
What each route's decomposition must yield at `n_tasks` tasks. `:at_least_tasks` is the contract —
a schedule with one slab pair still has thousands of lags, so there is always more work than
tasks here. A route that yields fewer is listed with the gap that owns it, and emptying that
column is what G30.1 delivers.
"""
const CM_WORK_UNITS = (
    (name = "gridded lag sweep",
     run = (be -> (s = zeros(4); c = zeros(Int, 4);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_WU_U, CM_WU_S, CM_WU_B, Val(2), Val(1), Val(0);
            backend = be); (s, c))),
     expect = :at_least_tasks),
    (name = "gridded transform",
     run = (be -> (s = zeros(4); c = zeros(Int, 4);
        SFC.gridded_sweep!(s, c, CM_OP, CM_WU_U, CM_WU_S, CM_WU_B, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c))),
     expect = :at_least_tasks),
)

Test.@testset "a route decomposes into as many work units as it has tasks" begin
    n_tasks = SFC.sweep_tasks(CB.ThreadedBackend())
    Test.@test n_tasks == Threads.nthreads()

    for row in CM_WORK_UNITS
        Test.@testset "$(row.name)" begin
            probe = CMCountingBackend(CB.ThreadedBackend())
            got_s, got_c = row.run(probe)
            ref_s, ref_c = row.run(CB.SerialBackend())
            # The decomposition is only interesting if it still computes the right answer.
            Test.@test cm_agrees(got_s, ref_s)
            Test.@test cm_agrees(got_c, ref_c)

            if row.expect === :at_least_tasks
                Test.@test probe.n[] >= n_tasks
            else
                Test.@test !isempty(row.expect.reason)
                Test.@test probe.n[] == row.expect.units
            end
        end
    end
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

isempty(CM_WORKERS_ADDED_HERE) ||
    Distributed.rmprocs(CM_WORKERS_ADDED_HERE; waitfor = 30)
