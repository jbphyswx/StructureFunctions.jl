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

const CM_BACKENDS = (
    threaded = CB.ThreadedBackend(),
    distributed = CB.DistributedBackend(),
    dist_threaded = CB.DistributedBackend(CB.ThreadedBackend()),
    gpu = CB.GPUBackend(KA.CPU()),
)

# Every route as the entry a user calls, returning `(sums, counts)`, with the backends that reach its backend methods.
const CM_ROUTES = (
    ("point 1D", (:threaded, :distributed, :dist_threaded, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UP, CM_BINS, CM_RAW;
        backend = be); (r.sums, r.counts)))),
    ("point 1D in-place", (:threaded, :distributed, :gpu),
     (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.calculate_structure_function!(s, c, CM_OP, CM_XP, CM_UP, CM_BINS; backend = be);
            (s, c)))),
    ("point joint value", (:threaded, :distributed, :dist_threaded, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UP,
        CM_BINS, CM_VBINS; backend = be); (r.sums, r.counts)))),
    ("point joint angle", (:gpu,),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UP,
        CM_BINS, CM_ABINS; backend = be, second_axis = CM_AX);
        (r.sums, r.counts)))),
    ("point sorted line", (:threaded, :distributed, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_X1, CM_U1,
        CM_BINS, CM_RAW; backend = be); (r.sums, r.counts)))),
    ("point multi-field", (:threaded, :distributed, :dist_threaded, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP,
        MF.Fields(vectors = (CM_UP,)), CM_BINS, CM_RAW; backend = be);
        (r.sums, r.counts)))),
    ("moment tensor", (:threaded, :distributed, :dist_threaded, :gpu),
     (be -> (r = SFC.calculate_structure_function_tensor(Val(2), CM_XP, CM_UP,
        CM_BINS, CM_TRAW; backend = be); (r.sums, r.counts)))),
    ("moment tensor joint", (:distributed, :gpu),
     (be -> (r = SFC.calculate_structure_function_tensor(Val(2), CM_XP,
        CM_UP, CM_BINS, CM_ABINS; second_axis = CM_AX, backend = be); (r.sums, r.counts)))),
    ("single-pass 1D", (:threaded, :distributed, :dist_threaded, :gpu),
     (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB);
        SFC.calculate_structure_functions_single_pass!(s, c, CM_XP, CM_UP, CM_BINS; backend = be);
        (s, c)))),
    ("single-pass 2D", (:threaded, :distributed, :dist_threaded, :gpu),
     (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_NV);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_NV);
        SFC.calculate_structure_functions_single_pass_2d!(s, c, CM_XP, CM_UP, CM_BINS, CM_VBINS;
            backend = be); (s, c)))),
    ("aux axes 1D", (:threaded, :distributed, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_BINS, CM_RAW;
        backend = be); (r.sums, r.counts)))),
    ("aux axes joint", (:threaded, :distributed, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_BINS,
        CM_VBINS; backend = be); (r.sums, r.counts)))),
    ("aux axes joint angle", (:threaded,),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_BINS,
        CM_ABINS; backend = be, second_axis = CM_AX); (r.sums, r.counts)))),
    ("aux axes joint angle shared", (:gpu,),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UB,
        CM_BINS, CM_ABINS; backend = be, second_axis = CM_AX); (r.sums, r.counts)))),
    ("aux axes 1D culled", (:distributed, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XB, CM_UB, CM_TBINS,
        CM_RAW; backend = be, culling = CM_CULL); (r.sums, r.counts)))),
    ("aux axes 1D culled shared", (:threaded, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_XP, CM_UB,
        CM_TBINS, CM_RAW; backend = be, culling = CM_CULL); (r.sums, r.counts)))),
    ("slice batch 1D", (:threaded, :distributed, :gpu),
     (be -> (s = zeros(CM_NB, CM_T); c = zeros(Int, CM_NB, CM_T);
        SFC.calculate_structure_function_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS; backend = be);
        (s, c)))),
    ("slice batch joint", (:threaded, :distributed, :gpu),
     (be -> (s = zeros(CM_NB, CM_NV, CM_T);
        c = zeros(Int, CM_NB, CM_NV, CM_T);
        SFC.calculate_structure_function_2d_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS, CM_VBINS;
            backend = be); (s, c)))),
    ("slice batch joint angle", (:threaded,),
     (be -> (s = zeros(CM_NB, CM_NA, CM_T);
        c = zeros(Int, CM_NB, CM_NA, CM_T);
        SFC.calculate_structure_function_2d_batch!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS, CM_ABINS;
            backend = be, second_axis = CM_AX); (s, c)))),
    ("slice batch sp1d", (:threaded, :distributed, :gpu),
     (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_T);
        SFC.calculate_structure_functions_single_pass_batch!(s, c, CM_XB, CM_UB, CM_BINS;
            backend = be); (s, c)))),
    ("slice batch sp1d culled shared", (:gpu,),
     (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_T);
        SFC.calculate_structure_functions_single_pass_batch!(s, c, CM_XP, CM_UB, CM_TBINS;
            backend = be, culling = CM_CULL); (s, c)))),
    ("slice batch sp2d", (:threaded, :distributed, :gpu),
     (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB, CM_NV, CM_T);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB, CM_NV, CM_T);
        SFC.calculate_structure_functions_single_pass_2d_batch!(s, c, CM_XB, CM_UB, CM_BINS,
            CM_VBINS; backend = be); (s, c)))),
    ("aux axes 1D in-place", (:threaded,),
     (be -> (s = zeros(CM_NB, CM_T); c = zeros(Int, CM_NB, CM_T);
        SFC.calculate_structure_function!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS; backend = be); (s, c)))),
    ("aux axes joint in-place", (:threaded,),
     (be -> (s = zeros(CM_NB, CM_NV, CM_T); c = zeros(Int, CM_NB, CM_NV, CM_T);
        SFC.calculate_structure_function!(s, c, CM_OP, CM_XB, CM_UB, CM_BINS, CM_VBINS; backend = be); (s, c)))),
    ("aux axes single-pass", (:threaded,),
     (be -> (r = SFC.calculate_structure_functions_single_pass(CM_XB, CM_UB, CM_BINS; backend = be);
        (stack(r[k].sums for k in keys(r)), stack(r[k].counts for k in keys(r)))))),
    ("aux axes single-pass 2D", (:threaded,),
     (be -> (r = SFC.calculate_structure_functions_single_pass_2d(CM_XB, CM_UB, CM_BINS, CM_VBINS; backend = be);
        (stack(r[k].sums for k in keys(r)), stack(r[k].counts for k in keys(r)))))),
    ("gridded lag sweep", (:threaded, :gpu),
     (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded lag sweep joint value", (:gpu,),
     (be -> (s = zeros(CM_NB, CM_NV); c = zeros(CM_NB, CM_NV);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, CM_VBINS, Val(2), Val(1), Val(0);
            backend = be, second_axis = SFC.InvariantValueAxis()); (s, c)))),
    ("gridded lag sweep joint angle", (:gpu,),
     (be -> (s = zeros(CM_NB, CM_NA); c = zeros(CM_NB, CM_NA);
        SFC.gridded_lag_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, CM_ABINS, Val(2), Val(1), Val(0);
            backend = be, second_axis = CM_AX); (s, c)))),
    ("gridded lag sweep batch", (:distributed, :gpu),
     (be -> (s = zeros(CM_NB, 2); c = zeros(Int, CM_NB, 2);
        SFC.gridded_lag_sweep_batch!(s, c, CM_OP, CM_GUB, CM_GS, CM_GB, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded lag sweep batch joint value", (:gpu,),
     (be -> (s = zeros(CM_NB, CM_NV, 2); c = zeros(CM_NB, CM_NV, 2);
        SFC.gridded_lag_sweep_batch!(s, c, CM_OP, CM_GUB, CM_GS, CM_GB, CM_VBINS, Val(2), Val(1), Val(0);
            backend = be, second_axis = SFC.InvariantValueAxis()); (s, c)))),
    ("gridded transform", (:threaded, :distributed, :gpu),
     (be -> (s = zeros(CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_sweep!(s, c, CM_OP, CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded single pass", (:gpu,),
     (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB); c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB);
        SFC.gridded_lag_sweep!(s, c, SFT.SinglePassInvariants(), CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0);
            backend = be); (s, c)))),
    ("gridded single pass transform", (:gpu,),
     (be -> (s = zeros(SFC.SINGLE_PASS_N, CM_NB);
        c = zeros(Int, SFC.SINGLE_PASS_N, CM_NB);
        SFC.gridded_sweep!(s, c, SFT.SinglePassInvariants(), CM_GU, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded transform batch", (:gpu,),
     (be -> (s = zeros(CM_NB, 2); c = zeros(Int, CM_NB, 2);
        SFC.gridded_sweep_batch!(s, c, CM_OP, CM_GUB, CM_GS, CM_GB, Val(2), Val(1), Val(0), CM_FFT;
            backend = be); (s, c)))),
    ("gridded tensor", (:gpu,),
     (be -> (s = zeros(2, 2, CM_NB); c = zeros(Int, CM_NB);
        SFC.gridded_tensor_sweep!(s, c, Val(2), CM_GU, CM_GS, CM_GB, Val(2), CM_FFT;
            backend = be); (s, c)))),
    ("gridded tensor joint angle", (:gpu,),
     (be -> (s = zeros(2, 2, CM_NB, CM_NA); c = zeros(CM_NB, CM_NA);
        SFC.gridded_tensor_sweep!(s, c, Val(2), CM_GU, CM_GS, CM_GB, CM_ABINS, Val(2), CM_FFT;
            backend = be, second_axis = CM_AX); (s, c)))),
    ("harmonic direct sum", (:threaded, :distributed, :gpu),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_HX, CM_HU,
        CM_NODES, SB.DirectSumSpectralBackend(), CM_RAW; backend = be);
        (r.sums, r.counts)))),
    ("scattered modes NUFFT", (:gpu,),
     (be -> (r = SFC.calculate_structure_function(CM_OP, CM_SM, CM_UP,
        CM_BINS, CM_NUTAG, CM_RAW; backend = be);
        (r.sums, r.counts)))),
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

"""Whether a cell's `(sums, counts)` agree with the reference's."""
cm_reproduces((sums, counts), (ref_sums, ref_counts)) = cm_agrees(sums, ref_sums) && cm_agrees(counts, ref_counts)

const CM_WORKERS = let n_add = max(0, 3 - Distributed.nprocs())
    n_add == 0 ? Int[] :
    Distributed.addprocs(n_add; exeflags = ["--project=$(Base.active_project())", "-t", "1"])
end
try

Distributed.@everywhere using StructureFunctions: StructureFunctions
Distributed.@everywhere using OhMyThreads: OhMyThreads

# Every cell reproduces its route's serial reference, which bins at least one pair.
Test.@testset "every route reproduces its serial reference on the backends whose code it reaches" begin
    for (rname, bnames, run) in CM_ROUTES
        Test.@testset "$rname" begin
            ref = run(CB.SerialBackend())
            any(!iszero, ref[2]) || error("the serial reference of $rname bins no pair")
            for bname in bnames
                Test.@testset "$bname" begin
                    Test.@test cm_reproduces(run(CM_BACKENDS[bname]), ref)
                end
            end
        end
    end
end

finally
    isempty(CM_WORKERS) || Distributed.rmprocs(CM_WORKERS; waitfor = 30)
end
