using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC,
    StructureFunctionTypes as SFT, StructureFunctionObjects as SFO, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using StaticArrays: StaticArrays as SA
using Distances: Distances as DI
using OhMyThreads: OhMyThreads
using FFTW: FFTW
using Distributed: Distributed
using Random: Random

Random.seed!(11)
const PE_N, PE_T, PE_NB, PE_NV = 40, 3, 6, 5
const PE_OP = SFT.L2SFType()
const PE_RAW = SFO.StructureFunctionSumsAndCounts
const PE_X, PE_U = rand(2, PE_N), randn(2, PE_N)
const PE_XB, PE_UB = rand(2, PE_N, PE_T), randn(2, PE_N, PE_T)
const PE_X1, PE_U1 = reshape(sort(rand(PE_N)), 1, PE_N), randn(1, PE_N)
# Longitude and latitude in degrees, the metric in metres.
const PE_XS = permutedims(hcat(300 .* rand(PE_N) .- 150, 100 .* rand(PE_N) .- 50))
const PE_SPHERE = DI.Haversine(6.371e6)
const PE_W = 0.5 .+ rand(PE_N)
const PE_BINS = collect(range(0.0, 1.0; length = PE_NB + 1))
const PE_SBINS = collect(range(0.0, 9.0e6; length = PE_NB + 1))
const PE_VBINS = collect(range(-3.0, 3.0; length = PE_NV + 1))
const PE_ABINS = collect(range(prevfloat(0.0), π; length = 4))
const PE_NA = length(PE_ABINS) - 1
const PE_AX = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
const PE_F = MF.Fields(vectors = (PE_U,), scalars = (randn(PE_N),))
const PE_MIXED = SFT.MixedSFType{1, 0, 1}()
const PE_FFT = SB.FastFourierTransformSpectralBackend()
const PE_GS = SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true))
const PE_GU = reshape(Float64[sin(d + 3i + 7j) for d in 1:2, i in 1:8, j in 1:8], 2, :)
const PE_GB = collect(range(0.0, 0.5; length = PE_NB + 1))
const PE_HX = permutedims(hcat([2π * (i * 0.6180339887498949 % 1) for i in 1:PE_N],
    π / 2 .- acos.(clamp.(range(-0.95, 0.95; length = PE_N), -1, 1))))
const PE_HU = Float64[sin(d + 2i) for d in 1:2, i in 1:PE_N]
const PE_NODES = SF.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), 16)
const PE_F1 = MF.Fields(vectors = (PE_U1,), scalars = (randn(PE_N),))
# Bins short enough for a cull grid to exclude pairs on the unit square.
const PE_TBINS = collect(range(0.0, 0.25; length = PE_NB + 1))

const PE_BACKENDS = (
    threaded = CB.ThreadedBackend(),
    distributed = CB.DistributedBackend(),
    dist_threaded = CB.DistributedBackend(CB.ThreadedBackend()),
)

"""The six single-pass invariants of a public result stacked into one `(sums, counts)` pair."""
pe_stacked(r) = (stack([r[k].sums for k in keys(r) if k !== :helmholtz]),
                 stack([r[k].counts for k in keys(r) if k !== :helmholtz]))

"""`f!(s, c)` run twice into the buffers `s`, `c`, returned: a method that overwrites its buffers halves the answer."""
pe_twice(f!, s, c) = (f!(s, c); f!(s, c); (s, c))

# As the entry a user calls, returning `(sums, counts)`: one call per method of the threaded extension (allocating and
# in-place forms, flat and spherical geometries and auxiliary axes reach different methods), and one per mechanism of
# the Distributed extension (each partial family's reduction, the batch executor plain and culled, the split lag sweep,
# the harmonic sum, and the in-place forwarding of geometry and weights).
const PE_ROUTES = (
    ("point", (:threaded, :distributed),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_X, PE_U, PE_BINS, PE_RAW; backend = be);
            (r.sums, r.counts))),
    ("point on a sphere, weighted, in place", (:threaded, :distributed),
     be -> pe_twice(zeros(PE_NB), zeros(PE_NB)) do s, c
         SFC.calculate_structure_function!(s, c, PE_OP, PE_XS, PE_U, PE_SBINS; backend = be,
                                           distance_metric = PE_SPHERE, weights = PE_W)
     end),
    ("point on a line", (:threaded,),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_X1, PE_U1, PE_BINS, PE_RAW; backend = be);
            (r.sums, r.counts))),
    ("joint over the angle", (:threaded, :distributed),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_X, PE_U, PE_BINS, PE_ABINS; backend = be,
                                                 second_axis = PE_AX); (r.sums, r.counts))),
    ("joint on a sphere, in place", (:threaded,),
     be -> pe_twice(zeros(PE_NB, PE_NV), zeros(Int, PE_NB, PE_NV)) do s, c
         SFC.calculate_structure_function!(s, c, PE_OP, PE_XS, PE_U, PE_SBINS, PE_VBINS; backend = be,
                                           distance_metric = PE_SPHERE)
     end),
    ("single pass", (:threaded, :distributed),
     be -> pe_stacked(SFC.calculate_structure_functions_single_pass(PE_X, PE_U, PE_BINS; backend = be))),
    ("single pass on a sphere, in place", (:threaded, :distributed),
     be -> pe_twice(zeros(SFC.SINGLE_PASS_N, PE_NB), zeros(Int, SFC.SINGLE_PASS_N, PE_NB)) do s, c
         SFC.calculate_structure_functions_single_pass!(s, c, PE_XS, PE_U, PE_SBINS; backend = be,
                                                        distance_metric = PE_SPHERE)
     end),
    ("single pass 2D", (:threaded, :distributed),
     be -> pe_stacked(SFC.calculate_structure_functions_single_pass_2d(PE_X, PE_U, PE_BINS, PE_VBINS;
                                                                       backend = be))),
    ("single pass 2D on a sphere, in place", (:threaded,),
     be -> pe_twice(zeros(SFC.SINGLE_PASS_N, PE_NB, PE_NV), zeros(Int, SFC.SINGLE_PASS_N, PE_NB, PE_NV)) do s, c
         SFC.calculate_structure_functions_single_pass_2d!(s, c, PE_XS, PE_U, PE_SBINS, PE_VBINS; backend = be,
                                                           distance_metric = PE_SPHERE)
     end),
    ("multi-field", (:threaded, :distributed),
     be -> (r = SFC.calculate_structure_function(PE_MIXED, PE_X, PE_F, PE_BINS, PE_RAW; backend = be);
            (r.sums, r.counts))),
    ("multi-field on a sphere", (:threaded,),
     be -> (r = SFC.calculate_structure_function(PE_MIXED, PE_XS, PE_F, PE_SBINS, PE_RAW; backend = be,
                                                 distance_metric = PE_SPHERE); (r.sums, r.counts))),
    ("multi-field on a line", (:threaded,),
     be -> (r = SFC.calculate_structure_function(PE_MIXED, PE_X1, PE_F1, PE_BINS, PE_RAW; backend = be);
            (r.sums, r.counts))),
    ("multi-field of one vector field", (:threaded,),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_X, MF.Fields(vectors = (PE_U,)), PE_BINS, PE_RAW;
                                                 backend = be); (r.sums, r.counts))),
    ("moment tensor", (:threaded, :distributed),
     be -> (r = SFC.calculate_structure_function_tensor(Val(2), PE_X, PE_U, PE_BINS, SFO.StructureFunctionTensorSumsAndCounts;
                                                        backend = be); (r.sums, r.counts))),
    ("auxiliary axes", (:threaded, :distributed),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_XB, PE_UB, PE_BINS, PE_RAW; backend = be);
            (r.sums, r.counts))),
    ("auxiliary axes, culled per slice", (:threaded, :distributed),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_XB, PE_UB, PE_TBINS, PE_RAW; backend = be,
                                                 culling = SFC.AlwaysCulling()); (r.sums, r.counts))),
    ("auxiliary axes joint over the angle", (:threaded, :dist_threaded),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_XB, PE_UB, PE_BINS, PE_ABINS; backend = be,
                                                 second_axis = PE_AX); (r.sums, r.counts))),
    ("auxiliary axes single pass", (:threaded,),
     be -> pe_stacked(SFC.calculate_structure_functions_single_pass(PE_XB, PE_UB, PE_BINS; backend = be))),
    ("auxiliary axes single pass 2D", (:threaded,),
     be -> pe_stacked(SFC.calculate_structure_functions_single_pass_2d(PE_XB, PE_UB, PE_BINS, PE_VBINS;
                                                                       backend = be))),
    ("harmonic direct sum", (:threaded, :distributed),
     be -> (r = SFC.calculate_structure_function(PE_OP, PE_HX, PE_HU, PE_NODES, SB.DirectSumSpectralBackend(),
                                                 PE_RAW; backend = be); (r.sums, r.counts))),
    # Two Distributed workers split the schedule's one slab pair's lags in two.
    ("gridded lag sweep", (:threaded, :distributed),
     be -> pe_twice(zeros(PE_NB), zeros(Int, PE_NB)) do s, c
         SFC.gridded_lag_sweep!(s, c, PE_OP, PE_GU, PE_GS, PE_GB, Val(2), Val(1), Val(0); backend = be)
     end),
    ("gridded transform", (:threaded,),
     be -> pe_twice(zeros(PE_NB), zeros(Int, PE_NB)) do s, c
         SFC.gridded_sweep!(s, c, PE_OP, PE_GU, PE_GS, PE_GB, Val(2), Val(1), Val(0), PE_FFT; backend = be)
     end),
)

"""Counts exactly when both are integer; a weighted or kernel-weighted count is a mass, compared like a sum, to the
`1e-12` of the largest value that a different summation order allows."""
function pe_agrees(got, ref)
    size(got) == size(ref) || return false
    eltype(got) <: Integer && eltype(ref) <: Integer && return got == ref
    return maximum(abs, got .- ref) <= 1e-12 * max(maximum(abs, ref), 1e-12)
end

const PE_WORKERS = Distributed.nprocs() == 1 ? Distributed.addprocs(2) : Int[]
try

Distributed.@everywhere using StructureFunctions: StructureFunctions
Distributed.@everywhere using OhMyThreads: OhMyThreads

# Every cell reproduces its route's serial reference, which bins at least one pair.
Test.@testset "every threaded and Distributed method reproduces the serial answer" begin
    for (name, backends, run) in PE_ROUTES
        ref = run(CB.SerialBackend())
        any(!iszero, ref[2]) || error("the serial reference of $name bins no pair")
        for b in backends
            got = run(PE_BACKENDS[b])
            Test.@test (name, b, pe_agrees(got[1], ref[1]) && pe_agrees(got[2], ref[2])) == (name, b, true)
        end
    end
end

# An unbounded last bin cannot be culled, so an explicit AlwaysCulling() raising shows the policy reached the batch
# kernels through each in-place Distributed batch method.
Test.@testset "an explicit culling policy reaches the distributed batch kernels" begin
    be, always = PE_BACKENDS.distributed, SFC.AlwaysCulling()
    open = SF.InfPaddedBinEdges(PE_BINS)
    no, NI = PE_NB + 2, SFC.SINGLE_PASS_N
    Test.@test_throws ArgumentError SFC.calculate_structure_function_batch!(zeros(no, PE_T), zeros(Int, no, PE_T),
        PE_OP, PE_XB, PE_UB, open; backend = be, culling = always)
    Test.@test_throws ArgumentError SFC.calculate_structure_function_2d_batch!(zeros(no, PE_NV, PE_T),
        zeros(Int, no, PE_NV, PE_T), PE_OP, PE_XB, PE_UB, open, PE_VBINS; backend = be, culling = always)
    Test.@test_throws ArgumentError SFC.calculate_structure_functions_single_pass_batch!(zeros(NI, no, PE_T),
        zeros(Int, NI, no, PE_T), PE_XB, PE_UB, open; backend = be, culling = always)
    Test.@test_throws ArgumentError SFC.calculate_structure_functions_single_pass_2d_batch!(zeros(NI, no, PE_NV, PE_T),
        zeros(Int, NI, no, PE_NV, PE_T), PE_XB, PE_UB, open, PE_VBINS; backend = be, culling = always)
    Test.@test_throws ArgumentError SFC.calculate_structure_function!(zeros(no, PE_T), zeros(Int, no, PE_T),
        PE_OP, PE_XB, PE_UB, open; backend = be, culling = always)
    Test.@test_throws ArgumentError SFC.calculate_structure_function!(zeros(no, PE_NV, PE_T),
        zeros(Int, no, PE_NV, PE_T), PE_OP, PE_XB, PE_UB, open, PE_VBINS; backend = be, culling = always)
end

finally
    isempty(PE_WORKERS) || Distributed.rmprocs(PE_WORKERS; waitfor = 30)
end

# A worker's share of the outer indices, on the inner backend a Distributed or MPI worker runs it on.
Test.@testset "the shares of every partial family add to the culled whole sweep" begin
    N, k = 60, 3
    x, u = rand(2, N), randn(2, N)
    f = MF.Fields(vectors = (u,))
    xv, uv = (x[1, :], x[2, :]), (u[1, :], u[2, :])
    g2 = SF.HelperFunctions.FlatGeometry{2}()
    family = (
        ("1d", (be, db, sh, kw) -> (r = SFC._partial_sums_counts(be, PE_OP, xv, uv, db, sh, Int; kw...);
                                    (r.sums, r.counts))),
        ("joint", (be, db, sh, kw) -> SFC._partial_2d_sums_counts(be, PE_OP, xv, uv, db, PE_VBINS, sh, Int; kw...)),
        ("sp1d", (be, db, sh, kw) -> SFC._partial_single_pass_1d(be, x, u, db, sh, Int; kw...)),
        ("sp2d", (be, db, sh, kw) -> SFC._partial_single_pass_2d(be, x, u, db, PE_VBINS, sh, Int; kw...)),
        ("tensor", (be, db, sh, kw) -> SFC.tensor_partial(be, Val(2), SFC.PointField{2}(), x, u, db, sh, Int; kw...)),
        ("multi-field", (be, db, sh, kw) -> SFC.field_partial(be, PE_OP, x, f, db, sh, Int; kw...)),
    )
    db = collect(range(0.0, 0.1; length = 7))
    for (name, part) in family
        whole = part(CB.SerialBackend(), db, (1, 1), (; geometry = g2, culling = SFC.NoCulling()))
        for be in (CB.SerialBackend(), CB.ThreadedBackend())
            shares = [part(be, db, (w, k), (; geometry = g2, culling = SFC.AutoCulling())) for w in 1:k]
            Test.@test (name, be, sum(s[2] for s in shares) == whole[2]) == (name, be, true)
            Test.@test (name, be, isapprox(sum(s[1] for s in shares), whole[1]; rtol = 1e-12)) == (name, be, true)
        end
    end
end
