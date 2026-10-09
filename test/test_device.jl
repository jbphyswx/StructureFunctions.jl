using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO, HelperFunctions as SFH, MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using Distances: Distances as DI
using Random: Random

const DV_DEV, DV_SER = CB.GPUBackend(KA.CPU()), CB.SerialBackend()
const DV_EXT = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)
const DV_CAP = DV_EXT.SF_GPU_MAX_BINS
const DV_RAW, DV_TRAW = SFO.StructureFunctionSumsAndCounts, SFO.StructureFunctionTensorSumsAndCounts
const DV_OP = SFT.L2SFType()
const DV_ALWAYS = SFC.AlwaysCulling()

Random.seed!(20261009)
# Two 128-point tiles, the second partial.
const DV_N, DV_B = 150, 3
const DV_X, DV_U = rand(2, DV_N), randn(2, DV_N)
const DV_XB, DV_UB = rand(2, DV_N, DV_B), randn(2, DV_N, DV_B)
const DV_U17 = randn(2, DV_N, 17)
const DV_X3, DV_U3 = rand(3, DV_N), randn(3, DV_N)
const DV_X1, DV_U1 = rand(1, DV_N), randn(1, DV_N)
const DV_W = 0.5 .+ rand(DV_N)
const DV_F = MF.Fields(vectors = (randn(2, DV_N), randn(2, DV_N)), scalars = (randn(DV_N), randn(DV_N)))
const DV_F1 = MF.Fields(vectors = (DV_U1,), scalars = (randn(DV_N),))
const DV_BINS = collect(range(0.0, 1.0; length = 9))
const DV_NB = length(DV_BINS) - 1
const DV_VBINS = collect(range(-3.0, 3.0; length = 7))
const DV_ABINS = collect(range(prevfloat(0.0), π; length = 5))
const DV_AX = SFC.SeparationAngleAxis(SA.SVector(1.0, 0.0))
# Bins short enough for a cull grid to exclude pairs on the unit square.
const DV_TIGHT = collect(range(0.0, 0.08; length = 9))
# Longitude and latitude in degrees, bins in metres short enough to cull on the Earth.
const DV_XS = permutedims(hcat(300 .* rand(DV_N) .- 150, 100 .* rand(DV_N) .- 50))
const DV_SPHERE = DI.Haversine(6.371e6)
const DV_SBINS = collect(range(0.0, 2.0e6; length = 9))
const DV_WIDE = 13
const DV_XW, DV_XWB, DV_UWB = rand(DV_WIDE, DV_N), rand(DV_WIDE, DV_N, DV_B), randn(DV_WIDE, DV_N, DV_B)
const DV_WBINS = collect(range(0.0, 2.0; length = 7))
const DV_WVBINS = collect(range(0.0, 8.0; length = 6))
const DV_LATTICE = reduce(hcat, [[0.1 * i, 0.1 * j] for i in 0:5 for j in 0:4])
const DV_LATTICE_U = randn(2, size(DV_LATTICE, 2))

"""A result as host arrays `(sums, counts)`: the six single-pass invariants stacked."""
dv_arrays(r::Tuple) = map(Array, r)
dv_arrays(r::NamedTuple) = (stack([Array(r[k].sums) for k in keys(r) if k !== :helmholtz]),
                            stack([Array(r[k].counts) for k in keys(r) if k !== :helmholtz]))
dv_arrays(r) = (Array(r.sums), Array(r.counts))

"""`f!(s, c)` run twice into the zeroed buffers `s`, `c`: a method that overwrites its buffers halves the answer."""
dv_twice(f!, s, c) = (f!(s, c); f!(s, c); (s, c))

"""Counts exactly when both are integer, a pair mass to rounding; sums to rounding at the result's scale."""
function dv_agrees((s, c), (rs, rc))
    size(s) == size(rs) && size(c) == size(rc) || return false
    counts = eltype(c) <: Integer && eltype(rc) <: Integer ? c == rc : isapprox(c, rc; rtol = 1e-10)
    return counts && isapprox(s, rs; rtol = 1e-10, atol = 1e-12 * max(maximum(abs, rs; init = 0.0), 1.0))
end

sp(x, u, bins; kw...) = SFC.calculate_structure_functions_single_pass(x, u, bins; kw...)
sp2(x, u, bins, vbins; kw...) = SFC.calculate_structure_functions_single_pass_2d(x, u, bins, vbins; kw...)
sf(op, x, u, bins; kw...) = SFC.calculate_structure_function(op, x, u, bins, DV_RAW; kw...)

# One call per device kernel family and per regime that selects another kernel, digitizer or launch; each on the
# device equals the same call on the serial backend.
const DV_ROUTES = (
    # point families
    ("point", be -> sf(DV_OP, DV_X, DV_U, DV_BINS; backend = be)),
    ("point, in place twice", be -> dv_twice(zeros(DV_NB), zeros(UInt32, DV_NB)) do s, c
        SFC.calculate_structure_function!(s, c, DV_OP, DV_X, DV_U, DV_BINS; backend = be)
    end),
    ("point, log edges", be -> sf(DV_OP, DV_X, DV_U, SF.LogBinEdges(0.01, 1.0, 9); backend = be)),
    ("point at the bin cap", be -> sf(DV_OP, DV_X, DV_U, collect(range(0.0, 1.0; length = DV_CAP + 1)); backend = be)),
    ("point past the bin cap", be -> sf(DV_OP, DV_X, DV_U, collect(range(0.0, 1.0; length = DV_CAP + 2)); backend = be)),
    ("point, weighted", be -> SFC.calculate_structure_function(DV_OP, DV_X, DV_U, DV_BINS, Float64, DV_RAW; backend = be,
                                                               weights = DV_W)),
    ("point on a sphere, culled", be -> sf(DV_OP, DV_XS, DV_U, DV_SBINS; backend = be, distance_metric = DV_SPHERE,
                                           culling = DV_ALWAYS)),
    ("point, separations on the edges", be -> sf(DV_OP, DV_LATTICE, DV_LATTICE_U, collect(0.0:0.1:0.5); backend = be)),
    ("signed transverse T3", be -> sf(SFT.T3SFType(), DV_X, DV_U, DV_BINS; backend = be)),
    ("projected (0, 3) on a reference basis in 3-D", be -> sf(SFT.ProjectedStructureFunctionType{0, 3}(
        SFH.ReferenceAxisTransverseBasis(LA.normalize(SA.SVector(1.0, sqrt(2.0), sqrt(3.0))))), DV_X3, DV_U3, DV_BINS;
        backend = be)),
    ("joint", be -> SFC.calculate_structure_function(DV_OP, DV_X, DV_U, DV_BINS, DV_VBINS; backend = be)),
    ("joint over the angle", be -> SFC.calculate_structure_function(DV_OP, DV_X, DV_U, DV_BINS, DV_ABINS; backend = be,
                                                                    second_axis = DV_AX)),
    ("joint, log distance and padded value edges", be -> SFC.calculate_structure_function(SFT.L3SFType(), DV_X, DV_U,
        SF.LogBinEdges(0.02, 1.0, 9), SF.InfPaddedBinEdges(SF.LinearBinEdges(range(-1.0, 2.0; length = 23)));
        backend = be)),
    ("joint, weighted", be -> SFC.calculate_structure_function(DV_OP, DV_X, DV_U, DV_BINS, DV_VBINS, Float64;
                                                               backend = be, weights = DV_W)),
    ("joint past the shared fit, weighted", be -> SFC.calculate_structure_function(DV_OP, DV_X, DV_U, DV_BINS,
        collect(range(-3.0, 3.0; length = 601)), Float64; backend = be, weights = DV_W)),
    ("joint, culled", be -> SFC.calculate_structure_function(DV_OP, DV_X, DV_U, DV_TIGHT, DV_VBINS; backend = be,
                                                             culling = DV_ALWAYS)),
    ("single pass", be -> sp(DV_X, DV_U, DV_BINS; backend = be)),
    ("single pass past the bin cap", be -> sp(DV_X, DV_U, collect(range(0.0, 1.0; length = DV_CAP + 2)); backend = be)),
    ("single pass, weighted", be -> SFC.calculate_structure_functions_single_pass(DV_X, DV_U, DV_BINS, Float64;
                                                                                  backend = be, weights = DV_W)),
    ("single pass, separations on the edges", be -> sp(DV_LATTICE, DV_LATTICE_U, collect(0.0:0.1:0.5); backend = be)),
    ("single pass 2D", be -> sp2(DV_X, DV_U, DV_BINS, DV_VBINS; backend = be)),
    ("single pass 2D, log distance and padded value edges", be -> sp2(DV_X, DV_U, SF.LogBinEdges(0.01, 1.0, 9),
        SF.InfPaddedBinEdges(SF.LinearBinEdges(range(-0.5, 1.5; length = 7))); backend = be)),
    ("single pass 2D past the distance cap", be -> sp2(DV_X, DV_U, collect(range(0.0, 1.0; length = DV_CAP + 2)),
                                                       DV_VBINS; backend = be)),
    ("single pass 2D, Float32 edges on Float64 data", be -> sp2(DV_X, DV_U, collect(range(0.0f0, 1.0f0; length = 9)),
                                                                collect(range(-3.0f0, 3.0f0; length = 7)); backend = be)),
    ("single pass 2D, culled", be -> sp2(DV_X, DV_U, DV_TIGHT, DV_VBINS; backend = be, culling = DV_ALWAYS)),
    ("single pass 2D on a sphere, culled", be -> sp2(DV_XS, DV_U, DV_SBINS, DV_VBINS; backend = be,
                                                     distance_metric = DV_SPHERE, culling = DV_ALWAYS)),
    ("tensor", be -> SFC.calculate_structure_function_tensor(Val(2), DV_X, DV_U, DV_BINS, DV_TRAW; backend = be)),
    ("tensor over the angle", be -> SFC.calculate_structure_function_tensor(Val(2), DV_X, DV_U, DV_BINS, DV_ABINS;
                                                                            second_axis = DV_AX, backend = be)),
    ("odd tensor over shared positions, in place twice", be -> dv_twice(zeros(2, 2, 2, DV_NB, DV_B),
                                                                        zeros(Int, DV_NB, DV_B)) do s, c
        SFC.calculate_structure_function_tensor!(s, c, Val(3), DV_X, DV_UB, DV_BINS; backend = be)
    end),
    ("multi-field, a vector with a scalar", be -> sf(SFT.MixedSFType{1, 0, 2}(), DV_X, DV_F, DV_BINS; backend = be)),
    ("multi-field on a sphere, two vectors", be -> sf(SFT.VectorDotSFType(1, 2), DV_XS, DV_F, DV_SBINS; backend = be,
                                                      distance_metric = DV_SPHERE)),
    ("line", be -> sf(DV_OP, DV_X1, DV_U1, DV_BINS; backend = be)),
    ("line, weighted", be -> SFC.calculate_structure_function(DV_OP, DV_X1, DV_U1, DV_BINS, Float64, DV_RAW;
                                                              backend = be, weights = DV_W)),
    ("line, multi-field", be -> sf(SFT.MixedSFType{1, 0, 1}(), DV_X1, DV_F1, DV_BINS; backend = be)),
    # batches
    ("batch over shared positions, more slices than a field strip", be -> sf(DV_OP, DV_X, DV_U17, DV_BINS;
                                                                            backend = be)),
    ("batch keeps two trailing axes", be -> sf(DV_OP, DV_X, reshape(DV_UB[:, :, 1:2], 2, DV_N, 2, 1), DV_BINS;
                                               backend = be)),
    ("batch over per-slice positions", be -> sf(DV_OP, DV_XB, DV_UB, DV_BINS; backend = be)),
    ("batch, in place twice", be -> dv_twice(zeros(DV_NB, DV_B), zeros(UInt32, DV_NB, DV_B)) do s, c
        SFC.calculate_structure_function_batch!(s, c, DV_OP, DV_XB, DV_UB, DV_BINS; backend = be)
    end),
    ("joint batch over the angle, per-slice positions", be -> SFC.calculate_structure_function(DV_OP, DV_XB, DV_UB,
        DV_BINS, DV_ABINS; backend = be, second_axis = DV_AX)),
    ("single pass 2D batch over shared positions", be -> sp2(DV_X, DV_UB, DV_BINS, DV_VBINS; backend = be)),
    ("batch past the bin cap, shared positions", be -> sf(DV_OP, DV_X, DV_UB,
                                                          collect(range(0.0, 1.0; length = DV_CAP + 2)); backend = be)),
    ("single pass batch past the bin cap, per-slice positions", be -> sp(DV_XB, DV_UB,
        collect(range(0.0, 1.0; length = DV_CAP + 2)); backend = be)),
    ("single pass 2D batch past the shared fit, shared positions", be -> sp2(DV_X, DV_UB,
        collect(range(0.01, 1.5; length = 51)), collect(range(-0.5, 1.5; length = 53)); backend = be)),
    ("single pass 2D batch past the shared fit, per-slice positions", be -> sp2(DV_XB, DV_UB,
        collect(range(0.01, 1.5; length = 51)), collect(range(-0.5, 1.5; length = 53)); backend = be)),
    ("batch, culled over shared positions", be -> sf(DV_OP, DV_X, DV_UB, DV_TIGHT; backend = be, culling = DV_ALWAYS)),
    ("single pass batch, culled per slice, weighted", be -> SFC.calculate_structure_functions_single_pass(DV_XB, DV_UB,
        DV_TIGHT, Float64; backend = be, weights = DV_W, culling = DV_ALWAYS)),
    # Four tiles of 13 Float64 coordinates exceed every device's static shared memory: the wide kernels run.
    ("wide batch, shared positions", be -> sf(DV_OP, DV_XW, DV_UWB, DV_WBINS; backend = be)),
    ("wide single pass 2D batch, per-slice positions", be -> sp2(DV_XWB, DV_UWB, DV_WBINS, DV_WVBINS; backend = be)),
)

Test.@testset "every device kernel family and regime gives the serial answer" begin
    for (name, run) in DV_ROUTES
        ref = dv_arrays(run(DV_SER))
        sum(ref[2]) > 0 || error("the serial reference of $name bins no pair")
        Test.@test (name, dv_agrees(dv_arrays(run(DV_DEV)), ref)) == (name, true)
    end
end

# A device backend holding a configured KernelAbstractions backend gives the default device backend's answer.
Test.@testset "a configured device backend gives the default's answer" begin
    configured = CB.GPUBackend(KA.CPU(; static = true))
    Test.@test dv_agrees(dv_arrays(sf(DV_OP, DV_X, DV_U, DV_BINS; backend = configured)),
                         dv_arrays(sf(DV_OP, DV_X, DV_U, DV_BINS; backend = DV_DEV)))
end

# Each accumulation mode of the single pass 2D histogram, weighted, gives the serial answer fresh and through a reused
# workspace, which then serves an unweighted call.
Test.@testset "the single pass 2D accumulation modes, through a reused workspace" begin
    for (nd, nv) in ((10, 8), (30, 30), (60, 60))
        dist = collect(range(0.0, 1.5; length = nd + 1))
        vals = collect(range(-3.0, 3.0; length = nv + 1))
        ws = SFC.GPUSFWorkspace(KA.CPU(), dist, vals; kind = :single_pass_2d)
        weighted(be; kw...) = dv_arrays(SFC.calculate_structure_functions_single_pass_2d(DV_X, DV_U, dist, vals, Float64;
                                                                                         backend = be, weights = DV_W, kw...))
        ref = weighted(DV_SER)
        agree = all(w -> dv_agrees(weighted(DV_DEV; workspace = w), ref), (nothing, ws, ws)) &&
                dv_agrees(dv_arrays(sp2(DV_X, DV_U, dist, vals; backend = DV_DEV, workspace = ws)),
                          dv_arrays(sp2(DV_X, DV_U, dist, vals; backend = DV_SER)))
        Test.@test (nd, nv, agree) == (nd, nv, true)
    end
end

# A joint workspace compiled at the widest histogram that fits gives the serial answer, one narrower than the histogram
# is refused, and a workspace built from bins alone serves two coordinate widths.
Test.@testset "joint workspaces: compile width and coordinate width" begin
    widest = SFC.joint2d_smem_max(KA.CPU(), 2, 2, Float64, Float64, UInt32)
    joint(x, u, be; kw...) = dv_arrays(SFC.calculate_structure_function(DV_OP, x, u, DV_BINS, DV_VBINS; backend = be,
                                                                        kw...))
    wide = SFC.GPUSFWorkspace(KA.CPU(), DV_BINS, DV_VBINS; joint2d_compile_cells = widest)
    Test.@test dv_agrees(joint(DV_X, DV_U, DV_DEV; workspace = wide), joint(DV_X, DV_U, DV_SER))
    Test.@test_throws ArgumentError SFC.GPUSFWorkspace(KA.CPU(), DV_BINS, DV_VBINS;
                                                       joint2d_compile_cells = DV_NB * (length(DV_VBINS) - 1) - 1)
    ws = SFC.GPUSFWorkspace(KA.CPU(), DV_BINS, DV_VBINS)
    Test.@test all(((x, u),) -> dv_agrees(joint(x, u, DV_DEV; workspace = ws), joint(x, u, DV_SER)),
                   ((DV_X, DV_U), (DV_X3, DV_U3)))
end

# A culling workspace gives the serial answer on reuse, after refresh!, for positions in a new array and after release!,
# refuses other bins, and serves a batch; a single-pass workspace holds log edges.
Test.@testset "workspaces give the answer computed without one" begin
    ws = SFC.GPUSFWorkspace(KA.CPU(), DV_TIGHT)
    run(be, x, u; kw...) = (s = zeros(DV_NB); c = zeros(UInt32, DV_NB);
                            SFC.calculate_structure_function!(s, c, DV_OP, x, u, DV_TIGHT; backend = be, kw...); (s, c))
    serial_answer(x, u) = dv_agrees(run(DV_DEV, x, u; workspace = ws), run(DV_SER, x, u))
    x = copy(DV_X)
    u2 = randn(2, DV_N)
    Test.@test serial_answer(x, DV_U)
    Test.@test serial_answer(x, u2)
    x .= rand(2, DV_N)
    SFC.refresh!(ws)
    Test.@test serial_answer(x, u2)
    Test.@test_throws ArgumentError run(DV_DEV, x, u2; workspace = SFC.GPUSFWorkspace(KA.CPU(), DV_BINS))
    xs = 0.1 .* x
    run(DV_DEV, xs, u2; workspace = ws)
    Test.@test serial_answer(xs, u2)
    SFC.release!(ws)
    Test.@test serial_answer(x, u2)
    Test.@test dv_agrees(dv_arrays(sf(DV_OP, DV_X, DV_UB, DV_TIGHT; backend = DV_DEV, workspace = ws)),
                         dv_arrays(sf(DV_OP, DV_X, DV_UB, DV_TIGHT; backend = DV_SER)))
    logb = SF.LogBinEdges(0.01, 1.0, 9)
    Test.@test dv_agrees(dv_arrays(sp(DV_X, DV_U, logb; backend = DV_DEV,
                                      workspace = SFC.GPUSFWorkspace(KA.CPU(), logb; kind = :single_pass))),
                         dv_arrays(sp(DV_X, DV_U, logb; backend = DV_SER)))
end

# On the device a Float32 line field on a large offset keeps its moments to Float32 rounding of the field scale.
Test.@testset "a Float32 line field on a large offset keeps its moments on the device" begin
    x32, u32 = Float32.(DV_X1), Float32.(1000 .+ 0.01 .* DV_U1)
    bins32 = Float32.(DV_BINS)
    run(be) = dv_arrays(SFC.calculate_structure_function(SFT.L3SFType(), x32, u32, bins32, DV_RAW; backend = be))
    s, c = run(DV_DEV)
    xv, uv, bins = vec(Float64.(x32)), vec(Float64.(u32)), Float64.(bins32)
    rs, rc = dv_arrays(SFC.calculate_structure_function(SFT.L3SFType(), reshape(xv, 1, :), reshape(uv, 1, :), bins,
                                                        DV_RAW; backend = DV_SER))
    # Σ|δu|³ per bin, the scale a sum of values of either sign is compared at.
    scale = zeros(DV_NB)
    for i in 1:(DV_N - 1), j in (i + 1):DV_N
        b = searchsortedfirst(bins, abs(xv[j] - xv[i])) - 1
        1 <= b <= DV_NB && (scale[b] += abs(uv[j] - uv[i])^3)
    end
    Test.@test c == rc
    Test.@test all(abs.(s .- rs) .<= 1e-3 .* scale)
end

# On a sphere each pair's direction lives in its own frame, so the device refuses an angle to a fixed axis.
Test.@testset "the device refuses an angle axis on a sphere" begin
    for u in (DV_U, DV_UB)
        Test.@test_throws ArgumentError SFC.calculate_structure_function(DV_OP, DV_XS, u, DV_SBINS, DV_ABINS;
            backend = DV_DEV, distance_metric = DV_SPHERE, second_axis = DV_AX)
    end
end

struct _EdgeAdaptProbe end
KA.Adapt.adapt_storage(::_EdgeAdaptProbe, a::Array) = view(a, :)
_arrays_adapted(p::SF.InfPaddedBinEdges) = _arrays_adapted(p.edges)
_arrays_adapted(p::SF.BucketedBinEdges) = p.edges isa SubArray && p.cells isa SubArray
_arrays_adapted(p) = p.edges isa SubArray

# Device helpers: digitizer plans adapt their arrays, the device cull grid is the host grid, the work list covers every
# in-range pair, and the device sort orders as sortperm does across its 2048-point tile.
Test.@testset "the device helpers" begin
    gen = SF.BinEdges(Float32[0, 0.2, 0.5, 1])
    bucket = SF.digitize_plan(SF.LogBinEdges(0.01f0, 1.0f0, 9))
    table = DV_EXT._device_plan(SF.LogBinEdges(0.01f0, 1.0f0, 9), Val(:sf1d))
    padded_table = DV_EXT._device_plan(SF.InfPaddedBinEdges(SF.LogBinEdges(0.01f0, 1.0f0, 9)), Val(:sf1d))
    plans = (gen, bucket, table, SF.InfPaddedBinEdges(gen), SF.InfPaddedBinEdges(bucket), padded_table)
    adapted = map(p -> KA.Adapt.adapt(_EdgeAdaptProbe(), p), plans)
    same_bins(p, a) = collect(a) == collect(p) &&
                      all(q -> searchsortedfirst(a, q) == searchsortedfirst(p, q), 0.0f0:0.01f0:1.1f0)
    Test.@test all(_arrays_adapted, adapted)
    Test.@test all(splat(same_bins), zip(plans, adapted))

    for (D, N, frac) in ((2, 300, 0.06), (3, 250, 0.15), (5, 200, 0.4))
        x = rand(D, N) .* ntuple(d -> d, D)
        geom = SFH.FlatGeometry{D}()
        bins = collect(range(0.0, frac; length = 5))
        host = SFC.cull_grid_for(ntuple(d -> view(x, d, :), D), geom, bins, DV_ALWAYS)
        dev = DV_EXT._gpu_cull_grid(KA.CPU(), x, geom, SFC.cull_cutoff_for(geom, bins, DV_ALWAYS), DV_ALWAYS)
        Test.@test (D, dev.origin, dev.dims, dev.inv_h, dev.cell_ids, dev.run_starts, dev.perm, dev.offsets) ==
                   (D, host.origin, host.dims, host.inv_h, host.cell_ids, host.run_starts, host.perm, host.offsets)
    end

    for (D, N, frac, tile) in ((2, 300, 0.06, 128), (3, 300, 0.15, 64), (1, 300, 0.02, 32))
        x = rand(D, N)
        xc = ntuple(d -> x[d, :], D)
        cut = SFC.cull_cutoff(SFH.FlatGeometry{D}(), frac)
        grid = SFC.build_cell_grid(xc, cut, 2)
        xp = SFC.apply_perm(xc, grid.perm)
        wl = SFC.gpu_tile_worklist(grid, N, tile)
        n_tiles = cld(N, tile)
        canonical = issorted(wl.pairs) && allunique(wl.pairs) &&
                    all(k -> (1 <= SFC.tile_for(wl, k)[1] <= SFC.tile_for(wl, k)[2] <= n_tiles), 1:SFC.n_pair_blocks(wl))
        listed = Set(wl.pairs)
        T = eltype(wl.pairs)
        missed = 0
        for i in 1:(N - 1), j in (i + 1):N
            sum(d -> (xp[d][i] - xp[d][j])^2, 1:D) <= cut^2 || continue
            ti, tj = minmax(cld(i, tile), cld(j, tile))
            SFC.pack_tile_pair(T(ti), T(tj), T(n_tiles)) in listed || (missed += 1)
        end
        Test.@test (D, tile, canonical, missed) == (D, tile, true, 0)
    end

    # Every key type, and 2049 points to cross the sort's fixed 2048-point tile.
    pool = [0.0, -0.0, NaN, -NaN, Inf, -Inf, 1.0, 1.0, -2.5, randn(8)...]
    sort_keys = (zeros(0), rand(Int32(-3):Int32(40), 7), rand(typemin(Int64):typemax(Int64), 7),
                 Float32.(rand(pool, 2049)), rand(pool, 2049))
    Test.@test all(v -> DV_EXT._gpu_sortperm(v) == sortperm(v), sort_keys)
end
