using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using StructureFunctions: StructureFunctionObjects as SFO
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using KernelAbstractions: KernelAbstractions as KA
using FlowGeometries: FlowGeometries as FG
using NonuniformFFTs: NonuniformFFTs
using FFTW: FFTW
using Random: Random

const FFT = SB.FastFourierTransformSpectralBackend()
const DEV = CB.GPUBackend(KA.CPU())

# Slice `t`'s cells holding a finite value in every component, or `AllValid` for a complete batch.
_slice_validity(valid::SFC.AllValid, u, t) = valid
_slice_validity(valid, u, t) = vec(all(isfinite, view(u, :, :, t); dims = 1))

# Integer counts are whole pairs and agree exactly; weighted ones agree to round-off.
_counts_agree(got::AbstractArray{<:Integer}, ref::AbstractArray{<:Integer}) = got == ref
_counts_agree(got, ref) = maximum(abs, got .- ref) <= 1e-12 * max(maximum(abs, ref), 1e-12)

# Whether the batch entry (`tag`, or the lag sweep for `nothing`) equals the single-slice entry run per slice.
function _batch_matches(sf, u, s, edges, D; valid = SFC.AllValid(), weights = nothing, tag = FFT,
                        backend = CB.SerialBackend(), atol = 1e-12)
    nt = size(u, 3)
    nb = length(edges) - 1
    CT = weights === nothing ? Int : Float64
    ref_s, ref_c = zeros(nb, nt), zeros(CT, nb, nt)
    got_s, got_c = zeros(nb, nt), zeros(CT, nb, nt)
    for t in 1:nt
        rs, rc, ut, vt = view(ref_s, :, t), view(ref_c, :, t), view(u, :, :, t), _slice_validity(valid, u, t)
        tag === nothing ?
            SFC.gridded_lag_sweep!(rs, rc, sf, ut, s, edges, Val(D), Val(1), Val(0); valid = vt, weights) :
            SFC.gridded_sweep!(rs, rc, sf, ut, s, edges, Val(D), Val(1), Val(0), tag; valid = vt, weights)
    end
    tag === nothing ?
        SFC.gridded_lag_sweep_batch!(got_s, got_c, sf, u, s, edges, Val(D), Val(1), Val(0); valid, weights, backend) :
        SFC.gridded_sweep_batch!(got_s, got_c, sf, u, s, edges, Val(D), Val(1), Val(0), tag; valid, weights, backend)
    scale = max(maximum(abs, ref_s), 1e-12)
    return sum(ref_c) > 0 && _counts_agree(got_c, ref_c) && maximum(abs, got_s .- ref_s) <= atol * scale
end

# A batch with a fraction of each slice's cells set to NaN, and the batch validity that names them.
function _masked_batch(u::AbstractArray{<:Any, 3}, frac; seed = 4)
    Random.seed!(seed)
    v = copy(u)
    for t in 1:size(v, 3)
        v[:, rand(size(v, 2)) .< frac, t] .= NaN
    end
    return v, SFC.batch_validity(v)
end

Test.@testset "a batch equals its slices on a wrapping, a bounded and a mixed grid" begin
    Random.seed!(21)
    edges = collect(range(0.0, 5.0; length = 11))
    agree = Bool[]
    for periodic in ((true, true), (false, false), (true, false))
        s = SFC.UniformLagSchedule((8, 6), (0.5, 0.7), periodic)
        u = randn(2, 8 * 6, 3)
        push!(agree, _batch_matches(SFT.L2SFType(), u, s, edges, 2))
        push!(agree, _batch_matches(SFT.L2SFType(), u, s, edges, 2; tag = nothing))
    end
    Test.@test all(agree)
end

Test.@testset "a batch carries one mask per slice, and cell weights" begin
    Random.seed!(22)
    s = SFC.UniformLagSchedule((8, 8), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 3.0; length = 7))
    u, valid = _masked_batch(randn(2, 8 * 8, 3), 0.2)
    w = rand(8 * 8) .+ 0.5
    agree = Bool[_batch_matches(SFT.L2SFType(), u, s, edges, 2; valid),
                 _batch_matches(SFT.L2SFType(), u, s, edges, 2; valid, weights = w, tag = nothing)]
    Test.@test all(agree)
end

Test.@testset "a batch equals its slices on a sphere, complete and masked, and on a stretched axis" begin
    Random.seed!(23)
    n_lon, n_lat = 15, 9
    zs = SFC.ZonalLagSchedule(collect(range(-1.1, 1.1; length = n_lat)), n_lon, 2π / n_lon, 1.0, true)
    zedges = collect(range(0.0, π; length = 9))
    zu = randn(2, n_lon * n_lat, 3)
    zm, zv = _masked_batch(zu, 0.15)
    rs = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((16,), (0.4,), (true,)), (collect(cumsum(rand(7) .+ 0.4)),),
                                    (2, 1))
    redges = collect(range(0.0, 3.0; length = 9))
    ru = randn(2, 16 * 7, 3)
    agree = Bool[_batch_matches(SFT.L2SFType(), zu, zs, zedges, 2),
                 _batch_matches(SFT.L2SFType(), zu, zs, zedges, 2; tag = nothing),
                 _batch_matches(SFT.L2SFType(), zm, zs, zedges, 2; valid = zv),
                 _batch_matches(SFT.L2SFType(), ru, rs, redges, 2),
                 _batch_matches(SFT.L2SFType(), ru, rs, redges, 2; tag = nothing)]
    Test.@test all(agree)
end

Test.@testset "a batch takes the grid-shaped field and the slice count from its axes" begin
    Random.seed!(24)
    dims = (12, 10)
    nt = 3
    s = SFC.UniformLagSchedule(dims, (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 3.0; length = 7))
    ugrid = randn(2, dims..., nt)
    nb = length(edges) - 1
    a, ca = zeros(nb, nt), zeros(Int, nb, nt)
    b, cb = zeros(nb, nt), zeros(Int, nb, nt)
    SFC.gridded_sweep_batch!(a, ca, SFT.L2SFType(), ugrid, s, edges, Val(2), FFT)
    SFC.gridded_sweep_batch!(b, cb, SFT.L2SFType(), reshape(ugrid, 2, prod(dims), nt), s, edges, Val(2), Val(1),
                             Val(0), FFT)
    Test.@test ca == cb
    Test.@test a == b
    c, cc = zeros(nb, nt), zeros(Int, nb, nt)
    SFC.gridded_lag_sweep_batch!(c, cc, SFT.L2SFType(), ugrid, s, edges, Val(2))
    Test.@test cc == ca
    Test.@test maximum(abs, c .- a) <= 1e-12 * max(maximum(abs, a), 1e-12)
end

Test.@testset "the automatic algorithm on a batch equals the lag sweep" begin
    Random.seed!(25)
    s = SFC.UniformLagSchedule((8, 8), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 3.0; length = 7))
    u = randn(2, 8 * 8, 3)
    nb = length(edges) - 1
    agree = Bool[]
    for sf in (SFT.L2SFType(), SFT.L3SFType())
        a, ca = zeros(nb, 3), zeros(Int, nb, 3)
        b, cb = zeros(nb, 3), zeros(Int, nb, 3)
        SFC.gridded_sweep_batch!(a, ca, sf, u, s, edges, Val(2), Val(1), Val(0), SB.AutoSpectralBackend())
        SFC.gridded_lag_sweep_batch!(b, cb, sf, u, s, edges, Val(2), Val(1), Val(0))
        push!(agree, ca == cb && sum(cb) > 0 && maximum(abs, a .- b) <= 1e-11 * max(maximum(abs, b), 1e-12))
    end
    Test.@test all(agree)
end

Test.@testset "a device-backend batch equals its slices: complete, masked, weighted and on a sphere" begin
    Random.seed!(26)
    s = SFC.UniformLagSchedule((8, 6), (0.5, 0.7), (true, true))
    edges = collect(range(0.0, 3.5; length = 8))
    u = randn(2, 8 * 6, 3)
    um, uv = _masked_batch(u, 0.2)
    w = rand(8 * 6) .+ 0.5
    n_lon, n_lat = 15, 9
    zs = SFC.ZonalLagSchedule(collect(range(-1.1, 1.1; length = n_lat)), n_lon, 2π / n_lon, 1.0, true)
    zu = randn(2, n_lon * n_lat, 3)
    agree = Bool[_batch_matches(SFT.L2SFType(), u, s, edges, 2; backend = DEV, atol = 1e-11),
                 _batch_matches(SFT.L2SFType(), um, s, edges, 2; valid = uv, backend = DEV, atol = 1e-11),
                 _batch_matches(SFT.L2SFType(), um, s, edges, 2; valid = uv, weights = w, backend = DEV, atol = 1e-11),
                 _batch_matches(SFT.L2SFType(), zu, zs, collect(range(0.0, π; length = 9)), 2; backend = DEV,
                                atol = 1e-11)]
    Test.@test all(agree)
end

Test.@testset "a batch histogram joint in separation and angle" begin
    # Host and device batches equal the slices, and the angle marginal is the distance histogram.
    Random.seed!(27)
    s = SFC.UniformLagSchedule((8, 8), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 3.0; length = 7))
    axis_be = collect(range(-1e-9, π; length = 5))
    second_axis = SFC.SeparationAngleAxis([1.0, 0.0])
    nt = 3
    u = randn(2, 8 * 8, nt)
    nb, na = length(edges) - 1, length(axis_be) - 1
    ref_s, ref_c = zeros(nb, na, nt), zeros(nb, na, nt)
    for t in 1:nt
        SFC.gridded_sweep!(view(ref_s, :, :, t), view(ref_c, :, :, t), SFT.L2SFType(), view(u, :, :, t), s, edges,
                           axis_be, Val(2), Val(1), Val(0), FFT; second_axis)
    end
    agree = Bool[]
    for kw in ((;), (; backend = DEV))
        got_s, got_c = zeros(nb, na, nt), zeros(nb, na, nt)
        SFC.gridded_sweep_batch!(got_s, got_c, SFT.L2SFType(), u, s, edges, axis_be, Val(2), Val(1), Val(0), FFT;
                                 second_axis, kw...)
        push!(agree, sum(ref_c) > 0 && _counts_agree(got_c, ref_c) &&
                     maximum(abs, got_s .- ref_s) <= 1e-11 * max(maximum(abs, ref_s), 1e-12))
    end
    Test.@test all(agree)
    flat_s, flat_c = zeros(nb, nt), zeros(Float64, nb, nt)
    SFC.gridded_sweep_batch!(flat_s, flat_c, SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT)
    Test.@test maximum(abs, dropdims(sum(ref_c; dims = 2); dims = 2) .- flat_c) <= 1e-9
    Test.@test maximum(abs, dropdims(sum(ref_s; dims = 2); dims = 2) .- flat_s) <=
               1e-11 * max(maximum(abs, flat_s), 1e-12)
end

Test.@testset "a batch refuses what it cannot represent" begin
    # Too few output columns, a single-slice validity, weighted integer counts, a norm, a non-tag, no slices, a sphere's angle.
    s = SFC.UniformLagSchedule((10, 10), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 3.0; length = 7))
    u = randn(2, 100, 3)
    nb = length(edges) - 1
    Test.@test_throws DimensionMismatch SFC.gridded_sweep_batch!(
        zeros(nb, 2), zeros(Int, nb, 2), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT)
    Test.@test_throws DimensionMismatch SFC.gridded_lag_sweep_batch!(
        zeros(nb, 2), zeros(Int, nb, 2), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0))
    Test.@test_throws DimensionMismatch SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT;
        valid = trues(100))
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT;
        weights = rand(100) .+ 0.5)
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.FullVectorStructureFunctionType{3}(), u, s, edges, Val(2), Val(1),
        Val(0), FFT)
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), :fft)
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 0), zeros(Int, nb, 0), SFT.L2SFType(), randn(2, 100, 0), s, edges, Val(2), Val(1), Val(0), FFT)
    zs = SFC.ZonalLagSchedule(collect(range(-1.1, 1.1; length = 7)), 11, 2π / 11, 1.0, true)
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 2, 3), zeros(nb, 2, 3), SFT.L2SFType(), randn(2, 77, 3), zs, edges,
        collect(range(-1e-9, π; length = 3)), Val(2), Val(1), Val(0), FFT;
        second_axis = SFC.SeparationAngleAxis([1.0, 0.0]))
end

Test.@testset "a threaded batch equals a serial one" begin
    Random.seed!(28)
    n_lon, n_lat = 15, 9
    zs = SFC.ZonalLagSchedule(collect(range(-1.1, 1.1; length = n_lat)), n_lon, 2π / n_lon, 1.0, true)
    edges = collect(range(0.0, π; length = 9))
    u = randn(2, n_lon * n_lat, 3)
    nb = length(edges) - 1
    ref, cref = zeros(nb, 3), zeros(Int, nb, 3)
    SFC.gridded_lag_sweep_batch!(ref, cref, SFT.L2SFType(), u, zs, edges, Val(2), Val(1), Val(0);
                                 backend = CB.SerialBackend())
    agree = Bool[]
    for (tag, backend) in ((FFT, CB.AutoBackend()), (nothing, CB.AutoBackend()), (FFT, CB.SerialBackend()))
        a, ca = zeros(nb, 3), zeros(Int, nb, 3)
        tag === nothing ?
            SFC.gridded_lag_sweep_batch!(a, ca, SFT.L2SFType(), u, zs, edges, Val(2), Val(1), Val(0); backend) :
            SFC.gridded_sweep_batch!(a, ca, SFT.L2SFType(), u, zs, edges, Val(2), Val(1), Val(0), tag; backend)
        push!(agree, ca == cref && sum(cref) > 0 && maximum(abs, a .- ref) <= 1e-11 * max(maximum(abs, ref), 1e-12))
    end
    Test.@test all(agree)
end

Test.@testset "the grid entry takes a slice batch" begin
    # Each tag's batch against the single-slice entry, the joint form's angle marginal, cell weights, and refusals.
    Random.seed!(29)
    nx, ny, hx, hy = 10, 8, 0.2, 0.3
    nt = 3
    geo = FG.Geometry.CartesianGeometry()
    grid = FG.Grids.StructuredGrid(geo, range(0.0, step = hx, length = nx),
                                   range(0.0, step = hy, length = ny))
    bins = collect(range(0.0, 1.5; length = 9))
    nb = length(bins) - 1
    u = randn(2, nx, ny, nt)
    refs = [SFC.calculate_structure_function(SFT.L2SFType(), grid, u[:, :, :, t], bins, FFT, Int,
                                             SFO.StructureFunctionSumsAndCounts) for t in 1:nt]
    ref_s, ref_c = reduce(hcat, [r.sums for r in refs]), reduce(hcat, [r.counts for r in refs])
    agree = Bool[]
    for tag in (SB.AutoSpectralBackend(), FFT)
        sums, counts = zeros(nb, nt), zeros(Int, nb, nt)
        SFC.calculate_structure_function_batch!(sums, counts, SFT.L2SFType(), grid, u, bins, tag)
        push!(agree, counts == ref_c && sum(ref_c) > 0 &&
                     maximum(abs, sums .- ref_s) <= 1e-11 * max(maximum(abs, ref_s), 1e-12))
    end
    w = rand(nx * ny) .+ 0.5
    wrefs = [SFC.calculate_structure_function(SFT.L2SFType(), grid, u[:, :, :, t], bins, Float64,
                                              SFO.StructureFunctionSumsAndCounts; weights = w) for t in 1:nt]
    ws, wc = zeros(nb, nt), zeros(Float64, nb, nt)
    SFC.calculate_structure_function_batch!(ws, wc, SFT.L2SFType(), grid, u, bins; weights = w)
    wref_s, wref_c = reduce(hcat, [r.sums for r in wrefs]), reduce(hcat, [r.counts for r in wrefs])
    push!(agree, _counts_agree(wc, wref_c) && maximum(abs, ws .- wref_s) <= 1e-11 * max(maximum(abs, wref_s), 1e-12))
    Test.@test all(agree)
    axis_be = collect(range(-1e-9, π; length = 4))
    second_axis = SFC.SeparationAngleAxis([1.0, 0.0])
    na = length(axis_be) - 1
    js, jc = zeros(nb, na, nt), zeros(nb, na, nt)
    SFC.calculate_structure_function_batch!(js, jc, SFT.L2SFType(), grid, u, bins, axis_be; second_axis)
    Test.@test maximum(abs, dropdims(sum(jc; dims = 2); dims = 2) .- ref_c) <= 1e-9
    Test.@test_throws DimensionMismatch SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(Int, nb, nt), SFT.L2SFType(), grid, randn(2, nx, ny + 1, nt), bins)
    Test.@test_throws DimensionMismatch SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(Int, nb, nt), SFT.L2SFType(), grid, randn(2, nx * ny), bins)
end

Test.@testset "a scattered-mode batch equals its slices" begin
    # Then integer counts of a fractional kernel-weighted pair mass, and an FFT tag, are refused.
    Random.seed!(30)
    N, nt = 100, 3
    pts = rand(2, N) .* 2.0
    s = SFC.ScatteredModesSchedule(pts, 0.5, (16, 16))
    bins = collect(range(0.0, 0.5; length = 6))
    nb = length(bins) - 1
    u = randn(2, N, nt)
    tag = SFC.NonuniformFFTsSpectralBackend()
    sums, counts = zeros(nb, nt), zeros(Float64, nb, nt)
    SFC.calculate_structure_function_batch!(sums, counts, SFT.L2SFType(), s, u, bins, tag)
    res = [SFC.calculate_structure_function(SFT.L2SFType(), s, u[:, :, t], bins, tag,
                                            SFO.StructureFunctionSumsAndCounts) for t in 1:nt]
    ref_s, ref_c = reduce(hcat, [r.sums for r in res]), reduce(hcat, [r.counts for r in res])
    Test.@test maximum(abs, counts .- ref_c) <= 1e-9 * max(maximum(abs, ref_c), 1e-12)
    Test.@test maximum(abs, sums .- ref_s) <= 1e-11 * max(maximum(abs, ref_s), 1e-12)
    Test.@test_throws ArgumentError SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(Int, nb, nt), SFT.L2SFType(), s, u, bins, tag)
    Test.@test_throws ArgumentError SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(nb, nt), SFT.L2SFType(), s, u, bins, FFT)
end
