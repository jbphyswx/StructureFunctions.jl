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

# Slice `t`'s validity of a batch mask, in the form the single-slice entries take.
_slice(v::SFC.AllValid, t) = v
_slice(v::AbstractMatrix, t) = view(v, :, t)

# Unweighted counts are whole pairs and must agree exactly; weighted ones are a pair mass summed in
# whatever order the work is decomposed, so they agree to round-off.
_counts_agree(got::AbstractArray{<:Integer}, ref::AbstractArray{<:Integer}) = got == ref
_counts_agree(got, ref) = maximum(abs, got .- ref) <= 1e-12 * max(maximum(abs, ref), 1e-12)

# The batch against the single-slice entry run once per slice: the same pairs, the same reading, so
# the two agree exactly when the arithmetic is the same and to round-off when the order differs.
function _batch_matches(sf, u, s, edges, D; valid = SFC.AllValid(), weights = nothing, tag = FFT,
                        backend = CB.SerialBackend(), atol = 1e-12)
    nt = size(u, 3)
    nb = length(edges) - 1
    CT = weights === nothing ? Int : Float64
    ref_s, ref_c = zeros(nb, nt), zeros(CT, nb, nt)
    for t in 1:nt
        SFC.gridded_sweep!(view(ref_s, :, t), view(ref_c, :, t), sf, view(u, :, :, t), s, edges, Val(D), Val(1),
                           Val(0), tag; valid = _slice(valid, t), weights)
    end
    got_s, got_c = zeros(nb, nt), zeros(CT, nb, nt)
    SFC.gridded_sweep_batch!(got_s, got_c, sf, u, s, edges, Val(D), Val(1), Val(0), tag;
                             valid, weights, backend)
    scale = max(maximum(abs, ref_s), 1e-12)
    Test.@test _counts_agree(got_c, ref_c)
    Test.@test maximum(abs, got_s .- ref_s) <= atol * scale
    return nothing
end

# The same, for the direct lag sweep, which the batch must reproduce slice by slice.
function _lag_batch_matches(sf, u, s, edges, D; valid = SFC.AllValid(), weights = nothing,
                            backend = CB.SerialBackend(), atol = 1e-12)
    nt = size(u, 3)
    nb = length(edges) - 1
    CT = weights === nothing ? Int : Float64
    ref_s, ref_c = zeros(nb, nt), zeros(CT, nb, nt)
    for t in 1:nt
        SFC.gridded_lag_sweep!(view(ref_s, :, t), view(ref_c, :, t), sf, view(u, :, :, t), s, edges, Val(D),
                               Val(1), Val(0); valid = _slice(valid, t), weights)
    end
    got_s, got_c = zeros(nb, nt), zeros(CT, nb, nt)
    SFC.gridded_lag_sweep_batch!(got_s, got_c, sf, u, s, edges, Val(D), Val(1), Val(0);
                                 valid, weights, backend)
    scale = max(maximum(abs, ref_s), 1e-12)
    Test.@test _counts_agree(got_c, ref_c)
    Test.@test maximum(abs, got_s .- ref_s) <= atol * scale
    return nothing
end

# A batch with a fraction of each slice's cells set to NaN, and the validity that names them.
function _masked_batch(u::AbstractArray{<:Any, 3}, frac; seed = 4)
    Random.seed!(seed)
    v = copy(u)
    for t in 1:size(v, 3)
        v[:, rand(size(v, 2)) .< frac, t] .= NaN
    end
    return v, SFC.batch_validity(v)
end

Test.@testset "a batch sweeps the lags once and sums every slice against them" begin
    Random.seed!(21)
    edges = collect(range(0.0, 7.0; length = 13))
    for periodic in ((true, true), (false, false), (true, false))
        s = SFC.UniformLagSchedule((18, 14), (0.5, 0.7), periodic)
        u = randn(2, 18 * 14, 4)
        for sf in (SFT.L2SFType(), SFT.T2SFType(), SFT.S3SFType(), SFT.L3SFType())
            _batch_matches(sf, u, s, edges, 2)
            _lag_batch_matches(sf, u, s, edges, 2)
        end
    end
end

Test.@testset "a batch carries one mask and one set of weights per slice" begin
    Random.seed!(22)
    s = SFC.UniformLagSchedule((16, 16), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 5.0; length = 11))
    u0 = randn(2, 16 * 16, 5)
    u, valid = _masked_batch(u0, 0.2)
    # a mask that differs from slice to slice is the ordinary case for a time series
    Test.@test valid isa AbstractMatrix
    Test.@test size(valid) == (16 * 16, 5)
    Test.@test any(t -> view(valid, :, t) != view(valid, :, 1), 2:5)
    for sf in (SFT.L2SFType(), SFT.S3SFType())
        _batch_matches(sf, u, s, edges, 2; valid)
        _lag_batch_matches(sf, u, s, edges, 2; valid)
    end
    w = rand(16 * 16) .+ 0.5
    _batch_matches(SFT.L2SFType(), u, s, edges, 2; valid, weights = w)
    _lag_batch_matches(SFT.L2SFType(), u, s, edges, 2; valid, weights = w)
    # a complete batch reports itself complete, so the count column is never taken
    Test.@test SFC.batch_validity(u0) isa SFC.AllValid
end

Test.@testset "a batch runs on every separable schedule" begin
    Random.seed!(23)
    # the sphere, where the frames are the field-independent work a batch reuses
    n_lon, n_lat = 15, 9
    lats = collect(range(-1.1, 1.1; length = n_lat))
    zs = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    zedges = collect(range(0.0, π; length = 9))
    zu = randn(2, n_lon * n_lat, 3)
    _batch_matches(SFT.L2SFType(), zu, zs, zedges, 2; atol = 1e-12)
    _batch_matches(SFT.S3SFType(), zu, zs, zedges, 2; atol = 1e-12)
    _lag_batch_matches(SFT.L2SFType(), zu, zs, zedges, 2)
    zm, zv = _masked_batch(zu, 0.15)
    _batch_matches(SFT.L2SFType(), zm, zs, zedges, 2; valid = zv, atol = 1e-12)
    # a stretched axis, whose slices are permuted into slab order one at a time
    su = SFC.UniformLagSchedule((16,), (0.4,), (true,))
    ys = collect(cumsum(rand(7) .+ 0.4))
    rs = SFC.RectilinearLagSchedule(su, (ys,), (2, 1))
    redges = collect(range(0.0, 3.0; length = 9))
    ru = randn(2, 16 * 7, 3)
    _batch_matches(SFT.L2SFType(), ru, rs, redges, 2)
    _lag_batch_matches(SFT.L2SFType(), ru, rs, redges, 2)
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
    # the lag sweep takes the same shapes
    c, cc = zeros(nb, nt), zeros(Int, nb, nt)
    SFC.gridded_lag_sweep_batch!(c, cc, SFT.L2SFType(), ugrid, s, edges, Val(2))
    Test.@test cc == ca
    Test.@test maximum(abs, c .- a) <= 1e-12 * max(maximum(abs, a), 1e-12)
end

Test.@testset "Auto picks an algorithm for a batch and both agree" begin
    Random.seed!(25)
    s = SFC.UniformLagSchedule((14, 14), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 4.0; length = 9))
    u = randn(2, 14 * 14, 3)
    nb = length(edges) - 1
    for sf in (SFT.L2SFType(), SFT.L3SFType())
        a, ca = zeros(nb, 3), zeros(Int, nb, 3)
        b, cb = zeros(nb, 3), zeros(Int, nb, 3)
        SFC.gridded_sweep_batch!(a, ca, sf, u, s, edges, Val(2), Val(1), Val(0), SB.AutoSpectralBackend())
        SFC.gridded_lag_sweep_batch!(b, cb, sf, u, s, edges, Val(2), Val(1), Val(0))
        Test.@test ca == cb
        Test.@test maximum(abs, a .- b) <= 1e-11 * max(maximum(abs, b), 1e-12)
    end
end

Test.@testset "the device engine batches slices" begin
    Random.seed!(26)
    s = SFC.UniformLagSchedule((16, 14), (0.5, 0.7), (true, true))
    edges = collect(range(0.0, 5.0; length = 11))
    u = randn(2, 16 * 14, 4)
    for sf in (SFT.L2SFType(), SFT.S3SFType())
        _batch_matches(sf, u, s, edges, 2; backend = DEV, atol = 1e-11)
    end
    um, uv = _masked_batch(u, 0.2)
    _batch_matches(SFT.L2SFType(), um, s, edges, 2; valid = uv, backend = DEV, atol = 1e-11)
    w = rand(16 * 14) .+ 0.5
    _batch_matches(SFT.L2SFType(), um, s, edges, 2; valid = uv, weights = w, backend = DEV, atol = 1e-11)
    # a sphere, where the device reads one lag box per slab pair
    n_lon, n_lat = 15, 9
    lats = collect(range(-1.1, 1.1; length = n_lat))
    zs = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    zu = randn(2, n_lon * n_lat, 3)
    _batch_matches(SFT.L2SFType(), zu, zs, collect(range(0.0, π; length = 9)), 2; backend = DEV, atol = 1e-11)
end

Test.@testset "a batch histogram joint in separation and angle" begin
    Random.seed!(27)
    s = SFC.UniformLagSchedule((14, 14), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 4.0; length = 7))
    axis_be = collect(range(-1e-9, π; length = 5))
    second_axis = SFC.SeparationAngleAxis([1.0, 0.0])
    nt = 3
    u = randn(2, 14 * 14, nt)
    nb, na = length(edges) - 1, length(axis_be) - 1
    ref_s, ref_c = zeros(nb, na, nt), zeros(nb, na, nt)
    for t in 1:nt
        SFC.gridded_sweep!(view(ref_s, :, :, t), view(ref_c, :, :, t), SFT.L2SFType(), view(u, :, :, t), s, edges,
                           axis_be, Val(2), Val(1), Val(0), FFT; second_axis)
    end
    for (nm, kw) in (("host", (;)), ("device", (; backend = DEV)))
        got_s, got_c = zeros(nb, na, nt), zeros(nb, na, nt)
        SFC.gridded_sweep_batch!(got_s, got_c, SFT.L2SFType(), u, s, edges, axis_be, Val(2), Val(1), Val(0), FFT;
                                 second_axis, kw...)
        Test.@test _counts_agree(got_c, ref_c)
        Test.@test maximum(abs, got_s .- ref_s) <= 1e-11 * max(maximum(abs, ref_s), 1e-12)
    end
    # the angle marginal is the distance histogram of the same batch
    got_s, got_c = zeros(nb, na, nt), zeros(nb, na, nt)
    SFC.gridded_sweep_batch!(got_s, got_c, SFT.L2SFType(), u, s, edges, axis_be, Val(2), Val(1), Val(0), FFT;
                             second_axis)
    flat_s, flat_c = zeros(nb, nt), zeros(Float64, nb, nt)
    SFC.gridded_sweep_batch!(flat_s, flat_c, SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT)
    Test.@test maximum(abs, dropdims(sum(got_c; dims = 2); dims = 2) .- flat_c) <= 1e-9
    Test.@test maximum(abs, dropdims(sum(got_s; dims = 2); dims = 2) .- flat_s) <=
               1e-11 * max(maximum(abs, flat_s), 1e-12)
end

Test.@testset "a batch refuses what it cannot represent" begin
    s = SFC.UniformLagSchedule((10, 10), (0.5, 0.5), (true, true))
    edges = collect(range(0.0, 3.0; length = 7))
    u = randn(2, 100, 3)
    nb = length(edges) - 1
    # the output must carry a column per slice
    Test.@test_throws DimensionMismatch SFC.gridded_sweep_batch!(
        zeros(nb, 2), zeros(Int, nb, 2), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT)
    Test.@test_throws DimensionMismatch SFC.gridded_lag_sweep_batch!(
        zeros(nb, 2), zeros(Int, nb, 2), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0))
    # a validity of the wrong shape names no slice
    Test.@test_throws DimensionMismatch SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT;
        valid = trues(100))
    # weights give a fractional pair mass
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), FFT;
        weights = rand(100) .+ 0.5)
    # an operator the transform cannot express
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.FullVectorStructureFunctionType{3}(), u, s, edges, Val(2), Val(1),
        Val(0), FFT)
    # a tag that is not a spectral tag at all
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 3), zeros(Int, nb, 3), SFT.L2SFType(), u, s, edges, Val(2), Val(1), Val(0), :fft)
    # a batch of no slices
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 0), zeros(Int, nb, 0), SFT.L2SFType(), randn(2, 100, 0), s, edges, Val(2), Val(1), Val(0), FFT)
    # a sphere has no angle to one fixed axis
    lats = collect(range(-1.1, 1.1; length = 7))
    zs = SFC.ZonalLagSchedule(lats, 11, 2π / 11, 1.0, true)
    Test.@test_throws ArgumentError SFC.gridded_sweep_batch!(
        zeros(nb, 2, 3), zeros(nb, 2, 3), SFT.L2SFType(), randn(2, 77, 3), zs, edges,
        collect(range(-1e-9, π; length = 3)), Val(2), Val(1), Val(0), FFT;
        second_axis = SFC.SeparationAngleAxis([1.0, 0.0]))
end

Test.@testset "a threaded batch equals a serial one" begin
    Random.seed!(28)
    n_lon, n_lat = 15, 9
    lats = collect(range(-1.1, 1.1; length = n_lat))
    zs = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    edges = collect(range(0.0, π; length = 9))
    u = randn(2, n_lon * n_lat, 4)
    nb = length(edges) - 1
    for backend in (CB.AutoBackend(), CB.SerialBackend())
        a, ca = zeros(nb, 4), zeros(Int, nb, 4)
        b, cb = zeros(nb, 4), zeros(Int, nb, 4)
        SFC.gridded_sweep_batch!(a, ca, SFT.L2SFType(), u, zs, edges, Val(2), Val(1), Val(0), FFT; backend)
        SFC.gridded_lag_sweep_batch!(b, cb, SFT.L2SFType(), u, zs, edges, Val(2), Val(1), Val(0); backend)
        Test.@test ca == cb
        Test.@test maximum(abs, a .- b) <= 1e-11 * max(maximum(abs, b), 1e-12)
    end
end

Test.@testset "the grid entry takes a slice batch" begin
    Random.seed!(29)
    nx, ny, hx, hy = 10, 8, 0.2, 0.3
    nt = 4
    geo = FG.Geometry.CartesianGeometry()
    grid = FG.Grids.StructuredGrid(geo, range(0.0, step = hx, length = nx),
                                   range(0.0, step = hy, length = ny))
    bins = collect(range(0.0, 1.5; length = 9))
    nb = length(bins) - 1
    u = randn(2, nx, ny, nt)
    for tag in (SB.AutoSpectralBackend(), FFT)
        sums, counts = zeros(nb, nt), zeros(Int, nb, nt)
        SFC.calculate_structure_function_batch!(sums, counts, SFT.L2SFType(), grid, u, bins, tag; verbose = false)
        # each slice must be what the single-slice grid entry returns for that slice
        for t in 1:nt
            res = SFC.calculate_structure_function(SFT.L2SFType(), grid, u[:, :, :, t], bins, Int, tag;
                                                  output_type = SFO.StructureFunctionSumsAndCounts,
                                                  verbose = false)
            Test.@test counts[:, t] == res.counts
            Test.@test maximum(abs, sums[:, t] .- res.sums) <= 1e-11 * max(maximum(abs, res.sums), 1e-12)
        end
    end
    # the joint form carries the angle axis
    axis_be = collect(range(-1e-9, π; length = 4))
    second_axis = SFC.SeparationAngleAxis([1.0, 0.0])
    na = length(axis_be) - 1
    js, jc = zeros(nb, na, nt), zeros(nb, na, nt)
    SFC.calculate_structure_function_batch!(js, jc, SFT.L2SFType(), grid, u, bins, axis_be;
                                            second_axis, verbose = false)
    fs, fc = zeros(nb, nt), zeros(Float64, nb, nt)
    SFC.calculate_structure_function_batch!(fs, fc, SFT.L2SFType(), grid, u, bins; verbose = false)
    Test.@test maximum(abs, dropdims(sum(jc; dims = 2); dims = 2) .- fc) <= 1e-9
    # cell weights belong to the grid, so one vector serves every slice
    ws, wc = zeros(nb, nt), zeros(Float64, nb, nt)
    SFC.calculate_structure_function_batch!(ws, wc, SFT.L2SFType(), grid, u, bins;
                                            weights = SFC.cell_measure(grid), verbose = false)
    Test.@test all(>=(0), wc)
    Test.@test any(>(0), wc)
    # a field that does not cover the grid, and one with no slice axis
    Test.@test_throws DimensionMismatch SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(Int, nb, nt), SFT.L2SFType(), grid, randn(2, nx, ny + 1, nt), bins; verbose = false)
    Test.@test_throws DimensionMismatch SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(Int, nb, nt), SFT.L2SFType(), grid, randn(2, nx * ny), bins; verbose = false)
end

Test.@testset "the scattered mode route takes a slice batch" begin
    Random.seed!(30)
    N, nt = 400, 3
    pts = rand(2, N) .* 2.0
    s = SFC.ScatteredModesSchedule(pts, 0.5, (16, 16))
    bins = collect(range(0.0, 0.5; length = 6))
    nb = length(bins) - 1
    u = randn(2, N, nt)
    tag = SFC.NonuniformFFTsSpectralBackend()
    sums, counts = zeros(nb, nt), zeros(Float64, nb, nt)
    SFC.calculate_structure_function_batch!(sums, counts, SFT.L2SFType(), s, u, bins, tag; verbose = false)
    for t in 1:nt
        res = SFC.calculate_structure_function(SFT.L2SFType(), s, u[:, :, t], bins, tag;
                                               output_type = SFO.StructureFunctionSumsAndCounts, verbose = false)
        Test.@test maximum(abs, counts[:, t] .- res.counts) <= 1e-9 * max(maximum(abs, res.counts), 1e-12)
        Test.@test maximum(abs, sums[:, t] .- res.sums) <= 1e-11 * max(maximum(abs, res.sums), 1e-12)
    end
    # the kernel-weighted pair mass is fractional, so integer counts are refused
    Test.@test_throws ArgumentError SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(Int, nb, nt), SFT.L2SFType(), s, u, bins, tag; verbose = false)
    # an FFT tag names no non-uniform transform for a scattered mode set
    Test.@test_throws ArgumentError SFC.calculate_structure_function_batch!(
        zeros(nb, nt), zeros(nb, nt), SFT.L2SFType(), s, u, bins, FFT; verbose = false)
end

Test.@testset "a schedule names how its batch shares each lag's geometry" begin
    Random.seed!(31)
    n_lon, n_lat = 15, 9
    lats = collect(range(-1.1, 1.1; length = n_lat))
    zs = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    us = SFC.UniformLagSchedule((16, 16), (0.5, 0.5), (true, true))
    su = SFC.UniformLagSchedule((16,), (0.4,), (true,))
    rs = SFC.RectilinearLagSchedule(su, (collect(cumsum(rand(7) .+ 0.4)),), (1, 2))
    # the frames and transport matrices of a curved schedule are the same for every slice
    Test.@test SFC.batch_shares_lag_geometry(zs)
    Test.@test SFC.lag_transport(zs) isa SFC.FrameTransport
    # a flat schedule's lag geometry is a displacement and a bin
    Test.@test !SFC.batch_shares_lag_geometry(us)
    Test.@test !SFC.batch_shares_lag_geometry(rs)
    # either answer must give the reference result, so the trait is a decomposition and not a result
    edges = collect(range(0.0, 3.0; length = 7))
    for s in (us, rs)
        u = randn(2, SFC.n_cells(s), 3)
        _batch_matches(SFT.L2SFType(), u, s, edges, 2)
    end
end
