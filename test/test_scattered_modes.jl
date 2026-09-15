using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    StructureFunctionObjects as SFO
using ComputationalBackends: ComputationalBackends as CB
using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using SpectralBackends: SpectralBackends as SB
using KernelAbstractions: KernelAbstractions as KA
using Random: Random

const PROVIDERS = (SFC.NonuniformFFTsSpectralBackend(), SFC.FINUFFTSpectralBackend())
const NonuniformFFTsSpectralBackend = SB.NonUniformFastFourierTransformSpectralBackend()
const FINUFFTSpectralBackend = SB.FINUFFTSpectralBackend()
const FFT_TAG = SB.FastFourierTransformSpectralBackend()
const SERIAL = CB.SerialBackend()
const DEVICE = CB.GPUBackend(KA.CPU())
const RAW = SFO.StructureFunctionSumsAndCounts

_provider_name(tag) = nameof(typeof(tag))

# The lattice of a periodic uniform schedule as a point list, in the cell order the packed field uses.
function _lattice_points(dims, spacing)
    Dg = length(dims)
    x = zeros(Float64, Dg, prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims)), d in 1:Dg
        x[d, k] = (I[d] - 1) * spacing[d]
    end
    return x
end

# A schedule whose box is exactly the lattice's periodic domain, so its modes are the lattice's own.
function _lattice_schedule(x, dims, spacing; taper = SF.NoTaper())
    Dg = length(dims)
    return SFC.ScatteredModesSchedule(x, ntuple(_ -> 0.0, Dg), ntuple(d -> dims[d] * spacing[d], Dg), dims, taper)
end

function _gridded_run(sf, data, sched, bins, vD, vV, vK; valid = SFC.AllValid(), weights = nothing, tag = FFT_TAG,
                      backend = SERIAL)
    nb = SFC.n_histogram_bins(SFC.squared_digitize_plan(bins))
    s = zeros(Float64, nb)
    c = zeros(Float64, nb)
    SFC.gridded_sweep!(s, c, sf, data, sched, bins, vD, vV, vK, tag; valid, weights, backend)
    return s, c
end

_close(a, b; rtol = 1e-9) = isapprox(a, b; rtol, atol = 1e-9 * max(1.0, maximum(abs, b)))

Test.@testset "points on the mode grid reproduce the periodic gridded transform ($(_provider_name(tag)))" for tag in PROVIDERS
    Random.seed!(9700)
    cases = (
        ((12,), (0.3,), (SFT.L2SFType(), SFT.L3SFType())),
        ((12, 8), (0.25, 0.4), (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.S3SFType(), SFT.T3SFType(),
                               SFT.ProjectedStructureFunctionType{2, 2}())),
        ((8, 8, 8), (0.2, 0.2, 0.3), (SFT.L2SFType(), SFT.S3SFType(), SFT.T3SFType())),
    )
    for (dims, spacing, ops) in cases
        Dg = length(dims)
        N = prod(dims)
        x = _lattice_points(dims, spacing)
        u = randn(Dg, dims...)
        data = reshape(u, Dg, N)
        grid = SFC.UniformLagSchedule(dims, spacing, ntuple(_ -> true, Dg))
        modes = _lattice_schedule(x, dims, spacing)
        r_max = 0.45 * minimum(dims .* spacing)
        bins = collect(range(0.0, r_max; length = 6))
        # every node held, a random subset held, and random weights
        held = rand(N) .< 0.7
        valid = SFC.field_validity(reshape(ifelse.(held', data, NaN), Dg, N))
        w = 0.5 .+ rand(N)
        for sf in ops
            for (v, wt) in ((SFC.AllValid(), nothing), (valid, nothing), (SFC.AllValid(), w), (valid, w))
                ref_s, ref_c = _gridded_run(sf, data, grid, bins, Val(Dg), Val(1), Val(0); valid = v, weights = wt)
                got_s, got_c = _gridded_run(sf, data, modes, bins, Val(Dg), Val(1), Val(0);
                                            valid = v, weights = wt, tag)
                Test.@test _close(got_c, ref_c)
                Test.@test _close(got_s, ref_s)
                Test.@test sum(ref_c) > 0
            end
        end
    end
    # a channel bundle on the lattice
    dims, spacing = (12, 8), (0.25, 0.4)
    x = _lattice_points(dims, spacing)
    f = SF.Channels.Fields(vectors = (randn(2, dims...),), scalars = (randn(dims...),))
    grid = SFC.UniformLagSchedule(dims, spacing, (true, true))
    modes = _lattice_schedule(x, dims, spacing)
    bins = collect(range(0.0, 1.3; length = 6))
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{2}(), SFT.MixedSFType{1, 0, 1}())
        ref_s, ref_c = _gridded_run(sf, SF.Channels.packed(f), grid, bins, Val(2), Val(1), Val(1))
        got_s, got_c = _gridded_run(sf, SF.Channels.packed(f), modes, bins, Val(2), Val(1), Val(1); tag)
        Test.@test _close(got_c, ref_c)
        Test.@test _close(got_s, ref_s)
    end
end

# The periodic kernel of M modes with the taper's squared mode weights, on a box of length L.
function _kernel_1d(M, φ, L, taper)
    acc = 0.0
    for k in (-((M - 1) ÷ 2)):(M ÷ 2)
        b = SF.mode_taper(taper, (2π * k / L)^2)
        acc += b * b * cos(k * φ)
    end
    return acc / M
end

# The soft-binned statistic written out for one-dimensional points: every ordered pair, the self
# pairs included, against the kernel centred on each representative lag; a half-turn lag is its own
# reverse and carries half of its pairs, and an odd operator averages to zero over its two images.
function _oracle_1d(order, x, u, w, s, bins)
    N = length(x)
    M = s.modes[1]
    L = s.box[1]
    Δ = L / M
    θ = 2π .* (x .- s.origin[1]) ./ L
    nb = length(bins) - 1
    sums = zeros(nb)
    cnts = zeros(nb)
    for h in 1:(M ÷ 2)
        val = 0.0
        cnt = 0.0
        for i in 1:N, j in 1:N
            K = _kernel_1d(M, θ[j] - θ[i] - 2π * h / M, L, s.taper)
            ww = w[i] * w[j]
            val += ww * (u[j] - u[i])^order * K
            cnt += ww * K
        end
        if iseven(M) && h == M ÷ 2
            cnt /= 2
            val = isodd(order) ? 0.0 : val / 2
        end
        b = searchsortedfirst(bins, h * Δ) - 1
        1 <= b <= nb || continue
        sums[b] += val
        cnts[b] += cnt
    end
    return sums, cnts
end

Test.@testset "off the grid: the kernel identity written out in one dimension ($(_provider_name(tag)))" for tag in PROVIDERS
    Random.seed!(9710)
    N = 7
    x = sort(rand(N)) .* 0.8
    u = randn(1, N)
    w = 0.5 .+ rand(N)
    for M in (9, 8), taper in (SF.NoTaper(), SF.GaussianTaper(0.05))
        s = SFC.ScatteredModesSchedule(reshape(x, 1, N), 0.6, (M,); taper)
        Test.@test s.box[1] ≈ maximum(x) - minimum(x) + 0.6
        bins = collect(range(0.0, 0.7; length = 5))
        for (sf, order) in ((SFT.L2SFType(), 2), (SFT.L3SFType(), 3))
            ref_s, ref_c = _oracle_1d(order, x, u[1, :], w, s, bins)
            got_s, got_c = _gridded_run(sf, u, s, bins, Val(1), Val(1), Val(0); weights = w, tag)
            Test.@test isapprox(got_c, ref_c; rtol = 1e-9, atol = 1e-10)
            Test.@test isapprox(got_s, ref_s; rtol = 1e-9, atol = 1e-10 * maximum(abs, ref_s))
            Test.@test sum(abs, ref_c) > 0
        end
    end
end

Test.@testset "the soft bins converge to the hard bins as the modes grow ($(_provider_name(tag)))" for tag in PROVIDERS
    Random.seed!(9720)
    N = 200
    x = rand(2, N)
    u = vcat(sin.(2π .* x[1:1, :]) .* cos.(2π .* x[2:2, :]), cos.(2π .* x[1:1, :]) .* sin.(2π .* x[2:2, :]))
    r_max = 0.6
    bins = collect(range(0.0, r_max; length = 5))
    hard = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, Float64; backend = SERIAL, verbose = false,
                                            show_progress = false, output_type = RAW)
    hard_avg = hard.sums ./ hard.counts
    errs = Float64[]
    for M in (16, 32, 64)
        s = SFC.ScatteredModesSchedule(x, r_max, (M, M); taper = SF.GaussianTaper(maximum(x) / M))
        soft = SFC.calculate_structure_function(SFT.L2SFType(), s, u, bins, tag; backend = SERIAL, verbose = false,
                                                output_type = RAW)
        Test.@test soft.distance isa SF.ModeBinEdges
        Test.@test soft.distance.schedule === s
        Test.@test sum(soft.counts) ≈ sum(hard.counts) rtol = 0.2
        push!(errs, maximum(abs.(soft.sums ./ soft.counts .- hard_avg) ./ abs.(hard_avg)))
    end
    Test.@test errs[2] < errs[1]
    Test.@test errs[3] < errs[2]
    Test.@test errs[3] < 0.1
end

Test.@testset "the two providers compute the same transforms" begin
    Random.seed!(9760)
    for (Dg, modes, r_max) in ((1, (64,), 0.4), (2, (32, 24), 0.4), (3, (12, 10, 8), 0.5))
        N = 150
        x = rand(Dg, N)
        u = randn(Dg, N)
        u[1, rand(N) .< 0.15] .= NaN
        valid = SFC.field_validity(u)
        w = 0.5 .+ rand(N)
        s = SFC.ScatteredModesSchedule(x, r_max, modes; taper = SF.GaussianTaper(0.02))
        bins = collect(range(0.0, r_max; length = 5))
        for sf in (SFT.L2SFType(), SFT.S3SFType())
            a_s, a_c = _gridded_run(sf, u, s, bins, Val(Dg), Val(1), Val(0); valid, weights = w, tag = PROVIDERS[1])
            b_s, b_c = _gridded_run(sf, u, s, bins, Val(Dg), Val(1), Val(0); valid, weights = w, tag = PROVIDERS[2])
            Test.@test isapprox(a_c, b_c; rtol = 1e-10, atol = 1e-10 * maximum(abs, a_c))
            Test.@test isapprox(a_s, b_s; rtol = 1e-10, atol = 1e-10 * maximum(abs, a_s))
            Test.@test sum(a_c) > 0
        end
    end
end

Test.@testset "the device engine runs the non-uniform route ($(_provider_name(tag)))" for tag in PROVIDERS
    Random.seed!(9730)
    N = 60
    x = rand(2, N) .* (1.0, 0.7)
    u = randn(2, N)
    uf = copy(u)
    uf[1, rand(N) .< 0.2] .= NaN
    valid = SFC.field_validity(uf)
    w = 0.5 .+ rand(N)
    s = SFC.ScatteredModesSchedule(x, 0.5, (24, 16); taper = SF.GaussianTaper(0.03))
    bins = collect(range(0.0, 0.5; length = 6))
    for sf in (SFT.L2SFType(), SFT.S3SFType(), SFT.T3SFType())
        ref_s, ref_c = _gridded_run(sf, uf, s, bins, Val(2), Val(1), Val(0); valid, weights = w, tag)
        dev_s, dev_c = _gridded_run(sf, uf, s, bins, Val(2), Val(1), Val(0); valid, weights = w, tag,
                                    backend = DEVICE)
        Test.@test _close(dev_c, ref_c)
        Test.@test _close(dev_s, ref_s)
        Test.@test all(isfinite, ref_s)
    end
end

Test.@testset "the box is padded so that no pair within r_max wraps ($(_provider_name(tag)))" for tag in PROVIDERS
    Random.seed!(9740)
    half = 20
    x = reshape(vcat(0.1 .* rand(half), 2.0 .+ 0.1 .* rand(half)), 1, 2half)
    u = randn(1, 2half)
    r_max = 0.3
    s = SFC.ScatteredModesSchedule(x, r_max, (512,); taper = SF.GaussianTaper(0.01))
    Test.@test s.box[1] ≈ maximum(x) - minimum(x) + r_max
    # bins starting above the kernel's width, so no self pair's mass is counted; the two clusters' pairs
    # sit at separations 1.9 … 2.1, which a box without the padding wraps to 0 … 0.2
    bins = collect(range(0.02, 0.25; length = 4))
    soft = SFC.calculate_structure_function(SFT.L2SFType(), s, u, bins, tag; backend = SERIAL, verbose = false,
                                            output_type = RAW)
    hard = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, Float64; backend = SERIAL, verbose = false,
                                            show_progress = false, output_type = RAW)
    unpadded = SFC.ScatteredModesSchedule(x, (minimum(x),), (maximum(x) - minimum(x),), (512,),
                                          SF.GaussianTaper(0.01))
    wrapped = SFC.calculate_structure_function(SFT.L2SFType(), unpadded, u, bins, tag; backend = SERIAL,
                                               verbose = false, output_type = RAW)
    inter = half * half
    Test.@test sum(hard.counts) > 0
    Test.@test abs(sum(soft.counts) - sum(hard.counts)) / sum(hard.counts) < 0.1
    Test.@test sum(wrapped.counts) - sum(soft.counts) > 0.5 * inter
end

Test.@testset "the route is asked for by name and refuses what it cannot mean" begin
    Random.seed!(9750)
    N = 30
    x = rand(2, N)
    u = randn(2, N)
    bins = collect(range(0.0, 0.5; length = 5))
    s = SFC.ScatteredModesSchedule(x, 0.5, (16, 16))
    Test.@test_throws ArgumentError SFC.ScatteredModesSchedule(x, 0.5, (16, 16); taper = SF.Bartlett())
    Test.@test_throws ArgumentError SFC.ScatteredModesSchedule(x, 0.0, (16, 16))
    Test.@test_throws ArgumentError SFC.ScatteredModesSchedule(x, 0.5, (1, 16))
    Test.@test_throws DimensionMismatch SFC.ScatteredModesSchedule(x, 0.5, (16,))
    # the transforms' accuracy is the provider's own knob, validated on its tag
    Test.@test_throws ArgumentError SFC.NonuniformFFTsSpectralBackend(half_support = 0)
    Test.@test SFC.NonuniformFFTsSpectralBackend(half_support = 4).half_support == 4
    Test.@test_throws ArgumentError SFC.FINUFFTSpectralBackend(tolerance = 0.0)
    Test.@test_throws ArgumentError SFC.FINUFFTSpectralBackend(tolerance = 1.0)
    Test.@test SFC.FINUFFTSpectralBackend(tolerance = 1e-9).tolerance == 1e-9
    # NonuniformFFTs' spreading kernel needs at least its half-support in modes; FINUFFT pads its own fine grid
    small = SFC.ScatteredModesSchedule(x, 0.5, (4, 16))
    Test.@test_throws ArgumentError _gridded_run(SFT.L2SFType(), u, small, bins, Val(2), Val(1), Val(0);
                                                 tag = SFC.NonuniformFFTsSpectralBackend())
    Test.@test all(isfinite, _gridded_run(SFT.L2SFType(), u, small, bins, Val(2), Val(1), Val(0);
                                          tag = SFC.NonuniformFFTsSpectralBackend(half_support = 4))[1])
    Test.@test all(isfinite, _gridded_run(SFT.L2SFType(), u, small, bins, Val(2), Val(1), Val(0);
                                          tag = SFC.FINUFFTSpectralBackend())[1])
    # the plain tag names the one loaded provider; with both loaded it names neither
    for tag in PROVIDERS
        Test.@test SFC.nufft_provider(tag) === tag
    end
    Test.@test_throws ArgumentError SFC.nufft_provider(PLAIN_NUFFT)
    Test.@test_throws ArgumentError _gridded_run(SFT.L2SFType(), u, s, bins, Val(2), Val(1), Val(0); tag = PLAIN_NUFFT)
    sums = zeros(4)
    Test.@test_throws ArgumentError SFC.gridded_sweep!(sums, zeros(4), SFT.L2SFType(), u, s, bins, Val(2), Val(1), Val(0), FFT_TAG)
    Test.@test_throws ArgumentError SFC.gridded_sweep!(sums, zeros(4), SFT.L2SFType(), u, s, bins, Val(2), Val(1), Val(0),
                                                        SB.AutoSpectralBackend())
    Test.@test_throws ArgumentError SFC.gridded_sweep!(sums, zeros(Int, 4), SFT.L2SFType(), u, s, bins, Val(2), Val(1), Val(0),
                                                        PROVIDERS[1])
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(sums, zeros(4), SFT.L2SFType(), u, s, bins, Val(2), Val(1), Val(0))
    grid = SFC.UniformLagSchedule((6, 5), (0.2, 0.2), (true, true))
    for tag in PROVIDERS
        Test.@test_throws ArgumentError SFC.gridded_sweep!(sums, zeros(4), SFT.L2SFType(), randn(2, 30), grid, bins, Val(2),
                                                            Val(1), Val(0), tag)
    end
    Test.@test_throws DimensionMismatch SFC.calculate_structure_function(SFT.L2SFType(), s, randn(2, N + 1), bins,
                                                                         PROVIDERS[1]; verbose = false)
    # a point list without a schedule stays on the exact pair loop
    plain = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins; backend = CB.AutoBackend(), verbose = false,
                                             show_progress = false, output_type = RAW)
    Test.@test !(plain.distance isa SF.ModeBinEdges)
    for tag in PROVIDERS
        res = SFC.calculate_structure_function(SFT.L2SFType(), s, u, bins, tag; verbose = false)
        Test.@test res isa SFO.StructureFunction
        Test.@test SF.midpoints(res.distance) == SF.midpoints(bins)
        Test.@test SF.n_histogram_bins(res.distance) == 4
    end
end
