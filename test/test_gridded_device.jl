using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, Fields
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using FFTW: FFTW
using Random: Random

const DEV = CB.GPUBackend(KA.CPU())
const FFT = SB.FastFourierTransformSpectralBackend()

# The device engine against the CPU engine on the same schedule; counts exact, sums to round-off.
function _device_matches(sf, u, s, edges, D; valid = SFC.AllValid(), count_type = Int, tag = FFT)
    nb = length(edges) - 1
    a, ca = zeros(nb), zeros(count_type, nb)
    b, cb = zeros(nb), zeros(count_type, nb)
    SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(D), FFT; valid)
    SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(D), tag; valid, backend = DEV)
    scale = max(maximum(abs, a), 1e-12)
    Test.@test ca == cb
    Test.@test maximum(abs.(a .- b)) <= 1e-11 * scale
    return nothing
end

# A copy of the packed field with a fraction of its cells set to NaN, and their validity.
function _masked(u::AbstractMatrix, frac; seed = 3)
    Random.seed!(seed)
    v = copy(u)
    v[:, rand(size(v, 2)) .< frac] .= NaN
    return v, SFC.field_validity(v)
end

Test.@testset "the device engine equals the CPU engine on a uniform grid" begin
    Random.seed!(11)
    dims = (24, 20)
    edges = collect(range(0.0, 7.0; length = 15))
    for periodic in ((true, true), (false, false), (true, false))
        s = SFC.UniformLagSchedule(dims, (0.5, 0.7), periodic)
        u = randn(2, dims...)
        for sf in (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.S3SFType(), SFT.L1T2SFType())
            _device_matches(sf, u, s, edges, 2)
        end
        um, valid = _masked(reshape(u, 2, :), 0.25)
        for sf in (SFT.L2SFType(), SFT.L3SFType())
            _device_matches(sf, reshape(um, 2, dims...), s, edges, 2; valid)
        end
    end
    # three directions, floating-point counts, log bins
    s3 = SFC.UniformLagSchedule((8, 9, 7), (1.0, 1.1, 0.9), (true, false, true))
    u3 = randn(3, 8, 9, 7)
    _device_matches(SFT.L2SFType(), u3, s3, collect(range(0.0, 6.0; length = 9)), 3; count_type = Float64)
    _device_matches(SFT.S3SFType(), u3, s3, SF.LogBinEdges(exp.(range(log(0.8), log(6.0); length = 8))), 3)
end

Test.@testset "channel bundles and higher moments ride the device engine" begin
    Random.seed!(12)
    dims = (18, 16)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    θ = randn(dims...)
    f = Fields(vectors = (u,), scalars = (θ,))
    edges = collect(range(0.0, 8.0; length = 12))
    nb = length(edges) - 1
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.ScalarSFType{2}(), SFT.ProjectedStructureFunctionType{4, 0}())
        a, ca = zeros(nb), zeros(Int, nb)
        b, cb = zeros(nb), zeros(Int, nb)
        SFC.gridded_sweep!(a, ca, sf, f, s, edges, FFT)
        SFC.gridded_sweep!(b, cb, sf, f, s, edges, FFT; backend = DEV)
        Test.@test ca == cb
        Test.@test maximum(abs.(a .- b)) <= 1e-11 * maximum(abs, a)
    end
end

Test.@testset "rectilinear and zonal schedules run on the device" begin
    Random.seed!(13)
    coords = cumsum(0.4 .+ 0.6 .* rand(9))
    sr = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((16,), (0.5,), (true,)), (coords,), (1, 2))
    ur = randn(2, 16, 9)
    edges = collect(range(0.0, 5.0; length = 11))
    for sf in (SFT.L2SFType(), SFT.L3SFType())
        _device_matches(sf, ur, sr, edges, 2)
    end
    um, valid = _masked(reshape(ur, 2, :), 0.2)
    _device_matches(SFT.S3SFType(), reshape(um, 2, 16, 9), sr, edges, 2; valid)
    # the stretched axis first in the field
    sr2 = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((16,), (0.5,), (true,)), (coords,), (2, 1))
    _device_matches(SFT.L2SFType(), randn(2, 9, 16), sr2, edges, 2)

    lats = collect(range(-1.2, 1.3; length = 11))
    n_lon = 20
    sz = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    uz = randn(2, n_lon, 11)
    zedges = collect(range(0.0, π; length = 13))
    for sf in (SFT.L2SFType(), SFT.T2SFType(), SFT.L3SFType(), SFT.S3SFType())
        _device_matches(sf, uz, sz, zedges, 2)
    end
    um, valid = _masked(reshape(uz, 2, :), 0.3)
    _device_matches(SFT.L2SFType(), reshape(um, 2, n_lon, 11), sz, zedges, 2; valid)
    fz = Fields(vectors = (uz,), scalars = (randn(n_lon, 11),))
    nb = length(zedges) - 1
    a, ca = zeros(nb), zeros(Int, nb)
    b, cb = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(a, ca, SFT.MixedSFType{1, 0, 2}(), fz, sz, zedges, FFT)
    SFC.gridded_sweep!(b, cb, SFT.MixedSFType{1, 0, 2}(), fz, sz, zedges, FFT; backend = DEV)
    Test.@test ca == cb
    Test.@test maximum(abs.(a .- b)) <= 1e-11 * maximum(abs, a)
end

Test.@testset "the joint histogram and the wide histogram take the global-atomic path" begin
    Random.seed!(14)
    dims = (20, 18)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    edges = collect(range(0.0, 12.0; length = 9))                # reaches the half-turn lags
    aedges = collect(range(prevfloat(0.0), π; length = 7))
    nb, na = length(edges) - 1, length(aedges) - 1
    axis = SFC.SeparationAngleAxis([1.0, 0.0])
    a, ca = zeros(nb, na), zeros(nb, na)
    b, cb = zeros(nb, na), zeros(nb, na)
    SFC.gridded_sweep!(a, ca, SFT.L2SFType(), u, s, edges, aedges, Val(2), FFT; second_axis = axis)
    SFC.gridded_sweep!(b, cb, SFT.L2SFType(), u, s, edges, aedges, Val(2), FFT; second_axis = axis, backend = DEV)
    Test.@test ca == cb
    Test.@test maximum(abs.(a .- b)) <= 1e-11 * maximum(abs, a)
    Test.@test_throws ArgumentError SFC.gridded_sweep!(zeros(nb, na), zeros(Int, nb, na), SFT.L2SFType(), u, s,
                                                      edges, aedges, Val(2), FFT; second_axis = axis, backend = DEV)
    # more bins than shared memory holds
    wide = collect(range(0.0, 12.0; length = 4001))
    _device_matches(SFT.L2SFType(), u, s, wide, 2)
end

Test.@testset "the device answers Auto with the transform and refuses what it cannot express" begin
    dims = (12, 10)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    edges = collect(range(0.0, 5.0; length = 8))
    _device_matches(SFT.L2SFType(), u, s, edges, 2; tag = SB.AutoSpectralBackend())
    nb = length(edges) - 1
    Test.@test_throws ArgumentError SFC.gridded_sweep!(zeros(nb), zeros(Int, nb), SFT.FullVectorStructureFunctionType{3}(),
                                                      u, s, edges, Val(2), SB.AutoSpectralBackend(); backend = DEV)
    Test.@test_throws ArgumentError SFC.gridded_lag_sweep!(zeros(nb), zeros(Int, nb), SFT.L2SFType(), u, s, edges,
                                                          Val(2); backend = DEV)
end
