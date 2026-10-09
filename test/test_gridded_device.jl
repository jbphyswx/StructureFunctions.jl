using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using StructureFunctions.MultiFields: Fields
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

# (operator, periodic flags): every topology, an odd operator across a half-turn.
const UNIFORM_CASES = ((SFT.L2SFType(), (true, true)), (SFT.T2SFType(), (false, false)),
                       (SFT.L3SFType(), (true, false)))

# Every topology, a masked field with a periodic and an open axis, and a 3D grid with log bins.
Test.@testset "the device engine equals the CPU engine on a uniform grid" begin
    Random.seed!(11)
    dims = (12, 10)
    edges = collect(range(0.0, 7.0; length = 8))
    for (sf, periodic) in UNIFORM_CASES
        _device_matches(sf, randn(2, dims...), SFC.UniformLagSchedule(dims, (0.5, 0.7), periodic), edges, 2)
    end
    um, valid = _masked(randn(2, prod(dims)), 0.25)
    _device_matches(SFT.L2SFType(), reshape(um, 2, dims...), SFC.UniformLagSchedule(dims, (0.5, 0.7), (true, false)),
                    edges, 2; valid)
    s3 = SFC.UniformLagSchedule((4, 5, 3), (1.0, 1.1, 0.9), (true, false, true))
    _device_matches(SFT.S3SFType(), randn(3, 4, 5, 3), s3, SF.LogBinEdges(0.8, 6.0, 8), 3; count_type = Float64)
end

Test.@testset "multi-fields and higher moments ride the device engine" begin
    Random.seed!(12)
    dims = (8, 6)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    θ = randn(dims...)
    f = Fields(vectors = (u,), scalars = (θ,))
    edges = collect(range(0.0, 8.0; length = 8))
    nb = length(edges) - 1
    for sf in (SFT.MixedSFType{1, 0, 2}(), SFT.ProjectedStructureFunctionType{4, 0}())
        a, ca = zeros(nb), zeros(Int, nb)
        b, cb = zeros(nb), zeros(Int, nb)
        SFC.gridded_sweep!(a, ca, sf, f, s, edges, FFT)
        SFC.gridded_sweep!(b, cb, sf, f, s, edges, FFT; backend = DEV)
        Test.@test ca == cb
        Test.@test maximum(abs.(a .- b)) <= 1e-11 * maximum(abs, a)
    end
end

# Rectilinear schedules in both axis orders, masked or not, and a masked zonal schedule.
Test.@testset "rectilinear and zonal schedules run on the device" begin
    Random.seed!(13)
    coords = cumsum(0.4 .+ 0.6 .* rand(5))
    sr = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((8,), (0.5,), (true,)), (coords,), (1, 2))
    ur = randn(2, 8, 5)
    edges = collect(range(0.0, 5.0; length = 11))
    um, valid = _masked(reshape(ur, 2, :), 0.2)
    _device_matches(SFT.L2SFType(), reshape(um, 2, 8, 5), sr, edges, 2; valid)
    sr2 = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((8,), (0.5,), (true,)), (coords,), (2, 1))
    _device_matches(SFT.L2SFType(), randn(2, 5, 8), sr2, edges, 2)

    lats = collect(range(-1.2, 1.3; length = 6))
    n_lon = 8
    sz = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    uz = randn(2, n_lon, 6)
    zedges = collect(range(0.0, π; length = 13))
    um, valid = _masked(reshape(uz, 2, :), 0.3)
    _device_matches(SFT.L2SFType(), reshape(um, 2, n_lon, 6), sz, zedges, 2; valid)
end

# A joint histogram, refusing integer counts on the device, and 4000 bins, past the shared fit, match the CPU.
Test.@testset "the joint histogram and a histogram past the shared-memory fit match the CPU" begin
    Random.seed!(14)
    dims = (8, 6)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    edges = collect(range(0.0, 6.0; length = 9))
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
    wide = collect(range(0.0, 6.0; length = 4001))
    _device_matches(SFT.L2SFType(), u, s, wide, 2)
end

# Auto on the device matches the CPU for a polynomial operator and for one with no transform.
Test.@testset "the device answers Auto for a polynomial operator and for one with no transform" begin
    dims = (12, 10)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    edges = collect(range(0.0, 5.0; length = 8))
    _device_matches(SFT.L2SFType(), u, s, edges, 2; tag = SB.AutoSpectralBackend())
    nb = length(edges) - 1
    op = SFT.FullVectorStructureFunctionType{3}()
    rs, rc = zeros(nb), zeros(Int, nb)
    SFC.gridded_lag_sweep!(rs, rc, op, u, s, edges, Val(2); backend = CB.SerialBackend())
    gs, gc = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(gs, gc, op, u, s, edges, Val(2), SB.AutoSpectralBackend(); backend = DEV)
    Test.@test sum(rc) > 0
    Test.@test gc == rc
    Test.@test isapprox(gs, rs; rtol = 1e-11, atol = 1e-12)
end

# (schedule, axis): every schedule once and every axis once; a sphere takes no angle axis.
const LAG_JOINT_CASES = ((:uniform, :angle), (:rectilinear, :value), (:zonal, :value))
const LAG_BATCH_CASES = ((:uniform, :none), (:rectilinear, :value), (:zonal, :value))

Test.@testset "the joint histograms and the batches run the lag sweep on the device" begin
    Random.seed!(9002)
    vbins = collect(range(-2.0, 2.0; length = 9))
    abins = collect(range(prevfloat(0.0), π; length = 5))
    edges = collect(range(0.0, 3.0; length = 7)) .+ 0.0137
    nb = length(edges) - 1
    ys = collect(range(0.0, 1.4; length = 5))
    lats = collect(range(-1.0, 1.0; length = 5))
    schedules = (uniform = (SFC.UniformLagSchedule((10, 8), (0.3, 0.4), (true, false)), (10, 8)),
                 rectilinear = (SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((10,), (0.3,), (false,)), (ys,),
                                                           (1, 2)), (10, 5)),
                 zonal = (SFC.ZonalLagSchedule(lats, 8, 2π / 8, 1.0, true), (8, 5)))
    axes = (value = (SFC.InvariantValueAxis(), vbins), angle = (SFC.SeparationAngleAxis([1.0, 0.0]), abins))
    sf = SFT.L2SFType()
    for (name, axis) in LAG_JOINT_CASES
        sched, fdims = schedules[name]
        ax, ab = axes[axis]
        u = randn(2, fdims...)
        na = length(ab) - 1
        rs, rc = zeros(nb, na), zeros(nb, na)
        SFC.gridded_lag_sweep!(rs, rc, sf, u, sched, edges, ab, Val(2); second_axis = ax, backend = CB.SerialBackend())
        gs, gc = zeros(nb, na), zeros(nb, na)
        SFC.gridded_lag_sweep!(gs, gc, sf, u, sched, edges, ab, Val(2); second_axis = ax, backend = DEV)
        Test.@test sum(rc) > 0
        Test.@test isapprox(gc, rc; rtol = 1e-12)
        Test.@test isapprox(gs, rs; rtol = 1e-11, atol = 1e-12)
    end
    # Each batch misses a different cell in two of its slices.
    for (name, axis) in LAG_BATCH_CASES
        sched, fdims = schedules[name]
        ub = randn(2, fdims..., 3)
        ub[:, 2, 3, 1] .= NaN
        ub[:, 4, 1, 3] .= NaN
        valid = SFC.batch_validity(ub)
        if axis === :none
            rs, rc = zeros(nb, 3), zeros(Int, nb, 3)
            SFC.gridded_lag_sweep_batch!(rs, rc, sf, ub, sched, edges, Val(2); valid, backend = CB.SerialBackend())
            gs, gc = zeros(nb, 3), zeros(Int, nb, 3)
            SFC.gridded_lag_sweep_batch!(gs, gc, sf, ub, sched, edges, Val(2); valid, backend = DEV)
            Test.@test gc == rc
        else
            ax, ab = axes[axis]
            na = length(ab) - 1
            rs, rc = zeros(nb, na, 3), zeros(nb, na, 3)
            SFC.gridded_lag_sweep_batch!(rs, rc, sf, ub, sched, edges, ab, Val(2); second_axis = ax, valid,
                                         backend = CB.SerialBackend())
            gs, gc = zeros(nb, na, 3), zeros(nb, na, 3)
            SFC.gridded_lag_sweep_batch!(gs, gc, sf, ub, sched, edges, ab, Val(2); second_axis = ax, valid,
                                         backend = DEV)
            Test.@test isapprox(gc, rc; rtol = 1e-12)
        end
        Test.@test sum(rc) > 0
        Test.@test isapprox(gs, rs; rtol = 1e-11, atol = 1e-12)
    end
end

# The norm of odd order, which has no transform, on every schedule.
const DIRECT_CASES = ((:uniform, SFT.FullVectorStructureFunctionType{3}()),
                      (:rectilinear, SFT.FullVectorStructureFunctionType{3}()),
                      (:zonal, SFT.FullVectorStructureFunctionType{3}()))

Test.@testset "the direct lag sweep runs on the device, for the operators the transform refuses" begin
    Random.seed!(9001)
    nb = 6
    lats = collect(range(-60.0, 60.0; length = 5)) .* (π / 180)
    cases = (uniform = (SFC.UniformLagSchedule((8, 8), (1 / 8, 1 / 8), (true, true)), reshape(randn(2, 8, 8), 2, :),
                        collect(range(0.0, 0.5; length = nb + 1))),
             rectilinear = (SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((7,), (0.15,), (false,)),
                                                       (collect(range(0.0, 1.0; length = 5)),), (2, 1)),
                            reshape(randn(2, 7, 5), 2, :), collect(range(0.0, 0.7; length = nb + 1))),
             zonal = (SFC.ZonalLagSchedule(lats, 12, 2π / 12, 1.0, true), reshape(randn(2, 12, 5), 2, :),
                      collect(range(0.0, 1.5; length = nb + 1))))
    for (name, op) in DIRECT_CASES
        sched, u, edges = cases[name]
        rs, rc = zeros(nb), zeros(Int, nb)
        SFC.gridded_lag_sweep!(rs, rc, op, u, sched, edges, Val(2), Val(1), Val(0); backend = CB.SerialBackend())
        gs, gc = zeros(nb), zeros(Int, nb)
        SFC.gridded_lag_sweep!(gs, gc, op, u, sched, edges, Val(2), Val(1), Val(0); backend = DEV)
        Test.@test sum(rc) > 0
        Test.@test gc == rc
        Test.@test isapprox(gs, rs; rtol = 1e-11, atol = 1e-12)
    end
end
