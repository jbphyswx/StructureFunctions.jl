using Test: Test
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    HelperFunctions as SFH
using StructureFunctions.MultiFields: Fields
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using KernelAbstractions: KernelAbstractions as KA
using StaticArrays: StaticArrays as SA
using CUDA: CUDA
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using Distances: Distances
using Random: Random

const FFT = SB.FastFourierTransformSpectralBackend()
const GPU = CB.GPUBackend(CUDA.CUDABackend())
const CPU = Threads.nthreads() > 1 ? CB.ThreadedBackend() : CB.SerialBackend()

"""The device engine and the CPU engine agree: counts exactly (relatively when weighted), sums to round-off."""
function compare(name, sf, u, s, edges, D; valid = SFC.AllValid(), weights = nothing)
    nb = length(edges) - 1
    CT = weights === nothing ? Int : Float64
    a, ca = zeros(nb), zeros(CT, nb)
    b, cb = CUDA.zeros(Float64, nb), CUDA.zeros(CT, nb)
    SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(D), FFT; valid, weights, backend = CPU)
    SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(D), FFT; valid, weights, backend = GPU)
    Test.@testset "$name $(nameof(typeof(sf)))" begin
        Test.@test maximum(abs.(ca .- Array(cb))) / (CT <: Integer ? 1 : max(maximum(abs, ca), 1e-12)) <=
                   (CT <: Integer ? 0 : 1e-12)
        Test.@test maximum(abs.(a .- Array(b))) / max(maximum(abs, a), 1e-12) <= 1e-10
    end
end

"""Every slice of the device batch equals the single-slice device entry and the CPU batch."""
function compare_batch(name, sf, u, s, edges, D; valid = SFC.AllValid(), weights = nothing)
    nt = size(u)[end]
    nb = length(edges) - 1
    CT = weights === nothing ? Int : Float64
    ref, cref = CUDA.zeros(Float64, nb, nt), CUDA.zeros(CT, nb, nt)
    for t in 1:nt
        us = selectdim(u, ndims(u), t)
        vs = valid isa SFC.AllValid ? valid : view(valid, :, t)
        SFC.gridded_sweep!(view(ref, :, t), view(cref, :, t), sf, us, s, edges, Val(D), FFT;
                           valid = vs, weights, backend = GPU)
    end
    a, ca = zeros(nb, nt), zeros(CT, nb, nt)
    b, cb = CUDA.zeros(Float64, nb, nt), CUDA.zeros(CT, nb, nt)
    SFC.gridded_sweep_batch!(a, ca, sf, u, s, edges, Val(D), FFT; valid, weights, backend = CPU)
    SFC.gridded_sweep_batch!(b, cb, sf, u, s, edges, Val(D), FFT; valid, weights, backend = GPU)
    ref, cref, b, cb = Array(ref), Array(cref), Array(b), Array(cb)
    scale = max(maximum(abs, ref), 1e-12)
    cscale = CT <: Integer ? 1 : max(maximum(abs, cref), 1e-12)
    Test.@testset "$name $(nameof(typeof(sf)))" begin
        Test.@test max(maximum(abs, cb .- cref), maximum(abs, ca .- cref)) / cscale <= (CT <: Integer ? 0 : 1e-12)
        Test.@test max(maximum(abs, b .- ref), maximum(abs, a .- ref)) / scale <= 1e-10
    end
end

Test.@testset "transform engine on CUDA against the CPU engine" begin
    Random.seed!(1)

    # Uniform schedules: periodic, masked, masked and weighted, bounded; second and third order.
    Test.@testset "uniform 2-D" begin
        dims = (48, 48)
        s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
        u = randn(2, dims...)
        edges = collect(range(0.0, 22.5; length = 41))
        compare("periodic", SFT.L2SFType(), u, s, edges, 2)
        compare("periodic", SFT.L3SFType(), u, s, edges, 2)
        compare("periodic", SFT.T3SFType(), u, s, edges, 2)
        um = copy(u)
        umf = reshape(um, 2, :)
        umf[:, rand(size(umf, 2)) .< 0.25] .= NaN
        valid = SFC.field_validity(um)
        compare("masked", SFT.L2SFType(), um, s, edges, 2; valid)
        compare("masked, weighted", SFT.L3SFType(), um, s, edges, 2; valid, weights = 0.5 .+ rand(prod(dims)))
        compare("bounded", SFT.S3SFType(), u, SFC.UniformLagSchedule(dims, (1.0, 1.0), (false, false)), edges, 2)
    end

    # A three-dimensional grid periodic along two axes, with a projection about a fixed axis.
    Test.@testset "uniform 3-D" begin
        dims = (16, 16, 12)
        s = SFC.UniformLagSchedule(dims, (1.0, 1.0, 1.0), (true, true, false))
        u = randn(3, dims...)
        edges = collect(range(0.0, 10.0; length = 21))
        compare("3-D", SFT.L2SFType(), u, s, edges, 3)
        compare("3-D", SFT.T3SFType(), u, s, edges, 3)
        about_a = SFH.ReferenceAxisTransverseBasis(SA.SVector(1.0, sqrt(2.0), sqrt(3.0)))
        compare("3-D about an axis", SFT.ProjectedStructureFunctionType{0, 3}(about_a), u, s, edges, 3)
    end

    Test.@testset "rectilinear" begin
        coords = cumsum(0.8 .+ 0.4 .* rand(32))
        s = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((64,), (1.0,), (true,)), (coords,), (1, 2))
        compare("rectilinear", SFT.L2SFType(), randn(2, 64, 32), s, collect(range(0.0, 20.0; length = 31)), 2)
    end

    # The sphere, with edges off the lattice's separations since device transcendentals round differently.
    Test.@testset "zonal" begin
        n_lon, n_lat = 72, 36
        lats = collect(range(-π / 2 + π / (2n_lat), π / 2 - π / (2n_lat); length = n_lat))
        s = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
        u = randn(2, n_lon, n_lat)
        edges = collect(range(0.0, π; length = 65)) .+ 1e-3
        compare("zonal", SFT.L2SFType(), u, s, edges, 2)
        compare("zonal weighted", SFT.L2SFType(), u, s, edges, 2; weights = 0.5 .+ rand(n_lon * n_lat))
        f = Fields(vectors = (u,), scalars = (randn(n_lon, n_lat),))
        nb = length(edges) - 1
        a, ca = zeros(nb), zeros(Int, nb)
        b, cb = CUDA.zeros(Float64, nb), CUDA.zeros(Int, nb)
        SFC.gridded_sweep!(a, ca, SFT.MixedSFType{1, 0, 2}(), f, s, edges, FFT; backend = CPU)
        SFC.gridded_sweep!(b, cb, SFT.MixedSFType{1, 0, 2}(), f, s, edges, FFT; backend = GPU)
        Test.@test maximum(abs.(ca .- Array(cb))) == 0
        Test.@test maximum(abs.(a .- Array(b))) / maximum(abs, a) <= 1e-10
    end

    # A batch over a trailing slice axis: complete, masked per slice, masked and weighted, and on the sphere.
    Test.@testset "slice batch" begin
        Random.seed!(2)
        dims, nt = (48, 48), 3
        s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
        edges = collect(range(0.0, 22.5; length = 41))
        u = randn(2, dims..., nt)
        compare_batch("periodic", SFT.L2SFType(), u, s, edges, 2)
        um = copy(u)
        umf = reshape(um, 2, prod(dims), nt)
        for t in 1:nt
            umf[:, rand(prod(dims)) .< 0.2, t] .= NaN
        end
        valid = SFC.batch_validity(um)
        compare_batch("masked", SFT.L3SFType(), um, s, edges, 2; valid)
        compare_batch("masked, weighted", SFT.L2SFType(), um, s, edges, 2; valid, weights = 0.5 .+ rand(prod(dims)))
        n_lon, n_lat = 72, 36
        lats = collect(range(-π / 2 + π / (2n_lat), π / 2 - π / (2n_lat); length = n_lat))
        sz = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
        zedges = collect(range(0.0, π / 2; length = 33)) .+ 1e-3
        compare_batch("zonal", SFT.L2SFType(), randn(2, n_lon, n_lat, 2), sz, zedges, 2)
    end

    # Scattered points: each provider's device transforms against the host, and the two providers against each other.
    Test.@testset "scattered modes $(nameof(typeof(sf)))" for sf in (SFT.L2SFType(), SFT.S3SFType())
        N = 2000
        x = rand(2, N) .* (100.0, 60.0)
        u = randn(2, N)
        s = SFC.ScatteredModesSchedule(x, 30.0, (128, 96); taper = SF.GaussianTaper(0.3))
        edges = collect(range(0.0, 30.0; length = 31))
        nb = length(edges) - 1
        host = map((SFC.NonuniformFFTsSpectralBackend(), SFC.FINUFFTSpectralBackend())) do tag
            a, ca = zeros(nb), zeros(nb)
            b, cb = CUDA.zeros(Float64, nb), CUDA.zeros(Float64, nb)
            SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(2), tag; backend = CPU)
            SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(2), tag; backend = GPU)
            Test.@testset "$(nameof(typeof(tag))) device against host" begin
                Test.@test maximum(abs.(ca .- Array(cb))) / maximum(abs, ca) <= 1e-10
                Test.@test maximum(abs.(a .- Array(b))) / maximum(abs, a) <= 1e-9
            end
            (a, ca)
        end
        Test.@testset "providers against each other on the host" begin
            Test.@test maximum(abs.(host[1][2] .- host[2][2])) / maximum(abs, host[1][2]) <= 1e-10
            Test.@test maximum(abs.(host[1][1] .- host[2][1])) / maximum(abs, host[1][1]) <= 1e-9
        end
    end

    # The point tensor kernel against the serial tensor, flat and on the sphere, and against the transform on a grid.
    Test.@testset "point tensor" begin
        N = 500
        RAW = SF.StructureFunctionTensorSumsAndCounts
        function tensor_compare(name, P, x, u, edges; distance_metric = Distances.Euclidean())
            ref = SFC.calculate_structure_function_tensor(Val(P), x, u, edges, RAW; backend = CB.SerialBackend(),
                                                          distance_metric)
            got = SF.to_host(SFC.calculate_structure_function_tensor(Val(P), x, u, edges, RAW; backend = GPU,
                                                                     distance_metric))
            Test.@testset "$name P=$P" begin
                Test.@test maximum(abs.(Int.(ref.counts) .- Int.(got.counts))) == 0
                Test.@test maximum(abs.(ref.sums .- got.sums)) / maximum(abs, ref.sums) <= 1e-10
            end
        end
        edges = collect(range(0.0, 5.0; length = 21)) .+ 1e-3
        x, u = rand(2, N) .* 10.0, randn(2, N)
        tensor_compare("flat", 2, x, u, edges)
        tensor_compare("flat", 3, x, u, edges)
        xs = Matrix(hcat(rand(N) .* 2π, asin.(2 .* rand(N) .- 1))')
        tensor_compare("sphere", 3, xs, randn(2, N), collect(range(0.0, π; length = 17)) .+ 1e-3;
                       distance_metric = SFH.SphericalDistance(1.0))
        dims, spacing = (24, 18), (0.5, 0.25)
        ug = randn(2, dims...)
        xg = Matrix(hcat([[(i - 1) * spacing[1], (j - 1) * spacing[2]] for i in 1:dims[1], j in 1:dims[2]]...))
        gedges = collect(range(0.0, 8.0; length = 17)) .+ 1e-3
        sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
        Test.@testset "grid P=$P" for P in (2, 3)
            tsums = zeros(ntuple(_ -> 2, P)..., length(gedges) - 1)
            tcounts = zeros(Int, length(gedges) - 1)
            SFC.gridded_tensor_sweep!(tsums, tcounts, Val(P), reshape(ug, 2, :), sched, gedges, Val(2), FFT;
                                      backend = CPU)
            got = SF.to_host(SFC.calculate_structure_function_tensor(Val(P), xg, reshape(ug, 2, :), gedges, RAW;
                                                                     backend = GPU))
            Test.@test maximum(abs.(tcounts .- Int.(got.counts))) == 0
            Test.@test maximum(abs.(tsums .- got.sums)) / maximum(abs, tsums) <= 1e-9
        end
    end

    # The grid tensor and the six single-pass invariants into device buffers, on the transform and the lag sweep.
    Test.@testset "grid tensor and single pass" begin
        Random.seed!(3)
        dims = (32, 24)
        s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, false))
        u = randn(2, dims...)
        edges = collect(range(0.0, 12.0; length = 21)) .+ 1e-3
        nb = length(edges) - 1
        Test.@testset "grid tensor P=$P" for P in (2, 3)
            a, ca = zeros(ntuple(_ -> 2, P)..., nb), zeros(Int, nb)
            SFC.gridded_tensor_sweep!(a, ca, Val(P), reshape(u, 2, :), s, edges, Val(2), FFT; backend = CPU)
            b, cb = CUDA.zeros(Float64, ntuple(_ -> 2, P)..., nb), CUDA.zeros(Int, nb)
            SFC.gridded_tensor_sweep!(b, cb, Val(P), reshape(u, 2, :), s, edges, Val(2), FFT; backend = GPU)
            Test.@test maximum(abs.(ca .- Array(cb))) == 0
            Test.@test maximum(abs.(a .- Array(b))) / maximum(abs, a) <= 1e-9
        end
        SP = SFT.SinglePassInvariants()
        Test.@testset "single pass $(nameof(typeof(tag)))" for tag in (FFT, SB.DirectSumSpectralBackend())
            a, ca = zeros(6, nb), zeros(Int, 6, nb)
            SFC.gridded_sweep!(a, ca, SP, u, s, edges, Val(2), tag; backend = CPU)
            b, cb = CUDA.zeros(Float64, 6, nb), CUDA.zeros(Int, 6, nb)
            SFC.gridded_sweep!(b, cb, SP, u, s, edges, Val(2), tag; backend = GPU)
            Test.@test maximum(abs.(ca .- Array(cb))) == 0
            Test.@test maximum(abs.(a .- Array(b))) / maximum(abs, a) <= 1e-9
        end
    end
end
