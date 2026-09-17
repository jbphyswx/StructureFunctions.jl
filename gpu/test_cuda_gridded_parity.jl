# =============================================================================
# Parity and timing of the transform engine on CUDA against the CPU engine: uniform, rectilinear and
# zonal schedules, masked and complete, second and third order. Counts must be exact, sums to
# round-off. Both engines run inside the same job, so the timings are comparable.
#   julia --project=gpu gpu/test_cuda_gridded_parity.jl
# =============================================================================
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
using Printf: Printf
using Random: Random

const FFT = SB.FastFourierTransformSpectralBackend()
const GPU = CB.GPUBackend(CUDA.CUDABackend())
const CPU = Threads.nthreads() > 1 ? CB.ThreadedBackend() : CB.SerialBackend()

println("device: ", CUDA.name(CUDA.device()), "   host threads: ", Threads.nthreads())
println("| case | operator | max |Δcount| (relative when weighted) | max |Δsum|/scale | cpu s | gpu s |")

failures = Ref(0)
function compare(name, sf, u, s, edges, D; valid = SFC.AllValid(), weights = nothing)
    nb = length(edges) - 1
    CT = weights === nothing ? Int : Float64
    a, ca = zeros(nb), zeros(CT, nb)
    b, cb = zeros(nb), zeros(CT, nb)
    SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(D), FFT; valid, weights, backend = CPU)
    SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(D), FFT; valid, weights, backend = GPU)
    fill!(a, 0); fill!(ca, 0); fill!(b, 0); fill!(cb, 0)
    tc = @elapsed SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(D), FFT; valid, weights, backend = CPU)
    tg = @elapsed (SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(D), FFT; valid, weights, backend = GPU); CUDA.synchronize())
    dc = maximum(abs.(ca .- cb)) / (CT <: Integer ? 1 : max(maximum(abs, ca), 1e-12))
    ds = maximum(abs.(a .- b)) / max(maximum(abs, a), 1e-12)
    Printf.@printf("| %s | %s | %.1e | %.1e | %.3f | %.3f |\n", name, nameof(typeof(sf)), dc, ds, tc, tg)
    (dc <= (CT <: Integer ? 0 : 1e-12) && ds <= 1e-10) || (failures[] += 1)
    return nothing
end

Random.seed!(1)
let dims = (256, 256)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    edges = collect(range(0.0, 120.0; length = 41))
    compare("uniform 256² periodic", SFT.L2SFType(), u, s, edges, 2)
    compare("uniform 256² periodic", SFT.L3SFType(), u, s, edges, 2)
    compare("uniform 256² periodic", SFT.T3SFType(), u, s, edges, 2)
    um = copy(u)
    umf = reshape(um, 2, :)
    umf[:, rand(size(umf, 2)) .< 0.25] .= NaN
    valid = SFC.field_validity(um)
    compare("uniform 256² masked", SFT.L2SFType(), um, s, edges, 2; valid)
    compare("uniform 256² masked, weighted", SFT.L3SFType(), um, s, edges, 2; valid, weights = 0.5 .+ rand(prod(dims)))
    sb = SFC.UniformLagSchedule(dims, (1.0, 1.0), (false, false))
    compare("uniform 256² bounded", SFT.S3SFType(), u, sb, edges, 2)
end

let dims = (48, 48, 40)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0, 1.0), (true, true, false))
    u = randn(3, dims...)
    compare("uniform 48×48×40", SFT.L2SFType(), u, s, collect(range(0.0, 30.0; length = 31)), 3)
    compare("uniform 48×48×40", SFT.T3SFType(), u, s, collect(range(0.0, 30.0; length = 31)), 3)
    about_a = SFH.ReferenceAxisTransverseBasis(SA.SVector(1.0, sqrt(2.0), sqrt(3.0)))
    compare("uniform 48×48×40 axis", SFT.ProjectedStructureFunctionType{0, 3}(about_a), u, s,
            collect(range(0.0, 30.0; length = 31)), 3)
end

let coords = cumsum(0.8 .+ 0.4 .* rand(128))
    s = SFC.RectilinearLagSchedule(SFC.UniformLagSchedule((256,), (1.0,), (true,)), (coords,), (1, 2))
    u = randn(2, 256, 128)
    compare("rectilinear 256×128", SFT.L2SFType(), u, s, collect(range(0.0, 60.0; length = 31)), 2)
end

let n_lon = 720, n_lat = 360
    lats = collect(range(-π / 2 + π / (2n_lat), π / 2 - π / (2n_lat); length = n_lat))
    s = SFC.ZonalLagSchedule(lats, n_lon, 2π / n_lon, 1.0, true)
    u = randn(2, n_lon, n_lat)
    # edges off the lattice's own separations: on the sphere the device's transcendentals round differently
    # from the host's, so a pair exactly on an edge may change bins
    edges = collect(range(0.0, π; length = 65)) .+ 1e-3
    compare("zonal 720×360", SFT.L2SFType(), u, s, edges, 2)
    compare("zonal 720×360 weighted", SFT.L2SFType(), u, s, edges, 2; weights = 0.5 .+ rand(n_lon * n_lat))
    f = Fields(vectors = (u,), scalars = (randn(n_lon, n_lat),))
    nb = length(edges) - 1
    a, ca = zeros(nb), zeros(Int, nb)
    b, cb = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(a, ca, SFT.MixedSFType{1, 0, 2}(), f, s, edges, FFT; backend = CPU)
    SFC.gridded_sweep!(b, cb, SFT.MixedSFType{1, 0, 2}(), f, s, edges, FFT; backend = GPU)
    dc = maximum(abs.(ca .- cb))
    ds = maximum(abs.(a .- b)) / maximum(abs, a)
    Printf.@printf("| %s | %s | %d | %.1e | - | - |\n", "zonal 720×360 multi-field", "Mixed{1,0,2}", dc, ds)
    (dc == 0 && ds <= 1e-10) || (failures[] += 1)
end

# The non-uniform FFT route: the same engine on scattered points, its forward transforms from each provider on
# the device the field lives on (cuFINUFFT for FINUFFT); counts are a kernel-weighted mass, compared relatively.
# The two providers are also held to each other on the host.
let N = 20_000
    x = rand(2, N) .* (100.0, 60.0)
    u = randn(2, N)
    s = SFC.ScatteredModesSchedule(x, 30.0, (256, 192); taper = SF.GaussianTaper(0.3))
    edges = collect(range(0.0, 30.0; length = 31))
    nb = length(edges) - 1
    providers = (SFC.NonuniformFFTsSpectralBackend(), SFC.FINUFFTSpectralBackend())
    for sf in (SFT.L2SFType(), SFT.S3SFType())
        host = map(providers) do tag
            a, ca = zeros(nb), zeros(nb)
            b, cb = zeros(nb), zeros(nb)
            SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(2), tag; backend = CPU)
            SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(2), tag; backend = GPU)
            fill!(a, 0); fill!(ca, 0); fill!(b, 0); fill!(cb, 0)
            tc = @elapsed SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(2), tag; backend = CPU)
            tg = @elapsed (SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(2), tag; backend = GPU); CUDA.synchronize())
            dc = maximum(abs.(ca .- cb)) / maximum(abs, ca)
            ds = maximum(abs.(a .- b)) / maximum(abs, a)
            Printf.@printf("| %s (%s) | %s | %.1e | %.1e | %.3f | %.3f |\n", "scattered 20k → 256×192 modes",
                           nameof(typeof(tag)), nameof(typeof(sf)), dc, ds, tc, tg)
            (dc <= 1e-10 && ds <= 1e-9) || (failures[] += 1)
            (a, ca)
        end
        dc = maximum(abs.(host[1][2] .- host[2][2])) / maximum(abs, host[1][2])
        ds = maximum(abs.(host[1][1] .- host[2][1])) / maximum(abs, host[1][1])
        Printf.@printf("| %s | %s | %.1e | %.1e | - | - |\n", "scattered: NonuniformFFTs vs FINUFFT on the host",
                       nameof(typeof(sf)), dc, ds)
        (dc <= 1e-10 && ds <= 1e-9) || (failures[] += 1)
    end
end

# The tensor kernel: every component of the rank-P moment tensor, an odd rank taking the canonical pair
# reading in a fixed frame and none in the sphere's geodesic frame, against the serial point tensor and
# against the transform tensor on a grid.
let N = 3000
    RAW = SF.StructureFunctionTensorSumsAndCounts
    function tensor_compare(name, P, x, u, edges; distance_metric = Distances.Euclidean())
        ref = SFC.calculate_structure_function_tensor(Val(P), x, u, edges; backend = CB.SerialBackend(),
                                                      distance_metric, output_type = RAW)
        got = SFC.calculate_structure_function_tensor(Val(P), x, u, edges; backend = GPU,
                                                      distance_metric, output_type = RAW)
        CUDA.synchronize()
        dc = maximum(abs.(Int.(ref.counts) .- Int.(got.counts)))
        ds = maximum(abs.(ref.sums .- got.sums)) / maximum(abs, ref.sums)
        Printf.@printf("| %s | tensor P=%d | %d | %.1e | - | - |\n", name, P, dc, ds)
        (dc == 0 && ds <= 1e-10) || (failures[] += 1)
        return nothing
    end
    x = rand(2, N) .* 10.0
    u = randn(2, N)
    edges = collect(range(0.0, 5.0; length = 21)) .+ 1e-3
    tensor_compare("points 3000 flat", 2, x, u, edges)
    tensor_compare("points 3000 flat", 3, x, u, edges)
    lam = rand(N) .* 2π
    phi = asin.(2 .* rand(N) .- 1)
    xs = Matrix(hcat(lam, phi)')
    us = randn(2, N)
    tensor_compare("points 3000 sphere", 3, xs, us, collect(range(0.0, π; length = 17)) .+ 1e-3;
                   distance_metric = SFH.SphericalDistance(1.0))
    dims, spacing = (64, 48), (0.5, 0.25)
    ug = randn(2, dims...)
    xg = Matrix(hcat([[(i - 1) * spacing[1], (j - 1) * spacing[2]] for i in 1:dims[1], j in 1:dims[2]]...))
    gedges = collect(range(0.0, 8.0; length = 17)) .+ 1e-3
    sched = SFC.UniformLagSchedule(dims, spacing, (false, false))
    for P in (2, 3)
        tsums = zeros(ntuple(_ -> 2, P)..., length(gedges) - 1)
        tcounts = zeros(Int, length(gedges) - 1)
        SFC.gridded_tensor_sweep!(tsums, tcounts, Val(P), reshape(ug, 2, :), sched, gedges, Val(2), FFT; backend = CPU)
        got = SFC.calculate_structure_function_tensor(Val(P), xg, reshape(ug, 2, :), gedges; backend = GPU,
                                                      output_type = RAW)
        CUDA.synchronize()
        dc = maximum(abs.(tcounts .- Int.(got.counts)))
        ds = maximum(abs.(tsums .- got.sums)) / maximum(abs, tsums)
        Printf.@printf("| %s | transform vs gpu tensor P=%d | %d | %.1e | - | - |\n", "grid 64×48", P, dc, ds)
        (dc == 0 && ds <= 1e-9) || (failures[] += 1)
    end
end

println(failures[] == 0 ? "PARITY OK" : "PARITY FAILED in $(failures[]) case(s)")
exit(failures[] == 0 ? 0 : 1)
