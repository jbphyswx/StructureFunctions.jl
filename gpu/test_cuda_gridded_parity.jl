# =============================================================================
# Parity and timing of the transform engine on CUDA against the CPU engine: uniform, rectilinear and
# zonal schedules, masked and complete, second and third order. Counts must be exact, sums to
# round-off. Both engines run inside the same job, so the timings are comparable.
#   julia --project=gpu gpu/test_cuda_gridded_parity.jl
# =============================================================================
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT, Fields
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using KernelAbstractions: KernelAbstractions as KA
using CUDA, FFTW, OhMyThreads
using Printf: @printf
using Random: Random

const FFT = SB.FastFourierTransformSpectralBackend()
const GPU = CB.GPUBackend(CUDA.CUDABackend())
const CPU = Threads.nthreads() > 1 ? CB.ThreadedBackend() : CB.SerialBackend()

println("device: ", CUDA.name(CUDA.device()), "   host threads: ", Threads.nthreads())
println("| case | operator | max |Δcount| | max |Δsum|/scale | cpu s | gpu s |")

failures = Ref(0)
function compare(name, sf, u, s, edges, D; valid = SFC.AllValid())
    nb = length(edges) - 1
    a, ca = zeros(nb), zeros(Int, nb)
    b, cb = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(D), FFT; valid, backend = CPU)
    SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(D), FFT; valid, backend = GPU)
    fill!(a, 0); fill!(ca, 0); fill!(b, 0); fill!(cb, 0)
    tc = @elapsed SFC.gridded_sweep!(a, ca, sf, u, s, edges, Val(D), FFT; valid, backend = CPU)
    tg = @elapsed (SFC.gridded_sweep!(b, cb, sf, u, s, edges, Val(D), FFT; valid, backend = GPU); CUDA.synchronize())
    dc = maximum(abs.(ca .- cb))
    ds = maximum(abs.(a .- b)) / max(maximum(abs, a), 1e-12)
    @printf("| %s | %s | %d | %.1e | %.3f | %.3f |\n", name, nameof(typeof(sf)), dc, ds, tc, tg)
    (dc == 0 && ds <= 1e-10) || (failures[] += 1)
    return nothing
end

Random.seed!(1)
let dims = (256, 256)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0), (true, true))
    u = randn(2, dims...)
    edges = collect(range(0.0, 120.0; length = 41))
    compare("uniform 256² periodic", SFT.L2SFType(), u, s, edges, 2)
    compare("uniform 256² periodic", SFT.L3SFType(), u, s, edges, 2)
    um = copy(u)
    umf = reshape(um, 2, :)
    umf[:, rand(size(umf, 2)) .< 0.25] .= NaN
    valid = SFC.field_validity(um)
    compare("uniform 256² masked", SFT.L2SFType(), um, s, edges, 2; valid)
    sb = SFC.UniformLagSchedule(dims, (1.0, 1.0), (false, false))
    compare("uniform 256² bounded", SFT.S3SFType(), u, sb, edges, 2)
end

let dims = (48, 48, 40)
    s = SFC.UniformLagSchedule(dims, (1.0, 1.0, 1.0), (true, true, false))
    u = randn(3, dims...)
    compare("uniform 48×48×40", SFT.L2SFType(), u, s, collect(range(0.0, 30.0; length = 31)), 3)
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
    f = Fields(vectors = (u,), scalars = (randn(n_lon, n_lat),))
    nb = length(edges) - 1
    a, ca = zeros(nb), zeros(Int, nb)
    b, cb = zeros(nb), zeros(Int, nb)
    SFC.gridded_sweep!(a, ca, SFT.MixedSFType{1, 0, 2}(), f, s, edges, FFT; backend = CPU)
    SFC.gridded_sweep!(b, cb, SFT.MixedSFType{1, 0, 2}(), f, s, edges, FFT; backend = GPU)
    dc = maximum(abs.(ca .- cb))
    ds = maximum(abs.(a .- b)) / maximum(abs, a)
    @printf("| %s | %s | %d | %.1e | - | - |\n", "zonal 720×360 bundle", "Mixed{1,0,2}", dc, ds)
    (dc == 0 && ds <= 1e-10) || (failures[] += 1)
end

println(failures[] == 0 ? "PARITY OK" : "PARITY FAILED in $(failures[]) case(s)")
exit(failures[] == 0 ? 0 : 1)
