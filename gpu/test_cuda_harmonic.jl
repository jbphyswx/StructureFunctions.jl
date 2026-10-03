# The harmonic route on real CUDA: the tiled direct sum of pseudo-coefficients against the host at several
# (N, lmax, spin), the route end to end on the direct sum and on NUFSHT's device plans with the NUFFT they run
# recorded, the spectra, device residency, and the time of the direct sum against the threaded host.
using CUDA: CUDA
using Random: Random
using Printf: Printf
using OhMyThreads: OhMyThreads
using FINUFFT: FINUFFT
using NUFSHT: NUFSHT
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const THR = CB.ThreadedBackend()
const DS = SB.DirectSumSpectralBackend()
const NU = SB.NUFSHTSpectralBackend()
const RAW = SF.StructureFunctionSumsAndCounts

failures = String[]
check(name, ok, detail) = (ok || push!(failures, name);
                           Printf.@printf("%-56s %s  %s\n", name, detail, ok ? "ok" : "FAILED"))
rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / (maximum(abs, Array(b)) + eps())
sphere(N) = (acos.(2 .* rand(N) .- 1), 2π .* rand(N))

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260926)

for (N, lmax, spins) in ((4000, 16, (0, 1, 2)), (20_000, 64, (0, 1, -1)), (60_000, 128, (0, 1)))
    θ, φ = sphere(N)
    for s in spins
        f = s == 0 ? complex.(randn(N)) : randn(ComplexF64, N)
        ref = SFC.pseudo_coefficients_direct(f, θ, φ, s, lmax; backend = THR)
        got = SFC.pseudo_coefficients_direct(f, θ, φ, s, lmax; backend = DEV)
        check("coefficients N=$N lmax=$lmax spin=$s", got isa CUDA.CuArray && rel(got, ref) < 1e-10,
              Printf.@sprintf("rel=%.3e", rel(got, ref)))
    end
end

let N = 4000, lmax = 24
    θ, φ = sphere(N)
    x = permutedims(hcat(φ, π / 2 .- θ))
    u = randn(2, N)
    valid = rand(N) .> 0.2
    w = 0.5 .+ rand(N)
    nodes = SF.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), lmax)
    ref = SFC.calculate_structure_function(SFT.L2SFType(), x, u, nodes, DS, RAW; weights = w, valid, backend = SER)
    for tag in (DS, NU)
        got = SFC.calculate_structure_function(SFT.L2SFType(), x, u, nodes, tag, RAW; weights = w, valid,
                                               backend = DEV)
        check("route $(nameof(typeof(tag))) on the device",
              got.sums isa CUDA.CuArray && rel(got.sums, ref.sums) < 1e-8 && rel(got.counts, ref.counts) < 1e-8,
              Printf.@sprintf("Δsum=%.3e Δcount=%.3e", rel(got.sums, ref.sums), rel(got.counts, ref.counts)))
        a = SFC.harmonic_spectra(x, u, lmax, tag; weights = w, valid, backend = SER)
        b = SFC.harmonic_spectra(x, u, lmax, tag; weights = w, valid, backend = DEV)
        check("spectra $(nameof(typeof(tag))) on the device",
              b.EE isa CUDA.CuArray && rel(b.EE, a.EE) < 1e-8 && rel(b.BB, a.BB) < 1e-8,
              Printf.@sprintf("ΔEE=%.3e ΔBB=%.3e", rel(b.EE, a.EE), rel(b.BB, a.BB)))
    end
    provider = Base.get_extension(SF, :StructureFunctionsNUFSHTExt)._nufsht_provider(CUDA.CuArray(θ), CUDA.CuArray(φ),
                                                                                       lmax)
    C = provider(CUDA.CuArray(randn(ComplexF64, N)), 1)
    kind = string(typeof(provider.plans[0]))
    println("NUFSHT device plan NUFFT: ", occursin("CuNUFFTPlan", kind) ? "cuFINUFFT" : kind)
    check("NUFSHT runs its plan on the device", C isa CUDA.CuArray && occursin("CuNUFFTPlan", kind), "")
end

let N = 200_000
    θ, φ = sphere(N)
    θd, φd = CUDA.CuArray(θ), CUDA.CuArray(φ)
    for (lmax, s) in ((64, 0), (128, 0), (128, 1))
        fd = CUDA.CuArray(randn(ComplexF64, N))
        dev() = CUDA.@sync SFC.pseudo_coefficients_direct(fd, θd, φd, s, lmax; backend = DEV)
        dev()
        t_dev = minimum(@elapsed(dev()) for _ in 1:3)
        f = Array(fd)
        t_host = @elapsed SFC.pseudo_coefficients_direct(f, θ, φ, s, lmax; backend = THR)
        Printf.@printf("coefficients N=%d lmax=%d spin=%d  device %.2f ms  host threaded (%d) %.2f ms\n", N, lmax, s,
                       1e3 * t_dev, Threads.nthreads(), 1e3 * t_host)
    end
end

if isempty(failures)
    println("\nCUDA HARMONIC OK")
else
    println("\nCUDA HARMONIC FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
