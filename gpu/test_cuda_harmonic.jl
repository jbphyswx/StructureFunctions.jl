using Test: Test
using CUDA: CUDA
using Random: Random
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

rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / (maximum(abs, Array(b)) + eps())
sphere(N) = (acos.(2 .* rand(N) .- 1), 2π .* rand(N))

# (points, degree, spin): every spin, the paired orders of spin 0, and points spanning several batches per order chunk
const COEFFICIENT_CASES = ((4000, 16, 0), (4000, 16, 2), (4000, 64, -1), (4000, 128, 1))

Test.@testset "harmonic route on the device" begin
    Random.seed!(20260926)

    # The tiled direct sum of pseudo-coefficients against the threaded host.
    Test.@testset "coefficients N=$N lmax=$lmax spin=$s" for (N, lmax, s) in COEFFICIENT_CASES
        θ, φ = sphere(N)
        f = s == 0 ? complex.(randn(N)) : randn(ComplexF64, N)
        ref = SFC.pseudo_coefficients_direct(f, θ, φ, s, lmax; backend = THR)
        got = SFC.pseudo_coefficients_direct(f, θ, φ, s, lmax; backend = DEV)
        Test.@test got isa CUDA.CuArray
        Test.@test rel(got, ref) < 1e-10
    end

    # The route end to end and its spectra on the direct sum and on NUFSHT's device plans, which run cuFINUFFT.
    Test.@testset "route" begin
        N, lmax = 2000, 24
        θ, φ = sphere(N)
        x = permutedims(hcat(φ, π / 2 .- θ))
        u = randn(2, N)
        valid = rand(N) .> 0.2
        w = 0.5 .+ rand(N)
        nodes = SF.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), lmax)
        ref = SFC.calculate_structure_function(SFT.L2SFType(), x, u, nodes, DS, RAW; weights = w, valid, backend = SER)
        Test.@testset "$(nameof(typeof(tag)))" for tag in (DS, NU)
            got = SFC.calculate_structure_function(SFT.L2SFType(), x, u, nodes, tag, RAW; weights = w, valid,
                                                   backend = DEV)
            Test.@test got.sums isa CUDA.CuArray
            Test.@test rel(got.sums, ref.sums) < 1e-8
            Test.@test rel(got.counts, ref.counts) < 1e-8
            a = SFC.harmonic_spectra(x, u, lmax, tag; weights = w, valid, backend = SER)
            b = SFC.harmonic_spectra(x, u, lmax, tag; weights = w, valid, backend = DEV)
            Test.@test b.EE isa CUDA.CuArray
            Test.@test rel(b.EE, a.EE) < 1e-8
            Test.@test rel(b.BB, a.BB) < 1e-8
        end
        provider = Base.get_extension(SF, :StructureFunctionsNUFSHTExt)._nufsht_provider(CUDA.CuArray(θ),
                                                                                           CUDA.CuArray(φ), lmax)
        C = provider(CUDA.CuArray(randn(ComplexF64, N)), 1)
        Test.@test C isa CUDA.CuArray
        Test.@test occursin("CuNUFFTPlan", string(typeof(provider.plans[0])))
    end
end
