using Test: Test
using CUDA: CUDA
using Bessels: Bessels
using Random: Random
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const OUT = SF.StructureFunctionSumsAndCounts
const N = 4000

"""`got` equals `ref` to 1e-10 of its largest finite value, `NaN` where `ref` is, on the device when `on_device`."""
function check(got, ref; on_device = true)
    g = Array(got)
    nan = isnan.(ref)
    Test.@test isequal(isnan.(g), nan)
    Test.@test maximum(abs.(ifelse.(nan, 0.0, g .- ref))) / (maximum(abs, filter(!isnan, ref)) + eps()) < 1e-10
    on_device && Test.@test got isa CUDA.CuArray
end

# Post-processing a device-resident result equals the same function of the result copied to the host.
Test.@testset "post-processing device results" begin
    Random.seed!(20261003)
    x = CUDA.CuArray(rand(2, N))
    u = CUDA.CuArray(randn(2, N) .+ 0.3 .* sin.(4 .* rand(2, N)))
    bins = collect(range(0.0, 0.8; length = 33))
    Ks = [1.5, 4.0, 9.0, 17.0]
    result(op) = SFC.calculate_structure_function(op, x, u, bins, OUT; backend = DEV)
    s2, s3, l3, l2, t2 = result.((SFT.S2SFType(), SFT.S3SFType(), SFT.L3SFType(), SFT.L2SFType(), SFT.T2SFType()))
    host(r) = SF.to_host(r)
    Test.@test s2.sums isa CUDA.CuArray
    variance = let uh = Array(u)
        sum(abs2, uh .- sum(uh; dims = 2) ./ N) / (N - 1)
    end

    Test.@testset "isotropic spectrum" begin
        check(SFC.isotropic_spectrum(s2, Ks, Val(2); variance), SFC.isotropic_spectrum(host(s2), Ks, Val(2); variance))
    end
    Test.@testset "covariance" begin
        check(last(SFC.covariance(s2, variance)), last(SFC.covariance(host(s2), variance)))
    end
    Test.@testset "spectral flux from S3" begin
        check(SFC.spectral_flux(s3, Ks), SFC.spectral_flux(host(s3), Ks))
    end
    Test.@testset "spectral flux from L3 and S3" begin
        check(SFC.spectral_flux(l3, s3, Ks), SFC.spectral_flux(host(l3), host(s3), Ks))
    end
    hd = SFC.helmholtz_spectra(l2, t2, Ks; variance)
    hh = SFC.helmholtz_spectra(host(l2), host(t2), Ks; variance)
    Test.@testset "Helmholtz rotational" begin
        check(hd.rotational, hh.rotational)
    end
    Test.@testset "Helmholtz divergent" begin
        check(hd.divergent, hh.divergent)
    end
    Test.@testset "fit of a device result" begin
        k_edges = collect(range(0.5, 20.0; length = 9))
        W = fill(1e-20, length(bins) - 1)
        fd = SFC.fit_spectrum(s2, k_edges, SFC.RegularizedLeastSquares(nothing), Val(2); W)
        fh = SFC.fit_spectrum(host(s2), k_edges, SFC.RegularizedLeastSquares(nothing), Val(2); W)
        check(fd.E, fh.E; on_device = false)
    end
    Test.@testset "independent pair variance" begin
        joint = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, collect(range(-4.0, 4.0; length = 17));
                                                 backend = DEV)
        check(SFC.independent_pair_variance(joint), SFC.independent_pair_variance(host(joint)))
    end
end
