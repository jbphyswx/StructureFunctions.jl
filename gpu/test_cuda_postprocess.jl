# Post-processing a device-resident result on CUDA: spectra, covariance, fluxes, the Helmholtz split, a fit and the
# independent-pair variance of a joint histogram, each against the same function of the result copied to the host.
using CUDA: CUDA
using Bessels: Bessels
using Random: Random
using Printf: Printf
using StructureFunctions: StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const OUT = SF.StructureFunctionSumsAndCounts

failures = String[]

function check(name, got, ref; on_device = true)
    g = Array(got)
    nan = isnan.(ref)
    d = isequal(isnan.(g), nan) ? maximum(abs.(ifelse.(nan, 0.0, g .- ref))) / (maximum(abs, filter(!isnan, ref)) + eps()) :
        Inf
    ok = d < 1e-10 && (!on_device || got isa CUDA.CuArray)
    ok || push!(failures, name)
    Printf.@printf("%-48s Δ=%.3e %-8s %s\n", name, d, nameof(typeof(got)), ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20261003)
const N = 4000
x = CUDA.CuArray(rand(2, N))
u = CUDA.CuArray(randn(2, N) .+ 0.3 .* sin.(4 .* rand(2, N)))
bins = collect(range(0.0, 0.8; length = 33))
Ks = [1.5, 4.0, 9.0, 17.0]

result(op) = SFC.calculate_structure_function(op, x, u, bins, OUT; backend = DEV)
s2, s3, l3, l2, t2 = result.((SFT.S2SFType(), SFT.S3SFType(), SFT.L3SFType(), SFT.L2SFType(), SFT.T2SFType()))
host(r) = SF.to_host(r)
s2.sums isa CUDA.CuArray || push!(failures, "the public result is not device-resident")

check("isotropic spectrum", SFC.isotropic_spectrum(s2, Ks, Val(2)), SFC.isotropic_spectrum(host(s2), Ks, Val(2)))
check("covariance", last(SFC.covariance(s2, 2.0)), last(SFC.covariance(host(s2), 2.0)))
check("spectral flux from S3", SFC.spectral_flux(s3, Ks), SFC.spectral_flux(host(s3), Ks))
check("spectral flux from L3 and S3", SFC.spectral_flux(l3, s3, Ks), SFC.spectral_flux(host(l3), host(s3), Ks))
let d = SFC.helmholtz_spectra(l2, t2, Ks), h = SFC.helmholtz_spectra(host(l2), host(t2), Ks)
    check("Helmholtz rotational", d.rotational, h.rotational)
    check("Helmholtz divergent", d.divergent, h.divergent)
end

k_edges = collect(range(0.5, 20.0; length = 9))
W = fill(1e-20, length(bins) - 1)
let d = SFC.fit_spectrum(s2, k_edges, SFC.RegularizedLeastSquares(nothing), Val(2); W),
    h = SFC.fit_spectrum(host(s2), k_edges, SFC.RegularizedLeastSquares(nothing), Val(2); W)
    check("fit of a device result", d.E, h.E; on_device = false)
end

vbins = collect(range(-4.0, 4.0; length = 17))
joint = SFC.calculate_structure_function(SFT.L2SFType(), x, u, bins, vbins; backend = DEV)
check("independent pair variance", SFC.independent_pair_variance(joint), SFC.independent_pair_variance(host(joint)))

if isempty(failures)
    println("\nCUDA POSTPROCESS OK")
else
    println("\nFAILED: ", join(failures, ", "))
    exit(1)
end
