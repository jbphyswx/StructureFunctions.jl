# Pseudo spherical-harmonic coefficients on real CUDA. Each thread carries its own Wigner column
# out of a global scratch matrix and accumulates with atomics on the real and imaginary parts, so
# a device compile is what establishes that the recurrence and the scratch view build at all.
using CUDA, Random, Printf
using StructureFunctions
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()

failures = String[]

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260918)
const N, LMAX = 4000, 16
const θ = acos.(clamp.(range(-0.98, 0.98; length = N), -1, 1))
const φ = [2π * (i * 0.6180339887498949 % 1) for i in 1:N]

for s in (0, 1, 2)
    f = s == 0 ? randn(N) : complex.(randn(N), randn(N))
    ref = SFC.pseudo_coefficients_direct(f, θ, φ, s, LMAX; backend = SER)
    got = SFC.pseudo_coefficients_direct(f, θ, φ, s, LMAX; backend = DEV)
    d = maximum(abs.(got .- ref)) / (maximum(abs, ref) + eps())
    ok = d < 1e-10
    ok || push!(failures, "pseudo-coefficients spin=$s")
    @printf("%-46s rel=%.3e  %s\n", "pseudo-coefficients spin=$s", d, ok ? "ok" : "FAILED")
end

# and the harmonic route end to end, which is the capability-matrix cell
let
    op = SFT.L2SFType()
    hx = permutedims(hcat(collect(φ), π / 2 .- collect(θ)))
    hu = Float64[sin(d + 2i) for d in 1:2, i in 1:N]
    nodes = StructureFunctions.HarmonicNodes(collect(range(0.2, 2.6; length = 9)), LMAX)
    raw = StructureFunctions.StructureFunctionSumsAndCounts
    ref = SFC.calculate_structure_function(op, hx, hu, nodes, SB.DirectSumSpectralBackend();
        backend = SER, verbose = false, output_type = raw)
    got = SFC.calculate_structure_function(op, hx, hu, nodes, SB.DirectSumSpectralBackend();
        backend = DEV, verbose = false, output_type = raw)
    ds = maximum(abs.(got.sums .- ref.sums)) / (maximum(abs, ref.sums) + eps())
    dc = maximum(abs.(got.counts .- ref.counts)) / (maximum(abs, ref.counts) + eps())
    ok = ds < 1e-10 && dc < 1e-10
    ok || push!(failures, "harmonic route")
    @printf("%-46s Δsum=%.3e Δcount=%.3e  %s\n", "harmonic route end to end", ds, dc,
            ok ? "ok" : "FAILED")
end

if isempty(failures)
    println("\nCUDA HARMONIC OK")
else
    println("\nCUDA HARMONIC FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
