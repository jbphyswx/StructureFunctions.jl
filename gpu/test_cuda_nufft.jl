# The non-uniform FFT route on real CUDA through the public entries: both providers against the host, the result
# device-resident, the plan each provider runs recorded, one plan kept for every slice of a batch on fixed points,
# and the time of a batch with its plan kept against one built per call.
using CUDA: CUDA
using Random: Random
using Printf: Printf
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const RAW = SF.StructureFunctionSumsAndCounts

failures = String[]
check(name, ok, detail) = (ok || push!(failures, name);
                           Printf.@printf("%-60s %s  %s\n", name, detail, ok ? "ok" : "FAILED"))
rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / (maximum(abs, Array(b)) + eps())

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260926)
N, nt = 20_000, 8
s = SFC.ScatteredModesSchedule(rand(2, N), 0.3, (256, 192); taper = SF.GaussianTaper(0.01))
bins = collect(range(0.0, 0.3; length = 9))
nb = length(bins) - 1
u = randn(2, N)
ub = randn(2, N, nt)

for tag in (SFC.NonuniformFFTsSpectralBackend(), SFC.FINUFFTSpectralBackend())
    name = nameof(typeof(tag))
    for sf in (SFT.L2SFType(), SFT.L3SFType())
        ref = SFC.calculate_structure_function(sf, s, u, bins, tag, RAW; backend = SER)
        got = SFC.calculate_structure_function(sf, s, u, bins, tag, RAW; backend = DEV)
        check("$name $(nameof(typeof(sf))) public entry on the device",
              got.sums isa CUDA.CuArray && rel(got.sums, ref.sums) < 1e-9 && rel(got.counts, ref.counts) < 1e-9,
              Printf.@sprintf("Δsum=%.3e Δcount=%.3e", rel(got.sums, ref.sums), rel(got.counts, ref.counts)))
    end
    rs, rc = zeros(nb, nt), zeros(nb, nt)
    SFC.calculate_structure_function_batch!(rs, rc, SFT.L2SFType(), s, ub, bins, tag; backend = SER)
    ws = SFC.TransformWorkspace()
    batch() = begin
        gs, gc = CUDA.zeros(Float64, nb, nt), CUDA.zeros(Float64, nb, nt)
        CUDA.@sync SFC.calculate_structure_function_batch!(gs, gc, SFT.L2SFType(), s, ub, bins, tag; backend = DEV,
                                                           workspace = ws)
        gs, gc
    end
    gs, gc = batch()
    plans = [last(e) for e in ws.pool if first(first(e)) in (:nonuniformffts, :finufft)]
    println("$name device plan: ", isempty(plans) ? "none" : typeof(first(plans).plan))
    check("$name batch on the device with one kept plan",
          length(plans) == 1 && rel(gs, rs) < 1e-9 && rel(gc, rc) < 1e-9,
          Printf.@sprintf("plans=%d Δsum=%.3e", length(plans), rel(gs, rs)))
    t_kept = minimum(@elapsed(batch()) for _ in 1:3)
    fresh() = begin
        gs, gc = CUDA.zeros(Float64, nb, nt), CUDA.zeros(Float64, nb, nt)
        CUDA.@sync SFC.calculate_structure_function_batch!(gs, gc, SFT.L2SFType(), s, ub, bins, tag; backend = DEV)
    end
    fresh()
    t_fresh = minimum(@elapsed(fresh()) for _ in 1:3)
    Printf.@printf("%s batch N=%d x %d  plan kept %.2f ms  plan per call %.2f ms\n", name, N, nt, 1e3 * t_kept,
                   1e3 * t_fresh)
    SFC._release_plans!(ws)
end

if isempty(failures)
    println("\nCUDA NUFFT OK")
else
    println("\nCUDA NUFFT FAILED in $(length(failures)) case(s): ", join(failures, ", "))
    exit(1)
end
