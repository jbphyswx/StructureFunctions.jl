# The sorted line on real CUDA: a polynomial operator over points on a line through the device sort, the
# centred monomial prefix sums and one work item per point, against the host sorted line; weighted,
# multi-field, a field on a large offset in Float32 into device buffers that accumulate, and the time of a
# million points.
using CUDA: CUDA
using Random: Random
using Printf: Printf
using StructureFunctions: StructureFunctions as SF
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.MultiFields: MultiFields as MF
using ComputationalBackends: ComputationalBackends as CB

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const SER = CB.SerialBackend()
const RAW = SF.StructureFunctionSumsAndCounts

failures = String[]

function compare(name, gs, gc, rs, rc; rtol = 1e-10)
    ds = maximum(abs.(Array(gs) .- rs)) / (maximum(abs, rs) + eps())
    dc = maximum(abs.(float.(Array(gc)) .- float.(rc))) / (maximum(abs, float.(rc)) + eps())
    ok = ds < rtol && dc < rtol
    ok || push!(failures, name)
    Printf.@printf("%-54s Δsum=%.3e Δcount=%.3e  %s\n", name, ds, dc, ok ? "ok" : "FAILED")
    return ok
end

println("device=", CUDA.name(CUDA.device()))
Random.seed!(20260926)
N = 20_000
x = reshape(rand(N) .* 100.0, 1, :)
u = randn(1, N)
θ = randn(N)
w = 0.5 .+ rand(N)
bins = collect(range(0.0, 2.0; length = 33))
nb = length(bins) - 1

for (name, sf) in (("S2", SFT.S2SFType()), ("L3", SFT.L3SFType()), ("L4", SFT.ProjectedStructureFunctionType{4, 0}()))
    ref = SFC.calculate_structure_function(sf, x, u, bins, RAW; backend = SER)
    got = SFC.calculate_structure_function(sf, CUDA.CuArray(x), CUDA.CuArray(u), bins, RAW; backend = DEV)
    compare("line $name", got.sums, got.counts, ref.sums, ref.counts)
    refw = SFC.calculate_structure_function(sf, x, u, bins, Float64, RAW; backend = SER, weights = w)
    gotw = SFC.calculate_structure_function(sf, x, u, bins, Float64, RAW; backend = DEV, weights = w)
    compare("line $name weighted", gotw.sums, gotw.counts, refw.sums, refw.counts)
end

f = MF.Fields(vectors = (u,), scalars = (θ,))
for (name, sf) in (("Mixed{1,0,2}", SFT.MixedSFType{1, 0, 2}()), ("Scalar{3}", SFT.ScalarSFType{3}()))
    ref = SFC.calculate_structure_function(sf, x, f, bins, RAW; backend = SER)
    got = SFC.calculate_structure_function(sf, x, f, bins, RAW; backend = DEV)
    compare("line multi-field $name", got.sums, got.counts, ref.sums, ref.counts)
end

# Float32 values on an offset five decades above their increments, accumulated twice into device buffers,
# against the Float64 answer on the same values.
x32, u32, b32 = Float32.(x), Float32.(1000 .+ 0.01 .* u), Float32.(bins)
s, c = CUDA.zeros(Float32, nb), CUDA.zeros(UInt32, nb)
for _ in 1:2
    SFC.calculate_structure_function!(s, c, SFT.L3SFType(), CUDA.CuArray(x32), CUDA.CuArray(u32), b32; backend = DEV)
end
ref = SFC.calculate_structure_function(SFT.L3SFType(), Float64.(x32), Float64.(u32), Float64.(b32), RAW; backend = SER)
compare("line L3 Float32 on an offset, added twice", s, c, 2 .* ref.sums, 2 .* ref.counts; rtol = 1e-3)

# A million points: the device line against the host line, best of 5 and of 2.
NL = 1_000_000
xl, ul = reshape(rand(NL) .* 1000.0, 1, :), randn(1, NL)
xd, ud = CUDA.CuArray(xl), CUDA.CuArray(ul)
for (name, sf) in (("S2", SFT.S2SFType()), ("L3", SFT.L3SFType()))
    dev_call() = CUDA.@sync SFC.calculate_structure_function(sf, xd, ud, bins, Int64, RAW; backend = DEV)
    dev_call()
    t_dev = minimum(@elapsed(dev_call()) for _ in 1:5)
    SFC.calculate_structure_function(sf, xl[:, 1:1000], ul[:, 1:1000], bins, Int64, RAW; backend = SER)
    t_host = minimum(@elapsed(SFC.calculate_structure_function(sf, xl, ul, bins, Int64, RAW; backend = SER))
                     for _ in 1:2)
    Printf.@printf("line %s N=%d  device %.2f ms  host serial %.2f ms\n", name, NL, 1e3 * t_dev, 1e3 * t_host)
end

if isempty(failures)
    println("\nCUDA SORTED LINE OK")
else
    println("\nFAILED: ", join(failures, ", "))
    exit(1)
end
