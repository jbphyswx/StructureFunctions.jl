# Within-job A/B on one device: the non-batch point kernels of `ext/gpu/kernels_1d.jl` and
# `ext/gpu/kernels_1d_single_pass.jl` against the unified `sf_tiled_1d_varying!` reached by the
# same data as a one-slice batch. Both answers and both times, same process, same inputs.
using CUDA, Random, Printf
using StructureFunctions
using StructureFunctions.Calculations: Calculations as SFC
using StructureFunctions.StructureFunctionTypes: StructureFunctionTypes as SFT
using StructureFunctions.StructureFunctionObjects: StructureFunctionObjects as SFO
using ComputationalBackends: ComputationalBackends as CB
using KernelAbstractions: KernelAbstractions as KA

const DEV = CB.GPUBackend(CUDA.CUDABackend())
const REPS = 7

function tmin(f, reps = REPS)
    f(); CUDA.synchronize()
    best = Inf
    for _ in 1:reps
        CUDA.synchronize(); t0 = time(); f(); CUDA.synchronize()
        best = min(best, time() - t0)
    end
    return best
end

println("device=", CUDA.name(CUDA.device()))
@printf("%-6s %-5s %-6s %-5s | %-10s %-10s %-7s | %s\n",
    "regime", "D", "N", "NB", "u32 (s)", "unified", "ratio", "agree")

Random.seed!(11)
for D in (2, 3), N in (20_000, 60_000), nb in (16, 64)
    bins = collect(range(0.0, 1.0; length = nb + 1))
    x = rand(Float64, D, N); u = rand(Float64, D, N)
    ub = reshape(u, D, N, 1)
    f_u32() = SFC.calculate_structure_function(SFT.S2SFType(), x, u, bins, UInt32;
        backend = DEV, verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    f_uni() = SFC.calculate_structure_function(SFT.S2SFType(), x, ub, bins, UInt32;
        backend = DEV, verbose = false, output_type = SFO.StructureFunctionSumsAndCounts)
    a = f_u32(); b = f_uni()
    bs = reshape(Array(b.sums), :); bc = reshape(Array(b.counts), :)
    as = Array(a.sums); ac = Array(a.counts)
    scale = maximum(abs, as) + eps()
    agree = (ac == bc) && maximum(abs.(as .- bs)) / scale < 1e-12
    t1 = tmin(f_u32); t2 = tmin(f_uni)
    @printf("%-6s %-5d %-6d %-5d | %-10.5f %-10.5f %-7.3f | %s\n",
        "indiv", D, N, nb, t1, t2, t2 / t1, agree ? "yes" : "NO")
end

Random.seed!(12)
for D in (2, 3), N in (20_000, 60_000), nb in (16, 64)
    bins = collect(range(0.0, 1.0; length = nb + 1))
    x = rand(Float64, D, N); u = rand(Float64, D, N)
    ub = reshape(u, D, N, 1)
    f_u32() = SFC.calculate_structure_functions_single_pass(x, u, bins; backend = DEV, verbose = false)
    f_uni() = SFC.calculate_structure_functions_single_pass(x, ub, bins; backend = DEV, verbose = false)
    a = f_u32(); b = f_uni()
    as = Array(a.S2.sums); bs = reshape(Array(b.S2.sums), :)
    ac = Array(a.S2.counts); bc = reshape(Array(b.S2.counts), :)
    scale = maximum(abs, as) + eps()
    agree = (ac == bc) && maximum(abs.(as .- bs)) / scale < 1e-12
    t1 = tmin(f_u32); t2 = tmin(f_uni)
    @printf("%-6s %-5d %-6d %-5d | %-10.5f %-10.5f %-7.3f | %s\n",
        "sp6", D, N, nb, t1, t2, t2 / t1, agree ? "yes" : "NO")
end

println("UNIFIED VS U32 A/B DONE")
