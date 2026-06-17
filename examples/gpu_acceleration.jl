"""
    gpu_acceleration.jl

Single-snapshot GPU structure function with optional `GPUSFWorkspace`.

Run from package root:
    julia --project=examples examples/gpu_acceleration.jl

With CUDA (recommended on GPU allocation):
    julia --project=gpu -e 'include("examples/gpu_acceleration.jl")'
"""

using StructureFunctions: StructureFunctions as SF, Calculations as SFC
using KernelAbstractions: KernelAbstractions as KA

using Random: Random

use_cuda = false
CUDA_mod = nothing
try
    @eval using CUDA: CUDA
    use_cuda = CUDA.functional()
    CUDA_mod = CUDA
catch
    use_cuda = false
end

const N = 2_000
const FT = Float32
backend = use_cuda ? CUDA_mod.CUDABackend() : KA.CPU()

Random.seed!(42)
x_cpu = rand(FT, 3, N)
u_cpu = rand(FT, 3, N)
x = use_cuda ? CUDA_mod.cu(x_cpu) : x_cpu
u = use_cuda ? CUDA_mod.cu(u_cpu) : u_cpu

bins = collect(FT, range(0.0f0, 1.5f0; length = 21))
sft = SF.LongitudinalSecondOrderStructureFunctionType()

println("Backend: ", typeof(backend))
println("N = $N  bins = $(length(bins) - 1)")

ws = SFC.GPUSFWorkspace(backend, bins)

result_fresh = @time SFC.gpu_calculate_structure_function(
    sft, backend, x, u, bins; return_sums_and_counts = true,
)
result_ws = @time SFC.gpu_calculate_structure_function(
    sft, backend, x, u, bins; return_sums_and_counts = true, workspace = ws,
)

println("Counts match (fresh vs workspace): ", result_fresh.counts == result_ws.counts)
println("Total pairs (approx): ", sum(result_ws.counts))
println("Mode: ", use_cuda ? "CUDA" : "KA.CPU() smoke")

SFC.release!(ws)
