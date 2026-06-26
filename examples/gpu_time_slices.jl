"""
    gpu_time_slices.jl

Time-slice batch: one device upload vs naive per-slice host loop (small demo).

Run from package root:
    julia --project=examples examples/gpu_time_slices.jl
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

const N = 200
const T = 4
const FT = Float32
Random.seed!(42)
backend = use_cuda ? CUDA_mod.CUDABackend() : KA.CPU()

bins = collect(FT, range(0.0f0, 1.0f0; length = 11))
NB = length(bins) - 1
sft = SF.LongitudinalSecondOrderStructureFunctionType()

x_batch = rand(FT, 3, N, T)
u_batch = rand(FT, 3, N, T)
x_dev = use_cuda ? CUDA_mod.cu(x_batch) : x_batch
u_dev = use_cuda ? CUDA_mod.cu(u_batch) : u_batch

sums_slice = zeros(FT, NB, T)
counts_slice = zeros(UInt32, NB, T)
ws = SFC.GPUSFWorkspace(backend, bins)

@time SFC.gpu_calculate_structure_function_batch!(
    sums_slice, counts_slice, sft, backend, x_dev, u_dev, bins; workspace = ws,
)

# Reference: one slice via serial CPU. Inputs are plain (D, N) matrices (tuple/SVector inputs
# are no longer accepted); pass output_type=StructureFunctionSumsAndCounts for the raw histogram.
x_t = x_batch[:, :, 1]
u_t = u_batch[:, :, 1]
ref = SFC.calculate_structure_function(
    sft, x_t, u_t, bins;
    verbose = false, show_progress = false, output_type = SF.StructureFunctionSumsAndCounts,
)

println("Slice 1 counts match serial CPU: ", counts_slice[:, 1] == ref.counts)
println("Backend: ", typeof(backend), "  N=$N  T=$T")
println("Use calculate_structure_function_batch! for production time series.")

SFC.release!(ws)
