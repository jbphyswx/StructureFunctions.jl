"""
    gpu_acceleration.jl

GPU backend example with strict qualified imports.

Run from package root:
    julia --project=examples examples/gpu_acceleration.jl
"""

using StructureFunctions: StructureFunctions as SF

use_cuda = false
CUDA_mod = nothing

try
    @eval using CUDA: CUDA
    use_cuda = CUDA.functional()
    CUDA_mod = CUDA
catch
    use_cuda = false
end

N = use_cuda ? 500_000 : 50_000
x_cpu = randn(Float32, N, 2)
u_cpu = randn(Float32, N, 2)

x = use_cuda ? CUDA_mod.cu(x_cpu) : x_cpu
u = use_cuda ? CUDA_mod.cu(u_cpu) : u_cpu

operator = SF.FullVectorStructureFunctionType{Float32}(order = 2)
bins = collect(Float32, 10.0:25.0:2000.0)

backend = SF.GPUBackend()

result = @time SF.calculate_structure_function(
    operator,
    x,
    u,
    bins;
    backend = backend,
    show_progress = true,
    verbose = true,
)

println("Computed bins: $(length(result.distance))")
println("First SF value: $(result.structure_function[1, 1])")
println("Mode: $(use_cuda ? "CUDA" : "CPU fallback")")
