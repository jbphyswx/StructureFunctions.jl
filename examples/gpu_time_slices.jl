# Calculate several snapshots with one upload of each input array. Run inside a GPU allocation.
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions
using Random: Random
using ComputationalBackends: GPUBackend, SerialBackend
using StructureFunctions: Calculations as C, StructureFunctionTypes as T, StructureFunctionSumsAndCounts
CUDA.functional() || error("This example requires a functioning allocated CUDA device")
CUDA.allowscalar(false)

function gpu_batch_example(; n=5000, snapshots=3)
    rng = Random.MersenneTwister(16)
    x, u = Random.rand(rng, Float32, 3, n, snapshots), Random.randn(rng, Float32, 3, n, snapshots)
    bins = range(0.0f0, 2.0f0; length=9)
    result = C.calculate_structure_function(T.L2SFType(), CUDA.CuArray(x), CUDA.CuArray(u), bins,
        StructureFunctionSumsAndCounts; backend=GPUBackend(CUDA.CUDABackend()))
    sums, counts = Array(result.sums), Array(result.counts)
    for t in 1:snapshots
        reference = C.calculate_structure_function(T.L2SFType(), x[:, :, t], u[:, :, t], bins,
            StructureFunctionSumsAndCounts; backend=SerialBackend())
        @assert counts[:, t] == reference.counts
        @assert isapprox(sums[:, t], reference.sums; rtol=2f-5)
    end
    println("Matched ", snapshots, " snapshots; output shape: ", size(sums))
    return result
end
result = gpu_batch_example()
