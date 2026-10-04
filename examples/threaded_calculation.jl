# Compare serial and threaded reductions of the same pairs. Start Julia with `--threads=N`.
using Random: Random
using OhMyThreads: OhMyThreads
using ComputationalBackends: SerialBackend, ThreadedBackend
using StructureFunctions: Calculations as C, StructureFunctionTypes as T, StructureFunctionSumsAndCounts

function threaded_example(; n=10_000)
    rng = Random.MersenneTwister(12)
    x, u = Random.rand(rng, 2, n), Random.randn(rng, 2, n)
    bins = range(0.0, 1.5; length=7)
    calculate(backend) = C.calculate_structure_function(T.S2SFType(), x, u, bins,
        StructureFunctionSumsAndCounts; backend)
    reference = calculate(SerialBackend())
    result = calculate(ThreadedBackend())
    @assert result.counts == reference.counts
    @assert result.sums ≈ reference.sums
    println("Matched ", sum(result.counts), " pairs using ", Threads.nthreads(), " Julia threads")
    return result
end
result = threaded_example()
