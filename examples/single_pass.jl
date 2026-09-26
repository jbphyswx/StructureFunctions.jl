# Share pair geometry across the six isotropic velocity moments.
using Random
using StructureFunctions: Calculations as C, StructureFunctionSumsAndCounts
using ComputationalBackends: SerialBackend
include("resources.jl")

function single_pass_example(; n=ExampleResources.points())
    rng = MersenneTwister(13)
    x, u = rand(rng, 2, n), randn(rng, 2, n)
    bins = range(0.0, 1.5; length=7)
    result = C.calculate_structure_functions_single_pass(x, u, bins, StructureFunctionSumsAndCounts;
        backend=SerialBackend())
    @assert result.S2.sums ≈ result.L2.sums + result.T2.sums
    @assert result.S3.sums ≈ result.L3.sums + result.L1T2.sums
    println("Moments: ", keys(result))
    println("Pairs: ", sum(result.S2.counts))
    return result
end
result = single_pass_example()
