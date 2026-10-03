# Second-order vector increments on a small point cloud.
using Random: Random
using StructureFunctions: Calculations as C, StructureFunctionTypes as T, midpoints
using ComputationalBackends: SerialBackend
include("resources.jl")

function simple_example(; n=ExampleResources.points())
    rng = Random.MersenneTwister(11)
    x = Random.rand(rng, 2, n)       # (coordinate, point), in metres
    u = Random.randn(rng, 2, n)      # (component, point), in metres per second
    bins = range(0.0, 1.5; length=7)
    result = C.calculate_structure_function(T.S2SFType(), x, u, bins;
        backend=SerialBackend())
    println("Separation [m]: ", collect(midpoints(result.distance)))
    println("S2 [m²/s²]: ", result.values)
    return result
end
result = simple_example()
