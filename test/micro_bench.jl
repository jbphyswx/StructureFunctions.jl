using StructureFunctions
using StructureFunctions.StructureFunctionTypes
using StructureFunctions.Calculations
using StaticArrays
using BenchmarkTools

function run_bench()
    # Synthetic data: 100 points
    N = 100
    x = (rand(N), rand(N))
    u = (rand(N), rand(N))
    bins = 10
    sf_type = LongitudinalSecondOrderStructureFunction()

    println("Benchmarking calculate_structure_function (100 points)...")
    try
        @btime calculate_structure_function(
            $x,
            $u,
            $bins,
            $sf_type,
            verbose = false,
            show_progress = false,
        )
    catch e
        println("Benchmark failed as expected: ", e)
    end
end

run_bench()
