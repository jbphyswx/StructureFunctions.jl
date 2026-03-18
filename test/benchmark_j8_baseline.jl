using BenchmarkTools
using StructureFunctions: StructureFunctions as SF
using StaticArrays: StaticArrays as SA

function run_benchmark()
    N = 2000
    FT = Float64
    x = ([rand(FT) for _ in 1:N], [rand(FT) for _ in 1:N])
    u = ([rand(FT) for _ in 1:N], [rand(FT) for _ in 1:N])
    bins = SA.SVector{10}([(i * 0.1, (i + 1) * 0.1) for i in 0:9])

    println("--- Benchmark: Tuple Variant (N=$N, Bins=10) ---")
    b_tuple = @benchmark SF.calculate_structure_function(
        SF.L2SF,
        $x,
        $u,
        $bins;
        verbose = false,
        show_progress = false,
    )
    display(b_tuple)

    x_arr = rand(FT, 2, N)
    u_arr = rand(FT, 2, N)
    println("\n--- Benchmark: Array Variant (N=$N, Bins=10) ---")
    b_array = @benchmark SF.calculate_structure_function(
        SF.L2SF,
        $x_arr,
        $u_arr,
        $bins;
        verbose = false,
        show_progress = false,
    )
    display(b_array)
end

run_benchmark()
